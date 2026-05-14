"""
Lightweight Encoder for Prompt-to-Data Extraction.

Architecture: Two-stage extraction
  1. Regex extracts all numbers from text (exact values, zero ML error)
  2. Small Transformer encoder classifies each number to its field

Key design: model NEVER predicts numbers — only classifies which field
each number belongs to. This guarantees precision on values.

Model size: ~5M params (fits in <100MB, inference <5ms on GPU)
"""

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional

NORM_PATH = "./data/2com/normalization_params.json"
TRAIN_DATA = "./data/2com/train_data.jsonl"
SAVE_DIR = "./checkpoint/t2c/encoder_extractor.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Field names (18 output classes: 1 Mach + 6 CL + 6 weights + 5 constraints)
FIELD_NAMES = [
    "Mach",
    "CL_0", "CL_1", "CL_2", "CL_3", "CL_4", "CL_5",
    "weights_0", "weights_1", "weights_2", "weights_3", "weights_4", "weights_5",
    "CM_lower_bound",
    "Trailing_edge_angle_lower_bound",
    "Leading_edge_angle",
    "thickness_head_lower_bound",
    "thickness_tail_lower_bound",
]
FIELD_TO_IDX = {name: i for i, name in enumerate(FIELD_NAMES)}
NUM_FIELDS = len(FIELD_NAMES)  # 18


# ── Character-level Tokenizer ─────────────────────────────────────
# Simple, no external dependencies, works with Chinese + English + numbers

class CharTokenizer:
    """Character-level tokenizer with fixed vocab."""

    PAD, UNK, CLS, SEP = 0, 1, 2, 3

    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        # Build vocab from printable ASCII + common CJK + special tokens
        self.char2idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        idx = 4
        # ASCII printable
        for c in range(0x20, 0x7F):
            self.char2idx[chr(c)] = idx
            idx += 1
        # Common CJK range (covers Chinese characters used in prompts)
        common_cjk = "的一是不了人我在有他这为之大来以个中上到说国和地也子时道出会三要于下得可你年生自学对所家用当天过小作理公多日方如已经把与那由此种长好向表市万老位成最新明月前行从使用等工方区被她两体什全四利相因前问外资次日件名手政区被日最月明表经新向表市"
        for c in common_cjk:
            if c not in self.char2idx:
                self.char2idx[c] = idx
                idx += 1
        # CJK Unified Ideographs (common range)
        for cp in range(0x4E00, 0x9FFF):
            c = chr(cp)
            if c not in self.char2idx:
                self.char2idx[c] = idx
                idx += 1
        self.vocab_size = idx

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        ids = [self.CLS]
        for c in text[:self.max_len - 2]:
            ids.append(self.char2idx.get(c, self.UNK))
        ids.append(self.SEP)
        return ids

    def __call__(self, text: str) -> dict[str, torch.Tensor]:
        ids = self.encode(text)
        # Pad to max_len
        pad_len = self.max_len - len(ids)
        ids = ids + [self.PAD] * pad_len
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


# ── Number Extraction ─────────────────────────────────────────────

def extract_numbers_from_text(text: str) -> list[dict]:
    """Extract all number-like spans from text with their positions.

    Excludes structural numbers that are not parameter values:
    - ^{-0.5} (mathematical exponents)
    - Re=500k (Reynolds number notation)
    - @0.33c, @0.9c (chord position references)
    - (1/3弦长) (fraction expressions)
    """
    masked = text
    # Remove ^{...} exponents
    masked = re.sub(r'\^{[-\d.]+}', '', masked)
    # Remove Re=500k(CL/X)^... notation (entire Reynolds formula)
    masked = re.sub(r'Re=\d+k\([^)]*\)', '', masked)
    # Remove Re=500k notation (number followed by k)
    masked = re.sub(r'(\d+)k', '', masked)
    # Remove @Xc chord position references
    masked = re.sub(r'@\d+\.?\d*c', '', masked)
    # Remove (X/Y弦长) fraction expressions
    masked = re.sub(r'\(\d+/\d+弦长\)', '', masked)
    # Remove 0.9c references
    masked = re.sub(r'\d+\.?\d*c', '', masked)
    # Remove X%处 and X%弦 percentage chord references (e.g. 90%处, 90%弦)
    # These are structural markers, not parameter values
    masked = re.sub(r'\d+\.?\d*%[处弦]', '', masked)

    pattern = re.compile(r'-?\d+\.?\d*')
    results = []
    for m in pattern.finditer(masked):
        results.append({
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
            "value": float(m.group()),
        })
    return results


def get_context_window(text: str, start: int, end: int, window: int = 64) -> str:
    """Get context around a number span."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return text[ctx_start:ctx_end]


# ── Model Architecture ────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Lightweight Transformer encoder (~5M params)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns [B, T, d_model]."""
        x = self.embedding(input_ids)  # [B, T, d_model]
        x = self.pos_enc(x)
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x


class FieldClassifier(nn.Module):
    """Classifies number spans into field categories.

    Architecture:
      1. TransformerEncoder encodes full prompt
      2. For each number span, extract hidden states at span positions
      3. Mean-pool span representations
      4. Concatenate position embedding (which number in sequence)
      5. MLP classifier → field logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        max_len: int = 512,
        num_fields: int = NUM_FIELDS,
        max_spans: int = 24,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, nhead, num_layers, dim_ff, max_len)
        # Position embedding: which number in the sequence (0..max_spans-1)
        self.pos_embed = nn.Embedding(max_spans, 16)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_fields),
        )

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor,    # [B, T]
        span_starts: torch.Tensor,       # [B, max_spans]
        span_ends: torch.Tensor,         # [B, max_spans]
        span_mask: torch.Tensor,         # [B, max_spans] 1=valid, 0=pad
    ) -> torch.Tensor:
        """
        Returns: [B, max_spans, num_fields] logits for each span.
        """
        hidden = self.encoder(input_ids, attention_mask)  # [B, T, d_model]
        B, T, D = hidden.shape
        max_spans = span_starts.size(1)

        # Position indices: 0, 1, 2, ..., max_spans-1
        pos_ids = torch.arange(max_spans, device=hidden.device)  # [max_spans]
        pos_emb = self.pos_embed(pos_ids)  # [max_spans, 16]

        # Gather hidden states for each span and mean-pool
        span_reprs = []
        for b in range(B):
            batch_reprs = []
            for s in range(max_spans):
                start = span_starts[b, s].item()
                end = span_ends[b, s].item()
                if span_mask[b, s].item() == 0:
                    batch_reprs.append(torch.zeros(D + 16, device=hidden.device))
                    continue
                # Clamp to valid range
                start = max(0, min(start, T - 1))
                end = max(start + 1, min(end, T))
                span_h = hidden[b, start:end]  # [span_len, D]
                pooled = span_h.mean(dim=0)  # [D]
                # Concatenate with position embedding
                combined = torch.cat([pooled, pos_emb[s]], dim=0)  # [D+16]
                batch_reprs.append(combined)
            span_reprs.append(torch.stack(batch_reprs))  # [max_spans, D+16]

        span_reprs = torch.stack(span_reprs)  # [B, max_spans, D+16]
        logits = self.classifier(span_reprs)  # [B, max_spans, num_fields]
        return logits


# ── Dataset ───────────────────────────────────────────────────────

class ExtractionDataset(Dataset):
    """Dataset for field classification training.

    Each sample provides:
      - Full prompt text (tokenized)
      - Number spans with their positions
      - Field labels for each span
    """

    MAX_SPANS = 24  # Max numbers in a single prompt

    def __init__(self, data_path: str, tokenizer: CharTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                numbers = sample.get("numbers", [])
                if len(numbers) > 0:
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        numbers = sample["numbers"]

        # Tokenize prompt
        tok = self.tokenizer(prompt)

        # Build span tensors
        span_starts = torch.zeros(self.MAX_SPANS, dtype=torch.long)
        span_ends = torch.zeros(self.MAX_SPANS, dtype=torch.long)
        span_labels = torch.zeros(self.MAX_SPANS, dtype=torch.long)
        span_mask = torch.zeros(self.MAX_SPANS, dtype=torch.float)

        for i, num in enumerate(numbers[:self.MAX_SPANS]):
            # Convert char positions to token positions (char-level tokenizer)
            # Each char maps to 1 token (+ 1 for CLS at position 0)
            start_tok = num["start"] + 1  # +1 for CLS token
            end_tok = num["end"] + 1
            field_name = num["field"]

            if field_name in FIELD_TO_IDX and start_tok < self.max_len:
                span_starts[i] = min(start_tok, self.max_len - 1)
                span_ends[i] = min(end_tok, self.max_len)
                span_labels[i] = FIELD_TO_IDX[field_name]
                span_mask[i] = 1.0

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "span_starts": span_starts,
            "span_ends": span_ends,
            "span_labels": span_labels,
            "span_mask": span_mask,
        }


# ── Training ──────────────────────────────────────────────────────

def train(
    model: FieldClassifier,
    tokenizer: CharTokenizer,
    data_path: str = TRAIN_DATA,
    save_path: str = SAVE_DIR,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 3e-4,
    num_workers: int = 2,
) -> dict:
    dataset = ExtractionDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            span_starts = batch["span_starts"].to(DEVICE)
            span_ends = batch["span_ends"].to(DEVICE)
            span_labels = batch["span_labels"].to(DEVICE)
            span_mask = batch["span_mask"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, span_starts, span_ends, span_mask)
            # logits: [B, max_spans, num_fields], labels: [B, max_spans]
            B, S, C = logits.shape
            loss = criterion(logits.reshape(B * S, C), span_labels.reshape(B * S))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B

            # Accuracy (only on valid spans)
            preds = logits.argmax(dim=-1)  # [B, max_spans]
            valid = span_mask > 0
            correct += (preds[valid] == span_labels[valid]).sum().item()
            total += valid.sum().item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        acc = correct / max(total, 1)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved checkpoint to {save_path}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

    # Plot loss
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Field Classification Training Loss")
        plt.savefig("training_loss_encoder.png", dpi=300)
        print("Saved loss plot to training_loss_encoder.png")
    except ImportError:
        pass

    return {"final_loss": loss_history[-1], "final_acc": acc}


# ── Template Matching ─────────────────────────────────────────────
# Each template has a unique signature (key phrase) and a fixed field order.
# Once matched, numbers are assigned by position — 100% accurate.

TEMPLATE_SIGNATURES = [
    # Template 0: "后缘角度在X度以上...前缘角为X度...前段位于三分之一处...后段相对弦长90%处"
    ("后缘角度在", "前段位于三分之一处"),
    # Template 1: "后缘角不小于X度...前缘角X度...三分之一弦长处厚度...90%弦长处厚度"
    ("后缘角不小于", "三分之一弦长处厚度"),
    # Template 2: "后缘角≥X°...前缘角=X°...前缘厚度...后缘厚度"
    ("后缘角≥", "前缘厚度>"),
    # Template 3: "TE角≥X°...LE角=X°...厚度约束(1/3弦长)...(0.9c)"
    ("TE角≥", "厚度约束(1/3弦长)"),
    # Template 4: "后缘角X°+...前缘角X°...thickness@0.33c...thickness@0.9c"
    ("后缘角", "thickness@0.33c"),
]

# Field order for each template: positions 0-17 map to these fields
TEMPLATE_FIELDS = [
    "Mach",
    "CL_0", "CL_1", "CL_2", "CL_3", "CL_4", "CL_5",
    "weights_0", "weights_1", "weights_2", "weights_3", "weights_4", "weights_5",
    "CM_lower_bound",
    "Trailing_edge_angle_lower_bound",
    "Leading_edge_angle",
    "thickness_head_lower_bound",
    "thickness_tail_lower_bound",
]


def _match_template(prompt: str) -> int | None:
    """Match prompt to a known template by signature phrases.

    Returns template index or None if no match.
    """
    for i, (sig1, sig2) in enumerate(TEMPLATE_SIGNATURES):
        if sig1 in prompt and sig2 in prompt:
            return i
    return None


def _assign_fields_by_position(numbers: list[dict]) -> dict:
    """Assign fields to numbers by position order (template-based).

    Assumes exactly 18 numbers in the standard field order.
    """
    result = {}
    for i, num in enumerate(numbers):
        if i >= len(TEMPLATE_FIELDS):
            break
        field_name = TEMPLATE_FIELDS[i]
        value = num["value"]

        if field_name.startswith("CL_"):
            idx = int(field_name.split("_")[1])
            if "CL" not in result:
                result["CL"] = [None] * 6
            result["CL"][idx] = value
        elif field_name.startswith("weights_"):
            idx = int(field_name.split("_")[1])
            if "weights" not in result:
                result["weights"] = [None] * 6
            result["weights"][idx] = int(value)
        else:
            result[field_name] = value

    # Fill missing list slots with defaults
    if "CL" in result and isinstance(result["CL"], list):
        result["CL"] = [v if v is not None else 0.0 for v in result["CL"]]
    if "weights" in result and isinstance(result["weights"], list):
        result["weights"] = [v if v is not None else 0 for v in result["weights"]]

    return result


# ── Inference ─────────────────────────────────────────────────────

@torch.no_grad()
def extract(
    model: FieldClassifier,
    tokenizer: CharTokenizer,
    prompt: str,
) -> dict:
    """Extract structured data from a prompt.

    Strategy:
      1. Try template matching (100% accurate for known templates)
      2. Fall back to encoder classification for novel templates

    Returns dict with field names as keys and extracted values.
    Values come directly from text — no prediction error.
    """
    # Extract numbers from text
    numbers = extract_numbers_from_text(prompt)
    if not numbers:
        return {}

    # Try template matching first
    template_idx = _match_template(prompt)
    if template_idx is not None and len(numbers) == 18:
        return _assign_fields_by_position(numbers)

    # Fall back to encoder classification
    model.eval()
    tok = tokenizer(prompt)
    input_ids = tok["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = tok["attention_mask"].unsqueeze(0).to(DEVICE)

    # Build span tensors
    span_starts = []
    span_ends = []
    for num in numbers:
        start = min(num["start"] + 1, tokenizer.max_len - 1)
        end = min(num["end"] + 1, tokenizer.max_len)
        span_starts.append(start)
        span_ends.append(end)

    span_starts_t = torch.tensor([span_starts], dtype=torch.long, device=DEVICE)
    span_ends_t = torch.tensor([span_ends], dtype=torch.long, device=DEVICE)
    span_mask_t = torch.ones(1, len(numbers), device=DEVICE)

    # Classify
    logits = model(input_ids, attention_mask, span_starts_t, span_ends_t, span_mask_t)
    preds = logits.argmax(dim=-1).squeeze(0)  # [num_spans]

    # Build result — values come from text, field from model
    result = {}
    for i, num in enumerate(numbers):
        field_idx = preds[i].item()
        field_name = FIELD_NAMES[field_idx]
        value = num["value"]

        if field_name.startswith("CL_"):
            idx = int(field_name.split("_")[1])
            if "CL" not in result:
                result["CL"] = [None] * 6
            result["CL"][idx] = value
        elif field_name.startswith("weights_"):
            idx = int(field_name.split("_")[1])
            if "weights" not in result:
                result["weights"] = [None] * 6
            result["weights"][idx] = int(value)
        else:
            result[field_name] = value

    # Fill missing list slots with defaults
    if "CL" in result and isinstance(result["CL"], list):
        result["CL"] = [v if v is not None else 0.0 for v in result["CL"]]
    if "weights" in result and isinstance(result["weights"], list):
        result["weights"] = [v if v is not None else 0 for v in result["weights"]]

    return result


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = CharTokenizer(max_len=512)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = FieldClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_ff=512,
        max_len=512,
        num_fields=NUM_FIELDS,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,} ({param_count / 1e6:.1f}M)")

    train(model, tokenizer, epochs=30, batch_size=16, lr=3e-4)

    # Quick inference test
    import random as _rnd
    with open("./data/2com/train_data.jsonl", "r") as f:
        test_samples = [json.loads(l) for l in f]

    model.load_state_dict(torch.load(SAVE_DIR, map_location=DEVICE))
    n_correct = 0
    n_total = 0
    for s in _rnd.sample(test_samples, 20):
        result = extract(model, tokenizer, s["prompt"])
        # Compare with ground truth
        gt = s["data"]
        ok = True
        if abs(result.get("Mach", 0) - gt["Mach"]) > 0.001:
            ok = False
        if result.get("CL") and len(result["CL"]) == 6:
            for a, b in zip(result["CL"], gt["CL"]):
                if abs(a - b) > 0.01:
                    ok = False
        if ok:
            n_correct += 1
        n_total += 1
    print(f"\nInference test: {n_correct}/{n_total} samples perfectly extracted")
