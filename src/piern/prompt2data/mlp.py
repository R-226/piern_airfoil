"""
Seq-Level Router for PiERN Airfoil.

This module implements a sequence-level router that learns to predict
the trigger boundary where LLM should switch from reasoning/explaining
to outputting actual results (e.g., airfoil coordinates).

Architecture Reference:
    LMClassifier1D from capacity_sample_PiERN.py (lines 34-49)

Key Differences from Reference:
    - Reference uses hidden_dim=128, we may need to tune
    - Reference uses embed_dim=1536 (Qwen2.5-0.5B), we need flexible vocab_size
    - Reference is binary classification, we are binary classification (same)

Data Flow:
    Input:  token_ids [B, T] + attention_mask [B, T]
            ↓
    Embed: embedding layer [B, T, E]
            ↓
    Mask:  apply attention_mask [B, T, E]
            ↓
    Pool:  mean pooling [B, E]
            ↓
    MLP:   5-layer MLP [B, 1]
            ↓
    Output: logits [B, 1] → sigmoid → binary decision

Training:
    - Labels: type=1 (trigger boundary) or type=0 (not trigger)
    - Loss: BCEWithLogitsLoss with pos_weight for imbalance
    - Optimizer: AdamW

Trigger Detection:
    The router learns to detect the semantic boundary where:
        - "推理结束标志" + "：" (reasoning ending + colon) = type=1
        - Everything else = type=0

    Examples:
        "分析完成，准备输出结果：" → type=1 (trigger!)
        "分析完成，准备输出结果"   → type=0 (still reasoning)
        "根据伯努利原理：气流..." → type=0 (explanation with colon, not trigger)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import json
import os
import time
from typing import Tuple

from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "./model/Qwen3.5-0.8B"  # Path to base model for embeddings (adjust as needed)

SAVE_DIR = "./checkpoint/t2c/seq_t2c_hidden.pt"
TRAIN_DATA = "./data/2com/train_data.jsonl"
TEST_DATA = "./data/2com/test_data.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SeqRouter(nn.Module):
    """Sequence-level Router for PiERN Airfoil.

    This router takes token sequences and predicts whether the current
    position is a trigger boundary (where Expert should be invoked).

    Architecture:
        1. Token Embedding Layer
        2. Attention Mask Application
        3. Mean Pooling
        4. 5-layer MLP → logits

    Input:
        - input_ids: [batch_size, seq_len] token IDs
        - attention_mask: [batch_size, seq_len] binary mask

    Output:
    """

    def __init__(
        self,
        base_model,
        embed_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        hidden_dim3: int,
        hidden_dim4: int,
        hidden_dim5: int,
    ):
        """Initialize SeqRouter.

        Args:
            vocab_size: Size of vocabulary (from tokenizer)
            embed_dim: Embedding dimension. 
            hidden_dim1: First hidden dimension for the MLP layer.
            hidden_dim2: Second hidden dimension for the MLP layer.
        """
        super().__init__()
        self.model = base_model
        # MLP classifier
        # Input: embedding dimension
        # Hidden: hidden_dim
        # Output: 1 (binary classification)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.ReLU(),
            nn.Linear(hidden_dim5, 18)  # Output logits for binary classification
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, T] token IDs
            attention_mask: [B, T] binary mask (optional)
        Returns:
            Tuple of (logits, probs) where each is [B, 18]
        """
        input_hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state  # [B, T, E]
        # 2. Mean pooling over sequence dimension
        # Sum over sequence: [B, T, E] → [B, E]
        sum_embed = input_hidden.sum(dim=1)
        count = attention_mask.sum(dim=1, keepdim=True) if attention_mask is not None else input_hidden.size(1)
        pooled = sum_embed / count

        return self.fc(pooled.to(torch.float))  # [B, 18]

    def get_data(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)  # [B, 18]
        # Denormalize to get real values
        denormalized = outputs * (SeqRouterDataset.STD.to(outputs.device) + 1e-8) + SeqRouterDataset.MEAN.to(outputs.device)
        dict = {
            "Mach" : denormalized[:, 0],
            "CL" : denormalized[:, 1:7],
            "weights" : denormalized[:, 7:13],
            "CM_lower_bound" : denormalized[:, 13],
            "Trailing_edge_angle_lower_bound" : denormalized[:, 14],
            "Leading_edge_angle" : denormalized[:, 15],
            "thickness_head_lower_bound" : denormalized[:, 16],
            "thickness_tail_lower_bound" : denormalized[:, 17]
        }
        return dict

NORM_PATH = "./data/2com/normalization_params.json"


def _load_normalization_params() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MEAN and STD from normalization_params.json."""
    with open(NORM_PATH, "r") as f:
        p = json.load(f)

    def vec(key: str) -> torch.Tensor:
        return torch.tensor([p[key]["mean"], p[key]["std"]])

    # Build per-element MEAN and STD arrays
    # Order: Mach, CL[6], weights[6], CM, TE, LE, th_h, th_t
    mean_vals = [
        p["Mach"]["mean"],
        *[p["CL"]["mean"]] * 6,
        *[p["weights"]["mean"]] * 6,
        p["CM_lower_bound"]["mean"],
        p["Trailing_edge_angle_lower_bound"]["mean"],
        p["Leading_edge_angle"]["mean"],
        p["thickness_head_lower_bound"]["mean"],
        p["thickness_tail_lower_bound"]["mean"],
    ]
    std_vals = [
        p["Mach"]["std"],
        *[p["CL"]["std"]] * 6,
        *[p["weights"]["std"]] * 6,
        p["CM_lower_bound"]["std"],
        p["Trailing_edge_angle_lower_bound"]["std"],
        p["Leading_edge_angle"]["std"],
        p["thickness_head_lower_bound"]["std"],
        p["thickness_tail_lower_bound"]["std"],
    ]
    return torch.tensor(mean_vals), torch.tensor(std_vals)


_MEAN, _STD = _load_normalization_params()


class SeqRouterDataset(data.Dataset):
    """Custom Dataset for SeqRouter training.

    Expects data in JSONL format with fields:
        - "prompt" : input text
        - "data"   : dict with Mach, CL, weights, CM_lower_bound, ...
    """

    MEAN: torch.Tensor = _MEAN
    STD: torch.Tensor = _STD

    def __init__(self, data_path: str, tokenizer):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL file containing training/test data
            tokenizer: Tokenizer for processing text into token IDs
        """
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input = self.tokenizer(sample['prompt'], return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        input_ids = input['input_ids'].squeeze(0)  # [T]
        attention_mask = input['attention_mask'].squeeze(0)  # [T]
        data_dict = sample['data']
        data_tensor = torch.tensor([
            data_dict["Mach"],
            *data_dict["CL"],
            *data_dict["weights"],
            data_dict["CM_lower_bound"],
            data_dict["Trailing_edge_angle_lower_bound"],
            data_dict["Leading_edge_angle"],
            data_dict["thickness_head_lower_bound"],
            data_dict["thickness_tail_lower_bound"]
        ], dtype=torch.float)  # [18]
        # Normalize to handle multi-scale targets
        normalized = (data_tensor - self.MEAN) / (self.STD + 1e-8)
        return input_ids, attention_mask, normalized

def train_router(
    model: SeqRouter,
    tokenizer: AutoTokenizer,
    save_path: str = "SAVE_DIR",
    train_data_path: str = "TRAIN_DATA",
    test_data_path: str = "TEST_DATA",
    epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 4,
) -> dict:
    """Train the SeqRouter model.

    Args:
        model: SeqRouter model to train
        tokenizer: Tokenizer for processing text
        save_path: Path to save the trained model checkpoint
        train_data_path: Path to training data (jsonl format)
        test_data_path: Path to test data (jsonl format)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        pos_weight: Weight for positive class (type=1). >1.0 if more negatives than positives.

    Returns:
        Dictionary with training metrics
    """
    dataset = SeqRouterDataset(train_data_path, tokenizer)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    criterion = nn.MSELoss()  # Using MSELoss for regression targets (Mach, CL, weights, etc.)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    avg_loss = 0.0
    loss_history = []
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            result = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))  # [B, 18]
            loss = criterion(result, labels.to(DEVICE))  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        scheduler.step(total_loss / len(dataset))
        last_loss = avg_loss
        avg_loss = total_loss / len(dataset)
        loss_history.append(last_loss)
        if(epoch % 100 == 0):
            torch.save(model.state_dict(), save_path)  # Save checkpoint every 100 epochs
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f}s")
        # if(abs(avg_loss - last_loss) < 1e-6):
        #     print("Early stopping due to minimal loss improvement.")
        #     break
    torch.save(model.state_dict(), save_path)
    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig(f"training_loss_t2c.png", dpi=300)

    test_dataset = SeqRouterDataset(test_data_path, tokenizer)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model.eval()
    criterion = nn.MSELoss()
    loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, data_tensor in test_dataloader:
            result = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))  # [B, 18]
            loss += criterion(result, data_tensor.to(DEVICE)).item() * input_ids.size(0)
    avg_test_loss = loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return {"train_loss": avg_loss, "test_loss": avg_test_loss}

def evaluate_router(model: SeqRouter, tokenizer: AutoTokenizer, test_data_path: str = "TEST_DATA", batch_size: int = 32, num_workers: int = 4) -> dict:
    """Evaluate the SeqRouter model on test data.

    Args:
        model: Trained SeqRouter model
        tokenizer: Tokenizer for processing text
        test_data_path: Path to test data (jsonl format)
        batch_size: Batch size for evaluation
        num_workers: Number of workers for DataLoader

    Returns:
        Dictionary with evaluation metrics
    """
    test_dataset = SeqRouterDataset(test_data_path, tokenizer)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model.eval()
    criterion = nn.MSELoss()
    loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, data_tensor in test_dataloader:
            result = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))  # [B, 18]
            loss += criterion(result, data_tensor.to(DEVICE)).item() * input_ids.size(0)
            print(result.cpu().numpy(), data_tensor.cpu().numpy())  # Debug: print predictions vs labels
    avg_test_loss = loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return {"test_loss": avg_test_loss}

if __name__ == "__main__":
    # Quick test with random inputs

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    embedding_dim = base_model.get_input_embeddings().embedding_dim

    # model = SeqRouter(base_model, embed_dim=embedding_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, hidden_dim4=64, hidden_dim5=32).to(DEVICE)

    # if os.path.exists(SAVE_DIR):
    #     model.load_state_dict(torch.load(SAVE_DIR))
    #     print(f"Loaded model from {SAVE_DIR}")
    # else:
    #     print(f"No checkpoint found at {SAVE_DIR}, using random initialized model.")

    input_ids = torch.randint(0, tokenizer.vocab_size, (4, 512)).to(DEVICE)  # [B, T]
    last_hidden_dim = base_model(input_ids).last_hidden_state.shape[-1]
    print(f"Base model last hidden dimension: {last_hidden_dim}")

    train_router(
        model=model,
        tokenizer=tokenizer,
        save_path=SAVE_DIR,
        train_data_path=TRAIN_DATA,
        test_data_path=TEST_DATA,
        epochs=1000,  # Adjust epochs as needed
        batch_size=32,  # Adjust batch size as needed
        lr=1e-4,  # Adjust learning rate as needed
        num_workers=2,
    )
    print(f"Training completed and model saved in {SAVE_DIR}")

    # eval_metrics = evaluate_router(model, tokenizer, test_data_path=TEST_DATA)
    # print(f"Evaluation Metrics: {eval_metrics}")

    # Evaluate on test set