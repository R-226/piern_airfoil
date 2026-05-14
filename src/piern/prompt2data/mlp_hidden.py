"""
Prompt-to-Data Regression Router (Hidden-State Variant) for PiERN Airfoil.

This module implements a regression model that predicts aerodynamic parameters
(Mach, CL, weights, CM, etc.) from natural language prompts describing airfoil
design requirements. It uses a frozen LLM as a feature extractor and trains
only a lightweight MLP head.

Architecture:
    Input:  token_ids [B, T] + attention_mask [B, T]
            ↓
    Frozen LLM: last_hidden_state [B, T, E]
            ↓
    Mask + Mean Pool: [B, E]
            ↓
    MLP:   6-layer MLP [B, 18]
            ↓
    Output: normalized predictions → denormalize to real values

Training:
    - Labels: 18-dim normalized aerodynamic parameters
    - Loss: MSELoss
    - Optimizer: AdamW
    - Scheduler: CosineAnnealingLR (lr decays from initial to 1% over all epochs)

Predicted Parameters (18-dim):
    - Mach (1), CL (6), weights (6), CM_lower_bound (1),
      Trailing_edge_angle_lower_bound (1), Leading_edge_angle (1),
      thickness_head_lower_bound (1), thickness_tail_lower_bound (1)
"""


import torch
import torch.nn as nn
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
    """MLP regression head that maps pooled LLM hidden states to aerodynamic parameters.

    Input:  [B, E] pooled hidden states from frozen base model
    Output: [B, 18] normalized predictions (denormalize via get_data)
    """

    def __init__(
        self,
        hidden_dim: int,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        hidden_dim3: int = 64,
        hidden_dim4: int = 32,
        dropout: float = 0.3,
    ):
        """Initialize SeqRouter.

        Args:
            hidden_dim: Input dimension (embedding dim from base model).
            hidden_dim1-4: Hidden dimensions for the MLP layers.
            dropout: Dropout probability (0 = no dropout).
        """
        super().__init__()
        # MLP regressor: hidden_dim → 18-dim output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim4, 18),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pooled_features: [B, E] pooled hidden states from the frozen base model
        Returns:
            predictions: [B, 18] normalized predictions
        """
        return self.fc(pooled_features.to(torch.float))  # [B, 18]

    def get_data(self, pooled_features: torch.Tensor) -> dict:
        """Forward pass + denormalize to real aerodynamic parameter values."""
        outputs = self.forward(pooled_features)  # [B, 18]
        denormalized = outputs * (SeqRouterDataset.STD.to(outputs.device) + 1e-8) + SeqRouterDataset.MEAN.to(outputs.device)
        return {
            "Mach" : denormalized[:, 0],
            "CL" : denormalized[:, 1:7],
            "weights" : denormalized[:, 7:13],
            "CM_lower_bound" : denormalized[:, 13],
            "Trailing_edge_angle_lower_bound" : denormalized[:, 14],
            "Leading_edge_angle" : denormalized[:, 15],
            "thickness_head_lower_bound" : denormalized[:, 16],
            "thickness_tail_lower_bound" : denormalized[:, 17]
        }


def encode_batch(base_model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Encode a batch with the frozen base model and mean-pool the last hidden state."""
    with torch.no_grad():
        input_hidden = base_model(input_ids, attention_mask=attention_mask).last_hidden_state  # [B, T, E]
        masked_hidden = input_hidden * attention_mask.unsqueeze(-1)
        sum_embed = masked_hidden.sum(dim=1)
        count = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = sum_embed / count
    return pooled

NORM_PATH = "./data/2com/normalization_params.json"


def _load_normalization_params() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MEAN and STD from normalization_params.json."""
    with open(NORM_PATH, "r") as f:
        p = json.load(f)

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
        encoded = self.tokenizer(sample['prompt'], return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        input_ids = encoded['input_ids'].squeeze(0)  # [T]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [T]
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
    base_model,
    tokenizer: AutoTokenizer,
    save_path: str = SAVE_DIR,
    train_data_path: str = TRAIN_DATA,
    test_data_path: str = TEST_DATA,
    epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 4,
) -> dict:
    """Train the SeqRouter model.

    Args:
        model: SeqRouter model to train
        base_model: Frozen LLM used for feature extraction
        tokenizer: Tokenizer for processing text
        save_path: Path to save the trained model checkpoint
        train_data_path: Path to training data (jsonl format)
        test_data_path: Path to test data (jsonl format)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        num_workers: Number of DataLoader workers

    Returns:
        Dictionary with training metrics (train_loss, test_loss)
    """
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    dataset = SeqRouterDataset(train_data_path, tokenizer)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    avg_loss = 0.0
    loss_history = []
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            pooled_features = encode_batch(base_model, input_ids, attention_mask)
            result = model(pooled_features)  # [B, 18]
            loss = criterion(result, labels)  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        avg_loss = total_loss / len(dataset)
        scheduler.step()
        loss_history.append(avg_loss)
        if(epoch % 100 == 0):
            torch.save(model.state_dict(), save_path)  # Save checkpoint every 100 epochs
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f}s")
    torch.save(model.state_dict(), save_path)
    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig(f"training_loss_t2c_hidden.png", dpi=300)

    test_dataset = SeqRouterDataset(test_data_path, tokenizer)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model.eval()
    criterion = nn.MSELoss()
    loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, data_tensor in test_dataloader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            data_tensor = data_tensor.to(DEVICE, non_blocking=True)
            pooled_features = encode_batch(base_model, input_ids, attention_mask)
            result = model(pooled_features)  # [B, 18]
            loss += criterion(result, data_tensor).item() * input_ids.size(0)
    avg_test_loss = loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return {"train_loss": avg_loss, "test_loss": avg_test_loss}

def evaluate_router(model: SeqRouter, base_model, tokenizer: AutoTokenizer, test_data_path: str = TEST_DATA, batch_size: int = 32, num_workers: int = 4) -> dict:
    """Evaluate the SeqRouter model on test data.

    Args:
        model: Trained SeqRouter model
        base_model: Base model for feature extraction
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
    base_model.eval()
    criterion = nn.MSELoss()
    loss = 0.0
    total_samples = len(test_dataset)
    total_time = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, data_tensor in test_dataloader:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            data_tensor = data_tensor.to(DEVICE, non_blocking=True)
            pooled_features = encode_batch(base_model, input_ids, attention_mask)
            result = model(pooled_features)  # [B, 18]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            batch_time = t1 - t0
            current_batch_size = input_ids.size(0)
            total_time += batch_time
            loss += criterion(result, data_tensor).item() * current_batch_size
    avg_test_loss = loss / total_samples if total_samples > 0 else float('nan')
    time_per_sample = total_time / total_samples if total_samples > 0 else float('nan')
    print(f"Test Loss: {avg_test_loss:.4f}, Average Time per Sample: {time_per_sample:.4f}s")
    return {"test_loss": avg_test_loss}

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    embedding_dim = base_model.get_input_embeddings().embedding_dim

    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    model = SeqRouter(hidden_dim=embedding_dim, dropout=0.3).to(DEVICE)

    if os.path.exists(SAVE_DIR):
        model.load_state_dict(torch.load(SAVE_DIR, map_location=DEVICE))
        print(f"Loaded model from {SAVE_DIR}")
    else:
        print(f"No checkpoint found at {SAVE_DIR}, using random initialized model.")


    input_ids = torch.randint(0, tokenizer.vocab_size, (4, 512)).to(DEVICE)  # [B, T]

    train_router(
        model=model,
        base_model=base_model,
        tokenizer=tokenizer,
        save_path=SAVE_DIR,
        train_data_path=TRAIN_DATA,
        test_data_path=TEST_DATA,
        epochs=500,  # Adjust epochs as needed
        batch_size=32,  # Adjust batch size as needed
        lr=1e-4,  # Adjust learning rate as needed
        num_workers=2,
    )
    print(f"Training completed and model saved in {SAVE_DIR}")

    eval_metrics = evaluate_router(model, base_model, tokenizer, test_data_path=TEST_DATA)
    print(f"Evaluation Metrics: {eval_metrics}")
