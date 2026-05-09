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
    MLP:   2-layer MLP [B, 1]
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

SAVE_DIR = "./checkpoint/router/seq_router.pt"
TRAIN_DATA = "./data/router/train_router.jsonl"
TEST_DATA = "./data/router/test_router.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SeqRouter(nn.Module):
    """Sequence-level Router for PiERN Airfoil.

    This router takes token sequences and predicts whether the current
    position is a trigger boundary (where Expert should be invoked).

    Architecture:
        1. Token Embedding Layer
        2. Attention Mask Application
        3. Mean Pooling
        4. 2-layer MLP → logits

    Input:
        - input_ids: [batch_size, seq_len] token IDs
        - attention_mask: [batch_size, seq_len] binary mask

    Output:
        - logits: [batch_size, 1] raw scores (sigmoid applied externally)
        - probs: [batch_size, 1] probabilities after sigmoid
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
        self.embedding = base_model.get_input_embeddings()  # Use base model's embedding layer
        self.embedding.requires_grad_(False)
        # MLP classifier
        # Input: embedding dimension
        # Hidden: hidden_dim
        # Output: 1 (binary classification)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.ReLU(),
            nn.Linear(hidden_dim5, 1)  # Output logits for binary classification
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
            Tuple of (logits, probs) where each is [B, 1]
        """
        input_embeddings = self.embedding(input_ids)  # [B, T, E]
        # 2. Mean pooling over sequence dimension
        # Sum over sequence: [B, T, E] → [B, E]
        sum_embed = input_embeddings.sum(dim=1)
        count = attention_mask.sum(dim=1, keepdim=True) if attention_mask is not None else input_embeddings.size(1)
        pooled = sum_embed / count

        # 3. MLP classifier - return LOGITS, not probs
        logits = self.fc(pooled.to(torch.float32)).squeeze(-1)  # [B, 1]

        # 4. Sigmoid for probabilities (separate for inference)
        probs = torch.sigmoid(logits).squeeze(-1)  # [B]

        # Return logits for loss computation
        return logits, probs

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Get binary predictions."""
        _, probs = self.forward(input_ids, attention_mask)  # [B]
        return (probs > 0.5)

class SeqRouterDataset(data.Dataset):
    """Custom Dataset for SeqRouter training.

    Expects data in JSONL format with fields:
        - "text" : original response text
        - "label": int (0 or 1)
    """

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
        input = self.tokenizer(sample['text'], return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        input_ids = input['input_ids'].squeeze(0)  # [T]
        attention_mask = input['attention_mask'].squeeze(0)  # [T]
        label = torch.tensor(sample['label'], dtype=torch.float)  # BCE loss expects float
        return input_ids, attention_mask, label

def train_router(
    model: SeqRouter,
    tokenizer: AutoTokenizer,
    save_path: str = "SAVE_DIR",
    train_data_path: str = "TRAIN_DATA",
    test_data_path: str = "TEST_DATA",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 4,
    pos_weight: float = 1.0,  # Weight for positive class in BCE loss
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    avg_loss = 0.0
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            logits, _ = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))  # [B], get logits for loss
            loss = criterion(logits, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        last_loss = avg_loss
        avg_loss = total_loss / len(dataset)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f}s")
        if(abs(avg_loss - last_loss) < 1e-6):
            print("Early stopping due to minimal loss improvement.")
            break
    torch.save(model.state_dict(), save_path)

    test_dataset = SeqRouterDataset(test_data_path, tokenizer)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_dataloader:
            predictions = model.predict(input_ids.to(DEVICE), attention_mask.to(DEVICE))  # [B]
            correct += (predictions == labels.bool().to(DEVICE)).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f}")
    return {"train_loss": avg_loss, "test_accuracy": accuracy}

def evaluate_router(model: SeqRouter, tokenizer: AutoTokenizer, test_data_path: str) -> dict:
    """Evaluate the SeqRouter model on test data.

    Args:
        model: Trained SeqRouter model
        tokenizer: Tokenizer for processing text
        test_data_path: Path to test data (jsonl format)

    Returns:
        Dictionary with evaluation metrics (e.g., accuracy)
    """
    test_dataset = SeqRouterDataset(test_data_path, tokenizer)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        start_time = time.time()
        for input_ids, attention_mask, labels in test_dataloader:
            predictions = model.predict(input_ids.to(DEVICE), attention_mask.to(DEVICE))  # [B]
            correct += (predictions == labels.bool().to(DEVICE)).sum().item()
            total += labels.size(0)
        end_time = time.time()
    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Evaluation Time per Sample: {(end_time - start_time) / total:.4f}s")
    return {"test_accuracy": accuracy}


if __name__ == "__main__":
    # Quick test with random inputs

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    embedding_dim = base_model.get_input_embeddings().embedding_dim

    model = SeqRouter(base_model, embed_dim=embedding_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, hidden_dim4=64, hidden_dim5=32).to(DEVICE)

    if os.path.exists(SAVE_DIR):
        model.load_state_dict(torch.load(SAVE_DIR))
        print(f"Loaded model from {SAVE_DIR}")
    else:
        print(f"No checkpoint found at {SAVE_DIR}, using random initialized model.")

    # train_router(
    #     model=model,
    #     tokenizer=tokenizer,
    #     save_path=SAVE_DIR,
    #     train_data_path=TRAIN_DATA,
    #     test_data_path=TEST_DATA,
    #     epochs=200,  # Adjust epochs as needed
    #     batch_size=256,  # Adjust batch size as needed
    #     lr=1e-5,  # Adjust learning rate as needed
    #     num_workers=16,
    #     pos_weight=1.0  # Adjust pos_weight based on class imbalance
    # )
    # print(f"Training completed and model saved in {SAVE_DIR}")

    # Evaluate on test set
    evaluate_router(model, tokenizer, TEST_DATA)
