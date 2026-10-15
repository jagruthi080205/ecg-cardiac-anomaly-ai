"""
CardioVision Model Training & Optimization Engine
Handles class-imbalance weighting, learning rate scheduling, and checkpointing.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, Tuple

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    NUM_CLASSES,
    CHECKPOINT_PATH,
    MODEL_DIR,
    CLASS_NAMES
)
from src.model import CardioVisionNet
from src.data_pipeline import build_or_load_dataset, ECGDataset


def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Computes inverse frequency class weights to counteract severe clinical imbalance
    (Normal beats vastly outnumber rare Ventricular/Supraventricular ectopics).
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Add smoothing to avoid division by zero for unrepresented classes
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(epochs: int = NUM_EPOCHS, 
                batch_size: int = BATCH_SIZE, 
                lr: float = LEARNING_RATE) -> Tuple[CardioVisionNet, Dict]:
    """
    Full training and validation routine for CardioVisionNet.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Running on device: {device}")

    # 1. Load Dataset
    X, y = build_or_load_dataset()
    full_dataset = ECGDataset(X, y)

    # 80/20 Train/Val Split
    val_size = int(len(full_dataset) * 0.20)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model, Loss & Optimizer
    model = CardioVisionNet(num_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_macro_f1": []
    }

    best_val_f1 = 0.0

    print(f"\n{'='*60}")
    print(f"  Starting CardioVisionNet Training ({epochs} Epochs)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for beats, labels in train_loader:
            beats = beats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(beats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)

        train_loss /= train_size

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for beats, labels in val_loader:
                beats = beats.to(device)
                labels = labels.to(device)

                logits, _ = model(beats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())

        val_loss /= val_size
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val Acc: {val_acc*100:.2f}% "
              f"| Val Macro-F1: {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_f1": best_val_f1,
                "history": history,
                "epoch": epoch
            }, CHECKPOINT_PATH)
            print(f"  --> Checkpoint saved (Best Macro-F1: {best_val_f1*100:.2f}%)")

    print(f"\n[Complete] Model trained and saved to {CHECKPOINT_PATH}")
    return model, history


if __name__ == "__main__":
    train_model(epochs=10)
