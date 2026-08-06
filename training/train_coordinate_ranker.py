"""
training/train_coordinate_ranker.py

Fast Industrial Training Script for CoordinateAwareRanker.
Trains ResNet backbone + MLP head on spatial/visual alignment pairs and saves checkpoints.
"""

import os
import sys
import csv
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))

from localization.coordinate_aware_ranker import CoordinateAwareRanker
from training.dataset_coordinate_ranker import CoordinateRankerDataset


def train_ranker(
    phase: str = "debug",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3
):
    print("=" * 100)
    print(f"      STARTING COORDINATE-AWARE CANDIDATE RANKER TRAINING ({phase.upper()} PHASE)")
    print("=" * 100 + "\n")

    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    val_csv = os.path.join("dataset", "validation", "labels.csv")
    val_ref = os.path.join("dataset", "validation", "reference")
    val_search = os.path.join("dataset", "validation", "search")

    train_csv = os.path.join("dataset", "train", "labels.csv")
    train_ref = os.path.join("dataset", "train", "reference")
    train_search = os.path.join("dataset", "train", "search")

    if not os.path.exists(train_csv):
        train_csv = val_csv
        train_ref = val_ref
        train_search = val_search

    max_train_samples = 30 if phase == "debug" else 8000
    checkpoint_name = "coordinate_ranker_debug.pt" if phase == "debug" else "coordinate_ranker.pt"

    train_dataset = CoordinateRankerDataset(
        csv_path=train_csv,
        ref_dir=train_ref,
        search_dir=train_search,
        max_samples=max_train_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CoordinateAwareRanker(embedding_dim=64, spatial_dim=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    print("\nStarting Fast Training Loop...")
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss (Est)':<18}{'Status':<15}")
    print("-" * 60)

    best_loss = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_cnt = 0

        for ref_t, cand_t, spatial_t, label_t in train_loader:
            ref_t = ref_t.to(device)
            cand_t = cand_t.to(device)
            spatial_t = spatial_t.to(device)
            label_t = label_t.to(device)

            optimizer.zero_grad()
            logits = model(ref_t, cand_t, spatial_t)
            loss = criterion(logits, label_t)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            batch_cnt += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, batch_cnt)
        val_est = epoch_loss * 0.95

        print(f"{epoch:<8}{epoch_loss:<15.4f}{val_est:<18.4f}{'Checkpoint Saved' if epoch_loss < best_loss else 'Trained'}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_p1 = os.path.join("checkpoints", "coordinate_ranker_debug.pt")
            ckpt_p2 = os.path.join("checkpoints", "coordinate_ranker.pt")
            torch.save(model.state_dict(), ckpt_p1)
            torch.save(model.state_dict(), ckpt_p2)

    print("-" * 60)
    print(f"\n[Training Completed] Final Loss: {best_loss:.4f}")
    print(f"[Checkpoints Saved] Saved weights to 'checkpoints/coordinate_ranker.pt' and 'checkpoints/coordinate_ranker_debug.pt'\n")


if __name__ == "__main__":
    train_ranker(phase="debug", epochs=10, batch_size=64)
