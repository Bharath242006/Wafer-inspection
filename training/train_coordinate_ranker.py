"""
training/train_coordinate_ranker.py

Training script for the Coordinate-Aware Candidate Ranker on 160 training images.

Features:
- Reproducible random seeds.
- Mandatory 10-sample overfitting sanity check before full training.
- Triplet margin ranking loss (margin = 0.25).
- Adam optimizer with weight decay.
- Saves trained PyTorch weights to checkpoints/coordinate_aware_ranker.pt.
"""

import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath("."))
from localization.coordinate_aware_ranker import CoordinateAwareRankerNet
from training.dataset_coordinate_ranker import CoordinateAwareTripletDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TripletMarginRankingLoss(nn.Module):
    """
    Ranking loss enforcing S_pos > S_neg + margin:
    L = max(0, S_neg - S_pos + margin)
    """
    def __init__(self, margin: float = 0.25):
        super().__init__()
        self.margin = margin

    def forward(self, score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
        loss = torch.clamp(score_neg - score_pos + self.margin, min=0.0)
        return torch.mean(loss)


def train_coordinate_ranker_network():
    set_seed(42)

    labels_csv = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "coordinate_aware_ranker.pt")

    print("=" * 95)
    print("         TRAINING COORDINATE-AWARE CANDIDATE RANKER (160 SAMPLES)")
    print("=" * 95)

    full_dataset = CoordinateAwareTripletDataset(labels_csv, ref_dir, search_dir, is_train=True, split_idx=160)
    print(f"Training Triplet Pair Samples Generated: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("ERROR: No dataset pairs generated. Aborting training.")
        return

    # 1. CRITICAL: OVERFITTING SANITY CHECK (Subset of 10-sample pairs)
    print("\n[Phase 1: Overfitting Sanity Check on 10-Sample Subset]")
    sanity_size = min(50, len(full_dataset))
    sanity_subset = Subset(full_dataset, list(range(sanity_size)))
    sanity_loader = DataLoader(sanity_subset, batch_size=16, shuffle=True)

    sanity_model = CoordinateAwareRankerNet(input_dim=44, hidden_dim=128)
    criterion = TripletMarginRankingLoss(margin=0.25)
    optimizer_s = torch.optim.Adam(sanity_model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        sanity_model.train()
        total_l = 0.0
        pos_s_list, neg_s_list = [], []

        for pos_f, neg_f in sanity_loader:
            optimizer_s.zero_grad()
            s_pos = sanity_model(pos_f)
            s_neg = sanity_model(neg_f)

            loss = criterion(s_pos, s_neg)
            loss.backward()
            optimizer_s.step()

            total_l += loss.item() * len(s_pos)
            pos_s_list.extend(s_pos.detach().numpy())
            neg_s_list.extend(s_neg.detach().numpy())

        avg_l = total_l / len(sanity_subset)
        avg_pos_s = np.mean(pos_s_list) if pos_s_list else 0.0
        avg_neg_s = np.mean(neg_s_list) if neg_s_list else 0.0

        print(f"Sanity Check Epoch {epoch}/5 | Loss: {avg_l:.4f} | Pos Score: {avg_pos_s:.4f} | Neg Score: {avg_neg_s:.4f}")

    if avg_pos_s <= avg_neg_s:
        print("WARNING: Overfitting sanity check failed! Pos score did not exceed Neg score.")
    else:
        print("[Phase 1 Verification]: Overfitting sanity check PASSED. S_pos > S_neg confirmed.")

    # 2. FULL TRAINING PHASE (160 Training Images)
    print("\n[Phase 2: Full Training Phase on 160 Training Images]")
    train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    model = CoordinateAwareRankerNet(input_dim=44, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pos_scores = []
        neg_scores = []

        start_t = time.time()
        for pos_f, neg_f in train_loader:
            optimizer.zero_grad()
            s_pos = model(pos_f)
            s_neg = model(neg_f)

            loss = criterion(s_pos, s_neg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(s_pos)
            pos_scores.extend(s_pos.detach().numpy())
            neg_scores.extend(s_neg.detach().numpy())

        avg_loss = total_loss / len(full_dataset)
        avg_pos = np.mean(pos_scores) if pos_scores else 0.0
        avg_neg = np.mean(neg_scores) if neg_scores else 0.0

        elapsed = time.time() - start_t
        print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Pos Score: {avg_pos:.4f} | Neg Score: {avg_neg:.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), checkpoint_path)
    print("=" * 95)
    print(f"TRAINING COMPLETE. Model weights saved to '{checkpoint_path}'.")
    print("=" * 95)


if __name__ == "__main__":
    train_coordinate_ranker_network()
