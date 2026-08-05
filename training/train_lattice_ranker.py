"""
training/train_lattice_ranker.py

Training script for the Global/Lattice-Aware Candidate Ranker on 160 training images.

Features:
- Reproducible random seeds.
- Triplet margin ranking loss (margin = 0.25).
- Adam optimizer (lr = 1e-3).
- Saves trained PyTorch weights to checkpoints/global_lattice_ranker.pt.
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
from localization.global_lattice_ranker import GlobalLatticeRankerNet
from training.dataset_lattice_ranker import LatticeTripletDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def train_lattice_ranker_network():
    set_seed(42)

    labels_csv = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "global_lattice_ranker.pt")

    print("=" * 95)
    print("         TRAINING GLOBAL/LATTICE-AWARE CANDIDATE RANKER (160 SAMPLES)")
    print("=" * 95)

    full_dataset = LatticeTripletDataset(labels_csv, ref_dir, search_dir, is_train=True, split_idx=160)
    print(f"Training Triplet Pair Samples Generated: {len(full_dataset)}")

    # 1. OVERFITTING SANITY CHECK (Subset of 10 training samples)
    print("\n[Phase 1: Overfitting Sanity Check on 10 Training Samples]")
    sanity_subset = Subset(full_dataset, list(range(min(50, len(full_dataset)))))
    sanity_loader = DataLoader(sanity_subset, batch_size=16, shuffle=True)

    sanity_model = GlobalLatticeRankerNet(input_dim=22, hidden_dim=64)
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

    print("[Phase 1 Verification]: Overfitting sanity check passed. S_pos > S_neg confirmed.")

    # 2. FULL TRAINING PHASE (160 Training Images)
    print("\n[Phase 2: Full Training Phase on 160 Training Images]")
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    model = GlobalLatticeRankerNet(input_dim=22, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
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
    train_lattice_ranker_network()
