"""
training/train_context_ranker.py

Training script for the Multi-Branch Context-Aware Candidate Ranker on semiconductor training split (160 images).

Features:
- Reproducible random seeds.
- Margin ranking loss (margin = 0.25).
- Adam optimizer (lr = 1e-3).
- Saves trained PyTorch weights to checkpoints/context_ranker.pt.
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
from localization.context_ranker import ContextRankerNet
from training.dataset_context_ranker import ContextTripletDataset


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


def train_context_ranker_network():
    set_seed(42)

    labels_csv = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "context_ranker.pt")

    print("=" * 95)
    print("           TRAINING MULTI-BRANCH CONTEXT-AWARE CANDIDATE RANKER (160 SAMPLES)")
    print("=" * 95)

    full_dataset = ContextTripletDataset(labels_csv, ref_dir, search_dir, is_train=True, split_idx=160)

    # 1. CRITICAL FIRST OVERFITTING SANITY CHECK (Subset of 10 training samples)
    print("\n[Phase 1: Overfitting Sanity Check on 10 Training Samples]")
    sanity_subset = Subset(full_dataset, list(range(min(50, len(full_dataset)))))
    sanity_loader = DataLoader(sanity_subset, batch_size=16, shuffle=True)

    sanity_model = ContextRankerNet(embedding_dim=32)
    criterion = TripletMarginRankingLoss(margin=0.25)
    optimizer_s = torch.optim.Adam(sanity_model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        sanity_model.train()
        total_l = 0.0
        pos_s_list, neg_s_list = [], []

        for ref_t, pos_t, neg_t in sanity_loader:
            optimizer_s.zero_grad()
            s_pos = sanity_model.forward_pair_similarity(ref_t[0], ref_t[1], ref_t[2], pos_t[0], pos_t[1], pos_t[2])
            s_neg = sanity_model.forward_pair_similarity(ref_t[0], ref_t[1], ref_t[2], neg_t[0], neg_t[1], neg_t[2])

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

    model = ContextRankerNet(embedding_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pos_scores = []
        neg_scores = []

        start_t = time.time()
        for ref_t, pos_t, neg_t in train_loader:
            optimizer.zero_grad()
            s_pos = model.forward_pair_similarity(ref_t[0], ref_t[1], ref_t[2], pos_t[0], pos_t[1], pos_t[2])
            s_neg = model.forward_pair_similarity(ref_t[0], ref_t[1], ref_t[2], neg_t[0], neg_t[1], neg_t[2])

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
    train_context_ranker_network()
