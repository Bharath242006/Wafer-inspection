"""
training/train_hybrid_ranker.py

Training script for DriftSense-X Final Hybrid Candidate Ranker.

Pipeline:
1. Phase 1 — 10-sample overfitting sanity check (50 pairs, 5 epochs).
   MUST achieve S_pos > S_neg before proceeding.
2. Phase 2 — Full training on images 00001–00160 (20 epochs).
   Triplet margin loss with margin=0.30.
   Adam optimizer with weight decay=1e-4.

Saves: checkpoints/hybrid_ranker.pt
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
from localization.hybrid_ranker import HybridRankerNet, HYBRID_FEATURE_DIM
from training.dataset_hybrid_ranker import HybridRankerTripletDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TripletMarginRankingLoss(nn.Module):
    """
    Pairwise triplet margin loss:
        L = max(0, S_neg - S_pos + margin)

    Forces S_pos > S_neg + margin.
    margin=0.30 (higher than coordinate_aware_ranker's 0.25) to push
    aliases further from the correct candidate.
    """
    def __init__(self, margin: float = 0.30):
        super().__init__()
        self.margin = margin

    def forward(self, score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
        loss = torch.clamp(score_neg - score_pos + self.margin, min=0.0)
        return torch.mean(loss)


def train_hybrid_ranker():
    set_seed(42)

labels_csv = "/kaggle/input/datasets/abineshsekar/training-wafer/dataset_small/train/labels.csv"  
ref_dir = "/kaggle/input/datasets/abineshsekar/training-wafer/dataset_small/train/reference"
search_dir = "/kaggle/input/datasets/abineshsekar/training-wafer/dataset_small/train/search"
    checkpoint_path = os.path.join("checkpoints", "hybrid_ranker.pt")
    os.makedirs("checkpoints", exist_ok=True)

    print("=" * 95)
    print("         TRAINING FINAL HYBRID CANDIDATE RANKER  (160 TRAINING IMAGES)")
    print("=" * 95)
    print(f"Feature dimension : {HYBRID_FEATURE_DIM}")
    print(f"Triplet margin    : 0.30")
    print(f"Epochs            : 30")
    print(f"Checkpoint        : {checkpoint_path}")
    print("=" * 95)

    # ── Load full training dataset ───────────────────────────────────────────
    t0 = time.time()
  full_dataset = HybridRankerTripletDataset( labels_csv, ref_dir, search_dir )
    print(f"Dataset built in {time.time() - t0:.1f}s | Pairs: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("ERROR: No dataset pairs generated. Check image paths. Aborting.")
        sys.exit(1)

    criterion = TripletMarginRankingLoss(margin=0.30)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: OVERFITTING SANITY CHECK
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Phase 1: 10-Sample Overfitting Sanity Check]")
    sanity_size = min(50, len(full_dataset))
    sanity_subset = Subset(full_dataset, list(range(sanity_size)))
    sanity_loader = DataLoader(sanity_subset, batch_size=16, shuffle=True)

    sanity_model = HybridRankerNet(input_dim=HYBRID_FEATURE_DIM, hidden_dim=128)
    opt_s = torch.optim.Adam(sanity_model.parameters(), lr=1e-3)

    sanity_passed = False
    for epoch in range(1, 11):
        sanity_model.train()
        total_l = 0.0
        pos_s_list, neg_s_list = [], []

        for pos_f, neg_f in sanity_loader:
            opt_s.zero_grad()
            s_pos = sanity_model(pos_f)
            s_neg = sanity_model(neg_f)
            loss = criterion(s_pos, s_neg)
            loss.backward()
            opt_s.step()
            total_l += loss.item() * len(s_pos)
            pos_s_list.extend(s_pos.detach().cpu().numpy().tolist())
            neg_s_list.extend(s_neg.detach().cpu().numpy().tolist())

        avg_l = total_l / len(sanity_subset)
        avg_pos = float(np.mean(pos_s_list)) if pos_s_list else 0.0
        avg_neg = float(np.mean(neg_s_list)) if neg_s_list else 0.0
        margin_achieved = avg_pos - avg_neg
        print(
            f"  Sanity Epoch {epoch:2d}/10 | Loss: {avg_l:.4f} "
            f"| Pos: {avg_pos:.4f} | Neg: {avg_neg:.4f} | Margin: {margin_achieved:+.4f}"
        )

        if avg_pos > avg_neg:
            sanity_passed = True

    if sanity_passed:
        print("[Phase 1] PASSED [OK] -- S_pos > S_neg confirmed after sanity training.")
    else:
        print("[Phase 1] WARNING [!!] -- S_pos did not consistently exceed S_neg.")
        print("  Proceeding to full training anyway (may need more epochs).")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: FULL TRAINING
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[Phase 2: Full Training on 160 Images | {len(full_dataset)} pairs]")
    train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True, drop_last=False)

    model = HybridRankerNet(input_dim=HYBRID_FEATURE_DIM, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )

    best_train_loss = float('inf')
    epochs = 30

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pos_scores, neg_scores = [], []
        t_ep = time.time()

        for pos_f, neg_f in train_loader:
            optimizer.zero_grad()
            s_pos = model(pos_f)
            s_neg = model(neg_f)
            loss = criterion(s_pos, s_neg)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(s_pos)
            pos_scores.extend(s_pos.detach().cpu().numpy().tolist())
            neg_scores.extend(s_neg.detach().cpu().numpy().tolist())

        scheduler.step()

        avg_loss = total_loss / len(full_dataset)
        avg_pos = float(np.mean(pos_scores)) if pos_scores else 0.0
        avg_neg = float(np.mean(neg_scores)) if neg_scores else 0.0
        elapsed = time.time() - t_ep
        lr_now = float(optimizer.param_groups[0]['lr'])

        print(
            f"  Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} "
            f"| Pos: {avg_pos:.4f} | Neg: {avg_neg:.4f} "
            f"| Margin: {avg_pos - avg_neg:+.4f} "
            f"| LR: {lr_now:.2e} | {elapsed:.1f}s"
        )

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss

    # Save final checkpoint
    torch.save(model.state_dict(), checkpoint_path)

    print("\n" + "=" * 95)
    print(f"TRAINING COMPLETE — Best training loss: {best_train_loss:.4f}")
    print(f"Model weights saved to '{checkpoint_path}'")
    print(f"Final margin (S_pos - S_neg): {avg_pos - avg_neg:+.4f}")
    if avg_pos > avg_neg:
        print("STATUS: PASS [OK] -- Positive candidates score higher than negatives.")
    else:
        print("STATUS: WARN [!!] -- Model may not have converged fully. Check features.")
    print("=" * 95)


if __name__ == "__main__":
    train_hybrid_ranker()
