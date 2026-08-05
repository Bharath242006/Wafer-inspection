"""
training/train_siamese.py

Training script for the Siamese CNN Candidate Ranker on semiconductor training split (160 images).

Features:
- Reproducible random seeds.
- Contrastive loss with margin m = 0.2.
- Adam optimizer (lr = 1e-3).
- Saves trained PyTorch weights to checkpoints/siamese_cnn.pt.
"""

import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))
from localization.cnn_candidate_ranker import SiameseNet
from training.dataset_siamese import SiameseDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks:
    L = y * (1 - sim)^2 + (1 - y) * max(0, sim - margin)^2
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, sim: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pos_loss = label * torch.square(1.0 - sim)
        neg_loss = (1.0 - label) * torch.square(torch.clamp(sim - self.margin, min=0.0))
        return torch.mean(pos_loss + neg_loss)


def train_siamese_network():
    set_seed(42)

    labels_csv = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "siamese_cnn.pt")

    print("=" * 90)
    print("                TRAINING SIAMESE CNN CANDIDATE RANKER (160 SAMPLES)")
    print("=" * 90)

    dataset = SiameseDataset(labels_csv, ref_dir, search_dir, is_train=True, split_idx=160)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

    print(f"Training Pair Samples Generated: {len(dataset)}")

    model = SiameseNet(embedding_dim=32)
    criterion = ContrastiveLoss(margin=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pos_sims = []
        neg_sims = []

        start_t = time.time()
        for x_ref, x_cand, label in dataloader:
            optimizer.zero_grad()
            sim = model(x_ref, x_cand)
            loss = criterion(sim, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)

            pos_mask = label > 0.5
            neg_mask = label <= 0.5

            if torch.any(pos_mask):
                pos_sims.extend(sim[pos_mask].detach().cpu().numpy())
            if torch.any(neg_mask):
                neg_sims.extend(sim[neg_mask].detach().cpu().numpy())

        avg_loss = total_loss / len(dataset)
        avg_pos_sim = np.mean(pos_sims) if pos_sims else 0.0
        avg_neg_sim = np.mean(neg_sims) if neg_sims else 0.0

        elapsed = time.time() - start_t
        print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Pos Pair Sim: {avg_pos_sim:+.4f} | Neg Pair Sim: {avg_neg_sim:+.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), checkpoint_path)
    print("=" * 90)
    print(f"TRAINING COMPLETE. Model weights saved to '{checkpoint_path}'.")
    print("=" * 90)


if __name__ == "__main__":
    train_siamese_network()
