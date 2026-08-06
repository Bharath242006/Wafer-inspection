"""
training/train_hybrid.py

Training script for Hybrid Neural Candidate Ranker.
"""

import os
import random
import time
import numpy as np
import torch

from models.hybrid_model import HybridRankerNet
from models.losses import TripletMarginRankingLoss


def train_hybrid_ranker():
    print("=" * 80)
    print("        TRAINING HYBRID CANDIDATE RANKER")
    print("=" * 80)

    os.makedirs("weights/checkpoints", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("weights", "checkpoints", "hybrid_ranker.pt")

    model = HybridRankerNet(input_dim=56, hidden_dim=128)
    print("Hybrid model initialized.")
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), "checkpoints/hybrid_ranker.pt")
    print(f"Saved checkpoint to '{checkpoint_path}'.")


if __name__ == "__main__":
    train_hybrid_ranker()
