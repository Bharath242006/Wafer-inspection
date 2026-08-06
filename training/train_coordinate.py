"""
training/train_coordinate.py

Training script for Coordinate-Aware Candidate Ranker.
"""

import os
import torch
from models.coordinate_model import CoordinateAwareRankerNet


def train_coordinate_ranker():
    print("=" * 80)
    print("        TRAINING COORDINATE-AWARE CANDIDATE RANKER")
    print("=" * 80)

    os.makedirs("weights/checkpoints", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("weights", "checkpoints", "coordinate_aware_ranker.pt")

    model = CoordinateAwareRankerNet(input_dim=44, hidden_dim=128)
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), "checkpoints/coordinate_aware_ranker.pt")
    print(f"Saved checkpoint to '{checkpoint_path}'.")


if __name__ == "__main__":
    train_coordinate_ranker()
