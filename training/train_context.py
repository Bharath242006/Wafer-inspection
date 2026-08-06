"""
training/train_context.py

Training script for Multi-Branch Context Candidate Ranker.
"""

import os
import torch
from models.context_model import ContextRankerNet


def train_context_ranker():
    print("=" * 80)
    print("        TRAINING CONTEXT-AWARE CANDIDATE RANKER")
    print("=" * 80)

    os.makedirs("weights/checkpoints", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("weights", "checkpoints", "context_ranker.pt")

    model = ContextRankerNet(embedding_dim=32)
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), "checkpoints/context_ranker.pt")
    print(f"Saved checkpoint to '{checkpoint_path}'.")


if __name__ == "__main__":
    train_context_ranker()
