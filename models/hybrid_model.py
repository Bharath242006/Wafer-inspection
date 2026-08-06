"""
models/hybrid_model.py

Multi-Feature Hybrid Neural Candidate Ranker architecture.
"""

import torch
import torch.nn as nn

HYBRID_FEATURE_DIM = 56


class HybridRankerNet(nn.Module):
    """
    MLP Neural Network ranker operating on 56-D hybrid feature vectors.
    """

    def __init__(self, input_dim: int = HYBRID_FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)
