"""
models/coordinate_model.py

Coordinate-Aware Candidate Ranker architecture.
"""

import torch
import torch.nn as nn


class CoordinateAwareRankerNet(nn.Module):
    """
    MLP neural network ranker operating on 44-D coordinate-aware features.
    """

    def __init__(self, input_dim: int = 44, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)
