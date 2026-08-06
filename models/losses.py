"""
models/losses.py

Loss functions for neural ranking models: Triplet Margin Loss, Contrastive Loss, BCE Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginRankingLoss(nn.Module):
    """
    Triplet margin ranking loss for candidate ranking models.
    Pushes positive candidate scores above negative candidate scores by margin.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        loss = F.relu(self.margin - (pos_scores - neg_scores))
        return torch.mean(loss)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese neural networks.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss_pos + loss_neg)
