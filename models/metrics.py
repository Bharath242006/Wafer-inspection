"""
models/metrics.py

Ranking and model accuracy evaluation metrics.
"""

import torch
import numpy as np


def compute_ranking_accuracy(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Computes fraction of pairs where positive score > negative score.
    """
    correct = (pos_scores > neg_scores).float()
    return float(torch.mean(correct).item())


def compute_center_error(pred_center: tuple, gt_center: tuple) -> float:
    """
    Computes Euclidean distance between predicted center and ground-truth center.
    """
    px, py = pred_center
    gx, gy = gt_center
    return float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
