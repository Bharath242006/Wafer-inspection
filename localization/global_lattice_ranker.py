"""
localization/global_lattice_ranker.py

Global/Lattice-Aware Candidate Ranker architecture module for DriftSense-X.

Model Architecture:
- Tabular MLP Neural Network operating on a 22-dimensional feature vector:
  1. Normalized search coordinates (cx / 1000, cy / 1000)
  2. Estimated lattice coordinates (cx / lx, cy / ly)
  3. Fractional lattice phase ((cx % lx) / lx, (cy % ly) / ly)
  4. Local structural signatures (7 Z-score normalized features)
  5. Surrounding macro context alignment score
  6. 8-Neighboring cell response consistency (at +/- lx, +/- ly)

Outputs candidate score in range [0.0, 1.0].
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.dataset_lattice_ranker import extract_lattice_candidate_features, estimate_lattice_period_2d


class GlobalLatticeRankerNet(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network ranker operating on 22-D global/lattice features.
    """
    def __init__(self, input_dim: int = 22, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handles single sample evaluation when batch size is 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


_lattice_model_cache = None


def load_trained_lattice_model(checkpoint_path: str = "checkpoints/global_lattice_ranker.pt") -> GlobalLatticeRankerNet:
    """Loads trained Global/Lattice Ranker model checkpoint."""
    global _lattice_model_cache
    if _lattice_model_cache is not None:
        return _lattice_model_cache

    model = GlobalLatticeRankerNet(input_dim=22, hidden_dim=64)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[GlobalLatticeRankerNet] Successfully loaded trained weights from '{checkpoint_path}'.")
    else:
        print(f"[GlobalLatticeRankerNet] Warning: Checkpoint '{checkpoint_path}' not found! Using random initialization.")

    _lattice_model_cache = model
    return model


def compute_global_lattice_scores(ref_img: np.ndarray, search_img: np.ndarray, candidates: list, checkpoint_path: str = "checkpoints/global_lattice_ranker.pt") -> list:
    """
    Calculates Global/Lattice ranking scores for candidate pool.

    Args:
        ref_img (np.ndarray): 1000x1000 reference image.
        search_img (np.ndarray): 1000x1000 search image.
        candidates (list): Candidate pool dictionaries.

    Returns:
        list: Float ranking scores in range [0.0, 1.0].
    """
    model = load_trained_lattice_model(checkpoint_path)
    lx, ly = estimate_lattice_period_2d(ref_img)

    feats = []
    for cand in candidates:
        f_vec = extract_lattice_candidate_features(ref_img, search_img, cand, lx, ly)
        feats.append(f_vec)

    if not feats:
        return []

    t_feats = torch.tensor(np.array(feats), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(t_feats)
        if scores.dim() == 0:
            scores_list = [float(scores.item())]
        else:
            scores_list = [float(s.item()) for s in scores]

    return scores_list
