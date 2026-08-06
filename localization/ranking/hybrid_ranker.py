"""
localization/ranking/hybrid_ranker.py

Hybrid Candidate Ranker evaluator wrapper and feature extraction pipeline.
"""

import math
import os
import cv2
import numpy as np
import torch

from models.hybrid_model import HybridRankerNet, HYBRID_FEATURE_DIM
from localization.features.fft_features import estimate_lattice_period_2d
from localization.features.edge_features import compute_sobel_gradient, compute_canny_edge
from localization.features.landmark_features import compute_global_landmark_heatmap
from localization.matching.template_matching import compute_local_variance_map, zmuv_ncc
from localization.matching.fft_matching import fft_phase_correlation_score


_hybrid_model_cache = None


def load_trained_hybrid_model(checkpoint_path: str = "weights/checkpoints/hybrid_ranker.pt") -> HybridRankerNet:
    """Loads trained Hybrid Ranker model checkpoint."""
    global _hybrid_model_cache
    if _hybrid_model_cache is not None:
        return _hybrid_model_cache

    if not os.path.exists(checkpoint_path) and os.path.exists("checkpoints/hybrid_ranker.pt"):
        checkpoint_path = "checkpoints/hybrid_ranker.pt"

    model = HybridRankerNet(input_dim=HYBRID_FEATURE_DIM, hidden_dim=128)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    _hybrid_model_cache = model
    return model
