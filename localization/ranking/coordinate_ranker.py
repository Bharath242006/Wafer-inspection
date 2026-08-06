"""
localization/ranking/coordinate_ranker.py

Coordinate-Aware Candidate Ranker evaluator wrapper.
"""

import math
import os
import cv2
import numpy as np
import torch

from models.coordinate_model import CoordinateAwareRankerNet
from localization.features.fft_features import estimate_lattice_period_2d


_coord_model_cache = None


def load_trained_coordinate_model(checkpoint_path: str = "weights/checkpoints/coordinate_aware_ranker.pt") -> CoordinateAwareRankerNet:
    """Loads trained Coordinate-Aware Ranker model checkpoint."""
    global _coord_model_cache
    if _coord_model_cache is not None:
        return _coord_model_cache

    if not os.path.exists(checkpoint_path) and os.path.exists("checkpoints/coordinate_aware_ranker.pt"):
        checkpoint_path = "checkpoints/coordinate_aware_ranker.pt"

    model = CoordinateAwareRankerNet(input_dim=44, hidden_dim=128)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    _coord_model_cache = model
    return model
