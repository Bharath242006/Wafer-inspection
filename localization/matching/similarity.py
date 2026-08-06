"""
localization/matching/similarity.py

Similarity metrics and multi-feature score normalization functions.
"""

import numpy as np


def normalize_zscore_tanh(scores: np.ndarray) -> np.ndarray:
    """
    Z-score normalizes array across candidates and squashes via tanh into [-1.0, 1.0].
    """
    mean = np.mean(scores)
    std = np.std(scores) + 1e-5
    z = (scores - mean) / std
    return np.tanh(z)
