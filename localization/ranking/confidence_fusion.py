"""
localization/ranking/confidence_fusion.py

Multi-feature candidate scoring and confidence score fusion module.
"""

import numpy as np


def fuse_candidate_scores(scores_dict: dict, weights_dict: dict = None) -> np.ndarray:
    """
    Fuses multiple candidate score vectors using weighted linear combination.
    """
    if weights_dict is None:
        weights_dict = {
            'cnn': 0.35,
            'hybrid': 0.35,
            'context': 0.15,
            'coordinate': 0.15
        }

    total_weight = 0.0
    fused_score = None

    for key, weight in weights_dict.items():
        if key in scores_dict and len(scores_dict[key]) > 0:
            arr = np.array(scores_dict[key], dtype=np.float32)
            if fused_score is None:
                fused_score = weight * arr
            else:
                fused_score += weight * arr
            total_weight += weight

    if fused_score is None or total_weight == 0:
        return np.array([], dtype=np.float32)

    return fused_score / total_weight
