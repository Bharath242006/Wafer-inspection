"""
localization/matching/attention_matching.py

Spatial attention-weighted template matching.
"""

import cv2
import numpy as np


def compute_attention_weighted_matching(patch: np.ndarray, tmpl: np.ndarray, attn_map: np.ndarray = None) -> float:
    """
    Computes spatial attention-weighted cross correlation.
    """
    p_f = patch.astype(np.float32) - np.mean(patch)
    t_f = tmpl.astype(np.float32) - np.mean(tmpl)

    if attn_map is None:
        attn_map = np.ones_like(p_f, dtype=np.float32)

    w_patch = p_f * attn_map
    w_tmpl = t_f * attn_map

    norm = (np.linalg.norm(w_patch) * np.linalg.norm(w_tmpl)) + 1e-6
    return float(np.sum(w_patch * w_tmpl) / norm)
