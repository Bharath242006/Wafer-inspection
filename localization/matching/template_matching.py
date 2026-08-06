"""
localization/matching/template_matching.py

Normalized Cross Correlation (NCC) and Zero-Mean Unit-Variance (ZMUV) matching.
"""

import cv2
import numpy as np


def compute_local_variance_map(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Computes local texture variance map in float32 [0, 1]."""
    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (ksize, ksize))
    sqr_mean = cv2.blur(img_f ** 2, (ksize, ksize))
    var = cv2.max(0.0, sqr_mean - mean ** 2)
    cv2.normalize(var, var, 0.0, 1.0, cv2.NORM_MINMAX)
    return var


def zmuv_ncc(patch: np.ndarray, tmpl: np.ndarray) -> float:
    """Calculates zero-mean unit-variance normalized cross correlation."""
    p_f = patch.astype(np.float32) - np.mean(patch)
    t_f = tmpl.astype(np.float32) - np.mean(tmpl)
    s_p = np.std(p_f)
    s_t = np.std(t_f)
    if s_p > 1e-5 and s_t > 1e-5:
        return float(np.mean(p_f * t_f) / (s_p * s_t))
    return 0.0
