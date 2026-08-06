"""
localization/fine_localization.py

Fine template matching search and 2D quadratic subpixel peak refinement.
"""

import cv2
import numpy as np


def refine_subpixel_peak(response_map: np.ndarray, tl_x: int, tl_y: int) -> tuple:
    """
    Refines integer peak coordinate (tl_x, tl_y) using 2D quadratic interpolation.
    """
    h, w = response_map.shape[:2]
    img_f = response_map.astype(np.float32)
    if 1 <= tl_x < w - 1 and 1 <= tl_y < h - 1:
        z0 = img_f[tl_y, tl_x]
        zx_prev = img_f[tl_y, tl_x - 1]
        zx_next = img_f[tl_y, tl_x + 1]
        zy_prev = img_f[tl_y - 1, tl_x]
        zy_next = img_f[tl_y + 1, tl_x]

        dx = (zx_next - zx_prev) / (2.0 * (2.0 * z0 - zx_prev - zx_next) + 1e-6)
        dy = (zy_next - zy_prev) / (2.0 * (2.0 * z0 - zy_prev - zy_next) + 1e-6)


        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)

        return float(tl_x + dx), float(tl_y + dy)
    return float(tl_x), float(tl_y)
