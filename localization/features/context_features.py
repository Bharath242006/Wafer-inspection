"""
localization/features/context_features.py

Context-aware multi-scale spatial crop extraction.
"""

import cv2
import numpy as np


def extract_multi_context_crops(img: np.ndarray, cx: float, cy: float) -> tuple:
    """
    Extracts local (100x100), medium (250x250 -> 100x100), and large (500x500 -> 100x100) context crops around (cx, cy).
    """
    h, w = img.shape[:2]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    
    pad = 500
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    px, py = cx_i + pad, cy_i + pad

    # Local: 100x100
    crop_loc = img_pad[py - 50:py + 50, px - 50:px + 50]
    # Medium: 250x250 -> 100x100
    crop_med_raw = img_pad[py - 125:py + 125, px - 125:px + 125]
    crop_med = cv2.resize(crop_med_raw, (100, 100), cv2.INTER_AREA)
    # Large: 500x500 -> 100x100
    crop_lrg_raw = img_pad[py - 250:py + 250, px - 250:px + 250]
    crop_lrg = cv2.resize(crop_lrg_raw, (100, 100), cv2.INTER_AREA)

    return crop_loc, crop_med, crop_lrg
