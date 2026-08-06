"""
localization/features/landmark_features.py

Global landmark macro heatmap and high-contrast structural landmark feature extraction.
"""

import cv2
import numpy as np
from localization.features.edge_features import compute_canny_edge


def compute_global_landmark_heatmap(ref_img: np.ndarray, search_img: np.ndarray) -> np.ndarray:
    """
    Computes a global landmark macro probability heatmap M(x, y).

    Uses downsampled low-frequency Gaussian correlation and edge density matching.
    """
    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)

    sh, sw = search_img.shape[:2]

    ref_macro_w = min(100, sw)
    ref_macro_h = min(100, sh)
    ref_macro = cv2.resize(ref_gray_f, (ref_macro_w, ref_macro_h), cv2.INTER_AREA)

    k_ref = min(31, (ref_macro_h // 2) * 2 + 1)
    k_search = min(101, (sh // 2) * 2 + 1)
    ref_blur = cv2.GaussianBlur(ref_macro, (k_ref, k_ref), 10.0)
    search_blur = cv2.GaussianBlur(search_gray_f, (k_search, k_search), 30.0)

    ref_edge = cv2.resize(compute_canny_edge(ref_img), (ref_macro_w, ref_macro_h), cv2.INTER_AREA)
    search_edge = compute_canny_edge(search_img)
    k_edge = min(41, (sh // 2) * 2 + 1)
    search_edge_blur = cv2.GaussianBlur(search_edge, (k_edge, k_edge), 10.0)

    r_blur = cv2.matchTemplate(search_blur, ref_blur, cv2.TM_CCOEFF_NORMED)
    r_edge = cv2.matchTemplate(search_edge_blur, ref_edge, cv2.TM_CCOEFF_NORMED)

    pad_h_top = max(0, (sh - r_blur.shape[0]) // 2)
    pad_h_bottom = max(0, sh - r_blur.shape[0] - pad_h_top)
    pad_w_left = max(0, (sw - r_blur.shape[1]) // 2)
    pad_w_right = max(0, sw - r_blur.shape[1] - pad_w_left)

    r_blur_pad = cv2.copyMakeBorder(r_blur, pad_h_top, pad_h_bottom,
                                    pad_w_left, pad_w_right, cv2.BORDER_REPLICATE)
    r_edge_pad = cv2.copyMakeBorder(r_edge, pad_h_top, pad_h_bottom,
                                    pad_w_left, pad_w_right, cv2.BORDER_REPLICATE)

    if r_blur_pad.shape[0] != sh or r_blur_pad.shape[1] != sw:
        r_blur_pad = cv2.resize(r_blur_pad, (sw, sh), cv2.INTER_LINEAR)
        r_edge_pad = cv2.resize(r_edge_pad, (sw, sh), cv2.INTER_LINEAR)

    cv2.normalize(r_blur_pad, r_blur_pad, 0.0, 1.0, cv2.NORM_MINMAX)
    cv2.normalize(r_edge_pad, r_edge_pad, 0.0, 1.0, cv2.NORM_MINMAX)

    heatmap = 0.60 * r_blur_pad + 0.40 * r_edge_pad
    cv2.normalize(heatmap, heatmap, 0.0, 1.0, cv2.NORM_MINMAX)

    return heatmap
