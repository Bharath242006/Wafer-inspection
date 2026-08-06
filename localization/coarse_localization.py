"""
localization/coarse_localization.py

Stage 1 Global Coarse Anchor Search using multi-resolution pyramidal correlation.
"""

import cv2
import numpy as np
from localization.features.edge_features import compute_sobel_gradient
from localization.matching.template_matching import compute_local_variance_map
from localization.candidate_generation import extract_local_peaks


def locate_global_coarse(ref_gray: np.ndarray, search_gray: np.ndarray) -> tuple:
    """
    Global Coarse Search: Establishes macro target location in 1000x1000 search space with uncertainty radius ~40 px.
    """
    search_h, search_w = search_gray.shape[:2]
    ref_h, ref_w = ref_gray.shape[:2]

    scaled_w = int(round(ref_w * 0.10))
    scaled_h = int(round(ref_h * 0.10))
    s_ref_gray = cv2.resize(ref_gray.astype(np.float32), (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    search_grad = compute_sobel_gradient(search_gray)
    ref_grad = compute_sobel_gradient(s_ref_gray)

    p_scale = 0.25
    s_search_grad_p = cv2.resize(search_grad, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
    s_ref_grad_p = cv2.resize(ref_grad, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)

    r_grad_p = cv2.matchTemplate(s_search_grad_p, s_ref_grad_p, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(r_grad_p)

    tl_x_pyramid, tl_y_pyramid = max_loc
    tl_x_search = int(round(tl_x_pyramid / p_scale))
    tl_y_search = int(round(tl_y_pyramid / p_scale))

    coarse_cx = tl_x_search + scaled_w / 2.0
    coarse_cy = tl_y_search + scaled_h / 2.0

    return coarse_cx, coarse_cy, float(max_val), 40.0, [], {}
