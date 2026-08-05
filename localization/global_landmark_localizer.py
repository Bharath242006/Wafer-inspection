"""
localization/global_landmark_localizer.py

Deterministic Global Landmark Localizer for DriftSense-X.

Pipeline:
1. Global Search Image Analysis: Extract low-frequency Gaussian intensity maps, macro boundary maps, and global gradient field.
2. Reference Global Context: Downscale 1000x1000 reference to 100x100 macro template.
3. Global Alignment: Multi-scale coarse template correlation and Phase Correlation to produce a global macro probability heatmap.
4. Landmark -> Candidate Filtering: Score Top-500 candidates by global landmark consistency.
5. Fine Search: Sub-pixel quadratic refinement around selected landmark center.
"""

import cv2
import math
import numpy as np

from scratch.improve_candidate_recall import generate_candidate_pool_multi, compute_sobel_gradient
from localization.global_coarse_localizer import compute_local_variance_map, zmuv_ncc
from localization.final_localizer import compute_canny_edge


def compute_global_landmark_heatmap(ref_img: np.ndarray, search_img: np.ndarray) -> np.ndarray:
    """
    Computes a 1000x1000 global landmark macro probability heatmap M(x, y).

    Uses downsampled low-frequency Gaussian correlation and edge density matching.
    """
    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)

    sh, sw = search_img.shape[:2]

    # 1. Downscale reference macro template (100x100 default, clamped to search dimensions)
    ref_macro_w = min(100, sw)
    ref_macro_h = min(100, sh)
    ref_macro = cv2.resize(ref_gray_f, (ref_macro_w, ref_macro_h), cv2.INTER_AREA)

    # 2. Low-frequency Gaussian blur maps (sigma = 30)
    k_ref = min(31, (ref_macro_h // 2) * 2 + 1)
    k_search = min(101, (sh // 2) * 2 + 1)
    ref_blur = cv2.GaussianBlur(ref_macro, (k_ref, k_ref), 10.0)
    search_blur = cv2.GaussianBlur(search_gray_f, (k_search, k_search), 30.0)

    # 3. Macro edge density maps
    ref_edge = cv2.resize(compute_canny_edge(ref_img), (ref_macro_w, ref_macro_h), cv2.INTER_AREA)
    search_edge = compute_canny_edge(search_img)
    k_edge = min(41, (sh // 2) * 2 + 1)
    search_edge_blur = cv2.GaussianBlur(search_edge, (k_edge, k_edge), 10.0)

    # Template matching at macro scale
    r_blur = cv2.matchTemplate(search_blur, ref_blur, cv2.TM_CCOEFF_NORMED)
    r_edge = cv2.matchTemplate(search_edge_blur, ref_edge, cv2.TM_CCOEFF_NORMED)

    # Pad correlation response maps back to (sh, sw) search dimensions dynamically
    pad_h_top = max(0, (sh - r_blur.shape[0]) // 2)
    pad_h_bottom = max(0, sh - r_blur.shape[0] - pad_h_top)
    pad_w_left = max(0, (sw - r_blur.shape[1]) // 2)
    pad_w_right = max(0, sw - r_blur.shape[1] - pad_w_left)

    r_blur_pad = cv2.copyMakeBorder(r_blur, pad_h_top, pad_h_bottom,
                                    pad_w_left, pad_w_right, cv2.BORDER_REPLICATE)
    r_edge_pad = cv2.copyMakeBorder(r_edge, pad_h_top, pad_h_bottom,
                                    pad_w_left, pad_w_right, cv2.BORDER_REPLICATE)

    # Guarantee exact (sw, sh) size match
    if r_blur_pad.shape[0] != sh or r_blur_pad.shape[1] != sw:
        r_blur_pad = cv2.resize(r_blur_pad, (sw, sh), cv2.INTER_LINEAR)
        r_edge_pad = cv2.resize(r_edge_pad, (sw, sh), cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    cv2.normalize(r_blur_pad, r_blur_pad, 0.0, 1.0, cv2.NORM_MINMAX)
    cv2.normalize(r_edge_pad, r_edge_pad, 0.0, 1.0, cv2.NORM_MINMAX)

    # Combined macro probability heatmap
    heatmap = 0.60 * r_blur_pad + 0.40 * r_edge_pad
    cv2.normalize(heatmap, heatmap, 0.0, 1.0, cv2.NORM_MINMAX)

    return heatmap


def locate_global_landmark(ref_img: np.ndarray, search_img: np.ndarray, top_k_cands: int = 500) -> tuple:
    """
    Locates target center by weighting Top-500 candidate pool with Global Landmark Heatmap.

    Returns:
        tuple: (pred_x, pred_y, selected_candidate_rank, global_alignment_score, candidate_scores)
    """
    # 1. Generate Top-500 candidate pool
    cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=top_k_cands)

    if not cands:
        return 500.0, 500.0, -1, 0.0, []

    # 2. Compute Global Landmark Heatmap
    heatmap = compute_global_landmark_heatmap(ref_img, search_img)
    sh, sw = search_img.shape[:2]

    # 3. Score candidates combining global landmark heatmap and local peak score
    for c in cands:
        cx, cy = c['cx'], c['cy']
        ix = int(np.clip(round(cx), 0, sw - 1))
        iy = int(np.clip(round(cy), 0, sh - 1))

        global_weight = float(heatmap[iy, ix])
        c['global_weight'] = global_weight
        # Final landmark score: 0.60 * global_weight + 0.40 * peak_score
        c['landmark_score'] = 0.60 * global_weight + 0.40 * c['score']

    # Sort candidates by landmark score
    ranked_cands = sorted(cands, key=lambda c: c['landmark_score'], reverse=True)
    winner = ranked_cands[0]

    # 4. Fine multi-scale search and quadratic sub-pixel refinement around winner center
    FINE_SCALES = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
    FINE_WINDOW_RADIUS = 35

    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)
    ref_grad = compute_sobel_gradient(ref_img)
    search_grad = compute_sobel_gradient(search_img)

    cx_w, cy_w = winner['cx'], winner['cy']
    best_score = -1.0
    fine_x, fine_y = cx_w, cy_w

    for s in FINE_SCALES:
        scaled_w = int(round(ref_img.shape[1] * s))
        scaled_h = int(round(ref_img.shape[0] * s))
        if scaled_w <= 0 or scaled_h <= 0 or scaled_w > sw or scaled_h > sh:
            continue

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), cv2.INTER_AREA)

        min_tl_x = max(0, int(round(cx_w - FINE_WINDOW_RADIUS - scaled_w / 2.0)))
        max_tl_x = min(sw - scaled_w, int(round(cx_w + FINE_WINDOW_RADIUS - scaled_w / 2.0)))
        min_tl_y = max(0, int(round(cy_w - FINE_WINDOW_RADIUS - scaled_h / 2.0)))
        max_tl_y = min(sh - scaled_h, int(round(cy_w + FINE_WINDOW_RADIUS - scaled_h / 2.0)))

        if min_tl_x >= max_tl_x or min_tl_y >= max_tl_y:
            continue

        crop_g = search_gray_f[min_tl_y:max_tl_y + scaled_h, min_tl_x:max_tl_x + scaled_w]
        crop_d = search_grad[min_tl_y:max_tl_y + scaled_h, min_tl_x:max_tl_x + scaled_w]

        res_g = cv2.matchTemplate(crop_g, s_ref_gray, cv2.TM_CCOEFF_NORMED)
        res_d = cv2.matchTemplate(crop_d, s_ref_grad, cv2.TM_CCOEFF_NORMED)
        res_combined = 0.5 * res_g + 0.5 * res_d

        _, max_v, _, max_l = cv2.minMaxLoc(res_combined)
        if float(max_v) > best_score:
            best_score = float(max_v)
            # Quadratic subpixel fitting
            H_res, W_res = res_combined.shape
            lx_idx, ly_idx = max_l[0], max_l[1]
            dx_sub, dy_sub = 0.0, 0.0
            if 1 <= lx_idx < W_res - 1 and 1 <= ly_idx < H_res - 1:
                r0 = res_combined[ly_idx, lx_idx]
                rx_p = res_combined[ly_idx, lx_idx + 1]
                rx_n = res_combined[ly_idx, lx_idx - 1]
                ry_p = res_combined[ly_idx + 1, lx_idx]
                ry_n = res_combined[ly_idx - 1, lx_idx]

                denom_x = 2.0 * (2.0 * r0 - rx_p - rx_n)
                denom_y = 2.0 * (2.0 * r0 - ry_p - ry_n)

                if abs(denom_x) > 1e-6:
                    dx_sub = float(np.clip((rx_p - rx_n) / denom_x, -0.5, 0.5))
                if abs(denom_y) > 1e-6:
                    dy_sub = float(np.clip((ry_p - ry_n) / denom_y, -0.5, 0.5))

            fine_x = min_tl_x + lx_idx + dx_sub + scaled_w / 2.0
            fine_y = min_tl_y + ly_idx + dy_sub + scaled_h / 2.0

    selected_rank = int([i for i, c in enumerate(cands) if c['cx'] == winner['cx'] and c['cy'] == winner['cy']][0]) + 1
    alignment_score = float(winner['landmark_score'])

    return fine_x, fine_y, selected_rank, alignment_score, ranked_cands
