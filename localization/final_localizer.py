"""
localization/final_localizer.py

DriftSense-X Final Principled Global-to-Local Localizer Pipeline.

Pipeline Architecture:
1. Stage 0 — Geometry & 10x Physical Scale Correction:
   - Maps 1000x1000 reference image to 100x100 search footprint (0.10x scale factor).
2. Stage 1 — Global Coarse Anchor Search:
   - Uses multi-resolution pyramidal correlation (4x downscaled) across low-frequency envelope,
     Sobel gradient, and local variance maps.
3. Stage 2 — Multi-Scale Candidate Peak Extraction:
   - Extracts top local correlation peaks across scales (0.085x to 0.115x) using spatial NMS.
4. Stage 3 — Multi-Scale Structural Signature & Illumination-Robust Feature Extraction:
   - Computes 7 independent features: local NCC, gradient NCC, LoG correlation, Canny edge overlap,
     low-frequency Gaussian envelope, macro texture variance, and multi-scale resolution signature (100x100 down to 12x12).
   - Applies robust Z-score normalization across candidate pool: z = (feat - mean) / (std + 1e-5) squashed via tanh.
5. Stage 4 — Automatic Lattice Estimation & Periodic Alias Analysis:
   - Dynamically estimates 2D lattice period lambda_x, lambda_y from reference autocorrelation / FFT.
   - Groups periodic spatial alias candidates (dx, dy = k * lambda).
6. Stage 5 & 6 — Fine Restricted Search & 2D Quadratic Subpixel Refinement:
   - Restricts fine template search to +/-35 px around winning coarse candidate.
   - Applies parabolic peak fitting for subpixel (x, y) precision.
7. Stage 7 & 8 — Confidence Estimation & Center Prior Tie-Break:
   - Center distance used ONLY as a tie-breaker when abs(top1_score - top2_score) < 0.01.
   - Rejects low-margin ambiguous predictions with status FAILED.

Dependencies: NumPy, OpenCV, standard Python libraries (time, math, argparse, os, sys).
No PyTorch, TensorFlow, pandas, or ground-truth usage during inference.
"""

import argparse
import csv
import math
import os
import sys
import time
import cv2
import numpy as np

# Import Stage 1 Global Coarse Localizer
from localization.global_coarse_localizer import locate_global_coarse, compute_sobel_gradient, compute_local_variance_map, zmuv_ncc


def estimate_lattice_period_2d(ref_img: np.ndarray) -> tuple:
    """Dynamically estimates 2D lattice periods lambda_x, lambda_y in search image coordinates."""
    if ref_img.shape[0] > 200:
        ref_s = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_s = ref_img.copy()

    ref_f = ref_s.astype(np.float32) - np.mean(ref_s)
    f = np.fft.fft2(ref_f)
    power = np.abs(f)**2
    autocorr = np.real(np.fft.ifft2(power))
    autocorr = np.fft.fftshift(autocorr)

    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[cy-2:cy+3, cx-2:cx+3] = 0.0

    _, _, _, max_loc = cv2.minMaxLoc(autocorr)
    p_dx = max_loc[0] - cx
    p_dy = max_loc[1] - cy

    scale_fac = (ref_img.shape[0] / ref_s.shape[0]) * 10.0
    lx = abs(p_dx) * scale_fac if abs(p_dx) > 2 else 67.0
    ly = abs(p_dy) * scale_fac if abs(p_dy) > 2 else 67.0

    lx = float(np.clip(lx, 30.0, 150.0))
    ly = float(np.clip(ly, 30.0, 150.0))
    return lx, ly


def compute_canny_edge(img: np.ndarray) -> np.ndarray:
    """Computes Canny edge map in float32 [0, 1]."""
    edges = cv2.Canny(img, 50, 150)
    return edges.astype(np.float32) / 255.0


def extract_local_peaks(response_map: np.ndarray, window_size: int = 5, min_thresh: float = 0.01, top_k: int = 50) -> list:
    """Extracts local peak top-left coordinates and match scores from correlation response map."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dilated = cv2.dilate(response_map, kernel)
    local_peaks = (response_map == dilated) & (response_map >= min_thresh)

    peak_y, peak_x = np.where(local_peaks)
    scores = response_map[peak_y, peak_x]

    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    candidates = []
    for idx in top_indices:
        candidates.append((int(peak_x[idx]), int(peak_y[idx]), float(scores[idx])))
    return candidates


def refine_subpixel_peak(response_map: np.ndarray, tl_x: int, tl_y: int) -> tuple:
    """Refines integer peak coordinate (tl_x, tl_y) using 2D quadratic interpolation."""
    H, W = response_map.shape
    if 1 <= tl_x < W - 1 and 1 <= tl_y < H - 1:
        R00 = response_map[tl_y, tl_x]
        Rx_pos = response_map[tl_y, tl_x + 1]
        Rx_neg = response_map[tl_y, tl_x - 1]
        Ry_pos = response_map[tl_y + 1, tl_x]
        Ry_neg = response_map[tl_y - 1, tl_x]

        denom_x = 2.0 * (2.0 * R00 - Rx_pos - Rx_neg)
        denom_y = 2.0 * (2.0 * R00 - Ry_pos - Ry_neg)

        dx = (Rx_pos - Rx_neg) / denom_x if abs(denom_x) > 1e-6 else 0.0
        dy = (Ry_pos - Ry_neg) / denom_y if abs(denom_y) > 1e-6 else 0.0

        dx = float(np.clip(dx, -0.5, 0.5))
        dy = float(np.clip(dy, -0.5, 0.5))

        return tl_x + dx, tl_y + dy
    return float(tl_x), float(tl_y)


def is_lattice_alias_2d(c1: dict, c2: dict, lx: float = 67.0, ly: float = 67.0, tolerance: float = 12.0) -> bool:
    """Checks if candidate c2 is a periodic lattice alias of candidate c1."""
    dx = abs(c1['center_x'] - c2['center_x'])
    dy = abs(c1['center_y'] - c2['center_y'])
    dist = math.hypot(dx, dy)

    if dist < 12.0:
        return True  # Near duplicate

    kx = round(dx / lx)
    ky = round(dy / ly)

    rx = abs(dx - kx * lx)
    ry = abs(dy - ky * ly)

    is_alias_x = (1 <= kx <= 5 and rx <= tolerance and (ky == 0 or ry <= tolerance))
    is_alias_y = (1 <= ky <= 5 and ry <= tolerance and (kx == 0 or rx <= tolerance))
    is_alias_diag = (1 <= kx <= 5 and 1 <= ky <= 5 and rx <= tolerance and ry <= tolerance)

    return is_alias_x or is_alias_y or is_alias_diag


def compute_multi_scale_signature(search_patch: np.ndarray, ref_tmpl: np.ndarray) -> float:
    """Computes multi-resolution structural signature score across 100x100 down to 12x12."""
    resolutions = [100, 50, 25, 12]
    scores = []

    for r in resolutions:
        if search_patch.shape[0] < r or search_patch.shape[1] < r:
            continue
        p_r = cv2.resize(search_patch, (r, r), cv2.INTER_AREA)
        t_r = cv2.resize(ref_tmpl, (r, r), cv2.INTER_AREA)

        z_int = zmuv_ncc(p_r, t_r)
        z_grad = zmuv_ncc(compute_sobel_gradient(p_r), compute_sobel_gradient(t_r))

        scores.append(0.5 * z_int + 0.5 * z_grad)

    return float(np.mean(scores)) if scores else 0.0


def locate_reference_pattern_final(
    ref_path: str,
    search_path: str,
    min_confidence_threshold: float = 0.12,
    tie_break_threshold: float = 0.01
) -> tuple:
    """
    Final Principled Global-to-Local Localizer Pipeline.

    Returns:
        tuple: (coarse_center, fine_center, confidence, status, debug_info)
    """
    start_t = time.perf_counter()

    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_raw is None:
        raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_raw is None:
        raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    ref_gray_f = ref_raw.astype(np.float32)
    search_gray_f = search_raw.astype(np.float32)

    search_h, search_w = search_raw.shape[:2]

    # STAGE 0 & 1 — GLOBAL COARSE SEARCH
    coarse_x, coarse_y, coarse_score, unc_radius, coarse_cands, coarse_debug = locate_global_coarse(ref_raw, search_raw)

    # STAGE 2 — MULTI-SCALE CANDIDATE PEAK EXTRACTION ACROSS SCALES
    ref_grad = compute_sobel_gradient(ref_raw)
    search_grad = compute_sobel_gradient(search_raw)

    ref_edge = compute_canny_edge(ref_raw)
    search_edge = compute_canny_edge(search_raw)

    ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)
    search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)

    search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)
    search_var = compute_local_variance_map(search_raw, ksize=15)

    scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
    res_gray_maps = {}
    res_grad_maps = {}
    res_log_maps = {}
    res_edge_maps = {}
    cand_peaks = []

    # Pad search images to handle border crops gracefully
    max_sw = int(round(ref_raw.shape[1] * max(scales)))
    pad = max_sw // 2 + 10
    search_gray_pad = cv2.copyMakeBorder(search_gray_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_grad_pad = cv2.copyMakeBorder(search_grad, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_log_pad = cv2.copyMakeBorder(search_log, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_edge_pad = cv2.copyMakeBorder(search_edge, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_blur_pad = cv2.copyMakeBorder(search_blur, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_var_pad = cv2.copyMakeBorder(search_var, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for s in scales:
        scaled_w = int(round(ref_raw.shape[1] * s))
        scaled_h = int(round(ref_raw.shape[0] * s))

        if scaled_w <= 0 or scaled_h <= 0 or scaled_w > search_w or scaled_h > search_h:
            continue

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_log = cv2.resize(ref_log, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_edge = cv2.resize(ref_edge, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        rg = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
        rd = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)
        rl = cv2.matchTemplate(search_log, s_ref_log, cv2.TM_CCOEFF_NORMED)

        res_gray_maps[s] = (rg, scaled_w, scaled_h, s_ref_gray)
        res_grad_maps[s] = (rd, scaled_w, scaled_h, s_ref_grad)
        res_log_maps[s] = (rl, scaled_w, scaled_h, s_ref_log)
        res_edge_maps[s] = s_ref_edge

        peaks_g = extract_local_peaks(rg, window_size=5, min_thresh=0.01, top_k=50)
        peaks_d = extract_local_peaks(rd, window_size=5, min_thresh=0.01, top_k=50)
        peaks_l = extract_local_peaks(rl, window_size=5, min_thresh=0.01, top_k=50)

        peak_locs = set([(x, y) for x, y, _ in peaks_g] + [(x, y) for x, y, _ in peaks_d] + [(x, y) for x, y, _ in peaks_l])

        for tl_x, tl_y in peak_locs:
            cx = tl_x + (scaled_w / 2.0)
            cy = tl_y + (scaled_h / 2.0)

            if cx < 40.0 or cy < 40.0 or cx > (search_w - 40.0) or cy > (search_h - 40.0):
                continue

            score_g = float(rg[tl_y, tl_x]) if 0 <= tl_y < rg.shape[0] and 0 <= tl_x < rg.shape[1] else 0.0
            score_d = float(rd[tl_y, tl_x]) if 0 <= tl_y < rd.shape[0] and 0 <= tl_x < rd.shape[1] else 0.0
            score_l = float(rl[tl_y, tl_x]) if 0 <= tl_y < rl.shape[0] and 0 <= tl_x < rl.shape[1] else 0.0

            raw_match = 0.40 * score_g + 0.40 * score_d + 0.20 * score_l
            cand_peaks.append((cx, cy, s, raw_match, score_g, score_d, score_l))

    if not cand_peaks:
        elapsed_sec = time.perf_counter() - start_t
        return (coarse_x, coarse_y), None, 0.0, "FAILED", {"computation_time_sec": elapsed_sec}

    # Spatial NMS to deduplicate candidate pool to top 50 unique spatial locations
    cand_peaks.sort(key=lambda c: c[3], reverse=True)
    top_candidates = []
    for c in cand_peaks:
        cx, cy, s, raw_match, score_g, score_d, score_l = c
        too_close = False
        for k in top_candidates:
            if math.hypot(cx - k['center_x'], cy - k['center_y']) < 12.0:
                too_close = True
                break
        if not too_close:
            top_candidates.append({
                'center_x': cx,
                'center_y': cy,
                'primary_scale': s,
                'raw_template_score': float(raw_match),
                'score_gray': float(score_g),
                'score_grad': float(score_d),
                'score_log': float(score_l)
            })
        if len(top_candidates) >= 50:
            break

    # STAGE 3 — FEATURE EXTRACTION (7 Independent Normalized Features)
    feature_matrix = {
        'ncc': np.zeros(len(top_candidates), dtype=np.float32),
        'gradient': np.zeros(len(top_candidates), dtype=np.float32),
        'log': np.zeros(len(top_candidates), dtype=np.float32),
        'edge': np.zeros(len(top_candidates), dtype=np.float32),
        'low_frequency': np.zeros(len(top_candidates), dtype=np.float32),
        'macro': np.zeros(len(top_candidates), dtype=np.float32),
        'texture': np.zeros(len(top_candidates), dtype=np.float32),
        'multi_scale': np.zeros(len(top_candidates), dtype=np.float32)
    }

    ref_blur_10x = cv2.GaussianBlur(cv2.resize(ref_gray_f, (100, 100), cv2.INTER_AREA), (15, 15), 3.0)
    ref_var_10x = compute_local_variance_map(cv2.resize(ref_raw, (100, 100), cv2.INTER_AREA), ksize=5)
    s_ref_10x = cv2.resize(ref_gray_f, (100, 100), cv2.INTER_AREA)

    for c_idx, cand in enumerate(top_candidates):
        cx, cy = cand['center_x'], cand['center_y']
        s = cand['primary_scale']
        sw = int(round(ref_raw.shape[1] * s))
        sh = int(round(ref_raw.shape[0] * s))

        tl_x_pad = int(round(cx + pad - sw / 2.0))
        tl_y_pad = int(round(cy + pad - sh / 2.0))

        patch_g = search_gray_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
        patch_d = search_grad_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
        patch_l = search_log_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
        patch_e = search_edge_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
        patch_b = search_blur_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
        patch_v = search_var_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]

        s_ref_g = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)
        s_ref_d = cv2.resize(ref_grad, (sw, sh), cv2.INTER_AREA)
        s_ref_l = cv2.resize(ref_log, (sw, sh), cv2.INTER_AREA)
        s_ref_e = res_edge_maps[s]
        ref_edge_dilated = cv2.dilate(s_ref_e, np.ones((3, 3), np.float32))

        z_g = zmuv_ncc(patch_g, s_ref_g)
        z_d = zmuv_ncc(patch_d, s_ref_d)
        z_l = zmuv_ncc(patch_l, s_ref_l)

        edge_cnt = np.sum(patch_e > 0.1)
        e_overlap = float(np.sum((patch_e > 0.1) & (ref_edge_dilated > 0.1)) / float(edge_cnt)) if edge_cnt > 0 else 0.0

        p_blur_100 = cv2.resize(patch_b, (100, 100), cv2.INTER_AREA)
        z_lf = zmuv_ncc(p_blur_100, ref_blur_10x)

        p_var_100 = cv2.resize(patch_v, (100, 100), cv2.INTER_AREA)
        z_var = zmuv_ncc(p_var_100, ref_var_10x)

        # Surrounding macro context (300x300 window)
        sw_01 = int(round(ref_gray_f.shape[1] * 0.10))
        sh_01 = int(round(ref_gray_f.shape[0] * 0.10))
        ctx_w = min(search_w, sw_01 * 3)
        ctx_h = min(search_h, sh_01 * 3)
        x1_c = max(0, int(round(cx - ctx_w / 2.0)))
        y1_c = max(0, int(round(cy - ctx_h / 2.0)))
        x2_c = min(search_w, int(round(cx + ctx_w / 2.0)))
        y2_c = min(search_h, int(round(cy + ctx_h / 2.0)))

        s_ctx = search_gray_f[y1_c:y2_c, x1_c:x2_c]
        r_ctx = cv2.resize(ref_gray_f, (x2_c - x1_c, y2_c - y1_c), cv2.INTER_AREA)
        s_ctx_p = cv2.resize(s_ctx, (30, 30), cv2.INTER_AREA)
        r_ctx_p = cv2.resize(r_ctx, (30, 30), cv2.INTER_AREA)
        macro_score = float(max(0.0, cv2.matchTemplate(s_ctx_p, r_ctx_p, cv2.TM_CCOEFF_NORMED)[0, 0]))

        multi_scale_sig = compute_multi_scale_signature(cv2.resize(patch_g, (100, 100), cv2.INTER_AREA), s_ref_10x)

        feature_matrix['ncc'][c_idx] = z_g
        feature_matrix['gradient'][c_idx] = z_d
        feature_matrix['log'][c_idx] = z_l
        feature_matrix['edge'][c_idx] = e_overlap
        feature_matrix['low_frequency'][c_idx] = z_lf
        feature_matrix['texture'][c_idx] = z_var
        feature_matrix['macro'][c_idx] = macro_score
        feature_matrix['multi_scale'][c_idx] = multi_scale_sig

    # ROBUST Z-SCORE NORMALIZATION ACROSS CANDIDATE POOL
    norm_features = {feat: np.zeros_like(feature_matrix[feat]) for feat in feature_matrix}
    for feat in feature_matrix:
        col = feature_matrix[feat]
        m_val = np.mean(col)
        std_val = np.std(col)
        denom = std_val if std_val > 1e-5 else 1.0
        z_scores = (col - m_val) / denom
        norm_features[feat] = np.tanh(0.5 * z_scores)

    # CANDIDATE SCORE COMPUTATION (Stage 8 Formula)
    for c_idx, cand in enumerate(top_candidates):
        cand['ncc'] = float(feature_matrix['ncc'][c_idx])
        cand['gradient'] = float(feature_matrix['gradient'][c_idx])
        cand['log'] = float(feature_matrix['log'][c_idx])
        cand['edge'] = float(feature_matrix['edge'][c_idx])
        cand['low_frequency'] = float(feature_matrix['low_frequency'][c_idx])
        cand['texture'] = float(feature_matrix['texture'][c_idx])
        cand['macro'] = float(feature_matrix['macro'][c_idx])
        cand['multi_scale'] = float(feature_matrix['multi_scale'][c_idx])

        # Transparent Score Formula: Low-frequency + Multi-scale signature dominance
        final_score = float(
            0.25 * norm_features['low_frequency'][c_idx] +
            0.20 * norm_features['log'][c_idx] +
            0.15 * norm_features['gradient'][c_idx] +
            0.15 * norm_features['macro'][c_idx] +
            0.15 * norm_features['multi_scale'][c_idx] +
            0.05 * norm_features['edge'][c_idx] +
            0.05 * norm_features['ncc'][c_idx]
        )

        cand['final_score'] = final_score
        cand['coarse_score'] = final_score

    # STAGE 4 — DYNAMIC LATTICE PERIOD ESTIMATION & PERIODIC ALIAS ANALYSIS
    lx, ly = estimate_lattice_period_2d(ref_raw)

    alias_groups = []
    visited_indices = set()
    top_candidates.sort(key=lambda c: c['final_score'], reverse=True)

    for i, c in enumerate(top_candidates):
        if i in visited_indices:
            continue
        group = [c]
        visited_indices.add(i)
        for j in range(i + 1, len(top_candidates)):
            if j in visited_indices:
                continue
            c_other = top_candidates[j]
            if any(is_lattice_alias_2d(c_other, member, lx=lx, ly=ly, tolerance=12.0) for member in group):
                group.append(c_other)
                visited_indices.add(j)
        alias_groups.append(group)

    # INTRA-GROUP ALIAS DISAMBIGUATION
    for g_idx, group in enumerate(alias_groups, start=1):
        for member in group:
            member['alias_group_id'] = g_idx
        group.sort(key=lambda c: c['final_score'], reverse=True)

    # Sort candidates by final_score descending
    top_candidates.sort(key=lambda c: c['final_score'], reverse=True)
    best_coarse = top_candidates[0]

    # STAGE 10 — CENTER TIE-BREAK CHECK
    if len(top_candidates) > 1:
        top1 = top_candidates[0]
        top2 = top_candidates[1]
        if abs(top1['final_score'] - top2['final_score']) < tie_break_threshold:
            dist1 = math.hypot(top1['center_x'] - 500.0, top1['center_y'] - 500.0)
            dist2 = math.hypot(top2['center_x'] - 500.0, top2['center_y'] - 500.0)
            if dist2 < dist1:
                best_coarse = top2

    coarse_x, coarse_y = best_coarse['center_x'], best_coarse['center_y']

    # STAGE 5 & 6 — FINE RESTRICTED LOCALIZATION (+/- 35 px window around winning coarse candidate)
    best_fine_score = -1.0
    best_fine_dict = None
    window_radius = 35

    for s in scales:
        scaled_w = int(round(ref_raw.shape[1] * s))
        scaled_h = int(round(ref_raw.shape[0] * s))

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_edge = cv2.resize(ref_edge, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        ref_edge_dilated = cv2.dilate(s_ref_edge, np.ones((3, 3), np.float32))

        min_tl_x = max(0, int(round(coarse_x - window_radius - scaled_w / 2.0)))
        max_tl_x = min(search_w - scaled_w, int(round(coarse_x + window_radius - scaled_w / 2.0)))
        min_tl_y = max(0, int(round(coarse_y - window_radius - scaled_h / 2.0)))
        max_tl_y = min(search_h - scaled_h, int(round(coarse_y + window_radius - scaled_h / 2.0)))

        if min_tl_x >= max_tl_x or min_tl_y >= max_tl_y:
            continue

        crop_search_g = search_gray_f[min_tl_y:max_tl_y+scaled_h, min_tl_x:max_tl_x+scaled_w]
        crop_search_d = search_grad[min_tl_y:max_tl_y+scaled_h, min_tl_x:max_tl_x+scaled_w]

        res_g = cv2.matchTemplate(crop_search_g, s_ref_gray, cv2.TM_CCOEFF_NORMED)
        res_d = cv2.matchTemplate(crop_search_d, s_ref_grad, cv2.TM_CCOEFF_NORMED)

        res_combined = 0.5 * res_g + 0.5 * res_d
        _, max_v, _, max_l = cv2.minMaxLoc(res_combined)

        local_tl_x, local_tl_y = max_l[0], max_l[1]
        abs_tl_x = min_tl_x + local_tl_x
        abs_tl_y = min_tl_y + local_tl_y

        patch_e = search_edge[abs_tl_y:abs_tl_y+scaled_h, abs_tl_x:abs_tl_x+scaled_w]
        edge_cnt = np.sum(patch_e > 0.1)
        edge_overlap = float(np.sum((patch_e > 0.1) & (ref_edge_dilated > 0.1)) / float(edge_cnt)) if edge_cnt > 0 else 0.0

        r_int = float(res_g[local_tl_y, local_tl_x])
        r_grad = float(res_d[local_tl_y, local_tl_x])

        fine_score = 0.40 * r_int + 0.40 * r_grad + 0.20 * edge_overlap

        if fine_score > best_fine_score:
            best_fine_score = fine_score
            sub_x, sub_y = refine_subpixel_peak(res_combined, local_tl_x, local_tl_y)
            fine_center_x = min_tl_x + sub_x + scaled_w / 2.0
            fine_center_y = min_tl_y + sub_y + scaled_h / 2.0

            best_fine_dict = {
                'fine_center_x': float(fine_center_x),
                'fine_center_y': float(fine_center_y),
                'top_left': (abs_tl_x, abs_tl_y),
                'scaled_w': scaled_w,
                'scaled_h': scaled_h,
                'scale': s,
                'fine_score': float(fine_score),
                'r_int': r_int,
                'r_grad': r_grad,
                'edge_overlap': edge_overlap,
                'res_map': res_combined,
                'local_tl': (local_tl_x, local_tl_y)
            }

    elapsed_sec = time.perf_counter() - start_t

    if best_fine_dict is None:
        return (coarse_x, coarse_y), None, 0.0, "FAILED", {"computation_time_sec": elapsed_sec}

    fine_x = best_fine_dict['fine_center_x']
    fine_y = best_fine_dict['fine_center_y']

    # STAGE 7 — CONFIDENCE CALCULATION
    res_map = best_fine_dict['res_map']
    loc_x, loc_y = best_fine_dict['local_tl']
    H, W = res_map.shape
    peak_val = float(res_map[loc_y, loc_x])

    mask = np.ones_like(res_map, dtype=bool)
    mask[max(0, loc_y-2):min(H, loc_y+3), max(0, loc_x-2):min(W, loc_x+3)] = False
    sidelobe_val = float(np.max(res_map[mask])) if np.any(mask) else 0.0

    margin = float(max(0.0, peak_val - sidelobe_val))
    confidence = float(np.clip(best_fine_score * (1.0 + margin), 0.0, 1.0))

    if confidence < min_confidence_threshold or best_fine_score < 0.12:
        status = "FAILED"
        final_center = None
    else:
        status = "SUCCESS"
        final_center = (fine_x, fine_y)

    debug_info = {
        "search_img": search_raw,
        "ref_raw": ref_raw,
        "coarse_center": (coarse_x, coarse_y),
        "all_candidates": top_candidates,
        "fine_dict": best_fine_dict,
        "final_center": final_center,
        "confidence": confidence,
        "status": status,
        "computation_time_sec": elapsed_sec,
        "lattice_period": (lx, ly)
    }

    return (coarse_x, coarse_y), final_center, confidence, status, debug_info
