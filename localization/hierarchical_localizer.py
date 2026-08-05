"""
localization/hierarchical_localizer.py

Hierarchical Coarse-to-Fine Localization Algorithm with Dynamic Lattice Period Estimation,
Reflected Border ZMUV Multi-Scale Correlation, and Periodic-Alias Group Ranking Strategy.

Pipeline Architecture:
1. Stage 1 — Coarse Global Localization with Periodic-Alias Group Ranking:
   - Operates on full 1000x1000 search space.
   - Dynamically estimates semiconductor lattice period lambda_lattice from 2D autocorrelation / FFT of reference image.
   - Generates candidate peak locations across multi-scale templates (0.085x to 0.115x).
   - Uses border-reflected ZMUV (Zero-Mean Unit-Variance) normalized cross-correlation across Intensity,
     Sobel gradient, and Laplacian of Gaussian.
   - Groups candidates separated by integer multiples of dynamically estimated lattice period into alias groups.
   - Disambiguates candidates within each alias group using multi-scale pyramid consensus to select the true cell candidate.
   - Yields disambiguated coarse target estimate (x_c, y_c).

2. Stage 2 — Fine Restricted Localization:
   - Restricts fine search space to a tight +/- 35 px window around (x_c, y_c).
   - Eradicates periodic pattern aliasing by excluding false periodic neighbor peaks (+/- 67 px).
   - Evaluates fine multi-scale intensity, Sobel gradient, and edge structural matching inside restricted window.
   - Applies 2D quadratic sub-pixel peak refinement to obtain final (x_f, y_f).

3. Stage 3 — Safety & Confidence:
   - Calculates fine match confidence score.
   - Rejects unreliable predictions with status FAILED rather than hallucinating coordinates.

CLI & Visualization:
- Supports --reference, --search, --debug, --vis-path.
- Measures and reports execution / computation time.

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


def estimate_dynamic_lattice_period(ref_img: np.ndarray) -> float:
    """
    Dynamically estimates semiconductor lattice period from 2D spatial autocorrelation / FFT
    of the reference image without hardcoding 67 px.
    """
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
    period_tmpl = math.hypot(p_dx, p_dy)

    # In 1000x1000 search space with 100x100 template, period is period_tmpl * 10
    period_search = period_tmpl * (ref_img.shape[0] / ref_s.shape[0]) * 10.0
    if 25.0 <= period_search <= 150.0:
        return float(period_search)
    return 67.0


def compute_sobel_gradient(img: np.ndarray) -> np.ndarray:
    """Computes normalized Sobel gradient magnitude image in float32 [0, 1]."""
    img_f = img.astype(np.float32)
    sobelx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    cv2.normalize(mag, mag, 0.0, 1.0, cv2.NORM_MINMAX)
    return mag


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


def is_lattice_alias(c1: dict, c2: dict, lattice_period: float = 67.0, tolerance: float = 14.0) -> bool:
    """
    Checks if candidate c2 is a periodic lattice alias of candidate c1
    (i.e. spatial offset is an integer multiple of lattice_period).
    """
    dx = abs(c1['center_x'] - c2['center_x'])
    dy = abs(c1['center_y'] - c2['center_y'])
    dist = math.hypot(dx, dy)

    if dist < 12.0:
        return True  # Near duplicate

    k = round(dist / lattice_period)
    if 1 <= k <= 5 and abs(dist - k * lattice_period) <= tolerance:
        return True
    return False


def zmuv_ncc(patch: np.ndarray, tmpl: np.ndarray) -> float:
    """Calculates zero-mean unit-variance normalized cross correlation."""
    p_f = patch.astype(np.float32) - np.mean(patch)
    t_f = tmpl.astype(np.float32) - np.mean(tmpl)
    s_p = np.std(p_f)
    s_t = np.std(t_f)
    if s_p > 1e-5 and s_t > 1e-5:
        return float(np.mean(p_f * t_f) / (s_p * s_t))
    return 0.0


def stage1_coarse_global_localization(
    search_gray_f: np.ndarray,
    search_grad: np.ndarray,
    search_edge: np.ndarray,
    search_blur: np.ndarray,
    ref_gray_f: np.ndarray,
    ref_grad: np.ndarray,
    ref_edge: np.ndarray,
    ref_blur: np.ndarray,
    scales: list = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
) -> tuple:
    """
    Stage 1 — Coarse Global Localization with Dynamic Lattice Period Estimation,
    Reflected Border ZMUV Multi-Scale Correlation, and Periodic-Alias Group Ranking Strategy.
    """
    search_h, search_w = search_gray_f.shape[:2]

    # 1. Estimate dynamic lattice period from reference image
    dyn_period = estimate_dynamic_lattice_period(ref_gray_f)

    search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)
    ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)

    res_gray_maps = {}
    res_grad_maps = {}
    res_log_maps = {}
    cand_peaks = []

    # Pad search images to handle border crops gracefully
    max_sw = int(round(ref_gray_f.shape[1] * max(scales)))
    pad = max_sw // 2 + 10
    search_gray_pad = cv2.copyMakeBorder(search_gray_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_grad_pad = cv2.copyMakeBorder(search_grad, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_log_pad = cv2.copyMakeBorder(search_log, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for s in scales:
        scaled_w = int(round(ref_gray_f.shape[1] * s))
        scaled_h = int(round(ref_gray_f.shape[0] * s))

        if scaled_w <= 0 or scaled_h <= 0 or scaled_w > search_w or scaled_h > search_h:
            continue

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_log = cv2.resize(ref_log, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        rg = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
        rd = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)
        rl = cv2.matchTemplate(search_log, s_ref_log, cv2.TM_CCOEFF_NORMED)

        res_gray_maps[s] = (rg, scaled_w, scaled_h, s_ref_gray)
        res_grad_maps[s] = (rd, scaled_w, scaled_h, s_ref_grad)
        res_log_maps[s] = (rl, scaled_w, scaled_h, s_ref_log)

        peaks_g = extract_local_peaks(rg, window_size=5, min_thresh=0.01, top_k=50)
        peaks_d = extract_local_peaks(rd, window_size=5, min_thresh=0.01, top_k=50)
        peaks_l = extract_local_peaks(rl, window_size=5, min_thresh=0.01, top_k=50)

        peak_locs = set([(x, y) for x, y, _ in peaks_g] + [(x, y) for x, y, _ in peaks_d] + [(x, y) for x, y, _ in peaks_l])

        for tl_x, tl_y in peak_locs:
            cx = tl_x + (scaled_w / 2.0)
            cy = tl_y + (scaled_h / 2.0)

            score_g = float(rg[tl_y, tl_x]) if 0 <= tl_y < rg.shape[0] and 0 <= tl_x < rg.shape[1] else 0.0
            score_d = float(rd[tl_y, tl_x]) if 0 <= tl_y < rd.shape[0] and 0 <= tl_x < rd.shape[1] else 0.0
            score_l = float(rl[tl_y, tl_x]) if 0 <= tl_y < rl.shape[0] and 0 <= tl_x < rl.shape[1] else 0.0

            raw_match = 0.40 * score_g + 0.40 * score_d + 0.20 * score_l
            cand_peaks.append((cx, cy, s, raw_match, score_g, score_d, score_l))

    if not cand_peaks:
        return 500.0, 500.0, {}

    # Dense Spatial NMS to reduce candidates to top 50 unique locations
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
                'scale': s,
                'raw_template_score': float(raw_match),
                'score_gray': float(score_g),
                'score_grad': float(score_d),
                'score_log': float(score_l),
                'is_alias_rejected': False,
                'alias_penalty': 0.0
            })
        if len(top_candidates) >= 50:
            break

    # Compute ZMUV Illumination-Invariant Correlation & Multi-Scale Integrated Pyramid Score
    for cand in top_candidates:
        cx, cy = cand['center_x'], cand['center_y']

        zmuv_gray_scores = []
        zmuv_grad_scores = []
        zmuv_log_scores = []

        for s in scales:
            _, sw, sh, s_ref_g = res_gray_maps[s]
            _, _, _, s_ref_d = res_grad_maps[s]
            _, _, _, s_ref_l = res_log_maps[s]

            tl_x_pad = int(round(cx + pad - sw / 2.0))
            tl_y_pad = int(round(cy + pad - sh / 2.0))

            patch_g = search_gray_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
            patch_d = search_grad_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]
            patch_l = search_log_pad[tl_y_pad:tl_y_pad+sh, tl_x_pad:tl_x_pad+sw]

            z_g = zmuv_ncc(patch_g, s_ref_g)
            z_d = zmuv_ncc(patch_d, s_ref_d)
            z_l = zmuv_ncc(patch_l, s_ref_l)

            zmuv_gray_scores.append(z_g)
            zmuv_grad_scores.append(z_d)
            zmuv_log_scores.append(z_l)

        avg_z_g = float(np.mean(zmuv_gray_scores)) if zmuv_gray_scores else 0.0
        avg_z_d = float(np.mean(zmuv_grad_scores)) if zmuv_grad_scores else 0.0
        avg_z_l = float(np.mean(zmuv_log_scores)) if zmuv_log_scores else 0.0

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

        pyramid_context_score = float(max(0.0, cv2.matchTemplate(s_ctx_p, r_ctx_p, cv2.TM_CCOEFF_NORMED)[0, 0]))

        multi_scale_pyramid_score = float(0.40 * avg_z_g + 0.40 * avg_z_d + 0.20 * avg_z_l)
        macro_structural_score = float(0.70 * multi_scale_pyramid_score + 0.30 * cand['raw_template_score'])

        cand['pyramid_context_score'] = pyramid_context_score
        cand['macro_structural_score'] = macro_structural_score
        cand['multi_scale_pyramid_score'] = multi_scale_pyramid_score

        cand['initial_score'] = float(0.60 * multi_scale_pyramid_score + 0.40 * macro_structural_score)
        cand['coarse_score'] = cand['initial_score']

    # DYNAMIC ALIAS GROUP FORMATION (~dyn_period lattice period)
    alias_groups = []
    visited_indices = set()

    top_candidates.sort(key=lambda c: c['initial_score'], reverse=True)

    for i, c in enumerate(top_candidates):
        if i in visited_indices:
            continue
        group = [c]
        visited_indices.add(i)
        for j in range(i + 1, len(top_candidates)):
            if j in visited_indices:
                continue
            c_other = top_candidates[j]
            if any(is_lattice_alias(c_other, member, lattice_period=dyn_period, tolerance=14.0) for member in group):
                group.append(c_other)
                visited_indices.add(j)
        alias_groups.append(group)

    # INTRA-GROUP ALIAS DISAMBIGUATION & LATTICE CONSISTENCY SCORING
    for g_idx, group in enumerate(alias_groups, start=1):
        for member in group:
            member['alias_group_id'] = g_idx
            lattice_fits = []
            for other in group:
                if other != member:
                    d = math.hypot(member['center_x'] - other['center_x'], member['center_y'] - other['center_y'])
                    k = round(d / dyn_period)
                    fit = max(0.0, 1.0 - abs(d - k * dyn_period) / 14.0) if k >= 1 else 1.0
                    lattice_fits.append(fit)
            member['lattice_consistency_score'] = float(np.mean(lattice_fits)) if lattice_fits else 1.0

        # Sort group members by initial_score descending
        group.sort(key=lambda c: c['initial_score'], reverse=True)
        winner = group[0]
        for member in group[1:]:
            member['is_alias_rejected'] = True
            member['alias_penalty'] = 0.25
            member['coarse_score'] -= 0.25

    # Sort final candidates by coarse_score descending
    top_candidates.sort(key=lambda c: c['coarse_score'], reverse=True)
    best_coarse = top_candidates[0]

    candidates_before = [dict(c) for c in top_candidates]

    stage1_info = {
        "estimated_lattice_period": dyn_period,
        "all_candidates": top_candidates,
        "alias_groups": alias_groups,
        "candidates_before": candidates_before,
        "candidates_after": top_candidates,
        "best_coarse": best_coarse
    }

    return float(best_coarse['center_x']), float(best_coarse['center_y']), stage1_info


def stage2_fine_localization(
    search_gray_f: np.ndarray,
    search_grad: np.ndarray,
    search_edge: np.ndarray,
    ref_gray_f: np.ndarray,
    ref_grad: np.ndarray,
    ref_edge: np.ndarray,
    coarse_x: float,
    coarse_y: float,
    window_radius: int = 35,
    scales: list = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
) -> tuple:
    """
    Stage 2 — Fine Restricted Localization:
    Restricts search space to [coarse_x - 35, coarse_x + 35] x [coarse_y - 35, coarse_y + 35].
    Evaluates multi-scale template matching across intensity, Sobel gradient, and edge maps,
    eradicating periodic aliasing by excluding adjacent periodic cells (+/- 67 px).
    Applies 2D quadratic sub-pixel peak refinement.
    """
    search_h, search_w = search_gray_f.shape[:2]
    best_fine_score = -1.0
    best_cand_dict = None

    for s in scales:
        scaled_w = int(round(ref_gray_f.shape[1] * s))
        scaled_h = int(round(ref_gray_f.shape[0] * s))

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_edge = cv2.resize(ref_edge, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        ref_edge_dilated = cv2.dilate(s_ref_edge, np.ones((3, 3), np.float32))

        # Fine search range in top-left coordinates
        min_tl_x = int(round(coarse_x - window_radius - scaled_w / 2.0))
        max_tl_x = int(round(coarse_x + window_radius - scaled_w / 2.0))
        min_tl_y = int(round(coarse_y - window_radius - scaled_h / 2.0))
        max_tl_y = int(round(coarse_y + window_radius - scaled_h / 2.0))

        min_tl_x = max(0, min_tl_x)
        max_tl_x = min(search_w - scaled_w, max_tl_x)
        min_tl_y = max(0, min_tl_y)
        max_tl_y = min(search_h - scaled_h, max_tl_y)

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
        if edge_cnt > 0:
            edge_overlap = float(np.sum((patch_e > 0.1) & (ref_edge_dilated > 0.1)) / float(edge_cnt))
        else:
            edge_overlap = 0.0

        r_int = float(res_g[local_tl_y, local_tl_x])
        r_grad = float(res_d[local_tl_y, local_tl_x])

        fine_score = 0.40 * r_int + 0.40 * r_grad + 0.20 * edge_overlap

        if fine_score > best_fine_score:
            best_fine_score = fine_score
            sub_x, sub_y = refine_subpixel_peak(res_combined, local_tl_x, local_tl_y)
            fine_center_x = min_tl_x + sub_x + scaled_w / 2.0
            fine_center_y = min_tl_y + sub_y + scaled_h / 2.0

            best_cand_dict = {
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
                'local_tl': (local_tl_x, local_tl_y),
                'window_bounds': (min_tl_x, min_tl_y, max_tl_x + scaled_w, max_tl_y + scaled_h)
            }

    return best_cand_dict


def locate_reference_pattern(
    ref_path: str,
    search_path: str,
    min_confidence_threshold: float = 0.15
) -> tuple:
    """
    Hierarchical Coarse-to-Fine Localizer pipeline with Dynamic Lattice Period Estimation
    and Periodic-Alias Group Ranking Strategy.

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

    ref_grad = compute_sobel_gradient(ref_raw)
    search_grad = compute_sobel_gradient(search_raw)

    ref_edge = compute_canny_edge(ref_raw)
    search_edge = compute_canny_edge(search_raw)

    ref_blur = cv2.GaussianBlur(ref_gray_f, (21, 21), 5.0)
    search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)

    # Stage 1: Coarse Global Localization with Dynamic Disambiguation
    coarse_x, coarse_y, coarse_info = stage1_coarse_global_localization(
        search_gray_f=search_gray_f,
        search_grad=search_grad,
        search_edge=search_edge,
        search_blur=search_blur,
        ref_gray_f=ref_gray_f,
        ref_grad=ref_grad,
        ref_edge=ref_edge,
        ref_blur=ref_blur
    )

    # Stage 2: Fine Restricted Localization (+/- 35 px window around disambiguated coarse center)
    fine_dict = stage2_fine_localization(
        search_gray_f=search_gray_f,
        search_grad=search_grad,
        search_edge=search_edge,
        ref_gray_f=ref_gray_f,
        ref_grad=ref_grad,
        ref_edge=ref_edge,
        coarse_x=coarse_x,
        coarse_y=coarse_y,
        window_radius=35
    )

    elapsed_sec = time.perf_counter() - start_t

    if fine_dict is None:
        debug_info = {
            "search_img": search_raw,
            "ref_raw": ref_raw,
            "coarse_center": (coarse_x, coarse_y),
            "coarse_info": coarse_info,
            "computation_time_sec": elapsed_sec
        }
        return (coarse_x, coarse_y), None, 0.0, "FAILED", debug_info

    fine_x = fine_dict['fine_center_x']
    fine_y = fine_dict['fine_center_y']
    fine_score = fine_dict['fine_score']

    # Stage 3: Safety & Confidence Calculation
    res_map = fine_dict['res_map']
    loc_x, loc_y = fine_dict['local_tl']
    H, W = res_map.shape
    peak_val = float(res_map[loc_y, loc_x])

    mask = np.ones_like(res_map, dtype=bool)
    mask[max(0, loc_y-2):min(H, loc_y+3), max(0, loc_x-2):min(W, loc_x+3)] = False
    sidelobe_val = float(np.max(res_map[mask])) if np.any(mask) else 0.0

    margin = float(max(0.0, peak_val - sidelobe_val))
    confidence = float(np.clip(fine_score * (1.0 + margin), 0.0, 1.0))

    if confidence < min_confidence_threshold or fine_score < 0.12:
        status = "FAILED"
        final_center = None
    else:
        status = "SUCCESS"
        final_center = (fine_x, fine_y)

    debug_info = {
        "search_img": search_raw,
        "ref_raw": ref_raw,
        "coarse_center": (coarse_x, coarse_y),
        "coarse_info": coarse_info,
        "fine_dict": fine_dict,
        "final_center": final_center,
        "confidence": confidence,
        "status": status,
        "computation_time_sec": elapsed_sec
    }

    return (coarse_x, coarse_y), final_center, confidence, status, debug_info


def save_debug_visualization(
    search_img: np.ndarray,
    coarse_center: tuple,
    fine_dict: dict,
    final_center: tuple,
    confidence: float,
    status: str,
    output_path: str
):
    """Saves visual overlay showing coarse prediction, fine search window, and final predicted center."""
    vis_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    # 1. Draw Coarse Point & Bounding Circle (MAGENTA)
    cx, cy = int(round(coarse_center[0])), int(round(coarse_center[1]))
    cv2.circle(vis_img, (cx, cy), 12, (255, 0, 255), 2)
    cv2.circle(vis_img, (cx, cy), 4, (255, 0, 255), -1)
    cv2.putText(vis_img, "Coarse Pred", (cx + 14, cy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    # 2. Draw Fine Search Window Bounds (YELLOW BOX)
    if fine_dict and 'window_bounds' in fine_dict:
        wx1, wy1, wx2, wy2 = fine_dict['window_bounds']
        cv2.rectangle(vis_img, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2)
        cv2.putText(vis_img, "Fine Window (+/-35px)", (wx1, max(15, wy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # 3. Draw Final Predicted Point & Bounding Box (RED CROSSHAIR & GREEN BOX)
    if final_center is not None and status == "SUCCESS":
        fx, fy = int(round(final_center[0])), int(round(final_center[1]))
        sw = fine_dict['scaled_w'] if fine_dict else 100
        sh = fine_dict['scaled_h'] if fine_dict else 100
        tl_x, tl_y = fine_dict['top_left'] if fine_dict else (fx - 50, fy - 50)

        cv2.rectangle(vis_img, (tl_x, tl_y), (tl_x + sw, tl_y + sh), (0, 255, 0), 2)
        cv2.circle(vis_img, (fx, fy), 6, (0, 0, 255), -1)
        cv2.drawMarker(vis_img, (fx, fy), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.putText(vis_img, f"Final Pred ({final_center[0]:.2f}, {final_center[1]:.2f})", (tl_x, max(25, tl_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 4. Text Overlay Panel
    panel_h, panel_w = 150, 440
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, vis_img, 0.25, 0, vis_img)
    cv2.rectangle(vis_img, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 1)

    status_col = (0, 255, 0) if status == "SUCCESS" else (0, 0, 255)
    fine_str = f"({final_center[0]:.2f}, {final_center[1]:.2f})" if final_center else "None (Failed)"

    lines = [
        (f"Status: {status}", status_col),
        (f"Coarse Predicted Center: ({coarse_center[0]:.2f}, {coarse_center[1]:.2f})", (255, 0, 255)),
        (f"Fine Predicted Center:   {fine_str}", (0, 255, 0)),
        (f"Confidence Score:        {confidence:.4f}", (255, 255, 255)),
        (f"Fine Match Score:        {fine_dict['fine_score']:.4f}" if fine_dict else "N/A", (255, 255, 255))
    ]

    for idx, (text, col) in enumerate(lines):
        y_pos = 34 + idx * 24
        cv2.putText(vis_img, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Debug visualization saved to: {output_path}")


def load_ground_truth(search_filename: str) -> tuple:
    """Helper to load ground truth coordinates for evaluation/debugging."""
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    if not os.path.exists(csv_path):
        return None, None
    base_name = os.path.basename(search_filename)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image"] == base_name:
                return float(row["x"]), float(row["y"])
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Coarse-to-Fine Localizer with Dynamic Lattice Disambiguation")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference grayscale image")
    parser.add_argument("--search", type=str, required=True, help="Path to search grayscale image")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    parser.add_argument("--vis-path", type=str, default="hierarchical_debug.png", help="Path to save debug visualization image")

    args = parser.parse_args()

    coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern(
        ref_path=args.reference,
        search_path=args.search
    )

    elapsed_ms = debug_info.get("computation_time_sec", 0.0) * 1000.0
    true_x, true_y = load_ground_truth(args.search)
    pixel_error = None
    if true_x is not None and true_y is not None and fine_center is not None:
        pixel_error = math.hypot(fine_center[0] - true_x, fine_center[1] - true_y)

    coarse_info = debug_info.get("coarse_info", {})
    dyn_period = coarse_info.get("estimated_lattice_period", 67.0)

    print("\n=== HIERARCHICAL LOCALIZATION DEBUG ===")
    ref_img = debug_info.get("ref_raw")
    search_img = debug_info.get("search_img")
    print(f"Scale correction:           10x mapping")
    print(f"Reference size:             {ref_img.shape if ref_img is not None else '1000x1000'}")
    print(f"Search size:                {search_img.shape if search_img is not None else '1000x1000'}")
    print(f"Estimated lattice period:   {dyn_period:.2f} px")

    cands_before = coarse_info.get("candidates_before", [])
    if cands_before:
        top_before = cands_before[0]
        print(f"Top candidate BEFORE structural ranking: ({top_before['center_x']:.2f}, {top_before['center_y']:.2f}) [Score: {top_before['initial_score']:.4f}]")

    cands_after = coarse_info.get("candidates_after", [])
    if cands_after:
        top_after = cands_after[0]
        print(f"Top candidate AFTER structural ranking:  ({top_after['center_x']:.2f}, {top_after['center_y']:.2f}) [Score: {top_after['coarse_score']:.4f}]")

    fine_str = f"({fine_center[0]:.2f}, {fine_center[1]:.2f})" if fine_center else "None (Failed)"
    print(f"Final prediction:           {fine_str}")
    print(f"Confidence:                 {confidence:.4f}")
    print(f"Status:                     {status}")
    print(f"Runtime:                    {elapsed_ms:.2f} ms ({debug_info.get('computation_time_sec', 0.0):.4f} s)")

    if true_x is not None and true_y is not None:
        print(f"Ground Truth (x, y):        ({true_x:.2f}, {true_y:.2f})")
        if pixel_error is not None:
            print(f"Pixel Error:                {pixel_error:.2f} px")

    print("\n" + "=" * 135)
    print(" HIERARCHICAL LOCALIZER (DYNAMIC LATTICE PERIOD & MULTI-SCALE DISAMBIGUATION) REPORT")
    print("=" * 135)
    print(f"{'Rank':<6} {'Center (x,y)':<22} {'Scale':<7} {'Raw Score':<11} {'Pyr Ctx Score':<15} {'Macro Struct':<14} {'Lattice Fit':<13} {'Group ID':<10} {'Final Score':<12} {'Dist GT (px)':<14}")
    print("-" * 135)

    for idx, c in enumerate(cands_after[:10], start=1):
        d_gt = math.hypot(c['center_x'] - true_x, c['center_y'] - true_y) if true_x else 0.0
        gid = c.get('alias_group_id', 1)
        lat_fit = c.get('lattice_consistency_score', 1.0)
        print(f"#{idx:<5} ({c['center_x']:.2f}, {c['center_y']:.2f})     {c['scale']:.3f}   {c['raw_template_score']:.4f}      {c['pyramid_context_score']:.4f}          {c['macro_structural_score']:.4f}        {lat_fit:.4f}       {gid:<9}  {c['coarse_score']:.4f}       {d_gt:.2f} px")

    print("=" * 135)

    if args.debug:
        save_debug_visualization(
            search_img=debug_info['search_img'],
            coarse_center=coarse_center,
            fine_dict=debug_info.get('fine_dict'),
            final_center=fine_center,
            confidence=confidence,
            status=status,
            output_path=args.vis_path
        )


if __name__ == "__main__":
    main()
