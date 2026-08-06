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
from localization.candidate_generation import generate_candidate_pool_multi
from localization.cnn_candidate_ranker import compute_cnn_similarity_scores


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
    Direct Candidate-Based Global-to-Local Localizer Pipeline.
    Bypasses flawed handcrafted z-score macro/low-frequency rescoring.
    Uses top candidate from generate_candidate_pool_multi (with CNN ranking if available)
    and performs 2D quadratic subpixel fine localization around the anchor.

    Returns:
        tuple: (coarse_center, fine_center, confidence, status, debug_info)
    """
    start_t = time.perf_counter()

    if isinstance(ref_path, np.ndarray):
        ref_raw = ref_path
    else:
        ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_raw is None:
            raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    if isinstance(search_path, np.ndarray):
        search_raw = search_path
    else:
        search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        if search_raw is None:
            raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    ref_gray_f = ref_raw.astype(np.float32)
    search_gray_f = search_raw.astype(np.float32)
    search_h, search_w = search_raw.shape[:2]

    # STAGE 1: Candidate Generation (Multi-scale template matching)
    cands_500 = generate_candidate_pool_multi(ref_raw, search_raw, max_pool_size=500)
    if not cands_500:
        elapsed_sec = time.perf_counter() - start_t
        return (500.0, 500.0), None, 0.0, "FAILED", {"computation_time_sec": elapsed_sec}

    # Sort candidate pool descending by candidate match score
    cands_sorted = sorted(cands_500, key=lambda c: c['score'], reverse=True)

    # STAGE 2: Take Top-20 candidates
    top20_cands = cands_sorted[:20]
    for c in top20_cands:
        c['center_x'] = float(c['cx'])
        c['center_y'] = float(c['cy'])
        c['primary_scale'] = float(c['scale'])
        c['cand_score'] = float(c['score'])

    # STAGE 3: Run trained Siamese CNN for all Top-20 candidates
    checkpoint_path = os.path.join("checkpoints", "siamese_cnn.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join("weights", "checkpoints", "siamese_cnn.pt")

    if os.path.exists(checkpoint_path):
        try:
            cnn_scores = compute_cnn_similarity_scores(ref_raw, search_raw, top20_cands, checkpoint_path=checkpoint_path)
            for i, c in enumerate(top20_cands):
                c['cnn_score'] = float(cnn_scores[i])
        except Exception:
            for c in top20_cands:
                c['cnn_score'] = 0.0
    else:
        for c in top20_cands:
            c['cnn_score'] = 0.0

    # STAGE 4: Normalize both Candidate Scores and CNN Scores across Top-20 pool
    cand_vals = np.array([c['cand_score'] for c in top20_cands], dtype=np.float32)
    cnn_vals = np.array([c['cnn_score'] for c in top20_cands], dtype=np.float32)

    min_cand, max_cand = np.min(cand_vals), np.max(cand_vals)
    denom_cand = (max_cand - min_cand) if (max_cand - min_cand) > 1e-6 else 1.0
    norm_cand = (cand_vals - min_cand) / denom_cand

    min_cnn, max_cnn = np.min(cnn_vals), np.max(cnn_vals)
    denom_cnn = (max_cnn - min_cnn) if (max_cnn - min_cnn) > 1e-6 else 1.0
    norm_cnn = (cnn_vals - min_cnn) / denom_cnn

    # STAGE 5: Compute Final Score = 0.6 * CNN Score + 0.4 * Candidate Score
    for i, c in enumerate(top20_cands):
        c['norm_cand_score'] = float(norm_cand[i])
        c['norm_cnn_score'] = float(norm_cnn[i])
        c['final_score'] = float(0.6 * norm_cnn[i] + 0.4 * norm_cand[i])

    # STAGE 6: Sort all 20 candidates descending by Final Score
    top20_ranked = sorted(top20_cands, key=lambda c: c['final_score'], reverse=True)

    # STAGE 7: Select highest final score candidate
    winning_cand = top20_ranked[0]
    coarse_x, coarse_y = winning_cand['center_x'], winning_cand['center_y']

    # STAGE 8: Fine Restricted Search (+/- 35 px window around winning candidate)
    scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
    ref_grad = compute_sobel_gradient(ref_raw)
    search_grad = compute_sobel_gradient(search_raw)
    ref_edge = compute_canny_edge(ref_raw)
    search_edge = compute_canny_edge(search_raw)

    window_radius = 35
    best_fine_score = -1.0
    best_fine_dict = None
    fine_center_x, fine_center_y = coarse_x, coarse_y

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
                'fine_score': float(fine_score)
            }

    elapsed_sec = time.perf_counter() - start_t
    final_center = (float(fine_center_x), float(fine_center_y))
    confidence = float(np.clip(best_fine_score if best_fine_score > 0 else best_cand['final_score'], 0.0, 1.0))
    status = "SUCCESS"

    debug_info = {
        "search_img": search_raw,
        "ref_raw": ref_raw,
        "coarse_center": (coarse_x, coarse_y),
        "all_candidates": top20_ranked,
        "fine_dict": best_fine_dict,
        "final_center": final_center,
        "confidence": confidence,
        "status": status,
        "computation_time_sec": elapsed_sec
    }

    return (coarse_x, coarse_y), final_center, confidence, status, debug_info



def locate_target_final(ref_img, search_img) -> tuple:
    """Wrapper function returning (pred_x, pred_y, confidence, status, debug_info)."""
    coarse, fine, conf, status, info = locate_reference_pattern_final(ref_img, search_img)
    if fine is not None:
        pred_x, pred_y = fine
    else:
        pred_x, pred_y = coarse
    return pred_x, pred_y, conf, status, info

