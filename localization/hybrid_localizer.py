"""
localization/hybrid_localizer.py

Standalone Hybrid Localization Algorithm for DriftSense-X Wafer Navigation with
Second-Stage Context Verification System.

Combines:
1. Multi-scale template matching across grayscale, Sobel gradient, Canny edge, and blurred representations.
2. Non-Maximum Suppression (NMS) for candidate peak extraction.
3. Candidate feature evaluation (intensity, gradient, edge overlap, SSIM).
4. SECOND-STAGE CONTEXT VERIFICATION:
   - Expands candidate search region to ~250x250 pixels surrounding context.
   - Constructs matching size context representation from reference image.
   - Evaluates gradient structure, edge structure, local intensity pattern, normalized cross-correlation,
     and multi-scale structural similarity (MS-SSIM) over the surrounding context.
   - Computes surrounding ring correlation to penalize candidates that match only inside the small periodic cell
     but disagree with surrounding macro-structure.
5. Periodicity / uniqueness scoring.
6. Center distance used strictly as a final TIE-BREAKER among top tied candidate scores.
7. Sub-pixel quadratic peak refinement.
8. Confidence-gated safe-failure mechanism.

Only depends on NumPy and OpenCV (standard Python libraries). No PyTorch, TensorFlow, or pandas.
"""

import argparse
import math
import os
import sys
import cv2
import numpy as np


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


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two equal-sized float32 image patches."""
    if img1.shape != img2.shape:
        return 0.0

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))


def extract_local_peaks(response_map: np.ndarray, window_size: int = 7, min_thresh: float = 0.05, top_k: int = 40) -> list:
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


def apply_nms(candidates: list, distance_threshold: float = 15.0) -> list:
    """Applies Non-Maximum Suppression (NMS) to merge nearby candidates."""
    sorted_cands = sorted(candidates, key=lambda c: c['raw_match_score'], reverse=True)
    kept = []

    for cand in sorted_cands:
        cx, cy = cand['center_x'], cand['center_y']
        too_close = False
        for k in kept:
            dist = math.hypot(cx - k['center_x'], cy - k['center_y'])
            if dist < distance_threshold:
                too_close = True
                break
        if not too_close:
            kept.append(cand)

    return kept


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


def crop_padded_region(img: np.ndarray, cx: float, cy: float, crop_w: int, crop_h: int) -> np.ndarray:
    """Crops a region of size (crop_h, crop_w) centered at (cx, cy) from image with reflection border padding."""
    H, W = img.shape[:2]
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(W, x2)
    crop_y2 = min(H, y2)

    patch = img[crop_y1:crop_y2, crop_x1:crop_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    return patch


def evaluate_second_stage_context(
    search_gray_f: np.ndarray,
    search_grad: np.ndarray,
    search_edge: np.ndarray,
    ref_gray_f: np.ndarray,
    ref_grad: np.ndarray,
    ref_edge: np.ndarray,
    cand_center_x: float,
    cand_center_y: float,
    scale: float,
    context_factor: float = 2.5
) -> dict:
    """
    Second-Stage Context Verification:
    Extracts ~250x250 surrounding context regions from search and reference images.
    Evaluates gradient structure, edge structure, local intensity, NCC, and MS-SSIM,
    plus surrounding ring correlation to penalize false periodic cell matches.
    """
    ref_h, ref_w = ref_gray_f.shape[:2]
    target_w = int(round(ref_w * scale))
    target_h = int(round(ref_h * scale))

    ctx_w = int(round(target_w * context_factor))
    ctx_h = int(round(target_h * context_factor))

    # 1. Crop search context region centered at candidate center
    s_ctx_gray = crop_padded_region(search_gray_f, cand_center_x, cand_center_y, ctx_w, ctx_h)
    s_ctx_grad = crop_padded_region(search_grad, cand_center_x, cand_center_y, ctx_w, ctx_h)
    s_ctx_edge = crop_padded_region(search_edge, cand_center_x, cand_center_y, ctx_w, ctx_h)

    # 2. Construct matching reference context representation
    ctx_scale = scale * context_factor
    r_ctx_gray = cv2.resize(ref_gray_f, (ctx_w, ctx_h), interpolation=cv2.INTER_AREA)
    r_ctx_grad = cv2.resize(ref_grad, (ctx_w, ctx_h), interpolation=cv2.INTER_AREA)
    r_ctx_edge = cv2.resize(ref_edge, (ctx_w, ctx_h), interpolation=cv2.INTER_AREA)

    # 3. Normalized cross-correlations over full context region
    ncc_int = float(max(0.0, cv2.matchTemplate(s_ctx_gray, r_ctx_gray, cv2.TM_CCOEFF_NORMED)[0, 0]))
    ncc_grad = float(max(0.0, cv2.matchTemplate(s_ctx_grad, r_ctx_grad, cv2.TM_CCOEFF_NORMED)[0, 0]))

    # Edge structure similarity
    ref_edge_dilated = cv2.dilate(r_ctx_edge, np.ones((3, 3), np.float32))
    edge_cnt = np.sum(s_ctx_edge > 0.1)
    if edge_cnt > 0:
        edge_overlap = float(np.sum((s_ctx_edge > 0.1) & (ref_edge_dilated > 0.1)) / float(edge_cnt))
    else:
        edge_overlap = 0.0

    # Multi-scale SSIM over context
    ssim_full = compute_ssim(s_ctx_gray, r_ctx_gray)
    s_ctx_half = cv2.resize(s_ctx_gray, (max(1, ctx_w // 2), max(1, ctx_h // 2)), interpolation=cv2.INTER_AREA)
    r_ctx_half = cv2.resize(r_ctx_gray, (max(1, ctx_w // 2), max(1, ctx_h // 2)), interpolation=cv2.INTER_AREA)
    ssim_half = compute_ssim(s_ctx_half, r_ctx_half)
    ms_ssim = float(0.5 * ssim_full + 0.5 * ssim_half)

    # 4. Surrounding Ring Correlation (evaluates structure OUTSIDE the central 100x100 target)
    ring_mask = np.ones((ctx_h, ctx_w), dtype=np.float32)
    cy_mid, cx_mid = ctx_h // 2, ctx_w // 2
    r_y1 = max(0, cy_mid - target_h // 2)
    r_y2 = min(ctx_h, cy_mid + target_h // 2)
    r_x1 = max(0, cx_mid - target_w // 2)
    r_x2 = min(ctx_w, cx_mid + target_w // 2)
    ring_mask[r_y1:r_y2, r_x1:r_x2] = 0.0

    s_ring = s_ctx_gray * ring_mask
    r_ring = r_ctx_gray * ring_mask
    ring_corr_val = float(cv2.matchTemplate(s_ring, r_ring, cv2.TM_CCOEFF_NORMED)[0, 0])
    ring_corr = float(max(0.0, ring_corr_val))

    # Context consistency score
    context_consistency = float(np.clip(
        0.25 * ncc_int + 0.25 * ncc_grad + 0.25 * edge_overlap + 0.25 * ms_ssim,
        0.0, 1.0
    ))

    # Disagreement penalty if surrounding ring correlation is low
    ring_disagreement_penalty = float(max(0.0, 1.0 - ring_corr))

    context_score = float(context_consistency - 0.30 * ring_disagreement_penalty)

    return {
        "context_consistency": context_consistency,
        "ncc_int": ncc_int,
        "ncc_grad": ncc_grad,
        "edge_overlap": edge_overlap,
        "ms_ssim": ms_ssim,
        "ring_corr": ring_corr,
        "context_score": context_score
    }


def locate_reference_pattern(
    ref_path: str,
    search_path: str,
    scales: list = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115],
    min_confidence_threshold: float = 0.15,
    tie_breaker_margin: float = 0.001
) -> tuple:
    """
    Locates reference pattern in search image using multi-stage hybrid ranking & context verification.

    Returns:
        tuple: (pred_center, confidence, status, debug_info)
    """
    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_raw is None:
        raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_raw is None:
        raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    search_h, search_w = search_raw.shape[:2]
    search_center_x = search_w / 2.0
    search_center_y = search_h / 2.0

    ref_gray_f = ref_raw.astype(np.float32)
    search_gray_f = search_raw.astype(np.float32)

    ref_grad = compute_sobel_gradient(ref_raw)
    search_grad = compute_sobel_gradient(search_raw)

    ref_edge = compute_canny_edge(ref_raw)
    search_edge = compute_canny_edge(search_raw)

    ref_blur = cv2.GaussianBlur(ref_gray_f, (3, 3), 0)
    search_blur = cv2.GaussianBlur(search_gray_f, (3, 3), 0)

    all_peaks = []

    # 1. Multi-scale candidate extraction
    for s in scales:
        scaled_w = int(round(ref_raw.shape[1] * s))
        scaled_h = int(round(ref_raw.shape[0] * s))

        if scaled_w <= 0 or scaled_h <= 0 or scaled_w > search_w or scaled_h > search_h:
            continue

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_edge = cv2.resize(ref_edge, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        s_ref_blur = cv2.resize(ref_blur, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        res_gray = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
        res_grad = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)
        res_edge = cv2.matchTemplate(search_edge, s_ref_edge, cv2.TM_CCOEFF_NORMED)
        res_blur = cv2.matchTemplate(search_blur, s_ref_blur, cv2.TM_CCOEFF_NORMED)

        peaks_g = extract_local_peaks(res_gray, window_size=7, min_thresh=0.05, top_k=30)
        peaks_d = extract_local_peaks(res_grad, window_size=7, min_thresh=0.05, top_k=30)
        peaks_e = extract_local_peaks(res_edge, window_size=7, min_thresh=0.05, top_k=30)

        peak_locs = set([(x, y) for x, y, _ in peaks_g] + [(x, y) for x, y, _ in peaks_d] + [(x, y) for x, y, _ in peaks_e])

        for tl_x, tl_y in peak_locs:
            if tl_x < 0 or tl_y < 0 or tl_x + scaled_w > search_w or tl_y + scaled_h > search_h:
                continue

            cx = tl_x + (scaled_w / 2.0)
            cy = tl_y + (scaled_h / 2.0)

            score_g = float(res_gray[tl_y, tl_x]) if 0 <= tl_y < res_gray.shape[0] and 0 <= tl_x < res_gray.shape[1] else 0.0
            score_d = float(res_grad[tl_y, tl_x]) if 0 <= tl_y < res_grad.shape[0] and 0 <= tl_x < res_grad.shape[1] else 0.0
            score_e = float(res_edge[tl_y, tl_x]) if 0 <= tl_y < res_edge.shape[0] and 0 <= tl_x < res_edge.shape[1] else 0.0
            score_b = float(res_blur[tl_y, tl_x]) if 0 <= tl_y < res_blur.shape[0] and 0 <= tl_x < res_blur.shape[1] else 0.0

            raw_score = max(score_g, 0.4 * score_g + 0.3 * score_d + 0.3 * score_e)

            all_peaks.append({
                'top_left': (tl_x, tl_y),
                'scaled_w': scaled_w,
                'scaled_h': scaled_h,
                'scale': s,
                'center_x': cx,
                'center_y': cy,
                's_ref_gray': s_ref_gray,
                's_ref_grad': s_ref_grad,
                's_ref_edge': s_ref_edge,
                'score_gray': score_g,
                'score_grad': score_d,
                'score_edge': score_e,
                'score_blur': score_b,
                'raw_match_score': raw_score,
                'res_gray_map': res_gray
            })

    if not all_peaks:
        return None, 0.0, "FAILED", {"candidates": [], "search_img": search_raw}

    # NMS candidate filtering
    candidates = apply_nms(all_peaks, distance_threshold=15.0)[:25]

    # First-stage local feature evaluation
    for cand in candidates:
        tl_x, tl_y = cand['top_left']
        sw, sh = cand['scaled_w'], cand['scaled_h']
        ref_g = cand['s_ref_gray']
        ref_d = cand['s_ref_grad']
        ref_e = cand['s_ref_edge']

        crop_g = search_gray_f[tl_y:tl_y+sh, tl_x:tl_x+sw]
        crop_d = search_grad[tl_y:tl_y+sh, tl_x:tl_x+sw]
        crop_e = search_edge[tl_y:tl_y+sh, tl_x:tl_x+sw]

        r_grad = cv2.matchTemplate(crop_d, ref_d, cv2.TM_CCOEFF_NORMED)[0, 0]
        score_grad_sim = float(max(0.0, r_grad))

        ref_e_dilated = cv2.dilate(ref_e, np.ones((3, 3), np.float32))
        edge_pixels = np.sum(crop_e > 0.1)
        if edge_pixels > 0:
            edge_overlap = float(np.sum((crop_e > 0.1) & (ref_e_dilated > 0.1)) / float(edge_pixels))
        else:
            edge_overlap = 0.0

        r_int = cv2.matchTemplate(crop_g, ref_g, cv2.TM_CCOEFF_NORMED)[0, 0]
        score_int_sim = float(max(0.0, r_int))

        score_ssim = float(max(0.0, compute_ssim(crop_g, ref_g)))

        # Periodicity / Uniqueness calculation
        res_map = cand['res_gray_map']
        y1 = max(0, tl_y - 12)
        y2 = min(res_map.shape[0], tl_y + 13)
        x1 = max(0, tl_x - 12)
        x2 = min(res_map.shape[1], tl_x + 13)
        local_patch = res_map[y1:y2, x1:x2].copy()

        cy_p = tl_y - y1
        cx_p = tl_x - x1
        local_patch[max(0, cy_p-2):min(local_patch.shape[0], cy_p+3), max(0, cx_p-2):min(local_patch.shape[1], cx_p+3)] = -1.0
        second_peak = float(np.max(local_patch)) if local_patch.size > 0 else 0.0

        uniqueness = float(np.clip(cand['score_gray'] - second_peak, 0.0, 1.0))
        dist_to_center = math.hypot(cand['center_x'] - search_center_x, cand['center_y'] - search_center_y)

        cand['feat_grad'] = score_grad_sim
        cand['feat_edge'] = edge_overlap
        cand['feat_int'] = score_int_sim
        cand['feat_ssim'] = score_ssim
        cand['uniqueness'] = uniqueness
        cand['dist_to_center'] = dist_to_center

        cand['initial_score'] = float(
            0.30 * score_int_sim + 0.30 * score_grad_sim + 0.20 * score_ssim + 0.10 * edge_overlap + 0.10 * uniqueness
        )

    # Sort initial candidate ranking (Before context verification)
    candidates_initial = sorted(candidates, key=lambda c: c['initial_score'], reverse=True)
    top_cand_before_context = candidates_initial[0]

    # 2. SECOND-STAGE CONTEXT VERIFICATION SYSTEM
    for cand in candidates:
        ctx_res = evaluate_second_stage_context(
            search_gray_f=search_gray_f,
            search_grad=search_grad,
            search_edge=search_edge,
            ref_gray_f=ref_gray_f,
            ref_grad=ref_grad,
            ref_edge=ref_edge,
            cand_center_x=cand['center_x'],
            cand_center_y=cand['center_y'],
            scale=cand['scale'],
            context_factor=2.5
        )

        cand['context_consistency'] = ctx_res['context_consistency']
        cand['context_score'] = ctx_res['context_score']
        cand['ring_corr'] = ctx_res['ring_corr']

        # Combined final score
        # Combine: initial candidate score + context consistency score + edge consistency + structural consistency + periodicity uniqueness
        final_score = (
            0.30 * cand['initial_score'] +
            0.35 * ctx_res['context_score'] +
            0.15 * ctx_res['edge_overlap'] +
            0.10 * ctx_res['ms_ssim'] +
            0.10 * cand['uniqueness']
        )
        cand['final_score'] = float(final_score)

    # Sort candidates by final score (After context verification)
    candidates_after_context = sorted(candidates, key=lambda c: c['final_score'], reverse=True)
    top_cand_after_context = candidates_after_context[0]

    top_final_score = top_cand_after_context['final_score']

    # 3. Final selection using center distance ONLY as a tie-breaker among top tied scores
    tied_candidates = [c for c in candidates_after_context if (top_final_score - c['final_score']) <= tie_breaker_margin]

    if len(tied_candidates) > 1:
        final_selected_cand = min(tied_candidates, key=lambda c: c['dist_to_center'])
    else:
        final_selected_cand = top_cand_after_context

    # 4. Sub-pixel quadratic peak refinement
    tl_x, tl_y = final_selected_cand['top_left']
    res_map = final_selected_cand['res_gray_map']
    sub_x, sub_y = refine_subpixel_peak(res_map, tl_x, tl_y)

    refined_center_x = float(sub_x + final_selected_cand['scaled_w'] / 2.0)
    refined_center_y = float(sub_y + final_selected_cand['scaled_h'] / 2.0)

    # Confidence calculation
    second_best = candidates_after_context[1] if len(candidates_after_context) > 1 else None
    second_score = second_best['final_score'] if second_best else 0.0
    score_margin = final_selected_cand['final_score'] - second_score

    confidence = float(np.clip(
        final_selected_cand['final_score'] * (1.0 + (score_margin / (final_selected_cand['final_score'] + 1e-6))),
        0.0, 1.0
    ))

    # Safety failure check
    if confidence < min_confidence_threshold or final_selected_cand['final_score'] < 0.10:
        status = "FAILED"
        pred_center = None
    else:
        status = "SUCCESS"
        pred_center = (refined_center_x, refined_center_y)

    debug_info = {
        "candidates": candidates_after_context,
        "top_cand_before_context": top_cand_before_context,
        "top_cand_after_context": top_cand_after_context,
        "final_selected_cand": final_selected_cand,
        "search_img": search_raw,
        "confidence": confidence,
        "top_score": final_selected_cand['final_score'],
        "second_score": second_score
    }

    return pred_center, confidence, status, debug_info


def save_debug_visualization(
    search_img: np.ndarray,
    candidates: list,
    top_before: dict,
    top_after: dict,
    final_selected: dict,
    pred_center: tuple,
    confidence: float,
    status: str,
    output_path: str
):
    """Generates visual debug overlay showing candidates, top choices before/after context, and final match."""
    vis_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    # Draw candidates
    for idx, cand in enumerate(candidates[:20], start=1):
        cx = int(round(cand['center_x']))
        cy = int(round(cand['center_y']))
        tl_x, tl_y = cand['top_left']
        sw, sh = cand['scaled_w'], cand['scaled_h']

        is_final = (cand == final_selected and status == "SUCCESS")
        color = (0, 255, 0) if is_final else (255, 200, 0)
        thickness = 2 if is_final else 1

        cv2.rectangle(vis_img, (tl_x, tl_y), (tl_x + sw, tl_y + sh), color, thickness)
        cv2.circle(vis_img, (cx, cy), 3, color, -1)
        cv2.putText(vis_img, f"#{idx}", (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Highlight top before context
    if top_before:
        cx = int(round(top_before['center_x']))
        cy = int(round(top_before['center_y']))
        cv2.circle(vis_img, (cx, cy), 8, (255, 0, 255), 2)  # Magenta circle

    # Highlight final selected candidate
    if pred_center is not None and status == "SUCCESS":
        px, py = int(round(pred_center[0])), int(round(pred_center[1]))
        cv2.circle(vis_img, (px, py), 6, (0, 0, 255), -1)
        cv2.drawMarker(vis_img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

    # Text overlay panel
    panel_h, panel_w = 190, 480
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, vis_img, 0.25, 0, vis_img)
    cv2.rectangle(vis_img, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 1)

    status_color = (0, 255, 0) if status == "SUCCESS" else (0, 0, 255)
    pred_str = f"({pred_center[0]:.2f}, {pred_center[1]:.2f})" if pred_center else "None"

    b_center = f"({top_before['center_x']:.1f}, {top_before['center_y']:.1f})" if top_before else "N/A"
    a_center = f"({top_after['center_x']:.1f}, {top_after['center_y']:.1f})" if top_after else "N/A"

    lines = [
        (f"Status: {status}", status_color),
        (f"Predicted center: {pred_str}", (255, 255, 255)),
        (f"Confidence: {confidence:.4f}", (255, 255, 255)),
        (f"Top Candidate Before Context: {b_center} Score={top_before['initial_score']:.4f}", (255, 200, 255)),
        (f"Top Candidate After Context:  {a_center} Score={top_after['final_score']:.4f}", (0, 255, 255)),
        (f"Context Score: {final_selected.get('context_score', 0.0):.4f} | Ring Corr: {final_selected.get('ring_corr', 0.0):.4f}", (255, 255, 255))
    ]

    for idx, (text, col) in enumerate(lines):
        y_pos = 32 + idx * 26
        cv2.putText(vis_img, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Localizer with Second-Stage Context Verification")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference grayscale image")
    parser.add_argument("--search", type=str, required=True, help="Path to search grayscale image")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization output")
    parser.add_argument("--vis-path", type=str, default="hybrid_debug.png", help="Output path for debug visualization")

    args = parser.parse_args()

    pred_center, confidence, status, debug_info = locate_reference_pattern(
        ref_path=args.reference,
        search_path=args.search
    )

    if pred_center is not None:
        print(f"Predicted center: ({pred_center[0]:.2f}, {pred_center[1]:.2f})")
    else:
        print("Predicted center: None")

    print(f"Confidence: {confidence:.4f}")
    print(f"Status: {status}")

    if args.debug:
        save_debug_visualization(
            search_img=debug_info['search_img'],
            candidates=debug_info['candidates'],
            top_before=debug_info.get('top_cand_before_context'),
            top_after=debug_info.get('top_cand_after_context'),
            final_selected=debug_info.get('final_selected_cand'),
            pred_center=pred_center,
            confidence=confidence,
            status=status,
            output_path=args.vis_path
        )


if __name__ == "__main__":
    main()
