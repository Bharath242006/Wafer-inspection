"""
localization/global_coarse_localizer.py

Global Coarse Localizer stage for DriftSense-X.

Pipeline Architecture:
1. Multi-Resolution Pyramidal Feature Extraction:
   - Downscales reference image by 0.10x (1000x1000 -> 100x100) to account for exact physical 10x scale transformation.
   - Extracts Sobel gradient magnitude, Laplacian of Gaussian (LoG), and local variance maps.
2. Band-Pass Pyramidal Correlation:
   - Evaluates structural envelope match in downsampled pyramid space (4x downscaling: 250x250 search space).
3. Candidate Anchor Peak Extraction:
   - Extracts macro structural peak candidates with uncertainty radius (~40 px).
"""

import math
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


def compute_local_variance_map(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Computes local texture variance map in float32 [0, 1]."""
    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (ksize, ksize))
    sqr_mean = cv2.blur(img_f**2, (ksize, ksize))
    var = cv2.max(0.0, sqr_mean - mean**2)
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


def extract_coarse_peaks(response_map: np.ndarray, window_size: int = 5, min_thresh: float = 0.01, top_k: int = 20) -> list:
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


def locate_global_coarse(ref_gray: np.ndarray, search_gray: np.ndarray) -> tuple:
    """
    Stage 1 — Global Coarse Search:
    Establishes macro target location in 1000x1000 search space with uncertainty radius ~40 px.

    Returns:
        tuple: (coarse_x, coarse_y, coarse_score, uncertainty_radius, coarse_candidates, debug_info)
    """
    search_h, search_w = search_gray.shape[:2]
    ref_h, ref_w = ref_gray.shape[:2]

    # 1. Physical 10x scale transformation (1000x1000 ref -> 100x100 search patch)
    scaled_w = int(round(ref_w * 0.10))
    scaled_h = int(round(ref_h * 0.10))

    s_ref_gray = cv2.resize(ref_gray.astype(np.float32), (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # 2. Extract illumination-robust feature maps
    search_grad = compute_sobel_gradient(search_gray)
    ref_grad = compute_sobel_gradient(s_ref_gray)

    search_log = cv2.Laplacian(search_gray.astype(np.float32), cv2.CV_32F, ksize=3)
    ref_log = cv2.Laplacian(s_ref_gray, cv2.CV_32F, ksize=3)

    search_var = compute_local_variance_map(search_gray, ksize=15)
    ref_var = compute_local_variance_map(s_ref_gray, ksize=5)

    # 3. Downsample to pyramid level 1 (4x downscaling) for macro structural search
    p_scale = 0.25

    p_search_grad = cv2.resize(search_grad, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
    p_ref_grad = cv2.resize(ref_grad, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)

    p_search_log = cv2.resize(search_log, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
    p_ref_log = cv2.resize(ref_log, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)

    p_search_var = cv2.resize(search_var, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
    p_ref_var = cv2.resize(ref_var, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)

    # 4. Multi-Feature Pyramidal Correlation Matching
    res_grad = cv2.matchTemplate(p_search_grad, p_ref_grad, cv2.TM_CCOEFF_NORMED)
    res_log = cv2.matchTemplate(p_search_log, p_ref_log, cv2.TM_CCOEFF_NORMED)
    res_var = cv2.matchTemplate(p_search_var, p_ref_var, cv2.TM_CCOEFF_NORMED)

    res_coarse = 0.45 * res_grad + 0.45 * res_log + 0.10 * res_var

    # Extract top coarse peaks in pyramid space
    p_peaks = extract_coarse_peaks(res_coarse, window_size=5, min_thresh=0.01, top_k=20)
    p_pw, p_ph = p_ref_grad.shape[1], p_ref_grad.shape[0]

    coarse_candidates = []
    for px, py, score in p_peaks:
        # Convert pyramid top-left coordinates back to full 1000x1000 search center coordinates
        cx = (px + p_pw / 2.0) / p_scale
        cy = (py + p_ph / 2.0) / p_scale

        coarse_candidates.append({
            'center_x': float(cx),
            'center_y': float(cy),
            'score': float(score),
            'pyramid_x': px,
            'pyramid_y': py
        })

    if not coarse_candidates:
        # Fallback to center if no correlation peak above threshold
        return 500.0, 500.0, 0.0, 100.0, [], {}

    # Spatial NMS to deduplicate coarse candidates (radius 25 px)
    unique_coarse = []
    for cand in coarse_candidates:
        too_close = False
        for uc in unique_coarse:
            if math.hypot(cand['center_x'] - uc['center_x'], cand['center_y'] - uc['center_y']) < 25.0:
                too_close = True
                break
        if not too_close:
            unique_coarse.append(cand)

    best_coarse = unique_coarse[0]
    coarse_x = float(best_coarse['center_x'])
    coarse_y = float(best_coarse['center_y'])
    coarse_score = float(best_coarse['score'])

    debug_info = {
        "p_search_grad": p_search_grad,
        "p_ref_grad": p_ref_grad,
        "res_coarse": res_coarse,
        "unique_coarse": unique_coarse
    }

    return coarse_x, coarse_y, coarse_score, 40.0, unique_coarse, debug_info
