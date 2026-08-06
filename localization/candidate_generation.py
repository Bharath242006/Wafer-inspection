"""
localization/candidate_generation.py

Industrial Semiconductor Multi-Feature Candidate Generation System for DriftSense-X.

Pipeline Architecture:
Stage 1: Multi-Scale Image Pyramid Transformation
Stage 2: Vectorized 8-Feature Response Maps (NCC, Phase Correlation, FFT Magnitude,
         Sobel Gradient, Directional Sobel X/Y, Canny Edge Distance, Local Variance, Local SSIM)
Stage 3: Min-Max Normalization & Configurable Weighted Fusion Engine
Stage 4: Dense Adaptive Peak Extraction (Min Distance = 8 px, Adaptive Thresholding)
Stage 5: Spatial Non-Maximum Suppression (NMS, Radius = 12 px)
Stage 6: Radius-Based Centroid Candidate Clustering
Stage 7: Top-500 Candidate Ranking with Rich Feature Metadata
"""

import math
import cv2
import numpy as np


# Default Feature Weights for Industrial Response Fusion
DEFAULT_FUSION_WEIGHTS = {
    'ncc': 0.20,
    'fft': 0.15,
    'phase': 0.15,
    'grad': 0.15,
    'sobel': 0.10,
    'canny': 0.10,
    'var': 0.10,
    'ssim': 0.05
}


def compute_sobel_gradient_magnitude(img: np.ndarray) -> np.ndarray:
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


def compute_canny_distance_map(img: np.ndarray) -> np.ndarray:
    """Computes Canny edge Euclidean distance transform map in float32."""
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img
    edges = cv2.Canny(img_u8, 50, 150)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0.0, 1.0, cv2.NORM_MINMAX)
    return dist.astype(np.float32)


def compute_fft_magnitude_spectrum(img: np.ndarray) -> np.ndarray:
    """Computes normalized log Fourier magnitude spectrum in float32 [0, 1]."""
    img_f = img.astype(np.float32) - np.mean(img)
    f_transform = np.fft.fft2(img_f)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    cv2.normalize(magnitude, magnitude, 0.0, 1.0, cv2.NORM_MINMAX)
    return magnitude.astype(np.float32)


def compute_phase_correlation_map(search_gray: np.ndarray, ref_tmpl: np.ndarray) -> np.ndarray:
    """Computes fast frequency domain phase correlation response map."""
    sh, sw = search_gray.shape[:2]
    th, tw = ref_tmpl.shape[:2]

    out_h, out_w = sh - th + 1, sw - tw + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((1, 1), dtype=np.float32)

    # Fast frequency domain phase correlation
    f_search = np.fft.fft2(search_gray.astype(np.float32) - np.mean(search_gray))
    tmpl_pad = np.zeros((sh, sw), dtype=np.float32)
    tmpl_pad[:th, :tw] = ref_tmpl.astype(np.float32) - np.mean(ref_tmpl)
    f_tmpl = np.fft.fft2(tmpl_pad)

    cps = f_search * np.conj(f_tmpl)
    cps_norm = cps / (np.abs(cps) + 1e-5)
    phase_resp = np.real(np.fft.ifft2(cps_norm))[:out_h, :out_w]

    return min_max_normalize(phase_resp).astype(np.float32)


def compute_ssim_response_map(search_gray: np.ndarray, ref_tmpl: np.ndarray) -> np.ndarray:
    """Computes fast local Structural Similarity (SSIM) response map."""
    sh, sw = search_gray.shape[:2]
    th, tw = ref_tmpl.shape[:2]

    out_h, out_w = sh - th + 1, sw - tw + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((1, 1), dtype=np.float32)

    s_f = search_gray.astype(np.float32)
    t_f = ref_tmpl.astype(np.float32)

    t_mean = float(np.mean(t_f))
    t_var = float(np.var(t_f)) + 1e-4

    s_mean = cv2.blur(s_f, (tw, th))[:out_h, :out_w]
    s_sqr_mean = cv2.blur(s_f**2, (tw, th))[:out_h, :out_w]
    s_var = cv2.max(0.0, s_sqr_mean - s_mean**2) + 1e-4

    cov = cv2.matchTemplate(s_f, t_f - t_mean, cv2.TM_CCORR) / float(th * tw)

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    num = (2 * s_mean * t_mean + c1) * (2 * cov + c2)
    den = (s_mean**2 + t_mean**2 + c1) * (s_var + t_var + c2)

    ssim_map = num / (den + 1e-5)
    return min_max_normalize(ssim_map).astype(np.float32)


def min_max_normalize(map_img: np.ndarray) -> np.ndarray:
    """Min-Max normalizes floating point response map to range [0.0, 1.0]."""
    m_min = np.min(map_img)
    m_max = np.max(map_img)
    denom = (m_max - m_min) if (m_max - m_min) > 1e-7 else 1.0
    return (map_img - m_min) / denom


def extract_local_peaks(
    response_map: np.ndarray,
    window_size: int = 5,
    min_thresh: float = 0.01,
    top_k: int = 50
) -> list:
    """Legacy helper function for extract_local_peaks backward compatibility."""
    kernel_size = window_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
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


def extract_dense_local_peaks(
    response_map: np.ndarray,
    min_dist: int = 8,
    k_std: float = 0.5
) -> list:
    """
    Stage 4: Dense Peak Extraction using morphological dilation and adaptive thresholding.
    Extracts all local maxima spaced at least `min_dist` pixels apart.
    """
    kernel_size = 2 * min_dist + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(response_map, kernel)

    mean_val = float(np.mean(response_map))
    std_val = float(np.std(response_map))
    thresh = float(max(0.05, mean_val + k_std * std_val))

    local_peaks = (response_map == dilated) & (response_map >= thresh)
    peak_y, peak_x = np.where(local_peaks)
    scores = response_map[peak_y, peak_x]

    candidates = []
    h, w = response_map.shape[:2]
    for py, px, sc in zip(peak_y, peak_x, scores):
        sub_x = float(px)
        sub_y = float(py)
        if 1 <= px < w - 1 and 1 <= py < h - 1:
            denom_x = 2.0 * (2.0 * sc - response_map[py, px + 1] - response_map[py, px - 1])
            denom_y = 2.0 * (2.0 * sc - response_map[py + 1, px] - response_map[py - 1, px])
            if abs(denom_x) > 1e-5:
                dx = (response_map[py, px + 1] - response_map[py, px - 1]) / denom_x
                if abs(dx) < 1.0:
                    sub_x += dx
            if abs(denom_y) > 1e-5:
                dy = (response_map[py + 1, px] - response_map[py - 1, px]) / denom_y
                if abs(dy) < 1.0:
                    sub_y += dy
        candidates.append((sub_x, sub_y, float(sc)))
    return candidates



def apply_spatial_nms_and_clustering(
    raw_candidates: list,
    radius: float = 12.0,
    max_candidates: int = 500
) -> list:
    """
    Stage 5 & 6: Vectorized Spatial Non-Maximum Suppression and Centroid Clustering.
    Fast NMS using vectorized NumPy array slicing.
    """
    if not raw_candidates:
        return []

    # Sort raw candidates by score descending
    raw_candidates.sort(key=lambda c: c['score'], reverse=True)
    raw_candidates = raw_candidates[:3000]

    coords = np.array([[c['cx'], c['cy']] for c in raw_candidates], dtype=np.float32)
    keep_mask = np.ones(len(raw_candidates), dtype=bool)

    final_candidates = []
    radius_sq = radius * radius

    for i in range(len(raw_candidates)):
        if not keep_mask[i]:
            continue

        c_base = raw_candidates[i]
        final_candidates.append(c_base)

        if len(final_candidates) >= max_candidates:
            break

        if i + 1 < len(raw_candidates):
            dx = coords[i+1:, 0] - coords[i, 0]
            dy = coords[i+1:, 1] - coords[i, 1]
            dist_sq = dx * dx + dy * dy
            suppress_indices = np.where(dist_sq <= radius_sq)[0] + (i + 1)
            keep_mask[suppress_indices] = False

    return final_candidates


def generate_candidate_pool_multi(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    max_pool_size: int = 500,
    fusion_weights: dict = None
) -> list:
    """
    Industrial Semiconductor Multi-Feature Candidate Generation System for DriftSense-X.

    Returns:
        list of dict: Candidate dictionaries containing 'cx', 'cy', 'center_x', 'center_y',
                      'scale', 'score', 'score_ncc', 'score_phase', 'score_fft',
                      'score_grad', 'score_sobel', 'score_canny', 'score_var', 'score_ssim', 'tl_x', 'tl_y'.
    """
    if fusion_weights is None:
        fusion_weights = DEFAULT_FUSION_WEIGHTS

    w_ncc = fusion_weights.get('ncc', 0.20)
    w_fft = fusion_weights.get('fft', 0.15)
    w_phase = fusion_weights.get('phase', 0.15)
    w_grad = fusion_weights.get('grad', 0.15)
    w_sobel = fusion_weights.get('sobel', 0.10)
    w_canny = fusion_weights.get('canny', 0.10)
    w_var = fusion_weights.get('var', 0.10)
    w_ssim = fusion_weights.get('ssim', 0.05)

    # 2x Pyramid Acceleration (500x500 search space for fine grid resolution)
    p_scale = 0.50
    s_search_p = cv2.resize(search_img, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
    s_ref_p = cv2.resize(ref_img, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)

    sh_p, sw_p = s_search_p.shape[:2]
    ref_h_p, ref_w_p = s_ref_p.shape[:2]

    search_gray_f = s_search_p.astype(np.float32)
    ref_gray_f = s_ref_p.astype(np.float32)

    # Precompute search image feature maps in pyramid space
    search_grad = compute_sobel_gradient_magnitude(s_search_p)
    search_sx = cv2.Sobel(search_gray_f, cv2.CV_32F, 1, 0, ksize=3)
    search_sy = cv2.Sobel(search_gray_f, cv2.CV_32F, 0, 1, ksize=3)
    search_dist_canny = compute_canny_distance_map(s_search_p)
    search_var = compute_local_variance_map(s_search_p, ksize=11)
    search_fft_spec = compute_fft_magnitude_spectrum(s_search_p)

    # Multi-Scale Pyramid: 9 Scales (0.60x to 1.40x relative to 0.10x nominal footprint)
    relative_scales = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]
    scales = [rel * 0.10 for rel in relative_scales]

    raw_candidates = []

    for s in scales:
        scaled_w_p = int(round(ref_w_p * s))
        scaled_h_p = int(round(ref_h_p * s))

        if scaled_w_p <= 4 or scaled_h_p <= 4 or scaled_w_p > sw_p or scaled_h_p > sh_p:
            continue

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w_p, scaled_h_p), interpolation=cv2.INTER_AREA)

        # STAGE 2: Vectorized 8-Feature Response Maps
        # 1. NCC
        r_ncc = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)

        # 2. Phase Correlation
        r_phase = compute_phase_correlation_map(s_search_p, s_ref_gray)

        # 3. FFT Magnitude Correlation
        ref_fft_spec = compute_fft_magnitude_spectrum(s_ref_gray)
        r_fft = cv2.matchTemplate(search_fft_spec, ref_fft_spec, cv2.TM_CCOEFF_NORMED)

        # 4. Gradient Magnitude Matching
        s_ref_grad = compute_sobel_gradient_magnitude(s_ref_gray)
        r_grad = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)

        # 5. Directional Sobel X and Y Correlation
        ref_sx = cv2.Sobel(s_ref_gray, cv2.CV_32F, 1, 0, ksize=3)
        ref_sy = cv2.Sobel(s_ref_gray, cv2.CV_32F, 0, 1, ksize=3)
        r_sx = cv2.matchTemplate(search_sx, ref_sx, cv2.TM_CCOEFF_NORMED)
        r_sy = cv2.matchTemplate(search_sy, ref_sy, cv2.TM_CCOEFF_NORMED)
        r_sobel = 0.5 * r_sx + 0.5 * r_sy

        # 6. Canny Edge Distance Matching
        s_ref_u8 = cv2.normalize(s_ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        s_ref_canny = cv2.Canny(s_ref_u8, 50, 150).astype(np.float32) / 255.0
        r_canny_raw = cv2.matchTemplate(search_dist_canny, s_ref_canny, cv2.TM_SQDIFF_NORMED)
        r_canny = 1.0 - r_canny_raw

        # 7. Local Variance Map Matching
        s_ref_var = compute_local_variance_map(s_ref_u8, ksize=5)
        r_var = cv2.matchTemplate(search_var, s_ref_var, cv2.TM_CCOEFF_NORMED)

        # 8. Structural Similarity (SSIM) Response Map
        r_ssim = compute_ssim_response_map(s_search_p, s_ref_gray)

        # Resize feature maps to matching dimension if minor padding offsets exist
        out_h, out_w = r_ncc.shape
        def align_dim(m):
            return cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_AREA) if m.shape != (out_h, out_w) else m

        r_ncc_n = min_max_normalize(align_dim(r_ncc))
        r_fft_n = min_max_normalize(align_dim(r_fft))
        r_phase_n = min_max_normalize(align_dim(r_phase))
        r_grad_n = min_max_normalize(align_dim(r_grad))
        r_sobel_n = min_max_normalize(align_dim(r_sobel))
        r_canny_n = min_max_normalize(align_dim(r_canny))
        r_var_n = min_max_normalize(align_dim(r_var))
        r_ssim_n = min_max_normalize(align_dim(r_ssim))

        # STAGE 3: Min-Max Response Map Fusion
        r_fused = (
            w_ncc * r_ncc_n +
            w_fft * r_fft_n +
            w_phase * r_phase_n +
            w_grad * r_grad_n +
            w_sobel * r_sobel_n +
            w_canny * r_canny_n +
            w_var * r_var_n +
            w_ssim * r_ssim_n
        )

        # STAGE 4: Dense Adaptive Peak Extraction
        peaks = extract_dense_local_peaks(r_fused, min_dist=4, k_std=0.3)

        scaled_w_full = scaled_w_p / p_scale
        scaled_h_full = scaled_h_p / p_scale

        for px, py, fused_sc in peaks:
            tl_x_full = px / p_scale
            tl_y_full = py / p_scale
            cx = float(tl_x_full + scaled_w_full / 2.0)
            cy = float(tl_y_full + scaled_h_full / 2.0)

            # Skip border edge bounds
            if cx < 20.0 or cy < 20.0 or cx > (search_img.shape[1] - 20.0) or cy > (search_img.shape[0] - 20.0):
                continue

            px_i = int(np.clip(round(px), 0, out_w - 1))
            py_i = int(np.clip(round(py), 0, out_h - 1))

            raw_candidates.append({
                'cx': cx,
                'cy': cy,
                'center_x': cx,
                'center_y': cy,
                'tl_x': int(round(tl_x_full)),
                'tl_y': int(round(tl_y_full)),
                'scale': float(s),
                'score': float(fused_sc),
                'score_ncc': float(r_ncc_n[py_i, px_i]),
                'score_fft': float(r_fft_n[py_i, px_i]),
                'score_phase': float(r_phase_n[py_i, px_i]),
                'score_grad': float(r_grad_n[py_i, px_i]),
                'score_sobel': float(r_sobel_n[py_i, px_i]),
                'score_canny': float(r_canny_n[py_i, px_i]),
                'score_var': float(r_var_n[py_i, px_i]),
                'score_ssim': float(r_ssim_n[py_i, px_i])
            })


    # STAGE 5 & 6: Spatial Non-Maximum Suppression and Centroid Candidate Clustering
    final_candidates = apply_spatial_nms_and_clustering(
        raw_candidates,
        radius=12.0,
        max_candidates=max_pool_size
    )

    return final_candidates


def rank_top500_candidates(candidates: list) -> list:
    """Ranks candidate pool in descending order of fused match score."""
    return sorted(candidates, key=lambda c: c.get('score', 0.0), reverse=True)
