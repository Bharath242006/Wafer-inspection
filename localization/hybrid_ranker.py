"""
localization/hybrid_ranker.py

DriftSense-X Final Hybrid Candidate Ranker.

Extracts a 56-dimensional feature vector per candidate combining:
1.  Candidate raw pipeline score                          (1D)
2.  Local NCC (raw + pool-normalized)                     (2D)
3.  Sobel gradient NCC (raw + normalized)                 (2D)
4.  LoG similarity (raw + normalized)                     (2D)
5.  FFT / phase-correlation score (NEW)                   (1D)
6.  Edge/Canny overlap (raw + normalized)                 (2D)
7.  Low-frequency Gaussian similarity (raw + normalized)  (2D)
8.  Medium-context window similarity 150x150 (NEW)        (1D)
9.  Global landmark heatmap value                         (1D)
10. Coordinate features: cx, cy, cx/1000, cy/1000,
    cx/W, cy/H, dist_to_center                           (7D)
11. Lattice phase: cx/lx, cy/ly, phase_x, phase_y,
    sin/cos encodings x2                                  (8D)
12. Rank + score margins: rank/500, log(rank+1),
    percentile, margin_top1, margin_median                (5D)
13. Local candidate density: density_r30, density_r60,
    dist_nearest                                          (3D)
14. Neighbor consistency: 4 lattice-direction responses   (4D)
15. Multi-scale signature (100→50→25→12 NCC mean)        (1D)

Hard negatives: lattice-alias candidates (±k*lx, ±k*ly from GT).
Training objective: triplet margin ranking loss.
"""

import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from scratch.improve_candidate_recall import compute_sobel_gradient, generate_candidate_pool_multi
from localization.global_coarse_localizer import compute_local_variance_map, zmuv_ncc
from localization.final_localizer import compute_canny_edge, estimate_lattice_period_2d
from localization.global_landmark_localizer import compute_global_landmark_heatmap

HYBRID_FEATURE_DIM = 56


class HybridRankerNet(nn.Module):
    """
    Lightweight MLP ranker operating on 56-D hybrid feature vectors.

    Architecture: 56 → 128 → 64 → 32 → 1
    Uses BatchNorm + ReLU + Dropout for regularization.
    """
    def __init__(self, input_dim: int = HYBRID_FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def _fft_phase_correlation_score(patch: np.ndarray, ref_tmpl: np.ndarray) -> float:
    """
    Computes FFT phase-correlation peak height between a search patch and reference template.

    Phase-correlation measures TRANSLATIONAL alignment — true matches yield a sharp
    dominant peak near the origin (zero shift), while periodic aliases yield a peak
    shifted by (k*lx, k*ly), so their score is structurally lower.

    Returns the normalized peak value in the phase-correlation response.
    """
    if patch.size == 0 or ref_tmpl.size == 0:
        return 0.0

    # Resize to common size
    h = min(patch.shape[0], ref_tmpl.shape[0], 64)
    w = min(patch.shape[1], ref_tmpl.shape[1], 64)
    if h < 4 or w < 4:
        return 0.0

    p = cv2.resize(patch.astype(np.float32), (w, h), cv2.INTER_AREA)
    r = cv2.resize(ref_tmpl.astype(np.float32), (w, h), cv2.INTER_AREA)

    # Zero-mean
    p -= np.mean(p)
    r -= np.mean(r)

    # FFT of both
    Fp = np.fft.fft2(p)
    Fr = np.fft.fft2(r)

    # Cross-power spectrum
    cross = Fp * np.conj(Fr)
    denom = np.abs(cross) + 1e-8
    phase_corr = np.real(np.fft.ifft2(cross / denom))

    # Shift so zero-lag is at center
    phase_corr = np.fft.fftshift(phase_corr)
    cy, cx = phase_corr.shape[0] // 2, phase_corr.shape[1] // 2

    # Peak near center (within 5 px) vs. global peak
    center_region = phase_corr[max(0, cy-5):cy+6, max(0, cx-5):cx+6]
    center_peak = float(np.max(center_region)) if center_region.size > 0 else 0.0
    global_peak = float(np.max(phase_corr)) if phase_corr.size > 0 else 1.0

    # Ratio of center-peak to global-peak: 1.0 means perfect alignment, <1 means shifted
    return float(np.clip(center_peak / (global_peak + 1e-8), 0.0, 1.0))


def _multi_scale_ncc(patch: np.ndarray, ref_tmpl: np.ndarray) -> float:
    """Multi-resolution NCC: 100→50→25→12 px."""
    scores = []
    for r in [100, 50, 25, 12]:
        if patch.shape[0] < r or patch.shape[1] < r:
            continue
        p_r = cv2.resize(patch, (r, r), cv2.INTER_AREA)
        t_r = cv2.resize(ref_tmpl, (r, r), cv2.INTER_AREA)
        scores.append(zmuv_ncc(p_r, t_r))
    return float(np.mean(scores)) if scores else 0.0


def _pool_normalize(vals: np.ndarray) -> np.ndarray:
    """Robust pool normalization: tanh(0.5 * z-score)."""
    mu = float(np.mean(vals))
    std = float(np.std(vals)) + 1e-5
    return np.tanh(0.5 * (vals - mu) / std).astype(np.float32)


def extract_hybrid_features_pool(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    lx: float,
    ly: float
) -> np.ndarray:
    """
    Extract 56-D hybrid feature matrix for the candidate pool.

    Args:
        ref_img:    Grayscale reference image (1000x1000).
        search_img: Grayscale search image (1000x1000).
        candidates: List of candidate dicts with 'cx', 'cy', 'scale', 'score'.
        lx, ly:     Estimated lattice periods in search-image px.

    Returns:
        np.ndarray of shape (N, 56).
    """
    if not candidates:
        return np.zeros((0, HYBRID_FEATURE_DIM), dtype=np.float32)

    N = len(candidates)
    sh, sw = search_img.shape[:2]

    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)

    # ── Precompute image-level maps ──────────────────────────────────────────
    heatmap = compute_global_landmark_heatmap(ref_img, search_img)

    ref_grad = compute_sobel_gradient(ref_img)
    search_grad = compute_sobel_gradient(search_img)

    ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)
    search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)

    search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)

    ref_edge = compute_canny_edge(ref_img)
    search_edge = compute_canny_edge(search_img)

    ref_100 = cv2.resize(ref_gray_f, (100, 100), cv2.INTER_AREA)
    ref_blur_100 = cv2.GaussianBlur(ref_100, (15, 15), 3.0)

    # Medium-context reference (150x150 template at 0.10 scale = 100px wide)
    ref_med_w = max(10, int(round(ref_img.shape[1] * 0.10)))
    ref_med_h = max(10, int(round(ref_img.shape[0] * 0.10)))
    ref_med = cv2.resize(ref_gray_f, (ref_med_w, ref_med_h), cv2.INTER_AREA)
    ref_med_150 = cv2.resize(ref_med, (min(150, ref_med_w * 2), min(150, ref_med_h * 2)), cv2.INTER_AREA)

    # Padded search images for safe border extraction
    pad = 300
    search_gray_pad = cv2.copyMakeBorder(search_gray_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_grad_pad = cv2.copyMakeBorder(search_grad, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_log_pad = cv2.copyMakeBorder(search_log, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_blur_pad = cv2.copyMakeBorder(search_blur, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_edge_pad = cv2.copyMakeBorder(search_edge, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Pre-build templates per scale
    unique_scales = list(set([c.get('scale', 0.10) for c in candidates]))
    ref_templates = {}
    for s in unique_scales:
        cw = max(10, int(round(ref_img.shape[1] * s)))
        ch = max(10, int(round(ref_img.shape[0] * s)))
        tg = cv2.resize(ref_gray_f, (cw, ch), cv2.INTER_AREA)
        td = cv2.resize(ref_grad, (cw, ch), cv2.INTER_AREA)
        tl = cv2.resize(ref_log, (cw, ch), cv2.INTER_AREA)
        te = cv2.resize(ref_edge, (cw, ch), cv2.INTER_AREA)
        te_dil = cv2.dilate(te, np.ones((3, 3), np.float32))
        ref_templates[s] = (cw, ch, tg, td, tl, te, te_dil)

    # Scores for margin computation
    raw_scores = np.array([c.get('score', 0.0) for c in candidates], dtype=np.float32)
    top1_score = float(np.max(raw_scores)) if N > 0 else 0.0
    median_score = float(np.median(raw_scores)) if N > 0 else 0.0

    # ── Per-candidate raw feature extraction ────────────────────────────────
    raw = []
    for idx, c in enumerate(candidates):
        cx, cy = c['cx'], c['cy']
        s = c.get('scale', 0.10)
        cand_score = float(c.get('score', 0.0))

        cw, ch, tg, td, tl, te, te_dil = ref_templates[s]
        tl_x_p = int(round(cx + pad - cw / 2.0))
        tl_y_p = int(round(cy + pad - ch / 2.0))

        patch_g = search_gray_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_d = search_grad_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_l = search_log_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_b = search_blur_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_e = search_edge_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]

        z_g = zmuv_ncc(patch_g, tg) if patch_g.shape == tg.shape else 0.0
        z_d = zmuv_ncc(patch_d, td) if patch_d.shape == td.shape else 0.0
        z_l = zmuv_ncc(patch_l, tl) if patch_l.shape == tl.shape else 0.0

        # FFT phase-correlation score (NEW — discriminates periodic aliases)
        fft_score = _fft_phase_correlation_score(patch_g, tg)

        # Edge/Canny overlap
        e_cnt = np.sum(patch_e > 0.1)
        e_overlap = float(
            np.sum((patch_e > 0.1) & (te_dil > 0.1)) / float(e_cnt)
        ) if e_cnt > 0 else 0.0

        # Low-frequency Gaussian similarity
        p_b_100 = cv2.resize(patch_b, (100, 100), cv2.INTER_AREA) if patch_b.size > 0 else np.zeros((100, 100), np.float32)
        z_lf = zmuv_ncc(p_b_100, ref_blur_100)

        # Medium-context window (150 px crop)
        hw = min(75, sw // 2 - 1)
        hh = min(75, sh // 2 - 1)
        x1m = max(0, int(round(cx)) - hw)
        y1m = max(0, int(round(cy)) - hh)
        x2m = min(sw, int(round(cx)) + hw)
        y2m = min(sh, int(round(cy)) + hh)
        med_patch = search_gray_f[y1m:y2m, x1m:x2m]
        if med_patch.size > 0 and ref_med_150.size > 0:
            mp_r = cv2.resize(med_patch, (ref_med_150.shape[1], ref_med_150.shape[0]), cv2.INTER_AREA)
            med_score = zmuv_ncc(mp_r, ref_med_150)
        else:
            med_score = 0.0

        # Multi-scale NCC
        p_g_100 = cv2.resize(patch_g, (100, 100), cv2.INTER_AREA) if patch_g.size > 0 else np.zeros((100, 100), np.float32)
        multi_scale_sig = _multi_scale_ncc(p_g_100, ref_100)

        # Global heatmap value
        ix = int(np.clip(round(cx), 0, sw - 1))
        iy = int(np.clip(round(cy), 0, sh - 1))
        global_val = float(heatmap[iy, ix])

        raw.append({
            'cx': cx, 'cy': cy, 'score': cand_score,
            'z_g': z_g, 'z_d': z_d, 'z_l': z_l,
            'fft_score': fft_score,
            'e_overlap': e_overlap,
            'z_lf': z_lf,
            'med_score': med_score,
            'global_val': global_val,
            'multi_scale_sig': multi_scale_sig,
        })

    # ── Pool-level normalization ─────────────────────────────────────────────
    vis_keys = ['z_g', 'z_d', 'z_l', 'fft_score', 'e_overlap', 'z_lf', 'med_score', 'multi_scale_sig']
    norm_v = {}
    for vk in vis_keys:
        vals = np.array([r[vk] for r in raw], dtype=np.float32)
        norm_v[vk] = _pool_normalize(vals)

    # Coordinate arrays for density computations
    all_cxs = np.array([c['cx'] for c in candidates], dtype=np.float32)
    all_cys = np.array([c['cy'] for c in candidates], dtype=np.float32)

    # Lattice-direction sampling reference template
    lx_cw = max(10, int(round(ref_img.shape[1] * 0.10)))
    lx_ch = max(10, int(round(ref_img.shape[0] * 0.10)))
    ref_lx_tmpl = cv2.resize(ref_gray_f, (lx_cw, lx_ch), cv2.INTER_AREA)

    def _sample_lat_dir(cx: float, cy: float, dx: float, dy: float) -> float:
        tx = int(round(cx + dx + pad - lx_cw / 2.0))
        ty = int(round(cy + dy + pad - lx_ch / 2.0))
        p = search_gray_pad[ty:ty+lx_ch, tx:tx+lx_cw]
        return zmuv_ncc(p, ref_lx_tmpl) if p.shape == ref_lx_tmpl.shape else 0.0

    # ── Build 56-D feature matrix ────────────────────────────────────────────
    feat_matrix = np.zeros((N, HYBRID_FEATURE_DIM), dtype=np.float32)

    for i, c in enumerate(candidates):
        r = raw[i]
        cx, cy = r['cx'], r['cy']
        cand_score = r['score']

        # --- Group 1-9: visual features (15D) ---
        v_score       = cand_score                     # 1
        v_z_g         = r['z_g']                       # 2
        nz_g          = float(norm_v['z_g'][i])        # 3
        v_z_d         = r['z_d']                       # 4
        nz_d          = float(norm_v['z_d'][i])        # 5
        v_z_l         = r['z_l']                       # 6
        nz_l          = float(norm_v['z_l'][i])        # 7
        v_fft         = r['fft_score']                 # 8
        v_e_overlap   = r['e_overlap']                 # 9
        nz_e          = float(norm_v['e_overlap'][i])  # 10
        v_z_lf        = r['z_lf']                      # 11
        nz_lf         = float(norm_v['z_lf'][i])       # 12
        v_med         = r['med_score']                 # 13
        v_global      = r['global_val']                # 14
        v_multiscale  = r['multi_scale_sig']            # 15

        # --- Group 10: coordinates (7D) ---
        abs_x      = cx / 1000.0                                    # 16
        abs_y      = cy / 1000.0                                    # 17
        norm_x     = cx / float(sw)                                 # 18
        norm_y     = cy / float(sh)                                 # 19
        dist_ctr   = math.hypot(norm_x - 0.5, norm_y - 0.5)        # 20
        abs_cx_raw = cx / 1000.0                                    # 21
        abs_cy_raw = cy / 1000.0                                    # 22

        # --- Group 11: lattice phase (8D) ---
        lat_x    = cx / float(lx)                                   # 23
        lat_y    = cy / float(ly)                                   # 24
        phase_x  = (cx % float(lx)) / float(lx)                    # 25
        phase_y  = (cy % float(ly)) / float(ly)                    # 26
        sin_px   = math.sin(2.0 * math.pi * cx / float(lx))        # 27
        cos_px   = math.cos(2.0 * math.pi * cx / float(lx))        # 28
        sin_py   = math.sin(2.0 * math.pi * cy / float(ly))        # 29
        cos_py   = math.cos(2.0 * math.pi * cy / float(ly))        # 30

        # --- Group 12: rank + margins (5D) ---
        rank_idx    = float(i)
        rank_norm   = rank_idx / 500.0                              # 31
        rank_log    = math.log(rank_idx + 1.0) / 6.22              # 32
        percentile  = 1.0 - rank_norm                              # 33
        margin_top1 = cand_score - top1_score                      # 34
        margin_med  = cand_score - median_score                    # 35

        # --- Group 13: local density (3D) ---
        dists       = np.hypot(all_cxs - cx, all_cys - cy)
        density_r30 = float(np.sum((dists > 0.0) & (dists <= 30.0))) / 50.0  # 36
        density_r60 = float(np.sum((dists > 0.0) & (dists <= 60.0))) / 50.0  # 37
        dists_ns    = dists[dists > 0.0]
        dist_near   = float(np.min(dists_ns)) / 100.0 if len(dists_ns) > 0 else 1.0  # 38

        # --- Group 14: lattice-direction neighbor consistency (4D) ---
        resp_px = _sample_lat_dir(cx, cy, +lx,  0.0)              # 39
        resp_nx = _sample_lat_dir(cx, cy, -lx,  0.0)              # 40
        resp_py = _sample_lat_dir(cx, cy,  0.0, +ly)              # 41
        resp_ny = _sample_lat_dir(cx, cy,  0.0, -ly)              # 42

        # --- Group 15: multi-scale NCC (1D) ---
        v_ms_sig = v_multiscale                                    # 43 (already computed)

        # --- Fill in remaining to reach 56D: FFT normalized + extra padding ---
        nz_fft  = float(norm_v['fft_score'][i])                    # 44
        nz_med  = float(norm_v['med_score'][i])                    # 45
        nz_ms   = float(norm_v['multi_scale_sig'][i])              # 46

        # Score margin from neighbors (spatial top-5 mean)
        top5_idx = np.argsort(dists)[:min(6, N)]
        mean_top5_spatial = float(np.mean(raw_scores[top5_idx])) if len(top5_idx) > 0 else cand_score  # 47

        # Extra: diagonal lattice samples (2D)
        resp_pp = _sample_lat_dir(cx, cy, +lx, +ly)               # 48
        resp_nn = _sample_lat_dir(cx, cy, -lx, -ly)               # 49

        # Extra: half-period samples (2D) — aliases at ±0.5*lx,ly are detectable
        resp_hx = _sample_lat_dir(cx, cy, lx * 0.5, 0.0)         # 50
        resp_hy = _sample_lat_dir(cx, cy, 0.0, ly * 0.5)         # 51

        # Extra: 2-period samples (2D) — 2nd-order aliases
        resp_2x = _sample_lat_dir(cx, cy, 2.0 * lx, 0.0)         # 52
        resp_2y = _sample_lat_dir(cx, cy, 0.0, 2.0 * ly)         # 53

        # Extra: raw heatmap gradient (difference between this and neighbors)
        ix_p = int(np.clip(round(cx + 5), 0, sw - 1))
        iy_p = int(np.clip(round(cy + 5), 0, sh - 1))
        global_neighbor = float(heatmap[iy_p, ix_p])
        heatmap_grad = v_global - global_neighbor                  # 54

        # Extra: top-10 score mean (for pool context)
        top10_mean = float(np.mean(raw_scores[:min(10, N)]))       # 55
        margin_top10 = cand_score - top10_mean                     # 56

        vec = [
            # Visual: 1-15
            v_score, v_z_g, nz_g, v_z_d, nz_d, v_z_l, nz_l,
            v_fft, v_e_overlap, nz_e, v_z_lf, nz_lf,
            v_med, v_global, v_ms_sig,
            # Coordinates: 16-22
            abs_x, abs_y, norm_x, norm_y, dist_ctr, abs_cx_raw, abs_cy_raw,
            # Lattice phase: 23-30
            lat_x, lat_y, phase_x, phase_y, sin_px, cos_px, sin_py, cos_py,
            # Rank+margins: 31-35
            rank_norm, rank_log, percentile, margin_top1, margin_med,
            # Density: 36-38
            density_r30, density_r60, dist_near,
            # Neighbor consistency: 39-42
            resp_px, resp_nx, resp_py, resp_ny,
            # Multi-scale NCC normalized: 43
            v_ms_sig,
            # Normalized FFT/med/multiscale: 44-46
            nz_fft, nz_med, nz_ms,
            # Spatial neighborhood score: 47
            mean_top5_spatial,
            # Diagonal + half-period + 2nd-order samples: 48-53
            resp_pp, resp_nn, resp_hx, resp_hy, resp_2x, resp_2y,
            # Heatmap grad + top10 margin: 54-56
            heatmap_grad, top10_mean, margin_top10,
        ]

        feat_matrix[i, :] = np.array(vec[:HYBRID_FEATURE_DIM], dtype=np.float32)

    return feat_matrix


# ── Inference helpers ────────────────────────────────────────────────────────

_hybrid_model_cache = None


def load_trained_hybrid_model(
    checkpoint_path: str = "checkpoints/hybrid_ranker.pt"
) -> HybridRankerNet:
    """Loads trained HybridRankerNet checkpoint (singleton cache)."""
    global _hybrid_model_cache
    if _hybrid_model_cache is not None:
        return _hybrid_model_cache

    model = HybridRankerNet(input_dim=HYBRID_FEATURE_DIM, hidden_dim=128)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[HybridRankerNet] Loaded trained weights from '{checkpoint_path}'.")
    else:
        print(f"[HybridRankerNet] WARNING: '{checkpoint_path}' not found — random init.")

    _hybrid_model_cache = model
    return model


def compute_hybrid_scores(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    checkpoint_path: str = "checkpoints/hybrid_ranker.pt"
) -> list:
    """
    Computes Hybrid Ranker scores for a candidate pool.

    Args:
        ref_img:         Grayscale reference image.
        search_img:      Grayscale search image.
        candidates:      List of candidate dicts with 'cx', 'cy', 'scale', 'score'.
        checkpoint_path: Path to saved HybridRankerNet checkpoint.

    Returns:
        List of float scores in [0.0, 1.0], one per candidate.
    """
    model = load_trained_hybrid_model(checkpoint_path)
    lx, ly = estimate_lattice_period_2d(ref_img)

    feats = extract_hybrid_features_pool(ref_img, search_img, candidates, lx, ly)
    if feats.shape[0] == 0:
        return []

    t_feats = torch.tensor(feats, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        scores = model(t_feats)
        if scores.dim() == 0:
            return [float(scores.item())]
        return [float(s.item()) for s in scores]
