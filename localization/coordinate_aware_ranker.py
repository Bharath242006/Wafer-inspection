"""
localization/coordinate_aware_ranker.py

Fast, optimized Coordinate-Aware Candidate Ranker architecture module for DriftSense-X.

Extracts a 44-dimensional feature vector per candidate incorporating:
1. Candidate visual features & 7 Z-score normalized signature metrics
2. Candidate absolute coordinates (cx/1000, cy/1000)
3. Normalized coordinates (cx/W, cy/H, dist_to_center)
4. Estimated lattice phase (cx/lx, cy/ly, (cx % lx)/lx, (cy % ly)/ly)
5. Sin/Cos lattice phase encodings (sin(2pi*cx/lx), cos(2pi*cx/lx), sin(2pi*cy/ly), cos(2pi*cy/ly))
6. Candidate rank / percentile (rank/500, log(rank+1), percentile)
7. Local candidate density (count within R=30, R=60, dist_to_nearest)
8. Neighboring candidate consistency (mean top-5 spatial score, 4-lattice direction responses)
9. Score margins (margin to top-1, median, and mean top-10 candidate scores)
10. Global position / context features (global landmark heatmap value, macro context score)

Outputs scalar candidate score in [0.0, 1.0].
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


class CoordinateAwareRankerNet(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network ranker operating on 44-D coordinate-aware features.
    """
    def __init__(self, input_dim: int = 44, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def extract_coordinate_aware_features_pool(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    lx: float,
    ly: float
) -> np.ndarray:
    """
    Fast 44-D coordinate-aware feature matrix extraction for candidate pool.
    """
    if not candidates:
        return np.zeros((0, 44), dtype=np.float32)

    sh, sw = search_img.shape[:2]
    N = len(candidates)

    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)

    # Precompute image-level maps
    heatmap = compute_global_landmark_heatmap(ref_img, search_img)

    ref_grad = compute_sobel_gradient(ref_img)
    search_grad = compute_sobel_gradient(search_img)

    ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)
    search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)

    search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)
    search_var = compute_local_variance_map(search_img, ksize=15)

    ref_edge = compute_canny_edge(ref_img)
    search_edge = compute_canny_edge(search_img)

    ref_100 = cv2.resize(ref_gray_f, (100, 100), cv2.INTER_AREA)
    ref_blur_100 = cv2.GaussianBlur(ref_100, (15, 15), 3.0)
    ref_var_100 = compute_local_variance_map(cv2.resize(ref_img, (100, 100), cv2.INTER_AREA), ksize=5)

    pad = 300
    search_gray_pad = cv2.copyMakeBorder(search_gray_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_grad_pad = cv2.copyMakeBorder(search_grad, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_log_pad = cv2.copyMakeBorder(search_log, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_blur_pad = cv2.copyMakeBorder(search_blur, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_var_pad = cv2.copyMakeBorder(search_var, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_edge_pad = cv2.copyMakeBorder(search_edge, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Pre-build templates per scale
    unique_scales = list(set([c.get('scale', 0.10) for c in candidates]))
    ref_templates = {}
    for s in unique_scales:
        cw = max(10, int(round(ref_img.shape[1] * s)))
        ch = max(10, int(round(ref_img.shape[0] * s)))
        s_g = cv2.resize(ref_gray_f, (cw, ch), cv2.INTER_AREA)
        s_d = cv2.resize(ref_grad, (cw, ch), cv2.INTER_AREA)
        s_l = cv2.resize(ref_log, (cw, ch), cv2.INTER_AREA)
        s_e = cv2.resize(ref_edge, (cw, ch), cv2.INTER_AREA)
        ref_e_dil = cv2.dilate(s_e, np.ones((3, 3), np.float32))
        ref_templates[s] = (cw, ch, s_g, s_d, s_l, s_e, ref_e_dil)

    scores = np.array([c.get('score', 0.0) for c in candidates], dtype=np.float32)
    top1_score = float(np.max(scores)) if len(scores) > 0 else 0.0
    median_score = float(np.median(scores)) if len(scores) > 0 else 0.0
    top10_mean_score = float(np.mean(scores[:min(10, N)])) if len(scores) > 0 else 0.0

    raw_features_list = []

    for idx, c in enumerate(candidates):
        cx, cy = c['cx'], c['cy']
        s = c.get('scale', 0.10)
        cand_score = c.get('score', 0.0)

        cw, ch, s_ref_g, s_ref_d, s_ref_l, s_ref_e, ref_e_dilated = ref_templates[s]
        tl_x_p = int(round(cx + pad - cw / 2.0))
        tl_y_p = int(round(cy + pad - ch / 2.0))

        patch_g = search_gray_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_d = search_grad_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_l = search_log_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_b = search_blur_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_v = search_var_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
        patch_e = search_edge_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]

        z_g = zmuv_ncc(patch_g, s_ref_g) if patch_g.shape == s_ref_g.shape else 0.0
        z_d = zmuv_ncc(patch_d, s_ref_d) if patch_d.shape == s_ref_d.shape else 0.0
        z_l = zmuv_ncc(patch_l, s_ref_l) if patch_l.shape == s_ref_l.shape else 0.0

        p_b_100 = cv2.resize(patch_b, (100, 100), cv2.INTER_AREA) if patch_b.size > 0 else np.zeros((100, 100), np.float32)
        p_v_100 = cv2.resize(patch_v, (100, 100), cv2.INTER_AREA) if patch_v.size > 0 else np.zeros((100, 100), np.float32)
        p_g_100 = cv2.resize(patch_g, (100, 100), cv2.INTER_AREA) if patch_g.size > 0 else np.zeros((100, 100), np.float32)

        z_lf = zmuv_ncc(p_b_100, ref_blur_100)
        z_var = zmuv_ncc(p_v_100, ref_var_100)

        e_cnt = np.sum(patch_e > 0.1)
        e_overlap = float(np.sum((patch_e > 0.1) & (ref_e_dilated > 0.1)) / float(e_cnt)) if e_cnt > 0 else 0.0
        multi_scale_sig = zmuv_ncc(p_g_100, ref_100)

        raw_features_list.append({
            'cx': cx, 'cy': cy, 'score': cand_score,
            'z_g': z_g, 'z_d': z_d, 'z_l': z_l, 'z_lf': z_lf, 'z_var': z_var,
            'e_overlap': e_overlap, 'multi_scale_sig': multi_scale_sig
        })

    v_keys = ['z_g', 'z_d', 'z_l', 'z_lf', 'z_var', 'e_overlap', 'multi_scale_sig']
    norm_v = {}
    for vk in v_keys:
        vals = np.array([rf[vk] for rf in raw_features_list], dtype=np.float32)
        mu = float(np.mean(vals))
        std = float(np.std(vals)) + 1e-5
        norm_v[vk] = np.tanh(0.5 * (vals - mu) / std)

    feat_matrix = np.zeros((N, 44), dtype=np.float32)
    all_cxs = np.array([c['cx'] for c in candidates], dtype=np.float32)
    all_cys = np.array([c['cy'] for c in candidates], dtype=np.float32)

    cw_01 = int(round(ref_gray_f.shape[1] * 0.10))
    ch_01 = int(round(ref_gray_f.shape[0] * 0.10))
    s_ref_01 = cv2.resize(ref_gray_f, (cw_01, ch_01), cv2.INTER_AREA)

    for i in range(N):
        rf = raw_features_list[i]
        cx, cy = rf['cx'], rf['cy']
        cand_score = rf['score']

        v1 = cand_score
        v2 = rf['z_g']
        v3 = rf['z_d']
        v4 = rf['z_l']
        v5 = rf['z_lf']
        v6 = rf['z_var']
        v7 = rf['e_overlap']
        v8 = rf['multi_scale_sig']
        nz_g = norm_v['z_g'][i]
        nz_d = norm_v['z_d'][i]
        nz_l = norm_v['z_l'][i]
        nz_lf = norm_v['z_lf'][i]
        nz_var = norm_v['z_var'][i]
        ne_overlap = norm_v['e_overlap'][i]
        nmulti_scale = norm_v['multi_scale_sig'][i]

        abs_x = cx / 1000.0
        abs_y = cy / 1000.0

        norm_x = cx / float(sw)
        norm_y = cy / float(sh)
        dist_center = math.hypot(norm_x - 0.5, norm_y - 0.5)

        lat_x = cx / float(lx)
        lat_y = cy / float(ly)
        phase_x = (cx % float(lx)) / float(lx)
        phase_y = (cy % float(ly)) / float(ly)

        sin_px = math.sin(2.0 * math.pi * cx / float(lx))
        cos_px = math.cos(2.0 * math.pi * cx / float(lx))
        sin_py = math.sin(2.0 * math.pi * cy / float(ly))
        cos_py = math.cos(2.0 * math.pi * cy / float(ly))

        rank_idx = float(i)
        rank_norm = rank_idx / 500.0
        rank_log = math.log(rank_idx + 1.0) / 6.22
        percentile = 1.0 - rank_norm

        dists = np.hypot(all_cxs - cx, all_cys - cy)
        density_r30 = float(np.sum((dists > 0.0) & (dists <= 30.0))) / 50.0
        density_r60 = float(np.sum((dists > 0.0) & (dists <= 60.0))) / 50.0
        dists_no_self = dists[dists > 0.0]
        dist_nearest = float(np.min(dists_no_self)) / 100.0 if len(dists_no_self) > 0 else 1.0

        top5_spatial_indices = np.argsort(dists)[:min(6, len(dists))]
        mean_top5_spatial = float(np.mean(scores[top5_spatial_indices])) if len(top5_spatial_indices) > 0 else cand_score

        def sample_off(off_x, off_y):
            tx = int(round(cx + off_x + pad - cw_01 / 2.0))
            ty = int(round(cy + off_y + pad - ch_01 / 2.0))
            p = search_gray_pad[ty:ty+ch_01, tx:tx+cw_01]
            return zmuv_ncc(p, s_ref_01) if p.shape == s_ref_01.shape else 0.0

        resp_px = sample_off(lx, 0)
        resp_nx = sample_off(-lx, 0)
        resp_py = sample_off(0, ly)
        resp_ny = sample_off(0, -ly)

        margin_top1 = cand_score - top1_score
        margin_median = cand_score - median_score
        margin_top10 = cand_score - top10_mean_score

        ix = int(np.clip(round(cx), 0, sw - 1))
        iy = int(np.clip(round(cy), 0, sh - 1))
        global_val = float(heatmap[iy, ix])

        tl_x_m = int(round(cx + pad - cw_01 / 2.0))
        tl_y_m = int(round(cy + pad - ch_01 / 2.0))
        patch_m = search_gray_pad[tl_y_m:tl_y_m+ch_01, tl_x_m:tl_x_m+cw_01]
        macro_score = zmuv_ncc(patch_m, s_ref_01) if patch_m.shape == s_ref_01.shape else 0.0

        vec = [
            v1, v2, v3, v4, v5, v6, v7, v8, nz_g, nz_d, nz_l, nz_lf, nz_var, ne_overlap, nmulti_scale,
            abs_x, abs_y,
            norm_x, norm_y, dist_center,
            lat_x, lat_y, phase_x, phase_y,
            sin_px, cos_px, sin_py, cos_py,
            rank_norm, rank_log, percentile,
            density_r30, density_r60, dist_nearest,
            mean_top5_spatial, resp_px, resp_nx, resp_py, resp_ny,
            margin_top1, margin_median, margin_top10,
            global_val, macro_score
        ]
        feat_matrix[i, :] = np.array(vec, dtype=np.float32)

    return feat_matrix


_coordinate_model_cache = None


def load_trained_coordinate_model(checkpoint_path: str = "checkpoints/coordinate_aware_ranker.pt") -> CoordinateAwareRankerNet:
    """Loads trained Coordinate-Aware Candidate Ranker model checkpoint."""
    global _coordinate_model_cache
    if _coordinate_model_cache is not None:
        return _coordinate_model_cache

    model = CoordinateAwareRankerNet(input_dim=44, hidden_dim=128)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[CoordinateAwareRankerNet] Successfully loaded trained weights from '{checkpoint_path}'.")
    else:
        print(f"[CoordinateAwareRankerNet] Warning: Checkpoint '{checkpoint_path}' not found! Using random initialization.")

    _coordinate_model_cache = model
    return model


def compute_coordinate_aware_scores(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    checkpoint_path: str = "checkpoints/coordinate_aware_ranker.pt"
) -> list:
    """Calculates Coordinate-Aware ranking scores for candidate pool."""
    model = load_trained_coordinate_model(checkpoint_path)
    lx, ly = estimate_lattice_period_2d(ref_img)

    feats = extract_coordinate_aware_features_pool(ref_img, search_img, candidates, lx, ly)
    if feats.shape[0] == 0:
        return []

    t_feats = torch.tensor(feats, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(t_feats)
        if scores.dim() == 0:
            scores_list = [float(scores.item())]
        else:
            scores_list = [float(s.item()) for s in scores]

    return scores_list
