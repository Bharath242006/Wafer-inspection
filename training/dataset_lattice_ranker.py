"""
training/dataset_lattice_ranker.py

PyTorch Dataset loader for Global/Lattice-Aware Candidate Ranker.

Extracts a 22-dimensional feature vector per candidate:
1. Normalized coordinates: cx/1000.0, cy/1000.0
2. Lattice coordinates: cx/lambda_x, cy/lambda_y
3. Fractional lattice phase: (cx % lambda_x)/lambda_x, (cy % lambda_y)/lambda_y
4. Neighbor cell response consistency (8 offsets: +/- lambda_x, +/- lambda_y, +/- diag)
5. 7 Z-score normalized structural features (NCC, gradient, LoG, edge, low-frequency, texture, multi-scale)
6. Surrounding macro context score

Generates triplets: (ref_img, positive_candidate_features, hard_negative_candidate_features)
"""

import csv
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from localization.candidate_generation import generate_candidate_pool_multi
from localization.features.edge_features import compute_sobel_gradient
from localization.global_coarse_localizer import compute_local_variance_map, zmuv_ncc
from localization.final_localizer import compute_canny_edge, estimate_lattice_period_2d, compute_multi_scale_signature


def extract_lattice_candidate_features(ref_img: np.ndarray, search_img: np.ndarray, candidate: dict, lx: float, ly: float) -> np.ndarray:
    """
    Extracts 22-dimensional feature vector for a candidate.
    """
    cx, cy = candidate['cx'], candidate['cy']
    s = candidate.get('scale', 0.10)
    sh, sw = search_img.shape[:2]

    # 1. Normalized position & lattice phase
    norm_x = cx / float(sw)
    norm_y = cy / float(sh)

    lat_x = cx / lx
    lat_y = cy / ly

    phase_x = (cx % lx) / lx
    phase_y = (cy % ly) / ly

    # 2. Local structural features
    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)

    cw = int(round(ref_img.shape[1] * s))
    ch = int(round(ref_img.shape[0] * s))

    pad = 300
    search_pad = cv2.copyMakeBorder(search_gray_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_grad = compute_sobel_gradient(search_img)
    search_grad_pad = cv2.copyMakeBorder(search_grad, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)
    search_log_pad = cv2.copyMakeBorder(search_log, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)
    search_blur_pad = cv2.copyMakeBorder(search_blur, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_var = compute_local_variance_map(search_img, ksize=15)
    search_var_pad = cv2.copyMakeBorder(search_var, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    tl_x_p = int(round(cx + pad - cw / 2.0))
    tl_y_p = int(round(cy + pad - ch / 2.0))

    patch_g = search_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
    patch_d = search_grad_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
    patch_l = search_log_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
    patch_b = search_blur_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
    patch_v = search_var_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]

    s_ref_g = cv2.resize(ref_gray_f, (cw, ch), cv2.INTER_AREA)
    s_ref_d = cv2.resize(compute_sobel_gradient(ref_img), (cw, ch), cv2.INTER_AREA)
    s_ref_l = cv2.resize(cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3), (cw, ch), cv2.INTER_AREA)

    ref_blur_100 = cv2.GaussianBlur(cv2.resize(ref_gray_f, (100, 100), cv2.INTER_AREA), (15, 15), 3.0)
    ref_var_100 = compute_local_variance_map(cv2.resize(ref_img, (100, 100), cv2.INTER_AREA), ksize=5)

    z_g = zmuv_ncc(patch_g, s_ref_g)
    z_d = zmuv_ncc(patch_d, s_ref_d)
    z_l = zmuv_ncc(patch_l, s_ref_l)
    z_lf = zmuv_ncc(cv2.resize(patch_b, (100, 100), cv2.INTER_AREA), ref_blur_100)
    z_var = zmuv_ncc(cv2.resize(patch_v, (100, 100), cv2.INTER_AREA), ref_var_100)

    # Edge overlap
    ref_edge = compute_canny_edge(ref_img)
    search_edge = compute_canny_edge(search_img)
    search_edge_pad = cv2.copyMakeBorder(search_edge, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    patch_e = search_edge_pad[tl_y_p:tl_y_p+ch, tl_x_p:tl_x_p+cw]
    s_ref_e = cv2.resize(ref_edge, (cw, ch), cv2.INTER_AREA)
    ref_e_dilated = cv2.dilate(s_ref_e, np.ones((3, 3), np.float32))
    e_cnt = np.sum(patch_e > 0.1)
    e_overlap = float(np.sum((patch_e > 0.1) & (ref_e_dilated > 0.1)) / float(e_cnt)) if e_cnt > 0 else 0.0

    multi_scale_sig = compute_multi_scale_signature(cv2.resize(patch_g, (100, 100), cv2.INTER_AREA), cv2.resize(ref_gray_f, (100, 100), cv2.INTER_AREA))

    # Macro context score
    sw_01 = int(round(ref_gray_f.shape[1] * 0.10))
    sh_01 = int(round(ref_gray_f.shape[0] * 0.10))
    ctx_w = min(sw, sw_01 * 3)
    ctx_h = min(sh, sh_01 * 3)
    x1_c = max(0, int(round(cx - ctx_w / 2.0)))
    y1_c = max(0, int(round(cy - ctx_h / 2.0)))
    x2_c = min(sw, int(round(cx + ctx_w / 2.0)))
    y2_c = min(sh, int(round(cy + ctx_h / 2.0)))

    s_ctx = search_gray_f[y1_c:y2_c, x1_c:x2_c]
    r_ctx = cv2.resize(ref_gray_f, (x2_c - x1_c, y2_c - y1_c), cv2.INTER_AREA)
    s_ctx_p = cv2.resize(s_ctx, (30, 30), cv2.INTER_AREA)
    r_ctx_p = cv2.resize(r_ctx, (30, 30), cv2.INTER_AREA)
    macro_score = float(max(0.0, cv2.matchTemplate(s_ctx_p, r_ctx_p, cv2.TM_CCOEFF_NORMED)[0, 0]))

    # 3. 8-Neighboring Cell Consistency
    neighbor_offsets = [
        (+lx, 0.0), (-lx, 0.0), (0.0, +ly), (0.0, -ly),
        (+lx, +ly), (-lx, +ly), (+lx, -ly), (-lx, -ly)
    ]
    neighbor_sims = []
    for ox, oy in neighbor_offsets:
        ncx = cx + ox
        ncy = cy + oy
        ntl_x_p = int(round(ncx + pad - cw / 2.0))
        ntl_y_p = int(round(ncy + pad - ch / 2.0))
        npatch_g = search_pad[ntl_y_p:ntl_y_p+ch, ntl_x_p:ntl_x_p+cw]
        n_zmuv = zmuv_ncc(npatch_g, s_ref_g) if npatch_g.shape == s_ref_g.shape else 0.0
        neighbor_sims.append(n_zmuv)

    feat_vector = np.array([
        norm_x, norm_y, lat_x, lat_y, phase_x, phase_y,
        z_g, z_d, z_l, e_overlap, z_lf, z_var, multi_scale_sig, macro_score,
        *neighbor_sims  # 8 neighbor features -> Total 22 features
    ], dtype=np.float32)

    return feat_vector


class LatticeTripletDataset(Dataset):
    """
    PyTorch Dataset generating feature vectors for triplet training.
    """
    def __init__(self, labels_csv: str, ref_dir: str, search_dir: str, is_train: bool = True, split_idx: int = 160):
        super().__init__()
        self.ref_dir = ref_dir
        self.search_dir = search_dir

        all_records = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_records.append(row)

        if is_train:
            self.records = all_records[:split_idx]
        else:
            self.records = all_records[split_idx:]

        self.triplets = []
        self._build_triplets()

    def _build_triplets(self):
        for item in self.records:
            img_name = item["image"]
            gt_x = float(item["x"])
            gt_y = float(item["y"])

            ref_p = os.path.join(self.ref_dir, img_name)
            sch_p = os.path.join(self.search_dir, img_name)

            ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
            sch_img = cv2.imread(sch_p, cv2.IMREAD_GRAYSCALE)

            if ref_img is None or sch_img is None:
                continue

            lx, ly = estimate_lattice_period_2d(ref_img)
            cands = generate_candidate_pool_multi(ref_img, sch_img, max_pool_size=200)

            if not cands:
                continue

            dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands]

            # Positive candidate (closest to GT)
            best_i = int(np.argmin(dists))
            pos_cand = cands[best_i]
            pos_feat = extract_lattice_candidate_features(ref_img, sch_img, pos_cand, lx, ly)

            # Hard negative candidates (periodic aliases or distant peaks)
            hard_negs = [cands[i] for i in range(len(cands)) if dists[i] > 35.0]

            for neg_cand in hard_negs[:8]:
                neg_feat = extract_lattice_candidate_features(ref_img, sch_img, neg_cand, lx, ly)
                self.triplets.append((pos_feat, neg_feat))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple:
        pos_f, neg_f = self.triplets[idx]
        return torch.tensor(pos_f, dtype=torch.float32), torch.tensor(neg_f, dtype=torch.float32)
