"""
training/dataset_context_ranker_v2.py

PyTorch Dataset Loader for training ContextAwareRankerV2 with Pairwise Hard Negative Mining.
Extracts:
- Reference Local Patch (100x100)
- Reference Context Patch (300x300 downsampled to 100x100)
- Positive Candidate Patch (GT candidate, dist <= 15px)
- Hard Negative Candidate Patches (Top-10 highest-scoring false candidates, dist >= 30px)
- Relative Spatial Offsets & Candidate Generator Scores
"""

import os
import csv
import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from localization.candidate_generation import generate_candidate_pool_multi


class ContextRankerDatasetV2(Dataset):
    """
    Dataset loader for training ContextAwareRankerV2 with Pairwise Hard Negative Mining.
    """
    def __init__(
        self,
        csv_path: str,
        ref_dir: str,
        search_dir: str,
        max_samples: int = None,
        pos_dist_thresh: float = 15.0,
        neg_dist_thresh: float = 30.0,
        top_k_hard_negs: int = 10
    ):
        super().__init__()
        self.ref_dir = ref_dir
        self.search_dir = search_dir
        self.pairs = []

        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        if max_samples is not None:
            records = records[:max_samples]

        print(f"[ContextRankerDatasetV2] Extracting pairwise hard-negatives from {len(records)} image records...", flush=True)

        for idx, item in enumerate(records):
            img_name = item["image"]
            gt_x = float(item["x"])
            gt_y = float(item["y"])

            ref_path = os.path.join(ref_dir, img_name)
            search_path = os.path.join(search_dir, img_name)

            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

            if ref_img is None or search_img is None:
                continue

            sh, sw = search_img.shape[:2]
            ref_h, ref_w = ref_img.shape[:2]

            pad = 300
            ref_pad = cv2.copyMakeBorder(ref_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            search_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

            rcx, rcy = ref_w / 2.0 + pad, ref_h / 2.0 + pad
            ref_center_x, ref_center_y = ref_w / 2.0, ref_h / 2.0

            # Reference Local Patch (100x100)
            ref_loc = ref_pad[int(rcy-50):int(rcy+50), int(rcx-50):int(rcx+50)]
            if ref_loc.shape[0] != 100 or ref_loc.shape[1] != 100:
                ref_loc = cv2.resize(ref_loc, (100, 100), cv2.INTER_AREA)
            ref_loc_f = ref_loc.astype(np.float32)
            ref_loc_norm = (ref_loc_f - np.mean(ref_loc_f)) / (np.std(ref_loc_f) + 1e-5)

            # Reference Context Patch (300x300 -> 100x100)
            ref_ctx = cv2.resize(ref_pad[int(rcy-150):int(rcy+150), int(rcx-150):int(rcx+150)], (100, 100), cv2.INTER_AREA)
            ref_ctx_f = ref_ctx.astype(np.float32)
            ref_ctx_norm = (ref_ctx_f - np.mean(ref_ctx_f)) / (np.std(ref_ctx_f) + 1e-5)

            # Generate Top-500 Candidate Pool
            cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=500)

            # Helper to crop candidate
            def get_cand_data(cx, cy, s, score):
                cw = max(4, int(round(ref_w * s)))
                ch = max(4, int(round(ref_h * s)))

                tl_x = int(round(cx + pad - cw / 2.0))
                tl_y = int(round(cy + pad - ch / 2.0))

                crop = search_pad[tl_y:tl_y+ch, tl_x:tl_x+cw]
                if crop.shape[0] != 100 or crop.shape[1] != 100:
                    crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)

                crop_f = crop.astype(np.float32)
                crop_norm = (crop_f - np.mean(crop_f)) / (np.std(crop_f) + 1e-5)

                rel_dx = float((cx - ref_center_x) / sw)
                rel_dy = float((cy - ref_center_y) / sh)

                return crop_norm, [rel_dx, rel_dy, float(score)]

            # Locate positive candidate (nearest to GT)
            dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands]
            pos_cand = None
            if dists and np.min(dists) <= pos_dist_thresh:
                pos_idx = int(np.argmin(dists))
                pos_cand = cands[pos_idx]
            else:
                # Force exact Ground Truth candidate if candidate pool missed exact peak
                pos_cand = {'cx': gt_x, 'cy': gt_y, 'scale': 0.10, 'score': 1.0}

            pos_crop, pos_spatial = get_cand_data(
                pos_cand['cx'], pos_cand['cy'], pos_cand.get('scale', 0.10), pos_cand.get('score', 1.0)
            )

            # Hard Negative Mining: Sort candidates by score descending and take Top-K false candidates (dist >= neg_dist_thresh)
            neg_candidates = []
            cands_sorted_by_score = sorted(cands, key=lambda c: c.get('score', 0.0), reverse=True)

            for cand in cands_sorted_by_score:
                dist = math.hypot(cand['cx'] - gt_x, cand['cy'] - gt_y)
                if dist >= neg_dist_thresh:
                    neg_candidates.append(cand)
                if len(neg_candidates) >= top_k_hard_negs:
                    break

            # Create pairwise training samples (GT positive vs Hard Negative)
            for neg_cand in neg_candidates:
                neg_crop, neg_spatial = get_cand_data(
                    neg_cand['cx'], neg_cand['cy'], neg_cand.get('scale', 0.10), neg_cand.get('score', 0.0)
                )

                self.pairs.append({
                    'ref_loc': ref_loc_norm,
                    'ref_ctx': ref_ctx_norm,
                    'cand_pos': pos_crop,
                    'spatial_pos': pos_spatial,
                    'cand_neg': neg_crop,
                    'spatial_neg': neg_spatial,
                })

            if (idx + 1) % 20 == 0:
                print(f"[ContextRankerDatasetV2] Processed {idx+1}/{len(records)} records | Total pairs: {len(self.pairs)}", flush=True)

        print(f"[ContextRankerDatasetV2] Total Pairwise Training Pairs Mined: {len(self.pairs)}", flush=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        p = self.pairs[idx]

        t_ref_loc = torch.tensor(p['ref_loc'], dtype=torch.float32).unsqueeze(0)
        t_ref_ctx = torch.tensor(p['ref_ctx'], dtype=torch.float32).unsqueeze(0)

        t_cand_pos = torch.tensor(p['cand_pos'], dtype=torch.float32).unsqueeze(0)
        t_spatial_pos = torch.tensor(p['spatial_pos'], dtype=torch.float32)

        t_cand_neg = torch.tensor(p['cand_neg'], dtype=torch.float32).unsqueeze(0)
        t_spatial_neg = torch.tensor(p['spatial_neg'], dtype=torch.float32)

        return t_ref_loc, t_ref_ctx, t_cand_pos, t_spatial_pos, t_cand_neg, t_spatial_neg
