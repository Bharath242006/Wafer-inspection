"""
training/dataset_context_ranker.py

PyTorch Dataset loader for Multi-Context Candidate Ranker (Local, Medium, Large context).

Crops 3 spatial context fields per patch:
1. Local: 100x100 around candidate center (1x field of view).
2. Medium: 250x250 around candidate center, downsampled to 100x100 (2.5x field of view).
3. Large: 500x500 around candidate center, downsampled to 100x100 (5.0x field of view).

Generates triplets:
(reference_multi_context, positive_candidate_multi_context, hard_negative_candidate_multi_context)
"""

import csv
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from localization.candidate_generation import generate_candidate_pool_multi


class ContextTripletDataset(Dataset):
    """
    PyTorch Dataset generating multi-context triplets for training ranking models.
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

    def _crop_multi_context(self, img: np.ndarray, cx: float, cy: float) -> tuple:
        """Crops local (100x100), medium (250x250 -> 100x100), and large (500x500 -> 100x100) context patches."""
        pad = 300
        img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        cx_p = cx + pad
        cy_p = cy + pad

        # 1. Local 100x100
        x1_l = int(round(cx_p - 50))
        y1_l = int(round(cy_p - 50))
        c_local = img_pad[y1_l:y1_l+100, x1_l:x1_l+100]
        if c_local.shape[0] != 100 or c_local.shape[1] != 100:
            c_local = cv2.resize(c_local, (100, 100), cv2.INTER_AREA)

        # 2. Medium 250x250 -> 100x100
        x1_m = int(round(cx_p - 125))
        y1_m = int(round(cy_p - 125))
        c_med = cv2.resize(img_pad[y1_m:y1_m+250, x1_m:x1_m+250], (100, 100), cv2.INTER_AREA)

        # 3. Large 500x500 -> 100x100
        x1_g = int(round(cx_p - 250))
        y1_g = int(round(cy_p - 250))
        c_large = cv2.resize(img_pad[y1_g:y1_g+500, x1_g:x1_g+500], (100, 100), cv2.INTER_AREA)

        return c_local, c_med, c_large

    def _normalize(self, patch: np.ndarray) -> torch.Tensor:
        patch_f = patch.astype(np.float32)
        mean_val = np.mean(patch_f)
        std_val = np.std(patch_f) + 1e-5
        norm_patch = (patch_f - mean_val) / std_val
        return torch.tensor(norm_patch, dtype=torch.float32).unsqueeze(0)

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

            ref_loc, ref_med, ref_large = self._crop_multi_context(ref_img, 500.0, 500.0)

            cands = generate_candidate_pool_multi(ref_img, sch_img, max_pool_size=200)
            if not cands:
                continue

            dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands]
            min_dist = float(np.min(dists))

            # Find positive candidate (closest to GT)
            best_i = int(np.argmin(dists))
            pos_cand = cands[best_i]

            # Find hard negative candidates (periodic aliases or distant peaks)
            hard_negs = [cands[i] for i in range(len(cands)) if dists[i] > 35.0]

            if not hard_negs:
                continue

            pos_loc, pos_med, pos_large = self._crop_multi_context(sch_img, pos_cand['cx'], pos_cand['cy'])

            for neg_cand in hard_negs[:8]:  # Up to 8 hard negative pairs per reference
                neg_loc, neg_med, neg_large = self._crop_multi_context(sch_img, neg_cand['cx'], neg_cand['cy'])

                self.triplets.append((
                    (ref_loc, ref_med, ref_large),
                    (pos_loc, pos_med, pos_large),
                    (neg_loc, neg_med, neg_large)
                ))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple:
        ref_tuple, pos_tuple, neg_tuple = self.triplets[idx]

        t_ref_loc = self._normalize(ref_tuple[0])
        t_ref_med = self._normalize(ref_tuple[1])
        t_ref_lrg = self._normalize(ref_tuple[2])

        t_pos_loc = self._normalize(pos_tuple[0])
        t_pos_med = self._normalize(pos_tuple[1])
        t_pos_lrg = self._normalize(pos_tuple[2])

        t_neg_loc = self._normalize(neg_tuple[0])
        t_neg_med = self._normalize(neg_tuple[1])
        t_neg_lrg = self._normalize(neg_tuple[2])

        return (
            (t_ref_loc, t_ref_med, t_ref_lrg),
            (t_pos_loc, t_pos_med, t_pos_lrg),
            (t_neg_loc, t_neg_med, t_neg_lrg)
        )
