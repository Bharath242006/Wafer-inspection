"""
training/dataset_coordinate_ranker.py

Fast PyTorch Dataset loader for Coordinate-Aware Candidate Ranker.
"""

import csv
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scratch.improve_candidate_recall import generate_candidate_pool_multi
from localization.final_localizer import estimate_lattice_period_2d
from localization.coordinate_aware_ranker import extract_coordinate_aware_features_pool


class CoordinateAwareTripletDataset(Dataset):
    """
    Triplet Dataset generating positive and hard-negative candidate feature pairs.
    """
    def __init__(self, csv_file: str, ref_dir: str, search_dir: str, is_train: bool = True, split_idx: int = 160):
        super().__init__()
        self.csv_file = csv_file
        self.ref_dir = ref_dir
        self.search_dir = search_dir

        self.records = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

        if is_train:
            self.records = self.records[:split_idx]
        else:
            self.records = self.records[split_idx:]

        self.pairs = []
        self._build_triplets()

    def _build_triplets(self):
        print(f"[CoordinateAwareTripletDataset] Fast-generating candidate feature pairs for {len(self.records)} images...")

        for idx, item in enumerate(self.records, start=1):
            img_name = item["image"]
            gt_x = float(item["x"])
            gt_y = float(item["y"])

            ref_path = os.path.join(self.ref_dir, img_name)
            search_path = os.path.join(self.search_dir, img_name)

            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

            if ref_img is None or search_img is None:
                continue

            cands_500 = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=500)
            if not cands_500:
                continue

            lx, ly = estimate_lattice_period_2d(ref_img)
            feat_matrix = extract_coordinate_aware_features_pool(ref_img, search_img, cands_500, lx, ly)

            dists = np.array([math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_500], dtype=np.float32)

            pos_idx = int(np.argmin(dists))
            pos_dist = dists[pos_idx]

            if pos_dist > 100.0:
                continue

            pos_feat = feat_matrix[pos_idx]

            # Select top hard negatives (periodic aliases & top false candidates)
            neg_indices = []
            for c_i, c in enumerate(cands_500):
                if c_i == pos_idx:
                    continue
                d = dists[c_i]
                if d <= 25.0:
                    continue

                dx = abs(c['cx'] - gt_x)
                dy = abs(c['cy'] - gt_y)
                rem_x = abs((dx % lx) if lx > 0 else dx)
                rem_y = abs((dy % ly) if ly > 0 else dy)
                is_alias = (rem_x <= 10.0 or rem_x >= (lx - 10.0)) and (rem_y <= 10.0 or rem_y >= (ly - 10.0))
                is_top_false = (c_i < 15 and d > 50.0)

                if is_alias or is_top_false:
                    neg_indices.append(c_i)
                if len(neg_indices) >= 10:
                    break

            if not neg_indices:
                neg_indices = [c_i for c_i in range(len(cands_500)) if dists[c_i] > 50.0][:10]

            for neg_idx in neg_indices:
                neg_feat = feat_matrix[neg_idx]
                self.pairs.append((pos_feat, neg_feat))

            if idx % 40 == 0:
                print(f"Processed {idx}/{len(self.records)} dataset images...")

        print(f"[CoordinateAwareTripletDataset] Successfully built {len(self.pairs)} training triplet pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_f, neg_f = self.pairs[idx]
        return torch.tensor(pos_f, dtype=torch.float32), torch.tensor(neg_f, dtype=torch.float32)
