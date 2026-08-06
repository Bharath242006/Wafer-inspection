"""
training/dataset_coordinate_ranker.py

PyTorch Dataset Loader for training CoordinateAwareRanker.
Generates realistic pairs of positive (near GT) and negative (distractor) candidate patches
with spatial position and scale metadata.
"""

import os
import csv
import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from localization.candidate_generation import generate_candidate_pool_multi


class CoordinateRankerDataset(Dataset):
    """
    Dataset loader for training CoordinateAwareRanker on wafer alignment pairs.
    """
    def __init__(
        self,
        csv_path: str,
        ref_dir: str,
        search_dir: str,
        max_samples: int = None,
        pos_dist_thresh: float = 15.0,
        neg_dist_thresh: float = 30.0
    ):
        super().__init__()
        self.ref_dir = ref_dir
        self.search_dir = search_dir
        self.samples = []

        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        if max_samples is not None:
            records = records[:max_samples]

        print(f"[Dataset] Processing {len(records)} records for CoordinateRankerDataset...")

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

            # Standard 100x100 reference patch
            ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA).astype(np.float32)
            ref_norm = (ref_100 - np.mean(ref_100)) / (np.std(ref_100) + 1e-5)

            # Generate candidate pool
            cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=100)

            # Always add Ground Truth positive candidate
            cands.append({
                'cx': gt_x,
                'cy': gt_y,
                'scale': 0.10
            })

            # Add jittered Ground Truth positives for augmentation
            for _ in range(2):
                jx = gt_x + np.random.uniform(-3.0, 3.0)
                jy = gt_y + np.random.uniform(-3.0, 3.0)
                js = 0.10 + np.random.uniform(-0.01, 0.01)
                cands.append({'cx': jx, 'cy': jy, 'scale': js})

            for cand in cands:
                cx = float(cand['cx'])
                cy = float(cand['cy'])
                s = float(cand.get('scale', 0.10))

                dist = math.hypot(cx - gt_x, cy - gt_y)

                if dist <= pos_dist_thresh:
                    label = 1.0
                elif dist >= neg_dist_thresh:
                    label = 0.0
                else:
                    continue  # Ambiguous margin zone

                norm_x = float(np.clip(cx / sw, 0.0, 1.0))
                norm_y = float(np.clip(cy / sh, 0.0, 1.0))

                # Crop candidate patch
                pad = 60
                search_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
                cw = max(4, int(round(ref_w * s)))
                ch = max(4, int(round(ref_h * s)))

                tl_x_pad = int(round(cx + pad - cw / 2.0))
                tl_y_pad = int(round(cy + pad - ch / 2.0))

                crop = search_pad[tl_y_pad:tl_y_pad+ch, tl_x_pad:tl_x_pad+cw]
                if crop.shape[0] != 100 or crop.shape[1] != 100:
                    crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)

                crop_f = crop.astype(np.float32)
                crop_norm = (crop_f - np.mean(crop_f)) / (np.std(crop_f) + 1e-5)

                self.samples.append({
                    'ref': ref_norm,
                    'cand': crop_norm,
                    'spatial': [norm_x, norm_y, s],
                    'label': label
                })

        print(f"[Dataset] Total dataset size: {len(self.samples)} pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]

        ref_tensor = torch.tensor(sample['ref'], dtype=torch.float32).unsqueeze(0)
        cand_tensor = torch.tensor(sample['cand'], dtype=torch.float32).unsqueeze(0)
        spatial_tensor = torch.tensor(sample['spatial'], dtype=torch.float32)
        label_tensor = torch.tensor([sample['label']], dtype=torch.float32)

        return ref_tensor, cand_tensor, spatial_tensor, label_tensor
