"""
training/dataset_siamese.py

Dynamic, memory-efficient PyTorch Dataset loader for training the Siamese CNN Candidate Ranker.
Supports max_images parameter for fast debug and rapid testing.
"""

import csv
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    """
    Dynamic, memory-efficient PyTorch Dataset for training Siamese Neural Networks.
    Loads image metadata in __init__ and extracts image patches dynamically in __getitem__.
    """
    PAIRS_PER_IMAGE = 20  # 5 positive + 8 hard alias negatives + 7 random negatives

    def __init__(
        self,
        labels_csv: str = None,
        ref_dir: str = None,
        search_dir: str = None,
        data_dir: str = None,
        max_images: int = None
    ):
        super().__init__()
        if data_dir is not None:
            if labels_csv is None:
                labels_csv = os.path.join(data_dir, "labels.csv")
            if ref_dir is None:
                ref_dir = os.path.join(data_dir, "reference")
            if search_dir is None:
                search_dir = os.path.join(data_dir, "search")

        self.labels_csv = labels_csv
        self.ref_dir = ref_dir
        self.search_dir = search_dir

        self.records = []
        if labels_csv and os.path.exists(labels_csv):
            with open(labels_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.records.append(row)

        if max_images is not None and max_images > 0:
            self.records = self.records[:max_images]

        self.num_images = len(self.records)

    def _crop_patch(self, img: np.ndarray, cx: float, cy: float, patch_size: int = 100) -> np.ndarray:
        pad = patch_size
        img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        
        tl_x = int(round(cx + pad - patch_size / 2.0))
        tl_y = int(round(cy + pad - patch_size / 2.0))
        
        crop = img_pad[tl_y:tl_y+patch_size, tl_x:tl_x+patch_size]
        if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
            crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        return crop

    def _normalize(self, patch: np.ndarray) -> torch.Tensor:
        patch_f = patch.astype(np.float32)
        mean_val = np.mean(patch_f)
        std_val = np.std(patch_f) + 1e-5
        norm_patch = (patch_f - mean_val) / std_val
        return torch.tensor(norm_patch, dtype=torch.float32).unsqueeze(0)

    def __len__(self) -> int:
        return self.num_images * self.PAIRS_PER_IMAGE

    def __getitem__(self, idx: int) -> tuple:
        img_idx = idx // self.PAIRS_PER_IMAGE
        sub_idx = idx % self.PAIRS_PER_IMAGE

        item = self.records[img_idx]
        img_name = item["image"]
        gt_x = float(item["x"])
        gt_y = float(item["y"])

        ref_p = os.path.join(self.ref_dir, img_name)
        sch_p = os.path.join(self.search_dir, img_name)

        ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        sch_img = cv2.imread(sch_p, cv2.IMREAD_GRAYSCALE)

        if ref_img is None:
            ref_img = np.zeros((1000, 1000), dtype=np.uint8)
        if sch_img is None:
            sch_img = np.zeros((1000, 1000), dtype=np.uint8)

        ref_crop = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

        # Determine target candidate crop center and label dynamically
        if sub_idx < 5:
            # Positive jittered target pair (Label = 1.0)
            jitters = [(-2.0, -2.0), (-2.0, 2.0), (0.0, 0.0), (2.0, -2.0), (2.0, 2.0)]
            jx, jy = jitters[sub_idx]
            cand_cx, cand_cy = gt_x + jx, gt_y + jy
            label = 1.0
        elif sub_idx < 13:
            # Hard periodic alias negative pair (Label = 0.0)
            lx, ly = 67.0, 67.0
            offsets = [
                (+lx, 0.0), (-lx, 0.0), (0.0, +ly), (0.0, -ly),
                (+lx, +ly), (-lx, +ly), (+lx, -ly), (-lx, -ly)
            ]
            ox, oy = offsets[sub_idx - 5]
            cand_cx = float(np.clip(gt_x + ox, 40.0, 960.0))
            cand_cy = float(np.clip(gt_y + oy, 40.0, 960.0))
            label = 0.0
        else:
            # Random background negative pair (Label = 0.0)
            rng = np.random.RandomState(idx)
            rx = rng.uniform(50.0, 950.0)
            ry = rng.uniform(50.0, 950.0)
            if math.hypot(rx - gt_x, ry - gt_y) <= 50.0:
                rx = (rx + 150.0) % 900.0 + 50.0
            cand_cx, cand_cy = rx, ry
            label = 0.0

        cand_crop = self._crop_patch(sch_img, cand_cx, cand_cy, 100)

        t_ref = self._normalize(ref_crop)
        t_cand = self._normalize(cand_crop)
        t_label = torch.tensor(label, dtype=torch.float32)

        return t_ref, t_cand, t_label
