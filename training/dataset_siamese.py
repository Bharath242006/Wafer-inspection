"""
training/dataset_siamese.py

PyTorch Dataset loader for training the Siamese CNN Candidate Ranker.

Generates:
- Positive pairs: (reference 100x100 crop, target search 100x100 crop) [Label: 1.0]
- Hard periodic alias negatives: (reference crop, search crop at GT +/- lambda_x/y) [Label: 0.0]
- Random candidate negatives: (reference crop, candidate search crop at dist > 40 px) [Label: 0.0]
"""

import csv
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def estimate_lattice_period_2d(ref_img: np.ndarray) -> tuple:
    """Estimates 2D lattice period lambda_x, lambda_y in search space."""
    if ref_img.shape[0] > 200:
        ref_s = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_s = ref_img.copy()

    ref_f = ref_s.astype(np.float32) - np.mean(ref_s)
    f = np.fft.fft2(ref_f)
    power = np.abs(f)**2
    autocorr = np.real(np.fft.ifft2(power))
    autocorr = np.fft.fftshift(autocorr)

    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[cy-2:cy+3, cx-2:cx+3] = 0.0

    _, _, _, max_loc = cv2.minMaxLoc(autocorr)
    p_dx = max_loc[0] - cx
    p_dy = max_loc[1] - cy

    scale_fac = (ref_img.shape[0] / ref_s.shape[0]) * 10.0
    lx = abs(p_dx) * scale_fac if abs(p_dx) > 2 else 67.0
    ly = abs(p_dy) * scale_fac if abs(p_dy) > 2 else 67.0

    lx = float(np.clip(lx, 30.0, 150.0))
    ly = float(np.clip(ly, 30.0, 150.0))
    return lx, ly


class SiameseDataset(Dataset):
    """
    Dataset class for training Siamese Neural Networks on semiconductor image pairs.
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

        self.samples = []
        self._build_pairs()

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

    def _build_pairs(self):
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

            ref_crop = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
            lx, ly = estimate_lattice_period_2d(ref_img)

            # 1. Positive pair (GT target centered with slight spatial jitter)
            for dx in [-2.0, 0.0, 2.0]:
                for dy in [-2.0, 0.0, 2.0]:
                    pos_crop = self._crop_patch(sch_img, gt_x + dx, gt_y + dy, 100)
                    self.samples.append((ref_crop, pos_crop, 1.0))

            # 2. Hard Periodic Alias Negatives (+/- lambda_x, +/- lambda_y)
            offsets = [
                (+lx, 0.0), (-lx, 0.0), (0.0, +ly), (0.0, -ly),
                (+lx, +ly), (-lx, +ly), (+lx, -ly), (-lx, -ly),
                (+2*lx, 0.0), (-2*lx, 0.0)
            ]
            for ox, oy in offsets:
                nx = gt_x + ox
                ny = gt_y + oy
                if 40.0 <= nx <= 960.0 and 40.0 <= ny <= 960.0:
                    neg_crop = self._crop_patch(sch_img, nx, ny, 100)
                    self.samples.append((ref_crop, neg_crop, 0.0))

            # 3. Random background negatives
            rng = np.random.RandomState(42)
            for _ in range(10):
                rx = rng.uniform(50.0, 950.0)
                ry = rng.uniform(50.0, 950.0)
                if math.hypot(rx - gt_x, ry - gt_y) > 50.0:
                    neg_crop = self._crop_patch(sch_img, rx, ry, 100)
                    self.samples.append((ref_crop, neg_crop, 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        ref_patch, cand_patch, label = self.samples[idx]
        t_ref = self._normalize(ref_patch)
        t_cand = self._normalize(cand_patch)
        t_label = torch.tensor(label, dtype=torch.float32)
        return t_ref, t_cand, t_label
