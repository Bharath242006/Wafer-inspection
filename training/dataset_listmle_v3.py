"""
training/dataset_listmle_v3.py

PyTorch ListMLE Dataset V3 for Semiconductor Wafer Candidate Ranking.
Reads image pairs from dataset_small (train and validation splits), generates K candidate patches per sample
centered around ground-truth and negative spatial locations, and formats inputs for ListMLE ranking model.
"""

import csv
import math
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Enable tolerant PIL image loading for safe handling
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_default_transform(img_size: int = 224) -> T.Compose:
    """Returns standard ImageNet normalization and resize transform pipeline."""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ListMLEDatasetV3(Dataset):
    """
    ListMLE Ranking Dataset V3 reading from dataset_small.
    Returns reference patch, K candidate patches, ground-truth candidate ranks, and candidate coordinates.
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "dataset_small",
        split: str = "train",
        num_candidates: int = 32,
        patch_size: int = 128,
        img_size: int = 224,
        seed: Optional[int] = 42,
        transform: Optional[T.Compose] = None,
    ) -> None:
        """
        Args:
            root_dir (str | Path): Base path to dataset_small. Default: 'dataset_small'.
            split (str): Split name ('train' or 'validation'). Default: 'train'.
            num_candidates (int): Number of candidate patches K per sample. Default: 32.
            patch_size (int): Spatial crop size for candidate patches before resizing. Default: 128.
            img_size (int): Target square dimension for model input. Default: 224.
            seed (int, optional): Random seed for deterministic candidate sampling. Default: 42.
            transform (T.Compose, optional): Custom torchvision transform. Default: ImageNet norm transform.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_candidates = num_candidates
        self.patch_size = patch_size
        self.img_size = img_size
        self.seed = seed
        self.transform = transform if transform is not None else get_default_transform(img_size)

        self.split_dir = self.root_dir / split
        self.ref_dir = self.split_dir / "reference"
        self.search_dir = self.split_dir / "search"
        self.labels_csv = self.split_dir / "labels.csv"

        self.records: List[Dict[str, str]] = []
        self._load_records()

    def _load_records(self) -> None:
        """Loads and validates records from labels.csv safely."""
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels file missing: {self.labels_csv}")

        try:
            with open(self.labels_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row.get("image", "").strip()
                    ref_path = self.ref_dir / img_name
                    search_path = self.search_dir / img_name

                    # Verify existence and non-zero size of image files
                    if ref_path.is_file() and ref_path.stat().st_size > 0 and search_path.is_file() and search_path.stat().st_size > 0:
                        self.records.append({
                            "image": img_name,
                            "x": float(row.get("x", 500.0)),
                            "y": float(row.get("y", 500.0)),
                            "style": row.get("style", "Unknown"),
                            "ref_path": str(ref_path),
                            "search_path": str(search_path),
                        })
        except Exception as e:
            print(f"Warning: Exception encountered while parsing {self.labels_csv}: {e}")

        if len(self.records) == 0:
            print(f"Warning: No valid records found in {self.labels_csv}")

    def __len__(self) -> int:
        return len(self.records)

    def _crop_patch(self, img: Image.Image, cx: float, cy: float, patch_size: int) -> Image.Image:
        """Crops a square patch centered at (cx, cy) with boundary padding if required."""
        w, h = img.size
        half = patch_size / 2.0
        left = int(round(cx - half))
        top = int(round(cy - half))
        right = left + patch_size
        bottom = top + patch_size

        # Zero-pad if crop extends beyond boundaries
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - w)
        pad_bottom = max(0, bottom - h)

        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(w, right)
        crop_bottom = min(h, bottom)

        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_w = cropped.width + pad_left + pad_right
            padded_h = cropped.height + pad_top + pad_bottom
            padded_img = Image.new(img.mode, (padded_w, padded_h), color=0)
            padded_img.paste(cropped, (pad_left, pad_top))
            return padded_img

        return cropped

    def _generate_candidates(
        self,
        gt_x: float,
        gt_y: float,
        img_w: int,
        img_h: int,
        idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates K candidate coordinates deterministically around GT and search space.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - candidate_coords: (K, 2) array of (x, y) coordinates.
                - distances: (K,) array of Euclidean distances to ground truth.
        """
        # Deterministic random state per sample index and global seed
        sample_seed = (self.seed + idx * 10007) if self.seed is not None else (idx * 10007)
        rng = np.random.RandomState(sample_seed % (2**31 - 1))

        K = self.num_candidates
        coords: List[Tuple[float, float]] = []

        # Candidate 0: Ground-truth location with tiny noise (best candidate)
        noise_x = float(rng.uniform(-1.0, 1.0))
        noise_y = float(rng.uniform(-1.0, 1.0))
        coords.append((gt_x + noise_x, gt_y + noise_y))

        # Candidates 1 to K//4: Near negatives (radius 10-50 px)
        num_near = max(1, K // 4)
        for _ in range(num_near):
            radius = float(rng.uniform(10.0, 50.0))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            cx = float(np.clip(gt_x + radius * math.cos(angle), 10.0, img_w - 10.0))
            cy = float(np.clip(gt_y + radius * math.sin(angle), 10.0, img_h - 10.0))
            coords.append((cx, cy))

        # Candidates K//4 to K//2: Medium negatives (radius 50-200 px)
        num_med = max(1, K // 4)
        for _ in range(num_med):
            radius = float(rng.uniform(50.0, 200.0))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            cx = float(np.clip(gt_x + radius * math.cos(angle), 10.0, img_w - 10.0))
            cy = float(np.clip(gt_y + radius * math.sin(angle), 10.0, img_h - 10.0))
            coords.append((cx, cy))

        # Remaining candidates: Uniform random negatives across search image
        while len(coords) < K:
            cx = float(rng.uniform(self.patch_size / 2.0, img_w - self.patch_size / 2.0))
            cy = float(rng.uniform(self.patch_size / 2.0, img_h - self.patch_size / 2.0))
            coords.append((cx, cy))

        cand_array = np.array(coords[:K], dtype=np.float32)  # (K, 2)
        gt_array = np.array([gt_x, gt_y], dtype=np.float32)

        # Compute Euclidean distance to ground truth for each candidate
        distances = np.linalg.norm(cand_array - gt_array, axis=1)  # (K,)

        return cand_array, distances

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single dataset item.

        Returns:
            Dict[str, torch.Tensor]:
                - 'reference_patch': [3, 224, 224]
                - 'candidate_patches': [K, 3, 224, 224]
                - 'target_rank': [K] (indices sorted by distance ascending)
                - 'candidate_coordinates': [K, 2]
        """
        rec = self.records[idx]

        try:
            ref_img = Image.open(rec["ref_path"]).convert("RGB")
            search_img = Image.open(rec["search_path"]).convert("RGB")
        except Exception as e:
            print(f"Error reading image files for index {idx} ({rec['image']}): {e}")
            # Fallback to black dummy images if corrupt
            ref_img = Image.new("RGB", (1000, 1000), color=0)
            search_img = Image.new("RGB", (1000, 1000), color=0)

        img_w, img_h = search_img.size
        gt_x, gt_y = rec["x"], rec["y"]

        # Generate K candidates and distance scores
        cand_coords, distances = self.generate_candidates_for_item(gt_x, gt_y, img_w, img_h, idx)

        # Crop and transform reference patch
        ref_patch_tensor = self.transform(ref_img)  # [3, 224, 224]

        # Crop and transform candidate patches
        cand_tensors: List[torch.Tensor] = []
        for i in range(self.num_candidates):
            cx, cy = cand_coords[i, 0], cand_coords[i, 1]
            cand_crop = self._crop_patch(search_img, cx, cy, self.patch_size)
            cand_tensor = self.transform(cand_crop)  # [3, 224, 224]
            cand_tensors.append(cand_tensor)

        candidate_patches_tensor = torch.stack(cand_tensors, dim=0)  # [K, 3, 224, 224]

        # Target rank order: sort candidate indices by ascending distance (closest = best = index 0)
        sorted_rank_indices = np.argsort(distances).astype(np.int64)  # [K]
        target_rank_tensor = torch.from_numpy(sorted_rank_indices)

        candidate_coords_tensor = torch.from_numpy(cand_coords)  # [K, 2]

        return {
            "reference_patch": ref_patch_tensor,
            "candidate_patches": candidate_patches_tensor,
            "target_rank": target_rank_tensor,
            "candidate_coordinates": candidate_coords_tensor,
        }

    def generate_candidates_for_item(
        self,
        gt_x: float,
        gt_y: float,
        img_w: int,
        img_h: int,
        idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Public helper to expose candidate generation for testing or evaluation."""
        return self._generate_candidates(gt_x, gt_y, img_w, img_h, idx)
