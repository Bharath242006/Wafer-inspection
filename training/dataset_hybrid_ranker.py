"""
training/dataset_hybrid_ranker.py

PyTorch Triplet Dataset for the DriftSense-X Final Hybrid Ranker.

Generates (positive_features, negative_features) pairs where:
- Positive:      The candidate closest to ground truth (within 100 px).
- Hard Negatives (priority order):
  1. Lattice aliases — candidates at approximately ±k*lx, ±k*ly from GT.
  2. Top-ranked false candidates — top-10 by score that are > 50 px from GT.
  3. Random far candidates — fill remaining to 10 negatives.

Training split: images 00001–00160 only.
Validation split (00161–00200) is NEVER loaded here.
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
from localization.hybrid_ranker import extract_hybrid_features_pool


class HybridRankerTripletDataset(Dataset):
    """
    Triplet Dataset for Hybrid Ranker training.

    Each sample is a (pos_feat, neg_feat) pair of 56-D feature vectors.
    Hard negatives are biased toward lattice aliases to teach the model
    to distinguish visually similar but periodically shifted candidates.
    """

    def __init__(
        self,
        csv_file: str,
        ref_dir: str,
        search_dir: str,
        is_train: bool = True,
        split_idx: int = 160,
        max_negatives_per_image: int = 10,
    ):
        super().__init__()
        self.pairs = []
        self._build_triplets(
            csv_file, ref_dir, search_dir, is_train, split_idx, max_negatives_per_image
        )

    def _build_triplets(
        self,
        csv_file: str,
        ref_dir: str,
        search_dir: str,
        is_train: bool,
        split_idx: int,
        max_neg: int,
    ):
        # Load records
        all_records = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_records.append(row)

        records = all_records[:split_idx] if is_train else all_records[split_idx:]
        print(
            f"[HybridRankerTripletDataset] Building triplets for "
            f"{'train' if is_train else 'val'} split ({len(records)} images)..."
        )

        skipped = 0
        for idx, item in enumerate(records, start=1):
            img_name = item["image"]
            gt_x = float(item["x"])
            gt_y = float(item["y"])

            ref_path = os.path.join(ref_dir, img_name)
            search_path = os.path.join(search_dir, img_name)

            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

            if ref_img is None or search_img is None:
                skipped += 1
                continue

            # Generate top-500 candidate pool
            cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=500)
            if not cands:
                skipped += 1
                continue

            # Estimate lattice periods
            lx, ly = estimate_lattice_period_2d(ref_img)

            # Extract 56-D feature matrix for full pool
            feat_matrix = extract_hybrid_features_pool(ref_img, search_img, cands, lx, ly)

            # Distances to ground truth for each candidate
            dists = np.array(
                [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands],
                dtype=np.float32
            )

            # Find the positive: closest candidate within 100 px of GT
            pos_idx = int(np.argmin(dists))
            if dists[pos_idx] > 100.0:
                skipped += 1
                continue  # GT not in candidate pool — skip image

            pos_feat = feat_matrix[pos_idx]

            # ── Hard negative selection ──────────────────────────────────────
            # Priority 1: Lattice aliases (≈k*lx or k*ly from GT, k∈{1,2,3,4,5})
            alias_indices = []
            for c_i, c in enumerate(cands):
                if c_i == pos_idx:
                    continue
                d = dists[c_i]
                if d <= 25.0:
                    continue  # Too close to GT — not a hard negative

                dx = abs(c['cx'] - gt_x)
                dy = abs(c['cy'] - gt_y)

                # Check if candidate lies at a lattice alias position
                rem_x = dx % float(lx) if lx > 0 else dx
                rem_y = dy % float(ly) if ly > 0 else dy
                # Allow tolerance of 12 px on either side
                alias_x = (rem_x <= 12.0) or (rem_x >= (lx - 12.0))
                alias_y = (rem_y <= 12.0) or (rem_y >= (ly - 12.0))

                if alias_x and alias_y:
                    alias_indices.append(c_i)
                if len(alias_indices) >= max_neg:
                    break

            # Priority 2: Top-scored false candidates (high-ranking but far from GT)
            top_false_indices = []
            for c_i in range(len(cands)):
                if c_i == pos_idx or c_i in alias_indices:
                    continue
                if dists[c_i] > 50.0:
                    top_false_indices.append(c_i)
                if len(top_false_indices) >= 5:
                    break

            # Priority 3: Random far candidates to fill up to max_neg
            remaining_needed = max_neg - len(alias_indices) - len(top_false_indices)
            random_indices = []
            if remaining_needed > 0:
                far_pool = [
                    c_i for c_i in range(len(cands))
                    if c_i != pos_idx
                    and c_i not in alias_indices
                    and c_i not in top_false_indices
                    and dists[c_i] > 50.0
                ]
                rng = np.random.default_rng(seed=idx)
                chosen = rng.choice(
                    far_pool,
                    size=min(remaining_needed, len(far_pool)),
                    replace=False
                ).tolist() if far_pool else []
                random_indices = [int(x) for x in chosen]

            neg_indices = alias_indices + top_false_indices + random_indices

            # If still no negatives, use any far candidate
            if not neg_indices:
                fallback = [
                    c_i for c_i in range(len(cands))
                    if c_i != pos_idx and dists[c_i] > 50.0
                ][:max_neg]
                neg_indices = fallback

            # Build triplet pairs
            for neg_idx in neg_indices:
                neg_feat = feat_matrix[neg_idx]
                self.pairs.append(
                    (
                        torch.tensor(pos_feat, dtype=torch.float32),
                        torch.tensor(neg_feat, dtype=torch.float32),
                    )
                )

            if idx % 40 == 0:
                print(f"  Processed {idx}/{len(records)} images | pairs so far: {len(self.pairs)}")

        print(
            f"[HybridRankerTripletDataset] Built {len(self.pairs)} triplet pairs "
            f"({skipped} images skipped)."
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]
