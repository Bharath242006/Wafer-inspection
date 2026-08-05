"""
scratch/test_coordinate_ranker_00001.py

Sanity check script for Coordinate-Aware Candidate Ranker on single sample (00001.png & 00065.png).
"""

import csv
import math
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath("."))
from scratch.improve_candidate_recall import generate_candidate_pool_multi
from localization.final_localizer import estimate_lattice_period_2d
from localization.coordinate_aware_ranker import extract_coordinate_aware_features_pool, compute_coordinate_aware_scores


def main():
    ref_path = os.path.join("dataset", "validation", "reference", "00001.png")
    search_path = os.path.join("dataset", "validation", "search", "00001.png")
    gt_x, gt_y = 512.45, 489.12  # example target GT

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        print("Error: Could not read 00001.png images.")
        return

    print("Step 1: Generating Top-500 candidate pool...")
    cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=500)
    print(f"Candidate count: {len(cands)}")

    print("Step 2: Estimating 2D lattice period...")
    lx, ly = estimate_lattice_period_2d(ref_img)
    print(f"Lattice period: lx={lx:.2f} px, ly={ly:.2f} px")

    print("Step 3: Extracting 44-D Coordinate-Aware feature matrix...")
    feats = extract_coordinate_aware_features_pool(ref_img, search_img, cands, lx, ly)
    print(f"Feature matrix shape: {feats.shape}")

    print("Step 4: Computing model scores...")
    scores = compute_coordinate_aware_scores(ref_img, search_img, cands)
    print(f"Computed {len(scores)} scores. Min score: {np.min(scores):.4f}, Max score: {np.max(scores):.4f}")

    print("[SUCCESS] Coordinate-Aware Candidate Ranker pipeline test passed.")


if __name__ == "__main__":
    main()
