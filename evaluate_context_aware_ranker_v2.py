"""
evaluate_context_aware_ranker_v2.py

Evaluation benchmark script for Trained Context-Aware Candidate Ranker V2
on the first 100 validation images.

Metrics printed:
- GT Mean Rank
- GT Median Rank
- Mean Error
- Median Error
- Accuracy <= 5px
- Accuracy <= 10px
- Accuracy <= 20px
- Accuracy <= 50px

Compares directly against previous Coordinate-Aware Ranker (GT Mean Rank = 221 / 500).
"""

import os
import sys
import csv
import math
import time
import cv2
import numpy as np

sys.path.append(os.path.abspath("."))

from localization.candidate_generation import generate_candidate_pool_multi
from localization.context_aware_ranker_v2 import compute_context_aware_v2_scores
from localization.fine_localization import refine_subpixel_peak


def load_first_100_val_records() -> list:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records[:100]


def main():
    records = load_first_100_val_records()
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    checkpoint_path = os.path.join("checkpoints", "context_aware_ranker_v2.pt")

    errors = []
    gt_ranks = []
    inference_times = []

    print("=" * 110, flush=True)
    print("      CONTEXT-AWARE RANKER V2 BENCHMARK EVALUATION (FIRST 100 VALIDATION IMAGES)", flush=True)
    print("=" * 110 + "\n", flush=True)

    for idx, item in enumerate(records, start=1):
        img_name = item["image"]
        gt_x = float(item["x"])
        gt_y = float(item["y"])

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if ref_raw is None or search_raw is None:
            print(f"Error loading {img_name}", flush=True)
            continue

        # 1. Candidate Generation (Top-500)
        cands = generate_candidate_pool_multi(ref_raw, search_raw, max_pool_size=500)
        cand_cnt = len(cands)

        if cand_cnt == 0:
            errors.append(1000.0)
            gt_ranks.append(500)
            print(f"Image {idx:03d} ({img_name}) | GT: ({gt_x:6.2f}, {gt_y:6.2f}) | NO CANDIDATES GENERATED", flush=True)
            continue

        # 2. Compute Context-Aware Ranker V2 Scores
        t0 = time.perf_counter()
        scores = compute_context_aware_v2_scores(ref_raw, search_raw, cands, checkpoint_path=checkpoint_path)
        t_elapsed = time.perf_counter() - t0
        inference_times.append(t_elapsed)

        for c, sc in zip(cands, scores):
            c['v2_score'] = sc
            cg_sc = float(c.get('final_score', c.get('score', 0.0)))
            c['rank_score'] = 0.60 * sc + 0.40 * cg_sc

        # 3. Sort Candidates by Combined Ranker Score Descending
        cands_sorted = sorted(cands, key=lambda c: c['rank_score'], reverse=True)


        # 4. Calculate GT Candidate Rank in Context-Aware Ranker V2 List
        dists_to_gt = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_sorted]
        best_gt_idx = int(np.argmin(dists_to_gt))
        gt_rank = best_gt_idx + 1  # 1-indexed
        gt_ranks.append(gt_rank)

        # 5. Top-1 Candidate Selection & Fine Subpixel Refinement
        top1_cand = cands_sorted[0]
        coarse_x = top1_cand['cx']
        coarse_y = top1_cand['cy']

        fine_x, fine_y = refine_subpixel_peak(
            search_raw, int(round(coarse_x)), int(round(coarse_y))
        )

        # 6. Localization Error Calculation
        err = float(math.hypot(fine_x - gt_x, fine_y - gt_y))
        errors.append(err)

        if idx % 10 == 0:
            print(f"Image {idx:03d} ({img_name}) | GT: ({gt_x:6.2f}, {gt_y:6.2f}) | "
                  f"Pred: ({fine_x:6.2f}, {fine_y:6.2f}) | Error: {err:6.2f} px | "
                  f"Score: {top1_cand['rank_score']:6.3f} | GT Rank: {gt_rank:3d}/500 | "
                  f"Time: {t_elapsed*1000.0:5.1f} ms", flush=True)

    total_n = len(records)

    def calc_acc(err_list, thresh):
        return (sum(1 for e in err_list if e <= thresh) / total_n) * 100.0

    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))

    acc_5 = calc_acc(errors, 5.0)
    acc_10 = calc_acc(errors, 10.0)
    acc_20 = calc_acc(errors, 20.0)
    acc_50 = calc_acc(errors, 50.0)

    mean_gt_rank = float(np.mean(gt_ranks))
    median_gt_rank = float(np.median(gt_ranks))
    avg_inf_time_ms = float(np.mean(inference_times)) * 1000.0

    print("\n" + "=" * 110, flush=True)
    print("           CONTEXT-AWARE RANKER V2 FINAL BENCHMARK RESULTS (100 IMAGES)", flush=True)
    print("=" * 110, flush=True)
    print(f"GT Mean Rank:                  {mean_gt_rank:6.1f} / 500", flush=True)
    print(f"GT Median Rank:                {median_gt_rank:6.1f} / 500", flush=True)
    print(f"Mean Error:                    {mean_err:7.2f} px", flush=True)
    print(f"Median Error:                  {median_err:7.2f} px", flush=True)
    print(f"Accuracy <= 5 px:              {acc_5:5.1f}% ({int(round(acc_5*total_n/100)):02d}/100)", flush=True)
    print(f"Accuracy <= 10 px:             {acc_10:5.1f}% ({int(round(acc_10*total_n/100)):02d}/100)", flush=True)
    print(f"Accuracy <= 20 px:             {acc_20:5.1f}% ({int(round(acc_20*total_n/100)):02d}/100)", flush=True)
    print(f"Accuracy <= 50 px:             {acc_50:5.1f}% ({int(round(acc_50*total_n/100)):02d}/100)", flush=True)
    print("-" * 110, flush=True)

    # Performance Comparison Table
    old_gt_mean = 221.8
    old_mean_err = 340.71
    old_acc20 = 7.0

    print("=" * 110, flush=True)
    print("     PERFORMANCE COMPARISON MATRIX (COORDINATE-AWARE RANKER vs CONTEXT-AWARE RANKER V2)", flush=True)
    print("=" * 110, flush=True)
    print(f"{'Metric':<25}{'Coordinate-Aware Ranker':<30}{'Context-Aware Ranker V2':<30}{'Improvement':<25}", flush=True)
    print("-" * 110, flush=True)
    print(f"{'GT Mean Rank':<25}{'221.8 / 500':<30}{mean_gt_rank:5.1f} / 500{' ':<22}{'Promoted by ' + str(round(old_gt_mean - mean_gt_rank, 1)) + ' spots'}", flush=True)
    print(f"{'GT Median Rank':<25}{'208.5 / 500':<30}{median_gt_rank:5.1f} / 500{' ':<22}{'Promoted by ' + str(round(208.5 - median_gt_rank, 1)) + ' spots'}", flush=True)
    print(f"{'Mean Error':<25}{'340.71 px':<30}{mean_err:6.2f} px{' ':<23}{'Reduced by ' + str(round(old_mean_err - mean_err, 2)) + ' px'}", flush=True)
    print(f"{'Accuracy <= 20px':<25}{'7.0%':<30}{acc_20:5.1f}%{' ':<24}{'+' + str(round(acc_20 - old_acc20, 1)) + '%'}", flush=True)
    print("=" * 110 + "\n", flush=True)


if __name__ == "__main__":
    main()
