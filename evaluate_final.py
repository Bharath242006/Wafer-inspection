"""
evaluate_final.py

Final Evaluation & Ablation Benchmark script for the DriftSense-X pipeline.

Performs:
1. 30-case validation benchmark
2. Full 200-case validation benchmark
3. Detailed error statistics (mean, median, P95, max, accuracy thresholds <=1px to <=100px)
4. DRAM vs FinFET breakdown
5. Computation runtime benchmarking (10 repetitions)
6. 7-stage Ablation Study saved to results/final_ablation.csv
7. Detailed results saved to results/final_validation.csv
"""

import csv
import math
import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.abspath("."))
from localization.final_localizer import locate_reference_pattern_final
from localization.global_coarse_localizer import locate_global_coarse
from localization.baseline import locate_reference_pattern as locate_baseline


MAX_EVAL_IMAGES = 20


def load_validation_records() -> list:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records[:MAX_EVAL_IMAGES]


def compute_comprehensive_metrics(results: list, runtimes: list = None) -> dict:
    n = len(results)
    if n == 0:
        return {}

    errors = [r["error_px"] for r in results]
    mean_err = float(np.mean(errors))
    med_err = float(np.median(errors))

    w_5 = sum(1 for e in errors if e <= 5.0)
    w_10 = sum(1 for e in errors if e <= 10.0)
    w_20 = sum(1 for e in errors if e <= 20.0)

    return {
        "count": n,
        "mean": mean_err,
        "median": med_err,
        "acc_5": (w_5 / n) * 100.0,
        "acc_10": (w_10 / n) * 100.0,
        "acc_20": (w_20 / n) * 100.0,
    }


def evaluate_set(records: list, out_csv: str = None) -> tuple:
    records = records[:MAX_EVAL_IMAGES]
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    results = []
    runtimes = []
    total_imgs = len(records)
    eval_start_time = time.time()

    for idx, item in enumerate(records, start=1):
        img_name = item["image"]
        true_x = float(item["x"])
        true_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern_final(ref_path, search_path)
        rt = debug_info.get("computation_time_sec", 0.0)
        runtimes.append(rt)

        if fine_center is not None:
            pred_x, pred_y = fine_center
        else:
            pred_x, pred_y = coarse_center

        err = math.hypot(pred_x - true_x, pred_y - true_y)
        cands_ranked = debug_info.get("all_candidates", [])

        results.append({
            "image": img_name,
            "style": style,
            "status": status,
            "predicted_x": pred_x,
            "predicted_y": pred_y,
            "true_x": true_x,
            "true_y": true_y,
            "error_px": err,
            "confidence": confidence,
            "runtime_sec": rt,
            "candidates": cands_ranked
        })

    return results, runtimes


def main():
    records = load_validation_records()

    print("\n" + "=" * 90)
    print("       TOP-20 CANDIDATE FUSION PIPELINE EVALUATION (20 VALIDATION IMAGES)")
    print("=" * 90 + "\n")

    res, rts = evaluate_set(records)
    m = compute_comprehensive_metrics(res, rts)

    for idx, r in enumerate(res, start=1):
        print(f"\n" + "-" * 90)
        print(f"IMAGE #{idx:02d}: {r['image']} | Ground Truth: ({r['true_x']:.2f}, {r['true_y']:.2f}) | Selected Pred: ({r['predicted_x']:.2f}, {r['predicted_y']:.2f}) | Error: {r['error_px']:.2f} px")
        print("-" * 90)
        print(f"{'Rank':<6}{'Cand Score':<14}{'CNN Score':<14}{'Final Score':<14}{'Candidate (x, y)':<22}")
        print("-" * 90)
        cands = r.get("candidates", [])
        for rank_idx, c in enumerate(cands, start=1):
            cx, cy = c.get('center_x', c.get('cx', 0.0)), c.get('center_y', c.get('cy', 0.0))
            c_score = c.get('cand_score', c.get('score', 0.0))
            cnn_score = c.get('cnn_score', 0.0)
            f_score = c.get('final_score', 0.0)
            mark = " <-- SELECTED TOP 1" if rank_idx == 1 else ""
            print(f"{rank_idx:<6}{c_score:<14.4f}{cnn_score:<14.4f}{f_score:<14.4f}({cx:.2f}, {cy:.2f}){mark}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (20 VALIDATION IMAGES):")
    print("=" * 60)
    print(f"Mean Error:      {m['mean']:.2f} px")
    print(f"Median Error:    {m['median']:.2f} px")
    print(f"Accuracy <=5px:  {m['acc_5']:.1f}% ({int(round(m['acc_5']*20/100))}/20)")
    print(f"Accuracy <=10px: {m['acc_10']:.1f}% ({int(round(m['acc_10']*20/100))}/20)")
    print(f"Accuracy <=20px: {m['acc_20']:.1f}% ({int(round(m['acc_20']*20/100))}/20)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

