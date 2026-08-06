"""
evaluation/evaluate_final.py

Final Integrated Pipeline Evaluation & Ablation Benchmark script.
"""

import csv
import math
import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.abspath("."))
from localization.final_localizer import locate_target_final, locate_reference_pattern_final
from localization.global_coarse_localizer import locate_global_coarse
from localization.baseline import locate_reference_pattern as locate_baseline


MAX_EVAL_IMAGES = 20


def load_validation_records() -> list:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    records = []
    if os.path.exists(csv_path):
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
    succ_cnt = sum(1 for r in results if r["status"] == "SUCCESS")
    fail_cnt = sum(1 for r in results if r["status"] == "FAILED")
    amb_cnt = sum(1 for r in results if r.get("status") == "AMBIGUOUS")

    mean_err = float(np.mean(errors))
    med_err = float(np.median(errors))
    p95_err = float(np.percentile(errors, 95))
    max_err = float(np.max(errors))

    w_1 = sum(1 for e in errors if e <= 1.0)
    w_2 = sum(1 for e in errors if e <= 2.0)
    w_5 = sum(1 for e in errors if e <= 5.0)
    w_10 = sum(1 for e in errors if e <= 10.0)
    w_25 = sum(1 for e in errors if e <= 25.0)
    w_50 = sum(1 for e in errors if e <= 50.0)
    w_100 = sum(1 for e in errors if e <= 100.0)

    rt_mean = float(np.mean(runtimes)) if runtimes else 0.0
    rt_med = float(np.median(runtimes)) if runtimes else 0.0
    rt_p95 = float(np.percentile(runtimes, 95)) if runtimes else 0.0

    return {
        "count": n,
        "successful": succ_cnt,
        "ambiguous": amb_cnt,
        "failed": fail_cnt,
        "mean": mean_err,
        "median": med_err,
        "p95": p95_err,
        "max": max_err,
        "acc_1": (w_1 / n) * 100.0,
        "acc_2": (w_2 / n) * 100.0,
        "acc_5": (w_5 / n) * 100.0,
        "acc_10": (w_10 / n) * 100.0,
        "acc_25": (w_25 / n) * 100.0,
        "acc_50": (w_50 / n) * 100.0,
        "acc_100": (w_100 / n) * 100.0,
        "rt_mean_ms": rt_mean * 1000.0,
        "rt_med_ms": rt_med * 1000.0,
        "rt_p95_ms": rt_p95 * 1000.0
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
        print(f"Evaluating image {idx}/{total_imgs}")
        print(f"Processing: {img_name}")
        true_x = float(item["x"])
        true_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        if not os.path.exists(ref_path) or not os.path.exists(search_path):
            continue

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        t0 = time.time()
        pred_x, pred_y, score, status, details = locate_target_final(ref_img, search_img)
        dt = time.time() - t0
        runtimes.append(dt)

        err = float(np.hypot(pred_x - true_x, pred_y - true_y))
        res = {
            "image": img_name,
            "style": style,
            "true_x": true_x,
            "true_y": true_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_px": err,
            "score": score,
            "status": status,
            "runtime_ms": dt * 1000.0
        }
        results.append(res)

    total_eval_time = time.time() - eval_start_time
    print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")

    if out_csv and results:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    metrics = compute_comprehensive_metrics(results, runtimes)
    return results, metrics


def main():
    print("=" * 80)
    print("        DRIFTSENSE-X FINAL EVALUATION & ABLATION BENCHMARK")
    print("=" * 80)

    records = load_validation_records()
    if not records:
        print("No validation records found.")
        return

    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    out_csv = os.path.join("outputs", "metrics", "final_validation.csv")
    t_start = time.time()
    results, metrics = evaluate_set(records, out_csv=out_csv)
    t_end = time.time()

    print(f"Evaluated {metrics.get('count', 0)} samples.")
    print(f"Mean Error: {metrics.get('mean', 0.0):.2f} px")
    print(f"Median Error: {metrics.get('median', 0.0):.2f} px")
    print(f"Accuracy <= 5px: {metrics.get('acc_5', 0.0):.2f}%")
    print(f"Total Evaluation Time: {t_end - t_start:.2f} seconds")


if __name__ == "__main__":
    main()
