"""
evaluate_frequency.py

Evaluates the frequency localizer algorithm (localization/frequency_localizer.py)
on all 200 validation image pairs in dataset/validation.

Calculates error statistics (mean, median, max, accuracy thresholds, failure count)
overall and broken down by architecture (DRAM / FinFET), prints the 10 worst failures,
compares with Baseline and Structural V1 performance, and exports results to
results/frequency_validation.csv.
"""

import os
import csv
import math
import numpy as np

from localization.frequency_localizer import locate_reference_pattern


def evaluate_validation():
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "frequency_validation.csv")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    results = []

    print(f"Evaluating frequency localizer on {len(records)} validation pairs...")

    for idx, item in enumerate(records, start=1):
        img_name = item["image"]
        true_x = float(item["x"])
        true_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        pred_x, pred_y, metrics, _, _ = locate_reference_pattern(ref_path, search_path)

        if pred_x is not None and pred_y is not None:
            err = math.hypot(pred_x - true_x, pred_y - true_y)
            status = "SUCCESS"
        else:
            # Failed localization distance penalty (max search space dimension)
            err = 1000.0
            status = "FAILED"

        results.append({
            "image": img_name,
            "style": style,
            "status": status,
            "predicted_x": pred_x if pred_x is not None else -1.0,
            "predicted_y": pred_y if pred_y is not None else -1.0,
            "true_x": true_x,
            "true_y": true_y,
            "error_px": err,
            "phase_response": metrics.get("phase_response", 0.0),
            "freq_sim": metrics.get("best_freq_score", 0.0)
        })

        if idx % 20 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)} pairs...")

    # Save complete results to results/frequency_validation.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "style", "status", "predicted_x", "predicted_y", "true_x", "true_y", "error_px", "phase_response", "freq_sim"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image"],
                "style": r["style"],
                "status": r["status"],
                "predicted_x": f"{r['predicted_x']:.2f}" if r["status"] == "SUCCESS" else "None",
                "predicted_y": f"{r['predicted_y']:.2f}" if r["status"] == "SUCCESS" else "None",
                "true_x": f"{r['true_x']:.2f}",
                "true_y": f"{r['true_y']:.2f}",
                "error_px": f"{r['error_px']:.2f}",
                "phase_response": f"{r['phase_response']:.4f}",
                "freq_sim": f"{r['freq_sim']:.4f}"
            })

    print(f"\nSaved detailed evaluation results to: {out_csv}")

    def compute_metrics(subset):
        n = len(subset)
        if n == 0:
            return {}
        failed_cnt = sum(1 for r in subset if r["status"] == "FAILED")
        errors = [r["error_px"] for r in subset]

        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        max_err = float(np.max(errors))

        within_1 = sum(1 for e in errors if e <= 1.0)
        within_2 = sum(1 for e in errors if e <= 2.0)
        within_5 = sum(1 for e in errors if e <= 5.0)
        within_10 = sum(1 for e in errors if e <= 10.0)
        within_25 = sum(1 for e in errors if e <= 25.0)
        within_50 = sum(1 for e in errors if e <= 50.0)
        within_100 = sum(1 for e in errors if e <= 100.0)

        acc_5 = (within_5 / n) * 100.0

        return {
            "count": n,
            "failed_cnt": failed_cnt,
            "mean": mean_err,
            "median": median_err,
            "max": max_err,
            "within_1": within_1,
            "within_2": within_2,
            "within_5": within_5,
            "within_10": within_10,
            "within_25": within_25,
            "within_50": within_50,
            "within_100": within_100,
            "acc_5": acc_5
        }

    overall = compute_metrics(results)
    dram_subset = [r for r in results if r["style"] == "DRAM"]
    finfet_subset = [r for r in results if r["style"] == "FinFET"]
    dram_metrics = compute_metrics(dram_subset)
    finfet_metrics = compute_metrics(finfet_subset)

    print("\n==================================================")
    print("  FREQUENCY LOCALIZER EVALUATION REPORT")
    print("==================================================")

    def print_metric_block(title, m):
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Total Pairs:            {m['count']}")
        print(f"Failed Count:           {m['failed_cnt']}")
        print(f"Mean Error:             {m['mean']:.2f} px")
        print(f"Median Error:           {m['median']:.2f} px")
        print(f"Max Error:              {m['max']:.2f} px")
        print(f"Within 1 px:            {m['within_1']} ({m['within_1']/m['count']*100:.1f}%)")
        print(f"Within 2 px:            {m['within_2']} ({m['within_2']/m['count']*100:.1f}%)")
        print(f"Within 5 px:            {m['within_5']} ({m['within_5']/m['count']*100:.1f}%)")
        print(f"Within 10 px:           {m['within_10']} ({m['within_10']/m['count']*100:.1f}%)")
        print(f"Within 25 px:           {m['within_25']} ({m['within_25']/m['count']*100:.1f}%)")
        print(f"Within 50 px:           {m['within_50']} ({m['within_50']/m['count']*100:.1f}%)")
        print(f"Within 100 px:          {m['within_100']} ({m['within_100']/m['count']*100:.1f}%)")
        print(f"Accuracy (<= 5 px):     {m['acc_5']:.2f}%")

    print_metric_block("OVERALL METRICS", overall)
    print_metric_block("DRAM METRICS", dram_metrics)
    print_metric_block("FINFET METRICS", finfet_metrics)

    print("\nMETHOD COMPARISON")
    print("-----------------")
    print(f"Baseline Mean:          369.10 px | Acc (<= 5 px): 0.00%")
    print(f"Structural V1 Mean:     363.28 px | Acc (<= 5 px): 0.50%")
    print(f"Frequency Localizer:    {overall['mean']:.2f} px | Acc (<= 5 px): {overall['acc_5']:.2f}%")

    # 10 worst failures
    sorted_results = sorted(results, key=lambda x: x["error_px"], reverse=True)
    print("\n10 WORST FAILURES")
    print("-----------------")
    print(f"{'Image':<10} {'Style':<8} {'Status':<8} {'Predicted (x,y)':<22} {'True (x,y)':<22} {'Error (px)':<10}")
    print("-" * 83)
    for r in sorted_results[:10]:
        pred_str = f"({r['predicted_x']:.2f}, {r['predicted_y']:.2f})" if r["status"] == "SUCCESS" else "None"
        true_str = f"({r['true_x']:.2f}, {r['true_y']:.2f})"
        print(f"{r['image']:<10} {r['style']:<8} {r['status']:<8} {pred_str:<22} {true_str:<22} {r['error_px']:<10.2f}")


if __name__ == "__main__":
    evaluate_validation()
