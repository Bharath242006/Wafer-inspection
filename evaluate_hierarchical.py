"""
evaluate_hierarchical.py

Evaluates the Coarse-to-Fine Hierarchical Localization algorithm (localization/hierarchical_localizer.py)
on all 200 validation image pairs in dataset/validation.

Saves complete per-image evaluation results to results/hierarchical_validation.csv.
Computes comprehensive error statistics (mean, median, max, accuracy thresholds, failure count)
overall and for DRAM vs FinFET, and prints a 6-way comparison table across Baseline,
Structural V1, Frequency, Previous Hybrid, Context Hybrid, and Hierarchical algorithms.
"""

import csv
import math
import os
import sys
import numpy as np

# Import hierarchical localizer
sys.path.append(os.path.abspath("."))
from localization.hierarchical_localizer import locate_reference_pattern


def load_previous_results(csv_path: str) -> dict:
    """Loads and computes metrics from a previous validation CSV file."""
    if not os.path.exists(csv_path):
        return {}

    errors = []
    failed_cnt = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            err = float(row["error_px"])
            status = row.get("status", "SUCCESS")
            if status == "FAILED":
                failed_cnt += 1
            errors.append(err)

    if not errors:
        return {}

    n = len(errors)
    return {
        "count": n,
        "failed": failed_cnt,
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "within_1": sum(1 for e in errors if e <= 1.0),
        "within_2": sum(1 for e in errors if e <= 2.0),
        "within_5": sum(1 for e in errors if e <= 5.0),
        "within_10": sum(1 for e in errors if e <= 10.0),
        "within_25": sum(1 for e in errors if e <= 25.0),
        "within_50": sum(1 for e in errors if e <= 50.0),
        "within_100": sum(1 for e in errors if e <= 100.0),
        "acc_1": (sum(1 for e in errors if e <= 1.0) / n) * 100.0,
        "acc_2": (sum(1 for e in errors if e <= 2.0) / n) * 100.0,
        "acc_5": (sum(1 for e in errors if e <= 5.0) / n) * 100.0,
        "acc_10": (sum(1 for e in errors if e <= 10.0) / n) * 100.0,
        "acc_25": (sum(1 for e in errors if e <= 25.0) / n) * 100.0,
        "acc_50": (sum(1 for e in errors if e <= 50.0) / n) * 100.0,
        "acc_100": (sum(1 for e in errors if e <= 100.0) / n) * 100.0,
    }


def compute_metrics(subset: list) -> dict:
    """Computes comprehensive error and accuracy metrics for a list of result dicts."""
    n = len(subset)
    if n == 0:
        return {}

    errors = [r["error_px"] for r in subset]
    successful_cnt = sum(1 for r in subset if r["status"] == "SUCCESS")
    failed_cnt = sum(1 for r in subset if r["status"] == "FAILED")

    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    max_err = float(np.max(errors))

    w_1 = sum(1 for e in errors if e <= 1.0)
    w_2 = sum(1 for e in errors if e <= 2.0)
    w_5 = sum(1 for e in errors if e <= 5.0)
    w_10 = sum(1 for e in errors if e <= 10.0)
    w_25 = sum(1 for e in errors if e <= 25.0)
    w_50 = sum(1 for e in errors if e <= 50.0)
    w_100 = sum(1 for e in errors if e <= 100.0)

    return {
        "count": n,
        "successful": successful_cnt,
        "failed": failed_cnt,
        "mean": mean_err,
        "median": median_err,
        "max": max_err,
        "within_1": w_1,
        "within_2": w_2,
        "within_5": w_5,
        "within_10": w_10,
        "within_25": w_25,
        "within_50": w_50,
        "within_100": w_100,
        "acc_1": (w_1 / n) * 100.0,
        "acc_2": (w_2 / n) * 100.0,
        "acc_5": (w_5 / n) * 100.0,
        "acc_10": (w_10 / n) * 100.0,
        "acc_25": (w_25 / n) * 100.0,
        "acc_50": (w_50 / n) * 100.0,
        "acc_100": (w_100 / n) * 100.0,
    }


def evaluate_hierarchical_validation():
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "hierarchical_validation.csv")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    results = []

    print(f"Evaluating Hierarchical Coarse-to-Fine Localizer on {len(records)} validation pairs...")

    for idx, item in enumerate(records, start=1):
        img_name = item["image"]
        true_x = float(item["x"])
        true_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern(ref_path, search_path)

        if fine_center is not None and status == "SUCCESS":
            pred_x, pred_y = fine_center
            err = math.hypot(pred_x - true_x, pred_y - true_y)
        else:
            if coarse_center is not None:
                pred_x, pred_y = coarse_center
                err = math.hypot(pred_x - true_x, pred_y - true_y)
            else:
                pred_x = -1.0
                pred_y = -1.0
                err = 1000.0
            status = "FAILED"

        coarse_x, coarse_y = coarse_center if coarse_center else (-1.0, -1.0)
        coarse_err = math.hypot(coarse_x - true_x, coarse_y - true_y) if coarse_x >= 0 else 1000.0

        results.append({
            "image": img_name,
            "style": style,
            "status": status,
            "predicted_x": pred_x,
            "predicted_y": pred_y,
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "coarse_err_px": coarse_err,
            "true_x": true_x,
            "true_y": true_y,
            "error_px": err,
            "confidence": confidence
        })

        if idx % 20 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)} pairs...")

    # Save to results/hierarchical_validation.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "style", "status", "predicted_x", "predicted_y", "coarse_x", "coarse_y", "coarse_err_px", "true_x", "true_y", "error_px", "confidence"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image"],
                "style": r["style"],
                "status": r["status"],
                "predicted_x": f"{r['predicted_x']:.2f}" if r["predicted_x"] >= 0 else "None",
                "predicted_y": f"{r['predicted_y']:.2f}" if r["predicted_y"] >= 0 else "None",
                "coarse_x": f"{r['coarse_x']:.2f}" if r["coarse_x"] >= 0 else "None",
                "coarse_y": f"{r['coarse_y']:.2f}" if r["coarse_y"] >= 0 else "None",
                "coarse_err_px": f"{r['coarse_err_px']:.2f}",
                "true_x": f"{r['true_x']:.2f}",
                "true_y": f"{r['true_y']:.2f}",
                "error_px": f"{r['error_px']:.2f}",
                "confidence": f"{r['confidence']:.4f}"
            })

    print(f"\nSaved detailed evaluation results to: {out_csv}")

    overall = compute_metrics(results)
    dram_subset = [r for r in results if r["style"] == "DRAM"]
    finfet_subset = [r for r in results if r["style"] == "FinFET"]
    dram_metrics = compute_metrics(dram_subset)
    finfet_metrics = compute_metrics(finfet_subset)

    def print_metric_block(title, m):
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Total Pairs:             {m['count']}")
        print(f"Successful Localizations:{m['successful']}")
        print(f"Failed Localizations:    {m['failed']}")
        print(f"Mean Error:              {m['mean']:.2f} px")
        print(f"Median Error:            {m['median']:.2f} px")
        print(f"Max Error:               {m['max']:.2f} px")
        print(f"Within 1 px:             {m['within_1']} ({m['acc_1']:.1f}%)")
        print(f"Within 2 px:             {m['within_2']} ({m['acc_2']:.1f}%)")
        print(f"Within 5 px:             {m['within_5']} ({m['acc_5']:.1f}%)")
        print(f"Within 10 px:            {m['within_10']} ({m['acc_10']:.1f}%)")
        print(f"Within 25 px:            {m['within_25']} ({m['acc_25']:.1f}%)")
        print(f"Within 50 px:            {m['within_50']} ({m['acc_50']:.1f}%)")
        print(f"Within 100 px:           {m['within_100']} ({m['acc_100']:.1f}%)")

    print("\n" + "#" * 70)
    print("      HIERARCHICAL COARSE-TO-FINE LOCALIZER EVALUATION REPORT")
    print("#" * 70)

    print_metric_block("OVERALL METRICS", overall)
    print_metric_block("DRAM METRICS", dram_metrics)
    print_metric_block("FINFET METRICS", finfet_metrics)

    # Load all previous evaluation CSVs
    base_m = load_previous_results(os.path.join("results", "baseline_validation.csv"))
    struct_m = load_previous_results(os.path.join("results", "structural_validation.csv"))
    freq_m = load_previous_results(os.path.join("results", "frequency_validation.csv"))
    prev_hybrid_m = load_previous_results(os.path.join("results", "hybrid_validation.csv"))
    ctx_hybrid_m = load_previous_results(os.path.join("results", "hybrid_context_validation.csv"))

    print("\n" + "=" * 125)
    print("                               ALGORITHM PERFORMANCE COMPARISON TABLE")
    print("=" * 125)
    header = f"{'Metric':<24} | {'Baseline':<10} | {'Structural':<10} | {'Frequency':<10} | {'Prev Hybrid':<11} | {'Ctx Hybrid':<11} | {'Hierarchical':<13}"
    print(header)
    print("-" * 125)

    def format_val(m, key, fmt=".2f", is_pct=False):
        if not m or key not in m:
            return "N/A"
        val = m[key]
        if is_pct:
            return f"{val:{fmt}}%"
        return f"{val:{fmt}}"

    metrics_rows = [
        ("Total Pairs", "count", "d", False),
        ("Failed Count", "failed", "d", False),
        ("Mean Error (px)", "mean", ".2f", False),
        ("Median Error (px)", "median", ".2f", False),
        ("Max Error (px)", "max", ".2f", False),
        ("Accuracy (<= 1 px)", "acc_1", ".1f", True),
        ("Accuracy (<= 2 px)", "acc_2", ".1f", True),
        ("Accuracy (<= 5 px)", "acc_5", ".1f", True),
        ("Accuracy (<= 10 px)", "acc_10", ".1f", True),
        ("Accuracy (<= 25 px)", "acc_25", ".1f", True),
        ("Accuracy (<= 50 px)", "acc_50", ".1f", True),
        ("Accuracy (<= 100 px)", "acc_100", ".1f", True),
    ]

    for label, key, fmt, is_pct in metrics_rows:
        b_str = format_val(base_m, key, fmt, is_pct)
        s_str = format_val(struct_m, key, fmt, is_pct)
        f_str = format_val(freq_m, key, fmt, is_pct)
        p_str = format_val(prev_hybrid_m, key, fmt, is_pct)
        c_str = format_val(ctx_hybrid_m, key, fmt, is_pct)
        h_str = format_val(overall, key, fmt, is_pct)
        print(f"{label:<24} | {b_str:<10} | {s_str:<10} | {f_str:<10} | {p_str:<11} | {c_str:<11} | {h_str:<13}")

    print("=" * 125)


if __name__ == "__main__":
    evaluate_hierarchical_validation()
