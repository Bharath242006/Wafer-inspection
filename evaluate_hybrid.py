"""
evaluate_hybrid.py

Evaluates the CURRENT context-enhanced hybrid localization algorithm (localization/hybrid_localizer.py)
on all 200 validation image pairs in dataset/validation.

Saves complete per-image evaluation results to results/hybrid_context_validation.csv.
Computes comprehensive error statistics (mean, median, max, accuracy thresholds, failure count)
overall and for DRAM vs FinFET, and prints a 5-way comparison table across Baseline,
Structural V1, Frequency, Previous Hybrid, and Context-Enhanced Hybrid algorithms.
"""

import csv
import math
import os
import sys
import numpy as np

# Import hybrid localizer locate function
sys.path.append(os.path.abspath("."))
from localization.hybrid_localizer import locate_reference_pattern


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


def evaluate_hybrid_validation():
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "hybrid_context_validation.csv")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    results = []

    print(f"Evaluating Context-Enhanced Hybrid Localizer on {len(records)} validation pairs...")

    for idx, item in enumerate(records, start=1):
        img_name = item["image"]
        true_x = float(item["x"])
        true_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        pred_center, confidence, status, debug_info = locate_reference_pattern(ref_path, search_path)

        if pred_center is not None and status == "SUCCESS":
            pred_x, pred_y = pred_center
            err = math.hypot(pred_x - true_x, pred_y - true_y)
        else:
            best_cand = debug_info.get("final_selected_cand") or debug_info.get("best_candidate")
            if best_cand is not None:
                pred_x = float(best_cand["center_x"])
                pred_y = float(best_cand["center_y"])
                err = math.hypot(pred_x - true_x, pred_y - true_y)
            else:
                pred_x = -1.0
                pred_y = -1.0
                err = 1000.0
            status = "FAILED"

        results.append({
            "image": img_name,
            "style": style,
            "status": status,
            "predicted_x": pred_x,
            "predicted_y": pred_y,
            "true_x": true_x,
            "true_y": true_y,
            "error_px": err,
            "confidence": confidence
        })

        if idx % 20 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)} pairs...")

    # Save complete per-image results to results/hybrid_context_validation.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "style", "status", "predicted_x", "predicted_y", "true_x", "true_y", "error_px", "confidence"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image"],
                "style": r["style"],
                "status": r["status"],
                "predicted_x": f"{r['predicted_x']:.2f}" if r["predicted_x"] >= 0 else "None",
                "predicted_y": f"{r['predicted_y']:.2f}" if r["predicted_y"] >= 0 else "None",
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
    print("   CONTEXT-ENHANCED HYBRID LOCALIZER VALIDATION EVALUATION REPORT")
    print("#" * 70)

    print_metric_block("OVERALL METRICS", overall)
    print_metric_block("DRAM METRICS", dram_metrics)
    print_metric_block("FINFET METRICS", finfet_metrics)

    # Load baseline, structural, frequency, previous hybrid results
    base_m = load_previous_results(os.path.join("results", "baseline_validation.csv"))
    struct_m = load_previous_results(os.path.join("results", "structural_validation.csv"))
    freq_m = load_previous_results(os.path.join("results", "frequency_validation.csv"))
    prev_hybrid_m = load_previous_results(os.path.join("results", "hybrid_validation.csv"))

    print("\n" + "=" * 105)
    print("                    ALGORITHM PERFORMANCE COMPARISON TABLE")
    print("=" * 105)
    header = f"{'Metric':<24} | {'Baseline':<10} | {'Structural':<10} | {'Frequency':<10} | {'Prev Hybrid':<12} | {'Context Hybrid':<14}"
    print(header)
    print("-" * 105)

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
        h_str = format_val(overall, key, fmt, is_pct)
        print(f"{label:<24} | {b_str:<10} | {s_str:<10} | {f_str:<10} | {p_str:<12} | {h_str:<14}")

    print("=" * 105)

    # Detailed Improvement Summary Analysis
    prev_mean = prev_hybrid_m.get("mean", 350.20)
    prev_acc5 = prev_hybrid_m.get("acc_5", 0.0)
    prev_acc10 = prev_hybrid_m.get("acc_10", 0.5)
    prev_acc50 = prev_hybrid_m.get("acc_50", 2.0)
    prev_failed = prev_hybrid_m.get("failed", 68)

    curr_mean = overall["mean"]
    curr_acc5 = overall["acc_5"]
    curr_acc10 = overall["acc_10"]
    curr_acc50 = overall["acc_50"]
    curr_failed = overall["failed"]

    print("\n" + "*" * 70)
    print("        CONTEXT ENHANCEMENT IMPROVEMENT SUMMARY ANALYSIS")
    print("*" * 70)
    print(f"1. Mean Error:          {prev_mean:.2f} px -> {curr_mean:.2f} px (Change: {curr_mean - prev_mean:+.2f} px | Improved: {'YES' if curr_mean < prev_mean else 'NO'})")
    print(f"2. <= 5 px Accuracy:    {prev_acc5:.1f}% -> {curr_acc5:.1f}% (Change: {curr_acc5 - prev_acc5:+.1f}% | Improved: {'YES' if curr_acc5 > prev_acc5 else 'NO'})")
    print(f"3. <= 10 px Accuracy:   {prev_acc10:.1f}% -> {curr_acc10:.1f}% (Change: {curr_acc10 - prev_acc10:+.1f}% | Improved: {'YES' if curr_acc10 > prev_acc10 else 'NO'})")
    print(f"4. <= 50 px Accuracy:   {prev_acc50:.1f}% -> {curr_acc50:.1f}% (Change: {curr_acc50 - prev_acc50:+.1f}% | Improved: {'YES' if curr_acc50 > prev_acc50 else 'NO'})")
    print(f"5. Failed Count:        {prev_failed} -> {curr_failed} (Change: {curr_failed - prev_failed:+d} failures)")
    print("*" * 70)


if __name__ == "__main__":
    evaluate_hybrid_validation()
