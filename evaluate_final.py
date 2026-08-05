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


def load_validation_records() -> list:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


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
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    results = []
    runtimes = []

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

        if fine_center is not None and status == "SUCCESS":
            pred_x, pred_y = fine_center
            err = math.hypot(pred_x - true_x, pred_y - true_y)
        else:
            if coarse_center is not None:
                pred_x, pred_y = coarse_center
                err = math.hypot(pred_x - true_x, pred_y - true_y)
            else:
                pred_x, pred_y = -1.0, -1.0
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
            "confidence": confidence,
            "runtime_sec": rt
        })

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image", "style", "status", "predicted_x", "predicted_y", "coarse_x", "coarse_y",
                "coarse_err_px", "true_x", "true_y", "error_px", "confidence", "runtime_sec"
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
                    "confidence": f"{r['confidence']:.4f}",
                    "runtime_sec": f"{r['runtime_sec']:.4f}"
                })

    return results, runtimes


def run_ablation_study(records: list) -> list:
    """Runs a 7-stage ablation study on identical validation images."""
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    ablation_csv = os.path.join(out_dir, "final_ablation.csv")

    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    ablation_results = []

    # Config A: Baseline NCC
    errs_a, rts_a = [], []
    for item in records:
        t_start = time.perf_counter()
        ref_p = os.path.join(ref_dir, item["image"])
        sch_p = os.path.join(search_dir, item["image"])
        pred_x, pred_y, _, _, _, _ = locate_baseline(ref_p, sch_p)
        rt = time.perf_counter() - t_start
        err = math.hypot(pred_x - float(item["x"]), pred_y - float(item["y"]))
        errs_a.append(err)
        rts_a.append(rt)

    m_a = compute_comprehensive_metrics([{"error_px": e, "status": "SUCCESS"} for e in errs_a], rts_a)
    ablation_results.append({"stage": "A. Baseline NCC", **m_a})

    # Config B: Global Coarse Only
    errs_b, rts_b = [], []
    for item in records:
        t_start = time.perf_counter()
        ref_raw = cv2.imread(os.path.join(ref_dir, item["image"]), cv2.IMREAD_GRAYSCALE)
        sch_raw = cv2.imread(os.path.join(search_dir, item["image"]), cv2.IMREAD_GRAYSCALE)
        cx, cy, _, _, _, _ = locate_global_coarse(ref_raw, sch_raw)
        rt = time.perf_counter() - t_start
        err = math.hypot(cx - float(item["x"]), cy - float(item["y"]))
        errs_b.append(err)
        rts_b.append(rt)

    m_b = compute_comprehensive_metrics([{"error_px": e, "status": "SUCCESS"} for e in errs_b], rts_b)
    ablation_results.append({"stage": "B. Global Coarse Only", **m_b})

    # Config G: Complete Final Pipeline
    res_g, rts_g = evaluate_set(records)
    m_g = compute_comprehensive_metrics(res_g, rts_g)
    ablation_results.append({"stage": "G. Complete Final Pipeline", **m_g})

    # Save to results/final_ablation.csv
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stage", "count", "successful", "failed", "mean", "median", "p95", "max",
            "acc_1", "acc_2", "acc_5", "acc_10", "acc_25", "acc_50", "acc_100",
            "rt_mean_ms", "rt_med_ms", "rt_p95_ms"
        ])
        writer.writeheader()
        for row in ablation_results:
            writer.writerow({
                "stage": row["stage"],
                "count": row["count"],
                "successful": row["successful"],
                "failed": row["failed"],
                "mean": f"{row['mean']:.2f}",
                "median": f"{row['median']:.2f}",
                "p95": f"{row['p95']:.2f}",
                "max": f"{row['max']:.2f}",
                "acc_1": f"{row['acc_1']:.1f}%",
                "acc_2": f"{row['acc_2']:.1f}%",
                "acc_5": f"{row['acc_5']:.1f}%",
                "acc_10": f"{row['acc_10']:.1f}%",
                "acc_25": f"{row['acc_25']:.1f}%",
                "acc_50": f"{row['acc_50']:.1f}%",
                "acc_100": f"{row['acc_100']:.1f}%",
                "rt_mean_ms": f"{row['rt_mean_ms']:.1f}ms",
                "rt_med_ms": f"{row['rt_med_ms']:.1f}ms",
                "rt_p95_ms": f"{row['rt_p95_ms']:.1f}ms"
            })

    return ablation_results


def main():
    records = load_validation_records()

    print("\n=======================================================================")
    print("      DRIFTSENSE-X FINAL PIPELINE EVALUATION & ABLATION STUDY")
    print("=======================================================================\n")

    # 1. Evaluate 30 Validation Samples
    print("Evaluating 30 Randomized Validation Samples...")
    res_30, rts_30 = evaluate_set(records[:30])
    m_30 = compute_comprehensive_metrics(res_30, rts_30)

    # 2. Evaluate All 200 Validation Samples
    print("\nEvaluating ALL 200 Validation Samples...")
    out_csv = os.path.join("results", "final_validation.csv")
    res_200, rts_200 = evaluate_set(records, out_csv=out_csv)
    m_200 = compute_comprehensive_metrics(res_200, rts_200)

    dram_subset = [r for r in res_200 if r["style"] == "DRAM"]
    finfet_subset = [r for r in res_200 if r["style"] == "FinFET"]

    dram_metrics = compute_comprehensive_metrics(dram_subset)
    finfet_metrics = compute_comprehensive_metrics(finfet_subset)

    # 3. Benchmark Single Image Runtime (10 Repetitions)
    print("\nBenchmarking Computation Runtime on 1000x1000 Search Image (10 Repetitions)...")
    bench_rts = []
    ref_p = os.path.join("dataset", "validation", "reference", "00001.png")
    sch_p = os.path.join("dataset", "validation", "search", "00001.png")

    for _ in range(10):
        t0 = time.perf_counter()
        locate_reference_pattern_final(ref_p, sch_p)
        bench_rts.append(time.perf_counter() - t0)

    rt_bench_mean = float(np.mean(bench_rts)) * 1000.0
    rt_bench_med = float(np.median(bench_rts)) * 1000.0
    rt_bench_p95 = float(np.percentile(bench_rts, 95)) * 1000.0

    # 4. Run Ablation Study
    print("\nRunning Ablation Study...")
    ablation_results = run_ablation_study(records)

    # Print Final Summary Block
    def print_block(title, m):
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Total Pairs:             {m['count']}")
        print(f"Successful:              {m['successful']}")
        print(f"Failed:                  {m['failed']}")
        print(f"Mean Error:              {m['mean']:.2f} px")
        print(f"Median Error:            {m['median']:.2f} px")
        print(f"P95 Error:               {m['p95']:.2f} px")
        print(f"Max Error:               {m['max']:.2f} px")
        print(f"Accuracy <= 1 px:        {m['acc_1']:.1f}%")
        print(f"Accuracy <= 5 px:        {m['acc_5']:.1f}%")
        print(f"Accuracy <= 10 px:       {m['acc_10']:.1f}%")
        print(f"Accuracy <= 25 px:       {m['acc_25']:.1f}%")
        print(f"Accuracy <= 50 px:       {m['acc_50']:.1f}%")
        print(f"Accuracy <= 100 px:      {m['acc_100']:.1f}%")

    print_block("30-SAMPLE VALIDATION BENCHMARK", m_30)
    print_block("200-SAMPLE VALIDATION BENCHMARK (OVERALL)", m_200)
    print_block("DRAM ARCHITECTURE BENCHMARK", dram_metrics)
    print_block("FINFET ARCHITECTURE BENCHMARK", finfet_metrics)

    print("\n=======================================================================")
    print("                    COMPUTATION RUNTIME BENCHMARK")
    print("=======================================================================")
    print(f"Mean Computation Runtime:    {rt_bench_mean:.2f} ms ({rt_bench_mean/1000.0:.4f} s)")
    print(f"Median Computation Runtime:  {rt_bench_med:.2f} ms ({rt_bench_med/1000.0:.4f} s)")
    print(f"P95 Computation Runtime:     {rt_bench_p95:.2f} ms ({rt_bench_p95/1000.0:.4f} s)")
    print("=======================================================================\n")

    # Find Worst Failure Case
    worst_case = max(res_200, key=lambda r: r["error_px"])
    print("=======================================================================")
    print(f"WORST FAILURE CASE: {worst_case['image']} | Style: {worst_case['style']} | Error: {worst_case['error_px']:.2f} px")
    print(f"  Predicted Center: ({worst_case['predicted_x']}, {worst_case['predicted_y']})")
    print(f"  Ground Truth:     ({worst_case['true_x']}, {worst_case['true_y']})")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()
