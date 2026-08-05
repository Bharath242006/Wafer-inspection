"""
evaluate_context_ranker.py

Evaluation benchmark script for the Trained Multi-Branch Context-Aware Candidate Ranker
on held-out validation samples (images 00161 to 00200).

Compares directly against:
A. Existing Handcrafted Top-500 Ranker
B. Existing Siamese CNN Ranker
C. New Multi-Branch Context-Aware Ranker
D. Oracle Top-500 Upper Bound

Saves per-sample results to results/context_ranker_validation.csv
Saves summary report to results/context_ranker_report.md
"""

import csv
import math
import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.abspath("."))
from scratch.improve_candidate_recall import generate_candidate_pool_multi
from scratch.test_ranking_top500 import rank_top500_candidates
from localization.context_ranker import compute_context_ranker_scores
from localization.cnn_candidate_ranker import compute_cnn_similarity_scores


def load_validation_records(split_idx: int = 160) -> tuple:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    all_records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_records.append(row)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    return train_records, val_records


def main():
    _, val_records = load_validation_records(split_idx=160)

    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    ctx_checkpoint = os.path.join("checkpoints", "context_ranker.pt")
    siamese_checkpoint = os.path.join("checkpoints", "siamese_cnn.pt")

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "context_ranker_validation.csv")
    out_report = os.path.join("results", "context_ranker_report.md")

    print("=" * 110)
    print("   EVALUATING MULTI-BRANCH CONTEXT RANKER ON HELD-OUT VALIDATION SET (40 SAMPLES)")
    print("=" * 110)

    results = []

    errs_oracle = []
    errs_hc = []
    errs_siamese = []
    errs_context = []

    runtimes_context = []

    for idx, item in enumerate(val_records, start=161):
        img_name = item["image"]
        gt_x = float(item["x"])
        gt_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()

        # 1. Generate Top-500 candidate pool
        cands_500 = generate_candidate_pool_multi(ref_raw, search_raw, max_pool_size=500)

        if not cands_500:
            errs_oracle.append(1000.0)
            errs_hc.append(1000.0)
            errs_siamese.append(1000.0)
            errs_context.append(1000.0)
            continue

        # D. Oracle Candidate (Closest candidate to GT in Top-500 pool)
        oracle_dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_500]
        o_err = float(np.min(oracle_dists))
        errs_oracle.append(o_err)

        # A. Handcrafted Top-500 Ranker
        ranked_hc, t1_hc, t2_hc, margin_hc = rank_top500_candidates(ref_raw, search_raw, cands_500)
        win_hc = ranked_hc[0]
        hc_err = math.hypot(win_hc['center_x'] - gt_x, win_hc['center_y'] - gt_y)
        errs_hc.append(hc_err)

        # B. Siamese CNN Ranker
        siamese_scores = compute_cnn_similarity_scores(ref_raw, search_raw, cands_500, checkpoint_path=siamese_checkpoint)
        for i, c in enumerate(cands_500):
            c['siamese_score'] = siamese_scores[i]
            c['siamese_comb'] = 0.70 * c['final_score'] + 0.30 * c['siamese_score']
        cands_siamese = sorted(cands_500, key=lambda c: c['siamese_comb'], reverse=True)
        win_siamese = cands_siamese[0]
        siamese_err = math.hypot(win_siamese['cx'] - gt_x, win_siamese['cy'] - gt_y)
        errs_siamese.append(siamese_err)

        # C. New Multi-Branch Context-Aware Ranker
        t_ctx_0 = time.perf_counter()
        ctx_scores = compute_context_ranker_scores(ref_raw, search_raw, cands_500, checkpoint_path=ctx_checkpoint)
        t_ctx_1 = time.perf_counter()
        runtimes_context.append(t_ctx_1 - t_ctx_0)

        for i, c in enumerate(cands_500):
            c['ctx_score'] = ctx_scores[i]
            c['ctx_comb'] = 0.60 * c['ctx_score'] + 0.40 * c['final_score']

        cands_ctx = sorted(cands_500, key=lambda c: c['ctx_comb'], reverse=True)
        win_ctx = cands_ctx[0]
        ctx_err = math.hypot(win_ctx['cx'] - gt_x, win_ctx['cy'] - gt_y)
        errs_context.append(ctx_err)

        # Rank of closest GT candidate under Context Ranker
        ctx_dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_ctx]
        rank_closest_gt = int(np.argmin(ctx_dists)) + 1

        top1_ctx = cands_ctx[0]['ctx_comb']
        top2_ctx = cands_ctx[1]['ctx_comb'] if len(cands_ctx) > 1 else 0.0
        margin_ctx = top1_ctx - top2_ctx

        results.append({
            "image_id": img_name,
            "style": style,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "oracle_error": o_err,
            "hc_error": hc_err,
            "siamese_error": siamese_err,
            "context_error": ctx_err,
            "predicted_x": win_ctx['cx'],
            "predicted_y": win_ctx['cy'],
            "rank_of_closest_gt": rank_closest_gt,
            "top1_score": top1_ctx,
            "top2_score": top2_ctx,
            "score_margin": margin_ctx,
            "status": "SUCCESS" if ctx_err <= 50.0 else "FAILED"
        })

    # Save CSV results
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "style", "gt_x", "gt_y", "oracle_error", "hc_error", "siamese_error",
            "context_error", "predicted_x", "predicted_y", "rank_of_closest_gt",
            "top1_score", "top2_score", "score_margin", "status"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image_id": r["image_id"],
                "style": r["style"],
                "gt_x": f"{r['gt_x']:.2f}",
                "gt_y": f"{r['gt_y']:.2f}",
                "oracle_error": f"{r['oracle_error']:.2f}",
                "hc_error": f"{r['hc_error']:.2f}",
                "siamese_error": f"{r['siamese_error']:.2f}",
                "context_error": f"{r['context_error']:.2f}",
                "predicted_x": f"{r['predicted_x']:.2f}",
                "predicted_y": f"{r['predicted_y']:.2f}",
                "rank_of_closest_gt": r["rank_of_closest_gt"],
                "top1_score": f"{r['top1_score']:.4f}",
                "top2_score": f"{r['top2_score']:.4f}",
                "score_margin": f"{r['score_margin']:.4f}",
                "status": r["status"]
            })

    def calc_stats(err_list):
        n = len(err_list)
        return {
            "mean": float(np.mean(err_list)),
            "median": float(np.median(err_list)),
            "p95": float(np.percentile(err_list, 95)),
            "max": float(np.max(err_list)),
            "acc_5": (sum(1 for e in err_list if e <= 5.0) / n) * 100.0,
            "acc_10": (sum(1 for e in err_list if e <= 10.0) / n) * 100.0,
            "acc_25": (sum(1 for e in err_list if e <= 25.0) / n) * 100.0,
            "acc_50": (sum(1 for e in err_list if e <= 50.0) / n) * 100.0,
            "acc_75": (sum(1 for e in err_list if e <= 75.0) / n) * 100.0,
            "acc_100": (sum(1 for e in err_list if e <= 100.0) / n) * 100.0
        }

    st_o = calc_stats(errs_oracle)
    st_hc = calc_stats(errs_hc)
    st_siamese = calc_stats(errs_siamese)
    st_ctx = calc_stats(errs_context)

    avg_ctx_rt_ms = float(np.mean(runtimes_context)) * 1000.0

    # Write Markdown Report
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# DriftSense-X Multi-Branch Context-Aware Candidate Ranker Report\n\n")
        f.write("## Executive Summary\n")
        f.write("Evaluates the Multi-Branch Context-Aware Candidate Ranker on 40 held-out validation samples ")
        f.write("(`00161.png` - `00200.png`) and compares directly against Handcrafted, Siamese CNN, and Oracle Top-500 bounds.\n\n")

        f.write("## 4-Way Direct Comparative Performance Matrix\n\n")
        f.write("| Model / Approach | <= 5 px | <= 10 px | <= 25 px | <= 50 px | <= 100 px | Mean Error (px) | Median Error (px) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        f.write(f"| **D. Oracle Top-500 Upper Bound** | {st_o['acc_5']:.1f}% | {st_o['acc_10']:.1f}% | {st_o['acc_25']:.1f}% | {st_o['acc_50']:.1f}% | {st_o['acc_100']:.1f}% | {st_o['mean']:.2f} | {st_o['median']:.2f} |\n")
        f.write(f"| **A. Handcrafted Top-500 Ranker** | {st_hc['acc_5']:.1f}% | {st_hc['acc_10']:.1f}% | {st_hc['acc_25']:.1f}% | {st_hc['acc_50']:.1f}% | {st_hc['acc_100']:.1f}% | {st_hc['mean']:.2f} | {st_hc['median']:.2f} |\n")
        f.write(f"| **B. Siamese CNN Ranker** | {st_siamese['acc_5']:.1f}% | {st_siamese['acc_10']:.1f}% | {st_siamese['acc_25']:.1f}% | {st_siamese['acc_50']:.1f}% | {st_siamese['acc_100']:.1f}% | {st_siamese['mean']:.2f} | {st_siamese['median']:.2f} |\n")
        f.write(f"| **C. Context-Aware Ranker** | {st_ctx['acc_5']:.1f}% | {st_ctx['acc_10']:.1f}% | {st_ctx['acc_25']:.1f}% | {st_ctx['acc_50']:.1f}% | {st_ctx['acc_100']:.1f}% | {st_ctx['mean']:.2f} | {st_ctx['median']:.2f} |\n\n")

        f.write("## Inference Runtime Benchmark\n\n")
        f.write(f"- **Average Context Scoring Runtime per Image**: {avg_ctx_rt_ms:.2f} ms ({avg_ctx_rt_ms/1000.0:.4f} s)\n\n")

        f.write("## Decision & Verdict\n\n")
        improved = (st_ctx['acc_100'] > st_hc['acc_100']) or (st_ctx['mean'] < st_hc['mean'])
        if improved:
            f.write("**VERDICT**: The Context-Aware Candidate Ranker demonstrates empirical improvement on held-out validation data.")
        else:
            f.write("**VERDICT**: The Context-Aware Candidate Ranker does NOT demonstrate sufficient held-out validation improvement over handcrafted ranking. Per instructions, `final_localizer.py` will NOT be modified.")

    print("\n" + "=" * 110)
    print("             HELD-OUT 4-WAY DIRECT COMPARATIVE EVALUATION SUMMARY (40 SAMPLES)")
    print("=" * 110)
    print(f"{'Approach':<32} | {'<=50 px':<8} | {'<=100 px':<8} | {'Mean Error':<12} | {'Median Error':<12}")
    print("-" * 110)
    print(f"{'D. Oracle Top-500 Upper Bound':<32} | {st_o['acc_50']:<7.1f}% | {st_o['acc_100']:<7.1f}% | {st_o['mean']:<12.2f} | {st_o['median']:<12.2f}")
    print(f"{'A. Handcrafted Top-500 Ranker':<32} | {st_hc['acc_50']:<7.1f}% | {st_hc['acc_100']:<7.1f}% | {st_hc['mean']:<12.2f} | {st_hc['median']:<12.2f}")
    print(f"{'B. Siamese CNN Ranker':<32} | {st_siamese['acc_50']:<7.1f}% | {st_siamese['acc_100']:<7.1f}% | {st_siamese['mean']:<12.2f} | {st_siamese['median']:<12.2f}")
    print(f"{'C. Context-Aware Ranker':<32} | {st_ctx['acc_50']:<7.1f}% | {st_ctx['acc_100']:<7.1f}% | {st_ctx['mean']:<12.2f} | {st_ctx['median']:<12.2f}")
    print("=" * 110)
    print(f"Average Context Inference Time: {avg_ctx_rt_ms:.2f} ms ({avg_ctx_rt_ms/1000.0:.4f} s)")
    print(f"CSV report saved to:           {out_csv}")
    print(f"Markdown report saved:         {out_report}")


if __name__ == "__main__":
    main()
