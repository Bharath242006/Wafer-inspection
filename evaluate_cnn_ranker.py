"""
evaluate_cnn_ranker.py

Evaluation benchmark script for the Trained Siamese CNN Candidate Ranker on held-out validation samples (images 00161 to 00200).

Reports:
1. Candidate recall before CNN vs candidate ranking accuracy after CNN
2. Accuracy thresholds (<= 5 px, <= 10 px, <= 25 px, <= 50 px, <= 100 px)
3. Mean, Median, P95, and Max pixel error
4. Rank of the closest-to-GT candidate
5. Average inference runtime
6. Detailed per-sample results saved to results/cnn_validation.csv
7. Summary report saved to results/cnn_training_report.md
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
    train_records, val_records = load_validation_records(split_idx=160)

    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    checkpoint_path = os.path.join("checkpoints", "siamese_cnn.pt")

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "cnn_validation.csv")
    out_report = os.path.join("results", "cnn_training_report.md")

    print("=" * 100)
    print("      EVALUATING TRAINED SIAMESE CNN RANKER ON HELD-OUT VALIDATION SET (40 SAMPLES)")
    print("=" * 100)

    results = []
    cnn_runtimes = []
    total_runtimes = []

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
        coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern_final(ref_path, search_path)
        cands = debug_info.get("all_candidates", [])

        # Measure CNN scoring runtime
        t_cnn_0 = time.perf_counter()
        cnn_scores = compute_cnn_similarity_scores(ref_raw, search_raw, cands, checkpoint_path=checkpoint_path)
        t_cnn_1 = time.perf_counter()

        cnn_rt = t_cnn_1 - t_cnn_0
        tot_rt = time.perf_counter() - t0

        cnn_runtimes.append(cnn_rt)
        total_runtimes.append(tot_rt)

        if not cands:
            results.append({
                "image": img_name,
                "style": style,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "cand_recall_dist": 1000.0,
                "best_cand_rank": -1,
                "pred_x": -1.0,
                "pred_y": -1.0,
                "error_px": 1000.0,
                "cnn_score": 0.0,
                "status": "FAILED"
            })
            continue

        # Distance to GT for all candidates in candidate pool
        distances = [math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y) for c in cands]
        min_cand_dist = float(np.min(distances))
        best_gt_cand_idx = int(np.argmin(distances))

        for i, c in enumerate(cands):
            c['cnn_score'] = cnn_scores[i]
            c['combined_score'] = 0.70 * c['final_score'] + 0.30 * c['cnn_score']

        cands_ranked = sorted(cands, key=lambda c: c['combined_score'], reverse=True)
        winner_cand = cands_ranked[0]

        # Find rank of closest-to-GT candidate after CNN ranking
        ranked_distances = [math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y) for c in cands_ranked]
        rank_closest_gt = int(np.argmin(ranked_distances)) + 1

        pred_x = winner_cand['center_x']
        pred_y = winner_cand['center_y']
        pred_err = math.hypot(pred_x - gt_x, pred_y - gt_y)

        results.append({
            "image": img_name,
            "style": style,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "cand_recall_dist": min_cand_dist,
            "best_cand_rank": rank_closest_gt,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_px": pred_err,
            "cnn_score": winner_cand['cnn_score'],
            "status": "SUCCESS" if pred_err <= 50.0 else "FAILED"
        })

    # Save to results/cnn_validation.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "style", "gt_x", "gt_y", "cand_recall_dist", "best_cand_rank",
            "pred_x", "pred_y", "error_px", "cnn_score", "status"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image"],
                "style": r["style"],
                "gt_x": f"{r['gt_x']:.2f}",
                "gt_y": f"{r['gt_y']:.2f}",
                "cand_recall_dist": f"{r['cand_recall_dist']:.2f}",
                "best_cand_rank": r["best_cand_rank"],
                "pred_x": f"{r['pred_x']:.2f}",
                "pred_y": f"{r['pred_y']:.2f}",
                "error_px": f"{r['error_px']:.2f}",
                "cnn_score": f"{r['cnn_score']:.4f}",
                "status": r["status"]
            })

    # Metrics computation
    n = len(results)
    errors = [r["error_px"] for r in results]
    cand_recalls = [r["cand_recall_dist"] for r in results]

    rec_50 = (sum(1 for d in cand_recalls if d <= 50.0) / n) * 100.0
    rec_100 = (sum(1 for d in cand_recalls if d <= 100.0) / n) * 100.0

    acc_5 = (sum(1 for e in errors if e <= 5.0) / n) * 100.0
    acc_10 = (sum(1 for e in errors if e <= 10.0) / n) * 100.0
    acc_25 = (sum(1 for e in errors if e <= 25.0) / n) * 100.0
    acc_50 = (sum(1 for e in errors if e <= 50.0) / n) * 100.0
    acc_100 = (sum(1 for e in errors if e <= 100.0) / n) * 100.0

    mean_err = float(np.mean(errors))
    med_err = float(np.median(errors))
    p95_err = float(np.percentile(errors, 95))
    max_err = float(np.max(errors))

    avg_cnn_time_ms = float(np.mean(cnn_runtimes)) * 1000.0
    avg_tot_time_ms = float(np.mean(total_runtimes)) * 1000.0

    # Write Markdown Report
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# DriftSense-X Trained Siamese CNN Evaluation Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"Evaluated on {n} held-out validation samples (`00161.png` - `00200.png`) ")
        f.write("using Contrastive Loss trained Siamese Neural Network weights (`checkpoints/siamese_cnn.pt`).\n\n")

        f.write("## Candidate Recall vs CNN Ranking Accuracy\n\n")
        f.write(f"- **Candidate Recall (before CNN, <= 50 px)**: {rec_50:.1f}%\n")
        f.write(f"- **Candidate Recall (before CNN, <= 100 px)**: {rec_100:.1f}%\n")
        f.write(f"- **CNN Ranking Accuracy (<= 50 px)**: {acc_50:.1f}%\n")
        f.write(f"- **CNN Ranking Accuracy (<= 100 px)**: {acc_100:.1f}%\n\n")

        f.write("## Held-Out Validation Error Statistics\n\n")
        f.write(f"- **Mean Error**: {mean_err:.2f} px\n")
        f.write(f"- **Median Error**: {med_err:.2f} px\n")
        f.write(f"- **P95 Error**: {p95_err:.2f} px\n")
        f.write(f"- **Maximum Error**: {max_err:.2f} px\n\n")

        f.write("| Accuracy Threshold | Percentage (%) | Count |\n")
        f.write("|---|---|---|\n")
        f.write(f"| $\\le 5\\text{{ px}}$ | {acc_5:.1f}% | {sum(1 for e in errors if e <= 5.0)}/{n} |\n")
        f.write(f"| $\\le 10\\text{{ px}}$ | {acc_10:.1f}% | {sum(1 for e in errors if e <= 10.0)}/{n} |\n")
        f.write(f"| $\\le 25\\text{{ px}}$ | {acc_25:.1f}% | {sum(1 for e in errors if e <= 25.0)}/{n} |\n")
        f.write(f"| $\\le 50\\text{{ px}}$ | {acc_50:.1f}% | {sum(1 for e in errors if e <= 50.0)}/{n} |\n")
        f.write(f"| $\\le 100\\text{{ px}}$ | {acc_100:.1f}% | {sum(1 for e in errors if e <= 100.0)}/{n} |\n\n")

        f.write("## Inference Runtime Benchmark\n\n")
        f.write(f"- **Average CNN Scoring Time per Image**: {avg_cnn_time_ms:.2f} ms ({avg_cnn_time_ms/1000.0:.4f} s)\n")
        f.write(f"- **Average Total Pipeline Time per Image**: {avg_tot_time_ms:.2f} ms ({avg_tot_time_ms/1000.0:.4f} s)\n")

    print("\n" + "=" * 100)
    print("                HELD-OUT VALIDATION EVALUATION SUMMARY (40 SAMPLES)")
    print("=" * 100)
    print(f"Candidate Recall <= 50 px (Before CNN):   {rec_50:.1f}%")
    print(f"Candidate Recall <= 100 px (Before CNN):  {rec_100:.1f}%")
    print(f"Accuracy <= 5 px:                         {acc_5:.1f}%")
    print(f"Accuracy <= 10 px:                        {acc_10:.1f}%")
    print(f"Accuracy <= 25 px:                        {acc_25:.1f}%")
    print(f"Accuracy <= 50 px:                        {acc_50:.1f}%")
    print(f"Accuracy <= 100 px:                       {acc_100:.1f}%")
    print("-" * 100)
    print(f"Mean Pixel Error:                         {mean_err:.2f} px")
    print(f"Median Pixel Error:                       {med_err:.2f} px")
    print(f"P95 Pixel Error:                          {p95_err:.2f} px")
    print(f"Max Pixel Error:                          {max_err:.2f} px")
    print(f"Average CNN Inference Time:               {avg_cnn_time_ms:.2f} ms ({avg_cnn_time_ms/1000.0:.4f} s)")
    print("=" * 100)
    print(f"Detailed CSV saved to:                    {out_csv}")
    print(f"Detailed Report saved to:                 {out_report}")


if __name__ == "__main__":
    main()
