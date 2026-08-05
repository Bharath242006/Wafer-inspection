"""
scratch/oracle_candidate_recall.py

Diagnostic Oracle-Ranking Experiment for DriftSense-X.

Evaluates the exact candidate generation pool across all 200 validation images
to determine whether true target candidates are generated (Recall Bottleneck)
or generated but mis-ranked (Ranking Bottleneck).

Saves per-sample results to results/oracle_candidate_recall.csv
and comprehensive diagnostic report to results/oracle_candidate_recall_report.md
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


def load_validation_labels() -> list:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def run_oracle_candidate_experiment():
    labels = load_validation_labels()

    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    out_csv = os.path.join("results", "oracle_candidate_recall.csv")
    out_report = os.path.join("results", "oracle_candidate_recall_report.md")
    os.makedirs("results", exist_ok=True)

    results = []

    print("=" * 100)
    print("      RUNNING DIAGNOSTIC ORACLE CANDIDATE RECALL EXPERIMENT (200 SAMPLES)")
    print("=" * 100)

    for idx, item in enumerate(labels, start=1):
        img_name = item["image"]
        gt_x = float(item["x"])
        gt_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern_final(ref_path, search_path)
        cands = debug_info.get("all_candidates", [])
        num_cands = len(cands)

        if fine_center is not None and status == "SUCCESS":
            pred_x, pred_y = fine_center
            pred_err = math.hypot(pred_x - gt_x, pred_y - gt_y)
        else:
            pred_x, pred_y = coarse_center if coarse_center else (-1.0, -1.0)
            pred_err = math.hypot(pred_x - gt_x, pred_y - gt_y) if pred_x >= 0 else 1000.0

        if cands:
            distances = [math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y) for c in cands]
            min_dist_gt = float(np.min(distances))
            best_cand_idx = int(np.argmin(distances))
            best_cand = cands[best_cand_idx]
            best_cand_coord = (best_cand['center_x'], best_cand['center_y'])
        else:
            min_dist_gt = 1000.0
            best_cand_coord = (-1.0, -1.0)

        hit_5 = min_dist_gt <= 5.0
        hit_10 = min_dist_gt <= 10.0
        hit_25 = min_dist_gt <= 25.0
        hit_50 = min_dist_gt <= 50.0
        hit_75 = min_dist_gt <= 75.0
        hit_100 = min_dist_gt <= 100.0

        # Classification of bottleneck:
        if not hit_100:
            category = "CANDIDATE_GENERATION_FAILURE"
        elif hit_50 and pred_err > 100.0:
            category = "CANDIDATE_RANKING_FAILURE"
        elif hit_10 and pred_err > 10.0:
            category = "FINE_SEARCH_FAILURE"
        else:
            category = "SUCCESS"

        results.append({
            "image": img_name,
            "style": style,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "num_candidates": num_cands,
            "min_dist_gt": min_dist_gt,
            "best_cand_x": best_cand_coord[0],
            "best_cand_y": best_cand_coord[1],
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_err": pred_err,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "hit_25": hit_25,
            "hit_50": hit_50,
            "hit_75": hit_75,
            "hit_100": hit_100,
            "category": category
        })

        if idx % 50 == 0:
            print(f"Processed {idx}/200 samples...")

    # Write CSV output
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "style", "gt_x", "gt_y", "num_candidates", "min_dist_gt",
            "best_cand_x", "best_cand_y", "pred_x", "pred_y", "pred_err",
            "hit_5", "hit_10", "hit_25", "hit_50", "hit_75", "hit_100", "category"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image"],
                "style": r["style"],
                "gt_x": f"{r['gt_x']:.2f}",
                "gt_y": f"{r['gt_y']:.2f}",
                "num_candidates": r["num_candidates"],
                "min_dist_gt": f"{r['min_dist_gt']:.2f}",
                "best_cand_x": f"{r['best_cand_x']:.2f}",
                "best_cand_y": f"{r['best_cand_y']:.2f}",
                "pred_x": f"{r['pred_x']:.2f}",
                "pred_y": f"{r['pred_y']:.2f}",
                "pred_err": f"{r['pred_err']:.2f}",
                "hit_5": r["hit_5"],
                "hit_10": r["hit_10"],
                "hit_25": r["hit_25"],
                "hit_50": r["hit_50"],
                "hit_75": r["hit_75"],
                "hit_100": r["hit_100"],
                "category": r["category"]
            })

    # Metrics computation
    total = len(results)
    avg_cands = float(np.mean([r["num_candidates"] for r in results]))
    med_cands = float(np.median([r["num_candidates"] for r in results]))
    worst_cand_dist = float(np.max([r["min_dist_gt"] for r in results]))

    rec_5 = (sum(1 for r in results if r["hit_5"]) / total) * 100.0
    rec_10 = (sum(1 for r in results if r["hit_10"]) / total) * 100.0
    rec_25 = (sum(1 for r in results if r["hit_25"]) / total) * 100.0
    rec_50 = (sum(1 for r in results if r["hit_50"]) / total) * 100.0
    rec_75 = (sum(1 for r in results if r["hit_75"]) / total) * 100.0
    rec_100 = (sum(1 for r in results if r["hit_100"]) / total) * 100.0

    dram_res = [r for r in results if r["style"] == "DRAM"]
    finfet_res = [r for r in results if r["style"] == "FinFET"]

    dram_rec_50 = (sum(1 for r in dram_res if r["hit_50"]) / len(dram_res)) * 100.0 if dram_res else 0.0
    dram_rec_100 = (sum(1 for r in dram_res if r["hit_100"]) / len(dram_res)) * 100.0 if dram_res else 0.0

    finfet_rec_50 = (sum(1 for r in finfet_res if r["hit_50"]) / len(finfet_res)) * 100.0 if finfet_res else 0.0
    finfet_rec_100 = (sum(1 for r in finfet_res if r["hit_100"]) / len(finfet_res)) * 100.0 if finfet_res else 0.0

    gen_failures = [r for r in results if r["category"] == "CANDIDATE_GENERATION_FAILURE"]
    rank_failures = [r for r in results if r["category"] == "CANDIDATE_RANKING_FAILURE"]
    fine_failures = [r for r in results if r["category"] == "FINE_SEARCH_FAILURE"]

    # Write Markdown Report
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# DriftSense-X Diagnostic Oracle-Ranking & Candidate Recall Report\n\n")
        f.write("## Executive Summary\n")
        f.write("This diagnostic study evaluates the candidate generation pool across all 200 validation samples ")
        f.write("under an **Oracle Ranking** assumption (i.e. if an oracle selector always chose the candidate closest to Ground Truth).\n\n")

        f.write("## Candidate-Generation Recall Statistics\n\n")
        f.write(f"- **Total Validation Samples**: {total}\n")
        f.write(f"- **Average Candidates per Pool**: {avg_cands:.1f}\n")
        f.write(f"- **Median Candidates per Pool**: {med_cands:.1f}\n")
        f.write(f"- **Worst-Case Candidate Distance to GT**: {worst_cand_dist:.2f} px\n\n")

        f.write("| Distance Tolerance | Oracle Recall (%) | Count |\n")
        f.write("|---|---|---|\n")
        f.write(f"| $\\le 5\\text{{ px}}$ | {rec_5:.1f}% | {sum(1 for r in results if r['hit_5'])}/{total} |\n")
        f.write(f"| $\\le 10\\text{{ px}}$ | {rec_10:.1f}% | {sum(1 for r in results if r['hit_10'])}/{total} |\n")
        f.write(f"| $\\le 25\\text{{ px}}$ | {rec_25:.1f}% | {sum(1 for r in results if r['hit_25'])}/{total} |\n")
        f.write(f"| $\\le 50\\text{{ px}}$ | {rec_50:.1f}% | {sum(1 for r in results if r['hit_50'])}/{total} |\n")
        f.write(f"| $\\le 75\\text{{ px}}$ | {rec_75:.1f}% | {sum(1 for r in results if r['hit_75'])}/{total} |\n")
        f.write(f"| $\\le 100\\text{{ px}}$ | {rec_100:.1f}% | {sum(1 for r in results if r['hit_100'])}/{total} |\n\n")

        f.write("## Architecture Recall Breakdown (DRAM vs FinFET)\n\n")
        f.write("| Architecture Style | Sample Count | Recall $\\le 50\\text{ px}$ | Recall $\\le 100\\text{ px}$ |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **DRAM** | {len(dram_res)} | {dram_rec_50:.1f}% | {dram_rec_100:.1f}%\n")
        f.write(f"| **FinFET** | {len(finfet_res)} | {finfet_rec_50:.1f}% | {finfet_rec_100:.1f}%\n\n")

        f.write("## Error Taxonomy Breakdown\n\n")
        f.write(f"- **A. Candidate Generation Failures** (`min_dist > 100 px`): **{len(gen_failures)} samples ({len(gen_failures)/total*100:.1f}%)**\n")
        f.write(f"- **B. Candidate Ranking Failures** (`min_dist <= 50 px`, but prediction error `> 100 px`): **{len(rank_failures)} samples ({len(rank_failures)/total*100:.1f}%)**\n")
        f.write(f"- **C. Fine Search Failures** (`min_dist <= 10 px`, but prediction error `> 10 px`): **{len(fine_failures)} samples ({len(fine_failures)/total*100:.1f}%)**\n\n")

        f.write("### A. Samples with Candidate Generation Failure (No Candidate within 100 px)\n\n")
        if gen_failures:
            f.write("| Sample Image | Architecture | Nearest Candidate Dist (px) |\n")
            f.write("|---|---|---|\n")
            for r in gen_failures:
                f.write(f"| `{r['image']}` | {r['style']} | {r['min_dist_gt']:.2f} px |\n")
        else:
            f.write("*(None — Candidate generation covers 100% of samples within 100 px)*\n")

        f.write("\n### B. Candidate Ranking Failures (GT Candidate Exists, but False Periodic Alias Winner Selected)\n\n")
        if rank_failures:
            f.write("| Sample Image | Architecture | Nearest GT Candidate Dist (px) | Final Prediction Error (px) |\n")
            f.write("|---|---|---|---|\n")
            for r in rank_failures[:20]:
                f.write(f"| `{r['image']}` | {r['style']} | {r['min_dist_gt']:.2f} px | {r['pred_err']:.2f} px |\n")
            if len(rank_failures) > 20:
                f.write(f"*... and {len(rank_failures) - 20} more samples.*\n")

    print("\n" + "=" * 100)
    print("                     ORACLE CANDIDATE RECALL SUMMARY")
    print("=" * 100)
    print(f"Total Samples:                       {total}")
    print(f"Average Candidates per Pool:         {avg_cands:.1f}")
    print(f"Candidate Recall <= 5 px:           {rec_5:.1f}% ({sum(1 for r in results if r['hit_5'])}/{total})")
    print(f"Candidate Recall <= 10 px:          {rec_10:.1f}% ({sum(1 for r in results if r['hit_10'])}/{total})")
    print(f"Candidate Recall <= 25 px:          {rec_25:.1f}% ({sum(1 for r in results if r['hit_25'])}/{total})")
    print(f"Candidate Recall <= 50 px:          {rec_50:.1f}% ({sum(1 for r in results if r['hit_50'])}/{total})")
    print(f"Candidate Recall <= 75 px:          {rec_75:.1f}% ({sum(1 for r in results if r['hit_75'])}/{total})")
    print(f"Candidate Recall <= 100 px:         {rec_100:.1f}% ({sum(1 for r in results if r['hit_100'])}/{total})")
    print("-" * 100)
    print(f"A. Candidate Generation Failures:    {len(gen_failures)} ({len(gen_failures)/total*100:.1f}%)")
    print(f"B. Candidate Ranking Failures:       {len(rank_failures)} ({len(rank_failures)/total*100:.1f}%)")
    print(f"C. Fine Search Failures:             {len(fine_failures)} ({len(fine_failures)/total*100:.1f}%)")
    print("=" * 100)
    print(f"Detailed CSV saved to:               {out_csv}")
    print(f"Detailed Report saved to:            {out_report}")


if __name__ == "__main__":
    run_oracle_candidate_experiment()
