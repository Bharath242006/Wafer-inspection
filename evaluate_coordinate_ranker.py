"""
evaluate_coordinate_ranker.py

Evaluation benchmark script for Trained Coordinate-Aware Candidate Ranker
on held-out validation samples (images 00161.png to 00200.png).

Compares directly across 7 approaches:
1. Oracle Top-500 Upper Bound
2. Handcrafted Top-500 Ranker
3. Siamese CNN Ranker
4. Context CNN Ranker
5. Global/Lattice-Aware Ranker
6. Global Landmark Localizer
7. New Coordinate-Aware Candidate Ranker

Saves per-sample results to results/coordinate_ranker_validation.csv
Saves summary report to results/coordinate_ranker_report.md
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
from localization.global_landmark_localizer import locate_global_landmark
from localization.global_lattice_ranker import compute_global_lattice_scores
from localization.context_ranker import compute_context_ranker_scores
from localization.cnn_candidate_ranker import compute_cnn_similarity_scores
from localization.coordinate_aware_ranker import compute_coordinate_aware_scores


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

    coord_checkpoint = os.path.join("checkpoints", "coordinate_aware_ranker.pt")
    lat_checkpoint = os.path.join("checkpoints", "global_lattice_ranker.pt")
    ctx_checkpoint = os.path.join("checkpoints", "context_ranker.pt")
    siamese_checkpoint = os.path.join("checkpoints", "siamese_cnn.pt")

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "coordinate_ranker_validation.csv")
    out_report = os.path.join("results", "coordinate_ranker_report.md")

    print("=" * 110)
    print("   EVALUATING COORDINATE-AWARE RANKER ON HELD-OUT VALIDATION SET (40 SAMPLES)")
    print("=" * 110)

    results = []

    errs_oracle = []
    errs_hc = []
    errs_siamese = []
    errs_context = []
    errs_lattice = []
    errs_landmark = []
    errs_coord = []

    runtimes_coord = []

    for idx, item in enumerate(val_records, start=161):
        img_name = item["image"]
        gt_x = float(item["x"])
        gt_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if ref_raw is None or search_raw is None:
            continue

        # 1. Generate Top-500 candidate pool
        cands_500 = generate_candidate_pool_multi(ref_raw, search_raw, max_pool_size=500)

        if not cands_500:
            errs_oracle.append(1000.0)
            errs_hc.append(1000.0)
            errs_siamese.append(1000.0)
            errs_context.append(1000.0)
            errs_lattice.append(1000.0)
            errs_landmark.append(1000.0)
            errs_coord.append(1000.0)
            continue

        # 1. Oracle Candidate Upper Bound
        oracle_dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_500]
        o_err = float(np.min(oracle_dists))
        errs_oracle.append(o_err)

        # 2. Handcrafted Top-500 Ranker
        ranked_hc, _, _, _ = rank_top500_candidates(ref_raw, search_raw, cands_500)
        win_hc = ranked_hc[0]
        hc_err = math.hypot(win_hc['center_x'] - gt_x, win_hc['center_y'] - gt_y)
        errs_hc.append(hc_err)

        # 3. Siamese CNN Ranker
        siamese_scores = compute_cnn_similarity_scores(ref_raw, search_raw, cands_500, checkpoint_path=siamese_checkpoint)
        for i, c in enumerate(cands_500):
            c['siamese_score'] = siamese_scores[i]
            c['siamese_comb'] = 0.70 * c.get('final_score', c['score']) + 0.30 * c['siamese_score']
        cands_siamese = sorted(cands_500, key=lambda c: c['siamese_comb'], reverse=True)
        win_siamese = cands_siamese[0]
        siamese_err = math.hypot(win_siamese['cx'] - gt_x, win_siamese['cy'] - gt_y)
        errs_siamese.append(siamese_err)

        # 4. Context-Aware Ranker
        ctx_scores = compute_context_ranker_scores(ref_raw, search_raw, cands_500, checkpoint_path=ctx_checkpoint)
        for i, c in enumerate(cands_500):
            c['ctx_score'] = ctx_scores[i]
            c['ctx_comb'] = 0.60 * c['ctx_score'] + 0.40 * c.get('final_score', c['score'])
        cands_ctx = sorted(cands_500, key=lambda c: c['ctx_comb'], reverse=True)
        win_ctx = cands_ctx[0]
        ctx_err = math.hypot(win_ctx['cx'] - gt_x, win_ctx['cy'] - gt_y)
        errs_context.append(ctx_err)

        # 5. Global/Lattice Ranker
        lat_scores = compute_global_lattice_scores(ref_raw, search_raw, cands_500, checkpoint_path=lat_checkpoint)
        for i, c in enumerate(cands_500):
            c['lat_score'] = lat_scores[i]
            c['lat_comb'] = 0.60 * c['lat_score'] + 0.40 * c.get('final_score', c['score'])
        cands_lat = sorted(cands_500, key=lambda c: c['lat_comb'], reverse=True)
        win_lat = cands_lat[0]
        lat_err = math.hypot(win_lat['cx'] - gt_x, win_lat['cy'] - gt_y)
        errs_lattice.append(lat_err)

        # 6. Global Landmark Localizer
        lm_pred_x, lm_pred_y, _, _, _ = locate_global_landmark(ref_raw, search_raw, top_k_cands=500)
        lm_err = math.hypot(lm_pred_x - gt_x, lm_pred_y - gt_y)
        errs_landmark.append(lm_err)

        # 7. New Coordinate-Aware Candidate Ranker
        t_coord_0 = time.perf_counter()
        coord_scores = compute_coordinate_aware_scores(ref_raw, search_raw, cands_500, checkpoint_path=coord_checkpoint)
        t_coord_1 = time.perf_counter()
        runtimes_coord.append(t_coord_1 - t_coord_0)

        for i, c in enumerate(cands_500):
            c['coord_score'] = coord_scores[i]
            c['coord_comb'] = 0.60 * c['coord_score'] + 0.40 * c.get('final_score', c['score'])

        cands_coord = sorted(cands_500, key=lambda c: c['coord_comb'], reverse=True)
        win_coord = cands_coord[0]
        coord_err = math.hypot(win_coord['cx'] - gt_x, win_coord['cy'] - gt_y)
        errs_coord.append(coord_err)

        winner_orig_rank = int([i for i, c in enumerate(cands_500) if c['cx'] == win_coord['cx'] and c['cy'] == win_coord['cy']][0]) + 1
        coord_dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_coord]
        rank_closest_gt = int(np.argmin(coord_dists)) + 1

        results.append({
            "image_id": img_name,
            "style": style,
            "predicted_x": win_coord['cx'],
            "predicted_y": win_coord['cy'],
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pixel_error": coord_err,
            "cand_rank": winner_orig_rank,
            "closest_gt_rank": rank_closest_gt,
            "oracle_error": o_err,
            "hc_error": hc_err,
            "siamese_error": siamese_err,
            "context_error": ctx_err,
            "lattice_error": lat_err,
            "landmark_error": lm_err,
            "coord_score": float(win_coord['coord_score']),
            "status": "SUCCESS" if coord_err <= 50.0 else "FAILED"
        })

        if idx % 10 == 0:
            print(f"Evaluated {idx-160}/40 held-out samples | Image: {img_name} | Coord Err: {coord_err:.2f} px")

    # Save CSV results
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "style", "predicted_x", "predicted_y", "gt_x", "gt_y",
            "pixel_error", "cand_rank", "closest_gt_rank", "oracle_error", "hc_error",
            "siamese_error", "context_error", "lattice_error", "landmark_error", "coord_score", "status"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image_id": r["image_id"],
                "style": r["style"],
                "predicted_x": f"{r['predicted_x']:.2f}",
                "predicted_y": f"{r['predicted_y']:.2f}",
                "gt_x": f"{r['gt_x']:.2f}",
                "gt_y": f"{r['gt_y']:.2f}",
                "pixel_error": f"{r['pixel_error']:.2f}",
                "cand_rank": r["cand_rank"],
                "closest_gt_rank": r["closest_gt_rank"],
                "oracle_error": f"{r['oracle_error']:.2f}",
                "hc_error": f"{r['hc_error']:.2f}",
                "siamese_error": f"{r['siamese_error']:.2f}",
                "context_error": f"{r['context_error']:.2f}",
                "lattice_error": f"{r['lattice_error']:.2f}",
                "landmark_error": f"{r['landmark_error']:.2f}",
                "coord_score": f"{r['coord_score']:.4f}",
                "status": r["status"]
            })

    def calc_stats(err_list):
        n = len(err_list) if len(err_list) > 0 else 1
        return {
            "mean": float(np.mean(err_list)) if err_list else 0.0,
            "median": float(np.median(err_list)) if err_list else 0.0,
            "p95": float(np.percentile(err_list, 95)) if err_list else 0.0,
            "max": float(np.max(err_list)) if err_list else 0.0,
            "acc_5": (sum(1 for e in err_list if e <= 5.0) / n) * 100.0,
            "acc_10": (sum(1 for e in err_list if e <= 10.0) / n) * 100.0,
            "acc_25": (sum(1 for e in err_list if e <= 25.0) / n) * 100.0,
            "acc_50": (sum(1 for e in err_list if e <= 50.0) / n) * 100.0,
            "acc_100": (sum(1 for e in err_list if e <= 100.0) / n) * 100.0
        }

    st_o = calc_stats(errs_oracle)
    st_hc = calc_stats(errs_hc)
    st_siamese = calc_stats(errs_siamese)
    st_ctx = calc_stats(errs_context)
    st_lat = calc_stats(errs_lattice)
    st_lm = calc_stats(errs_landmark)
    st_coord = calc_stats(errs_coord)

    avg_coord_rt_ms = float(np.mean(runtimes_coord)) * 1000.0 if runtimes_coord else 0.0

    # Write Markdown Report
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# DriftSense-X Coordinate-Aware Candidate Ranker Benchmark Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"Evaluates the new 44-D **Coordinate-Aware Candidate Ranker** on 40 held-out validation samples ")
        f.write("(`00161.png` - `00200.png`) and compares directly across all 7 candidate ranking approaches.\n\n")

        f.write("## 7-Way Direct Comparative Performance Matrix\n\n")
        f.write("| Model / Approach | <= 5 px | <= 10 px | <= 25 px | <= 50 px | <= 100 px | Mean Error (px) | Median Error (px) | P95 Error (px) | Max Error (px) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        f.write(f"| **1. Oracle Top-500 Upper Bound** | {st_o['acc_5']:.1f}% | {st_o['acc_10']:.1f}% | {st_o['acc_25']:.1f}% | {st_o['acc_50']:.1f}% | {st_o['acc_100']:.1f}% | {st_o['mean']:.2f} | {st_o['median']:.2f} | {st_o['p95']:.2f} | {st_o['max']:.2f} |\n")
        f.write(f"| **2. Handcrafted Top-500 Ranker** | {st_hc['acc_5']:.1f}% | {st_hc['acc_10']:.1f}% | {st_hc['acc_25']:.1f}% | {st_hc['acc_50']:.1f}% | {st_hc['acc_100']:.1f}% | {st_hc['mean']:.2f} | {st_hc['median']:.2f} | {st_hc['p95']:.2f} | {st_hc['max']:.2f} |\n")
        f.write(f"| **3. Siamese CNN Ranker** | {st_siamese['acc_5']:.1f}% | {st_siamese['acc_10']:.1f}% | {st_siamese['acc_25']:.1f}% | {st_siamese['acc_50']:.1f}% | {st_siamese['acc_100']:.1f}% | {st_siamese['mean']:.2f} | {st_siamese['median']:.2f} | {st_siamese['p95']:.2f} | {st_siamese['max']:.2f} |\n")
        f.write(f"| **4. Context CNN Ranker** | {st_ctx['acc_5']:.1f}% | {st_ctx['acc_10']:.1f}% | {st_ctx['acc_25']:.1f}% | {st_ctx['acc_50']:.1f}% | {st_ctx['acc_100']:.1f}% | {st_ctx['mean']:.2f} | {st_ctx['median']:.2f} | {st_ctx['p95']:.2f} | {st_ctx['max']:.2f} |\n")
        f.write(f"| **5. Global/Lattice-Aware Ranker** | {st_lat['acc_5']:.1f}% | {st_lat['acc_10']:.1f}% | {st_lat['acc_25']:.1f}% | {st_lat['acc_50']:.1f}% | {st_lat['acc_100']:.1f}% | {st_lat['mean']:.2f} | {st_lat['median']:.2f} | {st_lat['p95']:.2f} | {st_lat['max']:.2f} |\n")
        f.write(f"| **6. Global Landmark Localizer** | {st_lm['acc_5']:.1f}% | {st_lm['acc_10']:.1f}% | {st_lm['acc_25']:.1f}% | {st_lm['acc_50']:.1f}% | {st_lm['acc_100']:.1f}% | {st_lm['mean']:.2f} | {st_lm['median']:.2f} | {st_lm['p95']:.2f} | {st_lm['max']:.2f} |\n")
        f.write(f"| **7. Coordinate-Aware Candidate Ranker** | {st_coord['acc_5']:.1f}% | {st_coord['acc_10']:.1f}% | {st_coord['acc_25']:.1f}% | {st_coord['acc_50']:.1f}% | {st_coord['acc_100']:.1f}% | {st_coord['mean']:.2f} | {st_coord['median']:.2f} | {st_coord['p95']:.2f} | {st_coord['max']:.2f} |\n\n")

        f.write("## Inference Runtime Benchmark\n\n")
        f.write(f"- **Average Coordinate-Aware Scoring Runtime per Image**: {avg_coord_rt_ms:.2f} ms ({avg_coord_rt_ms/1000.0:.4f} s)\n\n")

        f.write("## Analysis of Coordinate & Phase Features\n\n")
        improved = (st_coord['mean'] < st_hc['mean']) or (st_coord['acc_100'] > st_hc['acc_100'])
        if improved:
            f.write("**VERDICT**: The Coordinate-Aware Candidate Ranker achieved empirical accuracy improvements on held-out validation data!\n")
        else:
            f.write("**VERDICT**: The Coordinate-Aware Candidate Ranker did NOT demonstrate sufficient held-out validation improvement over existing baseline localizers.\n\n")
            f.write("### Diagnostic Feature Breakdown & Root Cause Analysis\n")
            f.write("1. **Absolute Coordinate Shift Invariance**: In synthetic SEM wafer dataset generation, reference pattern crops are sampled uniformly across arbitrary wafer locations. Therefore, absolute coordinates $(cx, cy)$ do not correlate directly with target crop positions across different test images.\n")
            f.write("2. **Lattice Phase Shift Invariance**: Homogeneous periodic cell arrays are invariant under discrete integer lattice shifts $(cx + k_x \\lambda_x, cy + k_y \\lambda_y)$. Sin/cos lattice phase encodings $\\sin(2\\pi cx / \\lambda_x)$ evaluate identically for both the true target cell and all false periodic alias candidates.\n")
            f.write("3. **Local Density & Neighbor Consistency**: Periodic cell fields feature identical local candidate density and identical neighbor responses across all valid lattice sites.\n")

    print("\n" + "=" * 110)
    print("             HELD-OUT 7-WAY DIRECT COMPARATIVE EVALUATION SUMMARY (40 SAMPLES)")
    print("=" * 110)
    print(f"{'Approach':<36} | {'<=50 px':<8} | {'<=100 px':<8} | {'Mean Error':<12} | {'Median Error':<12}")
    print("-" * 110)
    print(f"{'1. Oracle Top-500 Upper Bound':<36} | {st_o['acc_50']:<7.1f}% | {st_o['acc_100']:<7.1f}% | {st_o['mean']:<12.2f} | {st_o['median']:<12.2f}")
    print(f"{'2. Handcrafted Top-500 Ranker':<36} | {st_hc['acc_50']:<7.1f}% | {st_hc['acc_100']:<7.1f}% | {st_hc['mean']:<12.2f} | {st_hc['median']:<12.2f}")
    print(f"{'3. Siamese CNN Ranker':<36} | {st_siamese['acc_50']:<7.1f}% | {st_siamese['acc_100']:<7.1f}% | {st_siamese['mean']:<12.2f} | {st_siamese['median']:<12.2f}")
    print(f"{'4. Context CNN Ranker':<36} | {st_ctx['acc_50']:<7.1f}% | {st_ctx['acc_100']:<7.1f}% | {st_ctx['mean']:<12.2f} | {st_ctx['median']:<12.2f}")
    print(f"{'5. Global/Lattice-Aware Ranker':<36} | {st_lat['acc_50']:<7.1f}% | {st_lat['acc_100']:<7.1f}% | {st_lat['mean']:<12.2f} | {st_lat['median']:<12.2f}")
    print(f"{'6. Global Landmark Localizer':<36} | {st_lm['acc_50']:<7.1f}% | {st_lm['acc_100']:<7.1f}% | {st_lm['mean']:<12.2f} | {st_lm['median']:<12.2f}")
    print(f"{'7. Coordinate-Aware Ranker':<36} | {st_coord['acc_50']:<7.1f}% | {st_coord['acc_100']:<7.1f}% | {st_coord['mean']:<12.2f} | {st_coord['median']:<12.2f}")
    print("=" * 110)
    print(f"Average Coordinate Inference Time: {avg_coord_rt_ms:.2f} ms ({avg_coord_rt_ms/1000.0:.4f} s)")
    print(f"CSV report saved to:               {out_csv}")
    print(f"Markdown report saved:             {out_report}")


if __name__ == "__main__":
    main()
