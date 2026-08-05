"""
evaluate_hybrid_ranker.py

DriftSense-X Final Hybrid Ranker Evaluation Script.

Evaluates on held-out validation images 00161–00200 ONLY.
Compares 3 approaches:
  1. Oracle Top-500 Upper Bound
  2. Best Existing Baseline (Global Landmark, 431.93 px) — loaded from CSV
  3. New Hybrid Ranker

Does NOT recompute old rankers (Siamese CNN, Context CNN, Lattice, etc.).

Outputs:
  results/hybrid_ranker_validation.csv
  results/hybrid_ranker_report.md

Success criterion: Hybrid mean error < 431.93 px → update final_localizer.py.
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
from localization.hybrid_ranker import compute_hybrid_scores
from scratch.improve_candidate_recall import compute_sobel_gradient
from localization.final_localizer import (
    compute_canny_edge, estimate_lattice_period_2d, refine_subpixel_peak
)
from localization.global_coarse_localizer import zmuv_ncc


BASELINE_CSV = os.path.join("results", "global_landmark_validation.csv")
SPLIT_IDX = 160
HYBRID_CHECKPOINT = os.path.join("checkpoints", "hybrid_ranker.pt")
BASELINE_MEAN_ERROR = 431.93  # Global Landmark baseline

FINE_WINDOW_RADIUS = 35
FINE_SCALES = [0.090, 0.095, 0.100, 0.105, 0.110]


def load_validation_records() -> list:
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records[SPLIT_IDX:]  # 00161–00200 only


def load_baseline_errors() -> dict:
    """
    Loads per-image Global Landmark errors from saved CSV.
    Returns dict: {image_id: landmark_error}.
    Falls back to None if CSV missing (will recompute).
    """
    if not os.path.exists(BASELINE_CSV):
        print(f"[WARNING] Baseline CSV not found at '{BASELINE_CSV}'. "
              f"Will recompute Global Landmark baseline.")
        return {}

    baseline = {}
    with open(BASELINE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            try:
                baseline[img_id] = float(row["landmark_error"])
            except (KeyError, ValueError):
                baseline[img_id] = 1000.0
    return baseline


def _fine_search_around(
    search_img: np.ndarray,
    ref_img: np.ndarray,
    coarse_cx: float,
    coarse_cy: float,
) -> tuple:
    """
    Fine local search ±35 px around the winning coarse candidate.
    Returns (fine_x, fine_y, fine_score).
    """
    ref_gray_f = ref_img.astype(np.float32)
    search_gray_f = search_img.astype(np.float32)
    ref_grad = compute_sobel_gradient(ref_img)
    search_grad = compute_sobel_gradient(search_img)

    sh, sw = search_img.shape[:2]
    best_score = -1.0
    best_x, best_y = coarse_cx, coarse_cy

    for s in FINE_SCALES:
        scaled_w = int(round(ref_img.shape[1] * s))
        scaled_h = int(round(ref_img.shape[0] * s))
        if scaled_w <= 0 or scaled_h <= 0 or scaled_w > sw or scaled_h > sh:
            continue

        s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), cv2.INTER_AREA)
        s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), cv2.INTER_AREA)

        min_tl_x = max(0, int(round(coarse_cx - FINE_WINDOW_RADIUS - scaled_w / 2.0)))
        max_tl_x = min(sw - scaled_w, int(round(coarse_cx + FINE_WINDOW_RADIUS - scaled_w / 2.0)))
        min_tl_y = max(0, int(round(coarse_cy - FINE_WINDOW_RADIUS - scaled_h / 2.0)))
        max_tl_y = min(sh - scaled_h, int(round(coarse_cy + FINE_WINDOW_RADIUS - scaled_h / 2.0)))

        if min_tl_x >= max_tl_x or min_tl_y >= max_tl_y:
            continue

        crop_g = search_gray_f[min_tl_y:max_tl_y + scaled_h, min_tl_x:max_tl_x + scaled_w]
        crop_d = search_grad[min_tl_y:max_tl_y + scaled_h, min_tl_x:max_tl_x + scaled_w]

        res_g = cv2.matchTemplate(crop_g, s_ref_gray, cv2.TM_CCOEFF_NORMED)
        res_d = cv2.matchTemplate(crop_d, s_ref_grad, cv2.TM_CCOEFF_NORMED)
        res_combined = 0.5 * res_g + 0.5 * res_d

        _, max_v, _, max_l = cv2.minMaxLoc(res_combined)
        loc_x, loc_y = max_l[0], max_l[1]
        fine_score = float(max_v)

        if fine_score > best_score:
            best_score = fine_score
            sub_x, sub_y = refine_subpixel_peak(res_combined, loc_x, loc_y)
            best_x = min_tl_x + sub_x + scaled_w / 2.0
            best_y = min_tl_y + sub_y + scaled_h / 2.0

    return float(best_x), float(best_y), float(best_score)


def compute_confidence(
    search_img: np.ndarray,
    ref_img: np.ndarray,
    fine_x: float,
    fine_y: float,
    fine_score: float
) -> float:
    """Computes a confidence score in [0, 1] based on fine match quality."""
    ref_100 = cv2.resize(ref_img, (100, 100), cv2.INTER_AREA).astype(np.float32)
    sh, sw = search_img.shape[:2]
    tl_x = int(np.clip(round(fine_x - 50), 0, sw - 100))
    tl_y = int(np.clip(round(fine_y - 50), 0, sh - 100))
    search_crop = search_img.astype(np.float32)[tl_y:tl_y+100, tl_x:tl_x+100]

    if search_crop.shape == ref_100.shape:
        res = cv2.matchTemplate(search_crop, ref_100, cv2.TM_CCOEFF_NORMED)
        peak = float(res.max())
        # Zero out peak neighborhood for sidelobe estimate
        H, W = res.shape
        py, px = np.unravel_index(np.argmax(res), res.shape)
        mask = np.ones_like(res, dtype=bool)
        mask[max(0, py-2):min(H, py+3), max(0, px-2):min(W, px+3)] = False
        sidelobe = float(np.max(res[mask])) if np.any(mask) else 0.0
        margin = max(0.0, peak - sidelobe)
        return float(np.clip(fine_score * (1.0 + margin), 0.0, 1.0))
    return float(np.clip(fine_score, 0.0, 1.0))


def calc_stats(err_list: list) -> dict:
    n = len(err_list) if err_list else 1
    return {
        "mean":    float(np.mean(err_list))          if err_list else 0.0,
        "median":  float(np.median(err_list))        if err_list else 0.0,
        "p95":     float(np.percentile(err_list, 95)) if err_list else 0.0,
        "max":     float(np.max(err_list))           if err_list else 0.0,
        "acc_5":   sum(1 for e in err_list if e <= 5.0)   / n * 100.0,
        "acc_10":  sum(1 for e in err_list if e <= 10.0)  / n * 100.0,
        "acc_25":  sum(1 for e in err_list if e <= 25.0)  / n * 100.0,
        "acc_50":  sum(1 for e in err_list if e <= 50.0)  / n * 100.0,
        "acc_100": sum(1 for e in err_list if e <= 100.0) / n * 100.0,
    }


def main():
    val_records = load_validation_records()
    baseline_errors = load_baseline_errors()

    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    os.makedirs("results", exist_ok=True)

    out_csv = os.path.join("results", "hybrid_ranker_validation.csv")
    out_report = os.path.join("results", "hybrid_ranker_report.md")

    print("=" * 100)
    print("   EVALUATING FINAL HYBRID RANKER ON HELD-OUT VALIDATION SET (40 SAMPLES)")
    print("=" * 100)
    print(f"   Hybrid checkpoint  : {HYBRID_CHECKPOINT}")
    print(f"   Baseline CSV       : {BASELINE_CSV}")
    print(f"   Target to beat     : {BASELINE_MEAN_ERROR} px mean error (Global Landmark)")
    print("=" * 100)

    results = []
    errs_oracle  = []
    errs_baseline = []
    errs_hybrid  = []
    runtimes_hybrid = []

    for sample_i, item in enumerate(val_records, start=1):
        img_name = item["image"]
        gt_x = float(item["x"])
        gt_y = float(item["y"])
        style = item.get("style", "Unknown")

        ref_path    = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        ref_raw    = cv2.imread(ref_path,    cv2.IMREAD_GRAYSCALE)
        search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if ref_raw is None or search_raw is None:
            print(f"  [{sample_i:02d}/40] SKIP {img_name} — image load failed")
            errs_oracle.append(1000.0)
            errs_baseline.append(baseline_errors.get(img_name, 1000.0))
            errs_hybrid.append(1000.0)
            continue

        # ── 1. Generate Top-500 candidate pool ──────────────────────────────
        cands_500 = generate_candidate_pool_multi(ref_raw, search_raw, max_pool_size=500)

        if not cands_500:
            errs_oracle.append(1000.0)
            errs_baseline.append(baseline_errors.get(img_name, 1000.0))
            errs_hybrid.append(1000.0)
            continue

        # ── 2. Oracle error (best candidate in pool) ─────────────────────────
        oracle_dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_500]
        o_err = float(np.min(oracle_dists))
        errs_oracle.append(o_err)

        # ── 3. Baseline error (from saved CSV) ───────────────────────────────
        baseline_err = baseline_errors.get(img_name, None)
        if baseline_err is None:
            # Recompute using Global Landmark
            try:
                from localization.global_landmark_localizer import locate_global_landmark
                lm_x, lm_y, _, _, _ = locate_global_landmark(ref_raw, search_raw, top_k_cands=500)
                baseline_err = math.hypot(lm_x - gt_x, lm_y - gt_y)
            except Exception as e:
                baseline_err = 1000.0
                print(f"  [{sample_i:02d}/40] Baseline recompute failed: {e}")
        errs_baseline.append(float(baseline_err))

        # ── 4. Hybrid Ranker ─────────────────────────────────────────────────
        t0 = time.perf_counter()

        hybrid_scores = compute_hybrid_scores(ref_raw, search_raw, cands_500, HYBRID_CHECKPOINT)

        # Combine hybrid score with pipeline score (0.60 hybrid + 0.40 candidate score)
        for i, c in enumerate(cands_500):
            c['hybrid_score'] = float(hybrid_scores[i]) if i < len(hybrid_scores) else 0.0
            c['hybrid_comb'] = 0.60 * c['hybrid_score'] + 0.40 * c.get('score', 0.0)

        cands_ranked = sorted(cands_500, key=lambda c: c['hybrid_comb'], reverse=True)
        winner = cands_ranked[0]
        coarse_cx, coarse_cy = winner['cx'], winner['cy']

        # Fine local search ±35 px around winning candidate
        fine_x, fine_y, fine_score = _fine_search_around(search_raw, ref_raw, coarse_cx, coarse_cy)

        t1 = time.perf_counter()
        runtime_sec = t1 - t0
        runtimes_hybrid.append(runtime_sec)

        # Compute error and confidence
        hybrid_err = math.hypot(fine_x - gt_x, fine_y - gt_y)
        errs_hybrid.append(hybrid_err)

        confidence = compute_confidence(search_raw, ref_raw, fine_x, fine_y, fine_score)

        # Find winner's original rank in pool
        cand_rank = next(
            (i + 1 for i, c in enumerate(cands_500)
             if c['cx'] == winner['cx'] and c['cy'] == winner['cy']),
            1
        )

        results.append({
            "image_id":      img_name,
            "style":         style,
            "predicted_x":   fine_x,
            "predicted_y":   fine_y,
            "gt_x":          gt_x,
            "gt_y":          gt_y,
            "pixel_error":   hybrid_err,
            "confidence":    confidence,
            "candidate_rank": cand_rank,
            "oracle_error":  o_err,
            "baseline_error": float(baseline_err),
            "hybrid_score":  float(winner['hybrid_score']),
            "runtime_sec":   runtime_sec,
            "status":        "SUCCESS" if hybrid_err <= 50.0 else "FAILED",
        })

        delta = hybrid_err - float(baseline_err)
        delta_str = f"{delta:+.1f}" if delta != 0 else "±0"
        print(
            f"  [{sample_i:02d}/40] {img_name} | "
            f"Oracle: {o_err:6.1f}px | "
            f"Baseline: {float(baseline_err):6.1f}px | "
            f"Hybrid: {hybrid_err:6.1f}px ({delta_str}) | "
            f"conf={confidence:.2f} | {runtime_sec*1000:.0f}ms"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "style",
            "predicted_x", "predicted_y", "gt_x", "gt_y",
            "pixel_error", "confidence", "candidate_rank",
            "oracle_error", "baseline_error", "hybrid_score",
            "runtime_sec", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image_id":       r["image_id"],
                "style":          r["style"],
                "predicted_x":    f"{r['predicted_x']:.2f}",
                "predicted_y":    f"{r['predicted_y']:.2f}",
                "gt_x":           f"{r['gt_x']:.2f}",
                "gt_y":           f"{r['gt_y']:.2f}",
                "pixel_error":    f"{r['pixel_error']:.2f}",
                "confidence":     f"{r['confidence']:.4f}",
                "candidate_rank": r["candidate_rank"],
                "oracle_error":   f"{r['oracle_error']:.2f}",
                "baseline_error": f"{r['baseline_error']:.2f}",
                "hybrid_score":   f"{r['hybrid_score']:.4f}",
                "runtime_sec":    f"{r['runtime_sec']:.4f}",
                "status":         r["status"],
            })

    # ── Compute stats ─────────────────────────────────────────────────────────
    st_o  = calc_stats(errs_oracle)
    st_b  = calc_stats(errs_baseline)
    st_h  = calc_stats(errs_hybrid)
    avg_rt_ms = float(np.mean(runtimes_hybrid)) * 1000.0 if runtimes_hybrid else 0.0

    improved = st_h['mean'] < BASELINE_MEAN_ERROR

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("   FINAL HYBRID RANKER — VALIDATION RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Approach':<36} | {'≤5px':>6} | {'≤10px':>6} | {'≤25px':>6} | {'≤50px':>6} | {'≤100px':>7} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'Max':>8}")
    print("-" * 100)
    print(f"{'Oracle Top-500':<36} | {st_o['acc_5']:>5.1f}% | {st_o['acc_10']:>5.1f}% | {st_o['acc_25']:>5.1f}% | {st_o['acc_50']:>5.1f}% | {st_o['acc_100']:>6.1f}% | {st_o['mean']:>8.2f} | {st_o['median']:>8.2f} | {st_o['p95']:>8.2f} | {st_o['max']:>8.2f}")
    print(f"{'Global Landmark (best baseline)':<36} | {st_b['acc_5']:>5.1f}% | {st_b['acc_10']:>5.1f}% | {st_b['acc_25']:>5.1f}% | {st_b['acc_50']:>5.1f}% | {st_b['acc_100']:>6.1f}% | {st_b['mean']:>8.2f} | {st_b['median']:>8.2f} | {st_b['p95']:>8.2f} | {st_b['max']:>8.2f}")
    print(f"{'Hybrid Ranker (NEW)':<36} | {st_h['acc_5']:>5.1f}% | {st_h['acc_10']:>5.1f}% | {st_h['acc_25']:>5.1f}% | {st_h['acc_50']:>5.1f}% | {st_h['acc_100']:>6.1f}% | {st_h['mean']:>8.2f} | {st_h['median']:>8.2f} | {st_h['p95']:>8.2f} | {st_h['max']:>8.2f}")
    print("=" * 100)
    print(f"Average Hybrid Ranker Runtime: {avg_rt_ms:.2f} ms/image")
    print(f"Hybrid vs. Baseline mean improvement: {BASELINE_MEAN_ERROR - st_h['mean']:+.2f} px")
    if improved:
        print(f"VERDICT: ✓ IMPROVED — Hybrid Ranker ({st_h['mean']:.2f} px) beats baseline ({BASELINE_MEAN_ERROR} px)")
        print("  → Will integrate into localization/final_localizer.py")
    else:
        print(f"VERDICT: ✗ NOT IMPROVED — Hybrid Ranker ({st_h['mean']:.2f} px) did NOT beat baseline ({BASELINE_MEAN_ERROR} px)")
        print("  → Keeping Global Landmark as the final pipeline.")
    print("=" * 100)

    # ── Write Markdown report ─────────────────────────────────────────────────
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# DriftSense-X Final Hybrid Ranker — Validation Report\n\n")
        f.write("## Summary\n\n")
        f.write(
            f"Evaluated the new **56-D Hybrid Ranker** (HybridRankerNet MLP) on "
            f"40 held-out validation images (00161–00200).\n\n"
        )
        f.write(
            f"Target to beat: **{BASELINE_MEAN_ERROR} px** mean error (Global Landmark baseline).\n\n"
        )

        f.write("## 3-Way Comparative Performance\n\n")
        f.write("| Approach | ≤5px | ≤10px | ≤25px | ≤50px | ≤100px | Mean Error | Median | P95 | Max |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        f.write(
            f"| **Oracle Top-500** | {st_o['acc_5']:.1f}% | {st_o['acc_10']:.1f}% | "
            f"{st_o['acc_25']:.1f}% | {st_o['acc_50']:.1f}% | {st_o['acc_100']:.1f}% | "
            f"{st_o['mean']:.2f} | {st_o['median']:.2f} | {st_o['p95']:.2f} | {st_o['max']:.2f} |\n"
        )
        f.write(
            f"| **Global Landmark (baseline)** | {st_b['acc_5']:.1f}% | {st_b['acc_10']:.1f}% | "
            f"{st_b['acc_25']:.1f}% | {st_b['acc_50']:.1f}% | {st_b['acc_100']:.1f}% | "
            f"{st_b['mean']:.2f} | {st_b['median']:.2f} | {st_b['p95']:.2f} | {st_b['max']:.2f} |\n"
        )
        f.write(
            f"| **Hybrid Ranker (NEW)** | {st_h['acc_5']:.1f}% | {st_h['acc_10']:.1f}% | "
            f"{st_h['acc_25']:.1f}% | {st_h['acc_50']:.1f}% | {st_h['acc_100']:.1f}% | "
            f"**{st_h['mean']:.2f}** | {st_h['median']:.2f} | {st_h['p95']:.2f} | {st_h['max']:.2f} |\n"
        )
        f.write("\n")

        f.write("## Runtime\n\n")
        f.write(f"- **Average Hybrid Ranker runtime**: {avg_rt_ms:.2f} ms/image\n\n")

        f.write("## Hybrid Features (56-D)\n\n")
        f.write("| Group | Features | Dim |\n")
        f.write("|---|---|---|\n")
        f.write("| Visual (NCC/Grad/LoG) | raw + pool-normalized | 6 |\n")
        f.write("| FFT phase-correlation | NEW: discriminates periodic aliases | 1 |\n")
        f.write("| Edge/Canny overlap | raw + normalized | 2 |\n")
        f.write("| Low-freq Gaussian | raw + normalized | 2 |\n")
        f.write("| Medium-context (150px) | NEW vs. prior rankers | 1 |\n")
        f.write("| Global landmark heatmap | | 1 |\n")
        f.write("| Coordinates | cx, cy, cx/1000, cy/1000, cx/W, cy/H, dist_center | 7 |\n")
        f.write("| Lattice phase | cx/lx, cy/ly, phase_x/y, sin/cos encodings | 8 |\n")
        f.write("| Rank + margins | rank/500, log(rank), percentile, margin_top1/median | 5 |\n")
        f.write("| Local density | density_r30/r60, dist_nearest | 3 |\n")
        f.write("| Neighbor consistency | ±lx, ±ly lattice-direction NCC | 4 |\n")
        f.write("| Extended neighbors | diagonal ±(lx,ly), half-period, 2nd-order | 6 |\n")
        f.write("| Pool statistics | heatmap grad, top10 mean, top10 margin | 3 |\n")
        f.write("| Multi-scale NCC | 100→50→25→12 px (normalized) | 2 |\n")
        f.write("| Spatial neighbor score | mean top-5 spatial candidate scores | 1 |\n")
        f.write(f"| **TOTAL** | | **56** |\n\n")

        f.write("## Integration Verdict\n\n")
        if improved:
            f.write(
                f"**✓ VERDICT: IMPROVED** — Hybrid Ranker achieves {st_h['mean']:.2f} px mean error "
                f"vs. baseline {BASELINE_MEAN_ERROR} px.\n\n"
                f"Improvement: **{BASELINE_MEAN_ERROR - st_h['mean']:+.2f} px**.\n\n"
                f"→ Hybrid Ranker integrated into `localization/final_localizer.py`.\n"
            )
        else:
            delta = st_h['mean'] - BASELINE_MEAN_ERROR
            f.write(
                f"**✗ VERDICT: NOT IMPROVED** — Hybrid Ranker achieves {st_h['mean']:.2f} px mean error "
                f"(+{delta:.2f} px vs. baseline {BASELINE_MEAN_ERROR} px).\n\n"
            )
            f.write("### Root Cause Analysis\n\n")
            f.write(
                "The primary failure mode remains **periodic aliasing**: the reference pattern repeats "
                "at lattice intervals (~67 px), and candidates at alias locations produce nearly "
                "identical visual feature vectors. The FFT phase-correlation score provides "
                "a weak additional signal but is insufficient on its own when:\n\n"
                "1. The search patch is noisy and phase-correlation peaks are broad.\n"
                "2. Alias candidates differ by an exact integer number of lattice periods, "
                "   causing the FFT phase response to look identical to the true match.\n\n"
                "→ Keeping Global Landmark as the final pipeline.\n"
            )

    print(f"\nCSV  saved: {out_csv}")
    print(f"Report saved: {out_report}")
    return improved, st_h['mean']


if __name__ == "__main__":
    improved, mean_error = main()
    sys.exit(0 if improved else 1)
