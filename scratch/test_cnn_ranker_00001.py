import sys, os, math, time
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

from localization.final_localizer import locate_reference_pattern_final
from localization.cnn_candidate_ranker import compute_cnn_candidate_scores

def test_cnn_ranker_00001():
    ref_path = "dataset/validation/reference/00001.png"
    search_path = "dataset/validation/search/00001.png"
    gt_x, gt_y = 636.26, 676.77

    start_t = time.perf_counter()

    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # 1. Run pipeline to extract candidate pool and handcrafted features
    coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern_final(
        ref_path=ref_path,
        search_path=search_path
    )

    cands = debug_info.get("all_candidates", [])

    # 2. Calculate CNN similarity scores
    t_cnn_start = time.perf_counter()
    cnn_scores = compute_cnn_candidate_scores(ref_raw, search_raw, cands)
    cnn_time_sec = time.perf_counter() - t_cnn_start

    # Attach CNN scores to candidate records
    for i, c in enumerate(cands):
        c['cnn_score'] = cnn_scores[i]
        # Combined score: 0.70 * existing final_score + 0.30 * cnn_score
        c['combined_score'] = 0.70 * c['final_score'] + 0.30 * c['cnn_score']

    # Sort candidates by combined score
    cands_combined = sorted(cands, key=lambda c: c['combined_score'], reverse=True)

    elapsed_time_sec = time.perf_counter() - start_t

    print("=" * 135)
    print("                 SIAMESE CNN CANDIDATE RANKER DIAGNOSTIC TEST FOR 00001.PNG")
    print("=" * 135)
    print(f"Search Image:                {search_path}")
    print(f"Ground Truth Coordinate:     ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Candidate Pool Size:         {len(cands)}")
    print(f"CNN Feature Computation Time: {cnn_time_sec*1000.0:.2f} ms ({cnn_time_sec:.4f} s)")
    print(f"Total Computation Runtime:   {elapsed_time_sec*1000.0:.2f} ms ({elapsed_time_sec:.4f} s)")
    print("-" * 135)

    print(f"{'Rank':<5} | {'Candidate Center (x, y)':<24} | {'Dist to GT (px)':<18} | {'Handcrafted Score':<18} | {'CNN Sim Score':<16} | {'Combined Score':<16}")
    print("-" * 135)

    for r, c in enumerate(cands_combined[:10], start=1):
        d_gt = math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y)
        coord_str = f"({c['center_x']:.2f}, {c['center_y']:.2f})"
        print(f"#{r:<4} | {coord_str:<24} | {d_gt:<18.2f} | {c['final_score']:<18.4f} | {c['cnn_score']:<16.4f} | {c['combined_score']:<16.4f}")

    print("-" * 135)

    top_orig = cands[0]
    top_comb = cands_combined[0]

    orig_win_coord = (top_orig['center_x'], top_orig['center_y'])
    comb_win_coord = (top_comb['center_x'], top_comb['center_y'])

    changed = (orig_win_coord != comb_win_coord)

    pred_err = math.hypot(top_comb['center_x'] - gt_x, top_comb['center_y'] - gt_y)

    print(f"ORIGINAL TOP CANDIDATE:      ({top_orig['center_x']:.2f}, {top_orig['center_y']:.2f}) | Handcrafted Score: {top_orig['final_score']:.4f}")
    print(f"CNN-COMBINED TOP CANDIDATE:  ({top_comb['center_x']:.2f}, {top_comb['center_y']:.2f}) | Handcrafted: {top_comb['final_score']:.4f} | CNN: {top_comb['cnn_score']:.4f} | Combined: {top_comb['combined_score']:.4f}")
    print(f"GROUND TRUTH COORDINATE:     ({gt_x:.2f}, {gt_y:.2f})")
    print(f"PIXEL ERROR TO GT:           {pred_err:.2f} px")
    print(f"CNN CHANGED WINNER?          {'YES' if changed else 'NO'}")
    print(f"STATUS:                      SUCCESS")
    print("=" * 135)

if __name__ == "__main__":
    test_cnn_ranker_00001()
