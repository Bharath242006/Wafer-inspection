import sys, os, math, time
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

from scratch.improve_candidate_recall import generate_candidate_pool_multi
from scratch.test_ranking_top500 import rank_top500_candidates
from localization.context_ranker import compute_context_ranker_scores

def test_context_ranker_00001():
    ref_path = "dataset/validation/reference/00001.png"
    search_path = "dataset/validation/search/00001.png"
    gt_x, gt_y = 636.26, 676.77
    checkpoint_path = "checkpoints/context_ranker.pt"

    start_t = time.perf_counter()

    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # 1. Generate Top-500 candidate pool
    cands_500 = generate_candidate_pool_multi(ref_raw, search_raw, max_pool_size=500)

    # 2. Calculate Handcrafted scores
    ranked_hc, _, _, _ = rank_top500_candidates(ref_raw, search_raw, cands_500)

    # 3. Calculate Trained Multi-Context scores
    t_ctx_start = time.perf_counter()
    ctx_scores = compute_context_ranker_scores(ref_raw, search_raw, cands_500, checkpoint_path=checkpoint_path)
    ctx_time_sec = time.perf_counter() - t_ctx_start

    # Attach Context scores to candidate records
    for i, c in enumerate(cands_500):
        c['context_score'] = ctx_scores[i]
        # Combined score: 0.60 * context_score + 0.40 * final_score
        c['combined_score'] = 0.60 * c['context_score'] + 0.40 * c['final_score']

    # Sort candidates by combined score
    cands_combined = sorted(cands_500, key=lambda c: c['combined_score'], reverse=True)

    elapsed_time_sec = time.perf_counter() - start_t

    print("=" * 135)
    print("        TRAINED MULTI-BRANCH CONTEXT-AWARE CANDIDATE RANKER DIAGNOSTIC TEST FOR 00001.PNG")
    print("=" * 135)
    print(f"Search Image:                {search_path}")
    print(f"Ground Truth Coordinate:     ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Model Checkpoint Path:       {checkpoint_path}")
    print(f"Candidate Pool Size:         {len(cands_500)}")
    print(f"Context Scoring Time:        {ctx_time_sec*1000.0:.2f} ms ({ctx_time_sec:.4f} s)")
    print(f"Total Computation Runtime:   {elapsed_time_sec*1000.0:.2f} ms ({elapsed_time_sec:.4f} s)")
    print("-" * 135)

    print(f"{'Rank':<5} | {'Candidate Center (x, y)':<24} | {'Dist to GT (px)':<18} | {'Handcrafted Score':<18} | {'Context Score':<16} | {'Combined Score':<16}")
    print("-" * 135)

    for r, c in enumerate(cands_combined[:10], start=1):
        d_gt = math.hypot(c['cx'] - gt_x, c['cy'] - gt_y)
        coord_str = f"({c['cx']:.2f}, {c['cy']:.2f})"
        print(f"#{r:<4} | {coord_str:<24} | {d_gt:<18.2f} | {c['final_score']:<18.4f} | {c['context_score']:<16.4f} | {c['combined_score']:<16.4f}")

    print("-" * 135)

    top_orig = ranked_hc[0]
    top_comb = cands_combined[0]

    orig_win_coord = (top_orig['center_x'], top_orig['center_y'])
    comb_win_coord = (top_comb['cx'], top_comb['cy'])

    changed = (orig_win_coord != comb_win_coord)
    pred_err = math.hypot(top_comb['cx'] - gt_x, top_comb['cy'] - gt_y)

    print(f"HANDCRAFTED TOP CANDIDATE:   ({top_orig['center_x']:.2f}, {top_orig['center_y']:.2f}) | Score: {top_orig['final_score']:.4f}")
    print(f"CONTEXT-COMBINED TOP:        ({top_comb['cx']:.2f}, {top_comb['cy']:.2f}) | Handcrafted: {top_comb['final_score']:.4f} | Context: {top_comb['context_score']:.4f} | Combined: {top_comb['combined_score']:.4f}")
    print(f"GROUND TRUTH COORDINATE:     ({gt_x:.2f}, {gt_y:.2f})")
    print(f"PIXEL ERROR TO GT:           {pred_err:.2f} px")
    print(f"CONTEXT RANKER CHANGED WINNER? {'YES' if changed else 'NO'}")
    print(f"STATUS:                      SUCCESS")
    print("=" * 135)

if __name__ == "__main__":
    test_context_ranker_00001()
