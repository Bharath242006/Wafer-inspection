import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np
from localization.final_localizer import locate_reference_pattern_final

def test_final_00001():
    ref_path = "dataset/validation/reference/00001.png"
    search_path = "dataset/validation/search/00001.png"
    gt_x, gt_y = 636.26, 676.77

    coarse_center, fine_center, confidence, status, debug_info = locate_reference_pattern_final(
        ref_path=ref_path,
        search_path=search_path
    )

    cands = debug_info.get("all_candidates", [])
    lx, ly = debug_info.get("lattice_period", (67.0, 67.0))

    cand_gt_region = None
    cand_alias = None

    for c in cands:
        d_gt = math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y)
        if d_gt < 60.0 and cand_gt_region is None:
            cand_gt_region = c
        d_alias = math.hypot(c['center_x'] - 703.0, c['center_y'] - 674.0)
        if d_alias < 15.0 and cand_alias is None:
            cand_alias = c

    print("=" * 135)
    print("                     FINAL LOCALIZER DIAGNOSTIC TEST FOR 00001.PNG")
    print("=" * 135)
    print(f"Search Image:                {search_path}")
    print(f"Ground Truth Coordinate:     ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Dynamically Estimated Period: lambda_x = {lx:.2f} px, lambda_y = {ly:.2f} px")
    print("-" * 135)

    print(f"{'Feature Metric':<25} | {'GT Target Region Candidate':<32} | {'False Periodic Alias Candidate':<32}")
    print("-" * 135)

    c_gt_x = f"({cand_gt_region['center_x']:.2f}, {cand_gt_region['center_y']:.2f})" if cand_gt_region else "N/A"
    c_al_x = f"({cand_alias['center_x']:.2f}, {cand_alias['center_y']:.2f})" if cand_alias else "N/A"

    print(f"{'Candidate Coordinates':<25} | {c_gt_x:<32} | {c_al_x:<32}")
    print(f"{'Distance to GT (px)':<25} | {math.hypot(cand_gt_region['center_x']-gt_x, cand_gt_region['center_y']-gt_y):.2f} px{'':<25} | {math.hypot(cand_alias['center_x']-gt_x, cand_alias['center_y']-gt_y):.2f} px{'':<25}" if cand_gt_region and cand_alias else "")

    features = ['ncc', 'gradient', 'log', 'edge', 'low_frequency', 'macro', 'texture', 'multi_scale', 'final_score']

    for feat in features:
        v_gt = f"{cand_gt_region[feat]:.4f}" if cand_gt_region and feat in cand_gt_region else "N/A"
        v_al = f"{cand_alias[feat]:.4f}" if cand_alias and feat in cand_alias else "N/A"
        print(f"{feat.upper():<25} | {v_gt:<32} | {v_al:<32}")

    print("-" * 135)

    top_cand = cands[0]
    winner_str = f"({top_cand['center_x']:.2f}, {top_cand['center_y']:.2f})"
    winner_d_gt = math.hypot(top_cand['center_x'] - gt_x, top_cand['center_y'] - gt_y)

    margin = cands[0]['final_score'] - cands[1]['final_score'] if len(cands) > 1 else 0.0

    print(f"RANK #1 WINNER CANDIDATE:    {winner_str} | Final Score: {top_cand['final_score']:.4f} | Dist GT: {winner_d_gt:.2f} px")
    print(f"SCORE MARGIN (Top #1 - #2):  {margin:.4f}")
    print(f"FINAL PREDICTED CENTER:      ({fine_center[0]:.2f}, {fine_center[1]:.2f})" if fine_center else "None (Failed)")

    pred_err = math.hypot(fine_center[0] - gt_x, fine_center[1] - gt_y) if fine_center else 1000.0
    print(f"FINAL PREDICTION PIXEL ERROR: {pred_err:.2f} px")
    print(f"CONFIDENCE SCORE:            {confidence:.4f}")
    print(f"STATUS:                      {status}")
    print(f"COMPUTATION RUNTIME:         {debug_info.get('computation_time_sec', 0.0)*1000.0:.2f} ms ({debug_info.get('computation_time_sec', 0.0):.4f} s)")
    print("=" * 135)

if __name__ == "__main__":
    test_final_00001()
