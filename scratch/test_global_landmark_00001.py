import sys, os, math, time
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

from localization.global_landmark_localizer import locate_global_landmark

def test_global_landmark_00001():
    ref_path = "dataset/validation/reference/00001.png"
    search_path = "dataset/validation/search/00001.png"
    gt_x, gt_y = 636.26, 676.77

    start_t = time.perf_counter()

    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    pred_x, pred_y, rank, score, ranked_cands = locate_global_landmark(ref_raw, search_raw, top_k_cands=500)

    elapsed_time_sec = time.perf_counter() - start_t
    pixel_err = math.hypot(pred_x - gt_x, pred_y - gt_y)

    print("=" * 100)
    print("      DETERMINISTIC GLOBAL LANDMARK LOCALIZER TEST FOR 00001.PNG")
    print("=" * 100)
    print(f"Search Image:                {search_path}")
    print(f"Ground Truth Coordinate:     ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Predicted Coordinate:        ({pred_x:.2f}, {pred_y:.2f})")
    print(f"Pixel Error to GT:           {pixel_err:.2f} px")
    print(f"Selected Candidate Rank:     #{rank}")
    print(f"Global Alignment Score:      {score:.4f}")
    print(f"Total Runtime:               {elapsed_time_sec*1000.0:.2f} ms ({elapsed_time_sec:.4f} s)")
    print("-" * 100)
    print(f"Predicted X: {pred_x:.2f} px")
    print(f"Predicted Y: {pred_y:.2f} px")
    print(f"Pixel Error: {pixel_err:.2f} px")
    print("=" * 100)

if __name__ == "__main__":
    test_global_landmark_00001()
