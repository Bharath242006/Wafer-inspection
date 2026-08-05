import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]

# Compare Cell A (True, gt_x=636.26, gt_y=676.77) vs Cell B (Alias, x=703, y=674)
print("=== MULTI-SCALE & MULTI-FEATURE SCORE COMPARISON ===")
print(f"{'Scale':<7} {'Feature':<15} {'True Cell (636.26, 676.77)':<28} {'Alias Cell (703, 674)':<24} {'Winner':<10}")
print("-" * 88)

for s in scales:
    sw = int(round(ref_raw.shape[1] * s))
    sh = int(round(ref_raw.shape[0] * s))

    s_ref_g = cv2.resize(ref_raw, (sw, sh), cv2.INTER_AREA)

    # 1. Laplacian of Gaussian (LoG) - detects high-frequency blob/corner structures
    log_ref = cv2.Laplacian(s_ref_g, cv2.CV_32F, ksize=3)

    # True cell crop
    tl_a_x = int(round(gt_x - sw/2.0))
    tl_a_y = int(round(gt_y - sh/2.0))
    crop_a = search_raw[tl_a_y:tl_a_y+sh, tl_a_x:tl_a_x+sw]
    log_a = cv2.Laplacian(crop_a, cv2.CV_32F, ksize=3)

    # Alias cell crop
    tl_b_x = int(round(703.0 - sw/2.0))
    tl_b_y = int(round(674.0 - sh/2.0))
    crop_b = search_raw[tl_b_y:tl_b_y+sh, tl_b_x:tl_b_x+sw]
    log_b = cv2.Laplacian(crop_b, cv2.CV_32F, ksize=3)

    score_a_int = float(cv2.matchTemplate(crop_a.astype(np.float32), s_ref_g.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0])
    score_b_int = float(cv2.matchTemplate(crop_b.astype(np.float32), s_ref_g.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0])

    score_a_log = float(cv2.matchTemplate(log_a, log_ref, cv2.TM_CCOEFF_NORMED)[0, 0])
    score_b_log = float(cv2.matchTemplate(log_b, log_ref, cv2.TM_CCOEFF_NORMED)[0, 0])

    win_int = "TRUE" if score_a_int > score_b_int else "ALIAS"
    win_log = "TRUE" if score_a_log > score_b_log else "ALIAS"

    print(f"{s:<7.3f} {'Intensity':<15} {score_a_int:<28.4f} {score_b_int:<24.4f} {win_int:<10}")
    print(f"{s:<7.3f} {'Laplacian (LoG)':<15} {score_a_log:<28.4f} {score_b_log:<24.4f} {win_log:<10}")
