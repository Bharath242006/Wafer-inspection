import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

ref_gray_f = ref_raw.astype(np.float32)
search_gray_f = search_raw.astype(np.float32)

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]

print("=" * 105)
print("     GROUND TRUTH REGION MATCH SCORE vs CANDIDATE ALIASES FOR 00001.PNG")
print("=" * 105)

for s in scales:
    sw = int(round(ref_raw.shape[1] * s))
    sh = int(round(ref_raw.shape[0] * s))

    s_ref_gray = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)

    # 1. Ground Truth crop
    gt_tl_x = int(round(gt_x - sw / 2.0))
    gt_tl_y = int(round(gt_y - sh / 2.0))
    gt_patch = search_gray_f[gt_tl_y:gt_tl_y+sh, gt_tl_x:gt_tl_x+sw]

    res = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
    gt_score = res[gt_tl_y, gt_tl_x]

    # Find peak score in +/- 20 px window around Ground Truth (636.26, 676.77)
    window = res[max(0, gt_tl_y-20):min(res.shape[0], gt_tl_y+20), max(0, gt_tl_x-20):min(res.shape[1], gt_tl_x+20)]
    _, max_win_v, _, max_win_l = cv2.minMaxLoc(window)
    near_gt_x = max(0, gt_tl_x-20) + max_win_l[0] + sw / 2.0
    near_gt_y = max(0, gt_tl_y-20) + max_win_l[1] + sh / 2.0
    near_d_gt = math.hypot(near_gt_x - gt_x, near_gt_y - gt_y)

    # 2. Alias crop at (703, 674)
    a_tl_x = int(round(703.0 - sw / 2.0))
    a_tl_y = int(round(674.0 - sh / 2.0))
    alias_score = res[a_tl_y, a_tl_x]

    print(f"Scale {s:.3f}: GT Exact Score = {gt_score:.4f} | Local Peak Near GT = ({near_gt_x:.1f}, {near_gt_y:.1f}) [Score = {max_win_v:.4f}, Dist GT = {near_d_gt:.2f} px] | Alias (703,674) Score = {alias_score:.4f}")

print("=" * 105)
