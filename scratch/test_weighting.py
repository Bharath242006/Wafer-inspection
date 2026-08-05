import os
import sys
import csv
import math
import cv2
import numpy as np

sys.path.insert(0, '.')
from scratch.improve_candidate_recall import generate_candidate_pool_multi, compute_sobel_gradient
from localization.global_landmark_localizer import compute_global_landmark_heatmap
from localization.final_localizer import refine_subpixel_peak

FINE_SCALES = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
FINE_WINDOW_RADIUS = 35

def fine_search(search_img, ref_img, coarse_cx, coarse_cy):
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
        if float(max_v) > best_score:
            best_score = float(max_v)
            sub_x, sub_y = refine_subpixel_peak(res_combined, max_l[0], max_l[1])
            best_x = min_tl_x + sub_x + scaled_w / 2.0
            best_y = min_tl_y + sub_y + scaled_h / 2.0

    return float(best_x), float(best_y), float(best_score)


labels_path = "dataset/validation/labels.csv"
ref_dir = "dataset/validation/reference"
search_dir = "dataset/validation/search"

records = []
with open(labels_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        records.append(r)

val_records = records[160:]

formulas = {
    "original_60_40": lambda gw, sc: 0.60 * gw + 0.40 * sc,
    "balanced_30_70": lambda gw, sc: 0.30 * gw + 0.70 * sc,
    "multiplicative": lambda gw, sc: sc * (1.0 + 0.5 * gw),
    "pure_candidate": lambda gw, sc: sc,
}

results = {k: [] for k in formulas}

for item in val_records:
    img_name = item["image"]
    gt_x, gt_y = float(item["x"]), float(item["y"])

    ref_img = cv2.imread(os.path.join(ref_dir, img_name), cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(os.path.join(search_dir, img_name), cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        continue

    cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=500)
    if not cands:
        for k in formulas:
            results[k].append(1000.0)
        continue

    heatmap = compute_global_landmark_heatmap(ref_img, search_img)
    sh, sw = search_img.shape[:2]

    for c in cands:
        ix = int(np.clip(round(c['cx']), 0, sw - 1))
        iy = int(np.clip(round(c['cy']), 0, sh - 1))
        c['gw'] = float(heatmap[iy, ix])

    for form_name, form_fn in formulas.items():
        sorted_cands = sorted(cands, key=lambda c: form_fn(c['gw'], c['score']), reverse=True)
        winner = sorted_cands[0]
        fx, fy, fscore = fine_search(search_img, ref_img, winner['cx'], winner['cy'])
        err = math.hypot(fx - gt_x, fy - gt_y)
        results[form_name].append(err)

print("=" * 70)
print("  EVALUATING LANDMARK SCORE FORMULAS ON 40 HELD-OUT VALIDATION IMAGES")
print("=" * 70)
for k, errs in results.items():
    print(f"Form: {k:<15} | Mean Error: {np.mean(errs):6.2f} px | Median: {np.median(errs):6.2f} px | <=50px: {sum(1 for e in errs if e <= 50)/len(errs)*100:.1f}%")
print("=" * 70)
