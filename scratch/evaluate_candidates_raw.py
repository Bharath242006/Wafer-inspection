import os
import sys
import csv
import math
import cv2
import numpy as np

sys.path.insert(0, '.')
from scratch.improve_candidate_recall import generate_candidate_pool_multi
from localization.global_landmark_localizer import locate_global_landmark

labels_path = os.path.join("dataset", "validation", "labels.csv")
ref_dir = os.path.join("dataset", "validation", "reference")
search_dir = os.path.join("dataset", "validation", "search")

records = []
with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

val_records = records[160:]  # 00161-00200

raw_errs = []
landmark_errs = []

print("Evaluating 40 held-out images...")
for item in val_records:
    img_name = item["image"]
    gt_x = float(item["x"])
    gt_y = float(item["y"])

    ref_img = cv2.imread(os.path.join(ref_dir, img_name), cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(os.path.join(search_dir, img_name), cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        continue

    # Candidate #1 (Highest raw multi-scale peak score)
    cands = generate_candidate_pool_multi(ref_img, search_img, max_pool_size=500)
    if cands:
        top1 = cands[0]
        raw_err = math.hypot(top1['cx'] - gt_x, top1['cy'] - gt_y)
    else:
        raw_err = 1000.0

    # Global Landmark
    lm_x, lm_y, _, _, _ = locate_global_landmark(ref_img, search_img)
    lm_err = math.hypot(lm_x - gt_x, lm_y - gt_y)

    raw_errs.append(raw_err)
    landmark_errs.append(lm_err)

    print(f"[{img_name}] Raw Candidate #1 Error: {raw_err:6.1f} px | Global Landmark Error: {lm_err:6.1f} px")

print("=" * 60)
print(f"Raw Top-1 Candidate Mean Error : {np.mean(raw_errs):.2f} px | Median: {np.median(raw_errs):.2f} px")
print(f"Global Landmark Mean Error     : {np.mean(landmark_errs):.2f} px | Median: {np.median(landmark_errs):.2f} px")
print("=" * 60)
