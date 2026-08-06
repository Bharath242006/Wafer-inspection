"""
evaluation/evaluate_frequency.py

Frequency Domain / FFT Phase Localizer Evaluator script.
"""

import csv
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath("."))
from localization.frequency_localizer import locate_reference_pattern_frequency


MAX_EVAL_IMAGES = 20


def main():
    print("=" * 80)
    print("        EVALUATING FREQUENCY DOMAIN LOCALIZER")
    print("=" * 80)

    csv_path = os.path.join("dataset", "validation", "labels.csv")
    if not os.path.exists(csv_path):
        print(f"Labels CSV not found at '{csv_path}'. Exiting.")
        return

    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows = rows[:MAX_EVAL_IMAGES]
    results = []
    total_imgs = len(rows)
    eval_start_time = time.time()

    for idx, row in enumerate(rows, start=1):
        img_name = row["image"]
        print(f"Evaluating image {idx}/{total_imgs}")
        print(f"Processing: {img_name}")
        gt_x = float(row["x"])
        gt_y = float(row["y"])

        ref_path = os.path.join(ref_dir, img_name)
        search_path = os.path.join(search_dir, img_name)

        if not os.path.exists(ref_path) or not os.path.exists(search_path):
            continue

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        pred_x, pred_y, score = locate_reference_pattern_frequency(ref_img, search_img)
        err = float(np.hypot(pred_x - gt_x, pred_y - gt_y))

        results.append({
            "image": img_name,
            "true_x": gt_x,
            "true_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_px": err,
            "score": score
        })

    total_eval_time = time.time() - eval_start_time
    print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")

    os.makedirs("outputs/metrics", exist_ok=True)
    out_csv = os.path.join("outputs", "metrics", "frequency_validation.csv")

    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        errs = [r["error_px"] for r in results]
        print(f"Evaluated {len(results)} samples.")
        print(f"Mean Center Error: {np.mean(errs):.2f} px")
        print(f"Median Center Error: {np.median(errs):.2f} px")


if __name__ == "__main__":
    main()
