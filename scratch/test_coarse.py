import csv
import math
import os
import sys
import cv2
import numpy as np

ref_dir = 'dataset/validation/reference'
search_dir = 'dataset/validation/search'
csv_path = 'dataset/validation/labels.csv'

records = []
with open(csv_path, 'r', encoding='utf-8') as f:
    records = list(csv.DictReader(f))

coarse_errors = []

for item in records[:15]:
    img_name = item['image']
    gt_x = float(item['x'])
    gt_y = float(item['y'])
    style = item['style']

    ref = cv2.imread(os.path.join(ref_dir, img_name), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(os.path.join(search_dir, img_name), cv2.IMREAD_GRAYSCALE)

    sw, sh = 100, 100
    s_ref = cv2.resize(ref, (sw, sh), cv2.INTER_AREA)

    # Compute macro density / low-frequency representations
    # 1. Intensity downsampled / low-pass
    ref_blur = cv2.GaussianBlur(s_ref, (31, 31), 10.0)
    search_blur = cv2.GaussianBlur(search, (61, 61), 15.0)

    # 2. Gradient magnitude envelope
    ref_sobel_x = cv2.Sobel(s_ref, cv2.CV_32F, 1, 0, ksize=3)
    ref_sobel_y = cv2.Sobel(s_ref, cv2.CV_32F, 0, 1, ksize=3)
    ref_grad_mag = cv2.magnitude(ref_sobel_x, ref_sobel_y)
    ref_grad_blur = cv2.GaussianBlur(ref_grad_mag, (31, 31), 10.0)

    search_sobel_x = cv2.Sobel(search, cv2.CV_32F, 1, 0, ksize=3)
    search_sobel_y = cv2.Sobel(search, cv2.CV_32F, 0, 1, ksize=3)
    search_grad_mag = cv2.magnitude(search_sobel_x, search_sobel_y)
    search_grad_blur = cv2.GaussianBlur(search_grad_mag, (61, 61), 15.0)

    res_b = cv2.matchTemplate(search_blur, ref_blur, cv2.TM_CCOEFF_NORMED)
    res_g = cv2.matchTemplate(search_grad_blur, ref_grad_blur, cv2.TM_CCOEFF_NORMED)

    res_macro = 0.5 * res_b + 0.5 * res_g
    _, max_v, _, max_l = cv2.minMaxLoc(res_macro)

    cx = max_l[0] + sw / 2.0
    cy = max_l[1] + sh / 2.0
    err = math.hypot(cx - gt_x, cy - gt_y)
    coarse_errors.append(err)
    print(f"{img_name} ({style}) -> Coarse: ({cx:.1f}, {cy:.1f}), GT: ({gt_x:.1f}, {gt_y:.1f}), Err: {err:.2f} px")

print(f"\nMean Coarse Error (first 15): {np.mean(coarse_errors):.2f} px, Median: {np.median(coarse_errors):.2f} px")
