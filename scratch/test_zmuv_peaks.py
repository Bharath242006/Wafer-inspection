import sys, os, math, time
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

ref_gray_f = ref_raw.astype(np.float32)
search_gray_f = search_raw.astype(np.float32)

# Compute ZMUV Correlation Map directly via OpenCV or sliding window
sw, sh = 100, 100
s_ref = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)

# Subtract mean from reference template
t_zero = s_ref - np.mean(s_ref)
t_std = np.std(t_zero)

# Template match with TM_CCOEFF_NORMED is mathematically ZMUV NCC!
rg = cv2.matchTemplate(search_gray_f, t_zero, cv2.TM_CCOEFF_NORMED)

# Extract top 60 peaks from ZMUV correlation map
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(rg, kernel)
peaks = (rg == dilated) & (rg > 0.02)
py, px = np.where(peaks)
scores = rg[py, px]

top_idx = np.argsort(scores)[::-1][:60]

print("=" * 105)
print("     TOP ZMUV NCC PEAKS EXTRACTED FROM SEARCH IMAGE FOR 00001.PNG")
print("=" * 105)
print(f"{'Rank':<6} {'Center (x,y)':<24} {'ZMUV NCC Score':<18} {'Dist GT (px)':<14}")
print("-" * 105)

for rank, idx in enumerate(top_idx, start=1):
    cx = px[idx] + sw / 2.0
    cy = py[idx] + sh / 2.0
    d_gt = math.hypot(cx - gt_x, cy - gt_y)
    is_gt = " <--- TRUE CELL CANDIDATE!" if d_gt < 15.0 else ""
    print(f"#{rank:<5} ({cx:.2f}, {cy:.2f}){is_gt:<28} {scores[idx]:.4f}             {d_gt:.2f} px")

print("=" * 105)
