import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

# Downscale reference image (1000x1000) to 100x100 template
s_ref = cv2.resize(ref_raw, (100, 100), cv2.INTER_AREA)

# Cell A: True cell candidate at (633.26, 675.77) -> top-left = (583, 625)
tl_a_x, tl_a_y = 583, 625
crop_a = search_raw[tl_a_y:tl_a_y+100, tl_a_x:tl_a_x+100]

# Cell B: Alias cell candidate at (700.26, 673.77) -> top-left = (650, 623)
tl_b_x, tl_b_y = 650, 623
crop_b = search_raw[tl_b_y:tl_b_y+100, tl_b_x:tl_b_x+100]

print("=== CELL FEATURE COMPARISON ===")
print(f"Ref Template 100x100: mean={np.mean(s_ref):.2f}, std={np.std(s_ref):.2f}")
print(f"Cell A (True, dist 3.16px):  mean={np.mean(crop_a):.2f}, std={np.std(crop_a):.2f}")
print(f"Cell B (Alias, dist 64.11px): mean={np.mean(crop_b):.2f}, std={np.std(crop_b):.2f}")

# 1. Compare Sobel Gradient Histograms
def get_grad_mag(img):
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return mag

grad_ref = get_grad_mag(s_ref)
grad_a = get_grad_mag(crop_a)
grad_b = get_grad_mag(crop_b)

print(f"\nSobel Gradient Mean & Std:")
print(f"Ref Grad:    mean={np.mean(grad_ref):.2f}, std={np.std(grad_ref):.2f}")
print(f"Cell A Grad: mean={np.mean(grad_a):.2f}, std={np.std(grad_a):.2f}")
print(f"Cell B Grad: mean={np.mean(grad_b):.2f}, std={np.std(grad_b):.2f}")

# 2. Compare 2D Spatial Auto-Correlation / Correlation Response Profile
res_a = cv2.matchTemplate(crop_a.astype(np.float32), s_ref.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]
res_b = cv2.matchTemplate(crop_b.astype(np.float32), s_ref.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]

print(f"\nTM_CCOEFF_NORMED Scores:")
print(f"Cell A (True):  {res_a:.4f}")
print(f"Cell B (Alias): {res_b:.4f}")

# 3. Compare surrounding context correlation at 200x200 vs 300x300
ctx_a_200 = search_raw[tl_a_y-50:tl_a_y+150, tl_a_x-50:tl_a_x+150]
ctx_b_200 = search_raw[tl_b_y-50:tl_b_y+150, tl_b_x-50:tl_b_x+150]
ref_200 = cv2.resize(ref_raw, (200, 200), cv2.INTER_AREA)

res_ctx_a = cv2.matchTemplate(ctx_a_200.astype(np.float32), ref_200.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]
res_ctx_b = cv2.matchTemplate(ctx_b_200.astype(np.float32), ref_200.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]

print(f"\n200x200 Surrounding Context Match Scores:")
print(f"Cell A Context: {res_ctx_a:.4f}")
print(f"Cell B Context: {res_ctx_b:.4f}")
