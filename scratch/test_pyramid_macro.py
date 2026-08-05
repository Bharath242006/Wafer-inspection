import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

# Downscale search to 250x250 (4x) and reference to 25x25 (10x scaled, 4x pyramid)
search_p = cv2.resize(search_raw, (250, 250), cv2.INTER_AREA)

# Downscale reference to 25x25 template
ref_s = cv2.resize(ref_raw, (100, 100), cv2.INTER_AREA)
ref_p = cv2.resize(ref_s, (25, 25), cv2.INTER_AREA)

# Blur pyramid images to leave only macroscopic envelope
search_p_blur = cv2.GaussianBlur(search_p, (21, 21), 5.0)
ref_p_blur = cv2.GaussianBlur(ref_p, (9, 9), 2.0)

# Sobel magnitude on pyramid level
sobelx = cv2.Sobel(search_p_blur.astype(np.float32), cv2.CV_32F, 1, 0)
sobely = cv2.Sobel(search_p_blur.astype(np.float32), cv2.CV_32F, 0, 1)
search_p_grad = cv2.magnitude(sobelx, sobely)

sobelx_r = cv2.Sobel(ref_p_blur.astype(np.float32), cv2.CV_32F, 1, 0)
sobely_r = cv2.Sobel(ref_p_blur.astype(np.float32), cv2.CV_32F, 0, 1)
ref_p_grad = cv2.magnitude(sobelx_r, sobely_r)

res_p_int = cv2.matchTemplate(search_p_blur.astype(np.float32), ref_p_blur.astype(np.float32), cv2.TM_CCOEFF_NORMED)
res_p_grad = cv2.matchTemplate(search_p_grad, ref_p_grad, cv2.TM_CCOEFF_NORMED)

res_macro = 0.5 * res_p_int + 0.5 * res_p_grad
min_v, max_v, min_l, max_l = cv2.minMaxLoc(res_macro)

# Convert P2 location back to full resolution (1000x1000)
# Top-left max_l in 250x250 -> center in 250x250 = (max_l[0] + 12.5, max_l[1] + 12.5)
# Center in 1000x1000 = (max_l[0] + 12.5) * 4.0, (max_l[1] + 12.5) * 4.0
cx = (max_l[0] + 12.5) * 4.0
cy = (max_l[1] + 12.5) * 4.0

err = math.hypot(cx - gt_x, cy - gt_y)

print(f"Pyramid Level 2 Macro Coarse Location: ({cx:.2f}, {cy:.2f})")
print(f"Ground Truth: ({gt_x:.2f}, {gt_y:.2f})")
print(f"Coarse Pyramid Error: {err:.2f} px")
