import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

# Crop 300x300 around GT (636.26, 676.77)
gt_crop = search_raw[int(gt_y-150):int(gt_y+150), int(gt_x-150):int(gt_x+150)]

# Downscale reference to 100x100
scaled_ref = cv2.resize(ref_raw, (100, 100), cv2.INTER_AREA)

# Template match scaled_ref inside the 300x300 GT crop
res = cv2.matchTemplate(gt_crop, scaled_ref, cv2.TM_CCOEFF_NORMED)

# Plot response map inside GT crop
print(f"Match Response Map shape inside 300x300 GT crop: {res.shape}") # (201, 201)

# Find local maxima inside GT crop response map
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dilated = cv2.dilate(res, kernel)
peaks = (res == dilated) & (res > 0.10)
py, px = np.where(peaks)

print("\nLocal Peaks inside 300x300 GT Crop:")
for x, y in zip(px, py):
    # Candidate center in full search image coordinates
    # gt_crop top-left in search = (gt_x - 150, gt_y - 150)
    # top-left of match in search = (gt_x - 150 + x, gt_y - 150 + y)
    # center of match in search = (gt_x - 150 + x + 50, gt_y - 150 + y + 50) = (gt_x - 100 + x, gt_y - 100 + y)
    cand_cx = gt_x - 100 + x
    cand_cy = gt_y - 100 + y
    score = res[y, x]
    dist_gt = math.hypot(cand_cx - gt_x, cand_cy - gt_y)
    print(f"  Peak at Crop({x:3d}, {y:3d}) -> Search Center: ({cand_cx:6.2f}, {cand_cy:6.2f}) | Score: {score:.4f} | Dist to GT: {dist_gt:6.2f} px")
