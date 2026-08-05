import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

# 1. Inspect search image at GT (636.26, 676.77) vs (703, 674) vs (561, 836)
sw, sh = 100, 100
s_ref = cv2.resize(ref_raw, (sw, sh), cv2.INTER_AREA)

gt_tl_x = int(round(gt_x - 50))  # 586
gt_tl_y = int(round(gt_y - 50))  # 627

gt_patch = search_raw[gt_tl_y:gt_tl_y+100, gt_tl_x:gt_tl_x+100]
alias1_patch = search_raw[624:724, 653:753]  # (703, 674)
alias2_patch = search_raw[786:886, 511:611]  # (561, 836)

print(f"Ref 100x100 mean={np.mean(s_ref):.2f}, std={np.std(s_ref):.2f}")
print(f"GT Patch (636, 676) mean={np.mean(gt_patch):.2f}, std={np.std(gt_patch):.2f}")
print(f"Alias 1 (703, 674)  mean={np.mean(alias1_patch):.2f}, std={np.std(alias1_patch):.2f}")
print(f"Alias 2 (561, 836)  mean={np.mean(alias2_patch):.2f}, std={np.std(alias2_patch):.2f}")

# 2. Test Normalized Cross Correlation with zero-mean unit-variance
def ncc_zmuv(p1, p2):
    p1_f = p1.astype(np.float32) - np.mean(p1)
    p2_f = p2.astype(np.float32) - np.mean(p2)
    s1 = np.std(p1_f)
    s2 = np.std(p2_f)
    if s1 > 0 and s2 > 0:
        return float(np.mean(p1_f * p2_f) / (s1 * s2))
    return 0.0

print(f"\nZero-Mean Unit-Variance NCC Scores:")
print(f"GT Patch vs Ref:      {ncc_zmuv(gt_patch, s_ref):.4f}")
print(f"Alias 1 (703, 674):   {ncc_zmuv(alias1_patch, s_ref):.4f}")
print(f"Alias 2 (561, 836):   {ncc_zmuv(alias2_patch, s_ref):.4f}")

# 3. Test Phase Correlation (FFT) on crop vs reference
s_ref_f = s_ref.astype(np.float32)
gt_patch_f = gt_patch.astype(np.float32)
alias1_patch_f = alias1_patch.astype(np.float32)

shift_gt, resp_gt = cv2.phaseCorrelate(gt_patch_f, s_ref_f)
shift_a1, resp_a1 = cv2.phaseCorrelate(alias1_patch_f, s_ref_f)

print(f"\nFFT Phase Correlation Response:")
print(f"GT Patch:      response={resp_gt:.4f}, subpixel shift={shift_gt}")
print(f"Alias 1:       response={resp_a1:.4f}, subpixel shift={shift_a1}")

# 4. Test SIFT on full high-res reference image (1000x1000) vs search image (1000x1000)
# Notice: reference image is 1000x1000 at 10x magnification.
# If we downscale reference image to 100x100, SIFT has 100x100 keypoints.
# What if we detect SIFT keypoints on high-res reference (1000x1000) and scale keypoint coordinates by 0.10?
sift = cv2.SIFT_create(nfeatures=500)
kp_ref, des_ref = sift.detectAndCompute(ref_raw, None)
kp_search, des_search = sift.detectAndCompute(search_raw, None)

print(f"\nSIFT Keypoints found: Ref={len(kp_ref)}, Search={len(kp_search)}")

if des_ref is not None and des_search is not None:
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_search, k=2)

    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.70 * n.distance:
                good.append(m)

    print(f"Lowe ratio test good matches: {len(good)}")
    if len(good) >= 4:
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_search = np.float32([kp_search[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Scale pts_ref by 0.10 to map to search image space
        # Target in search = pts_search - pts_ref * 0.10 + (50, 50)
        est_centers = pts_search[:, 0, :] - pts_ref[:, 0, :] * 0.10 + np.array([50.0, 50.0])

        med_center_x = float(np.median(est_centers[:, 0]))
        med_center_y = float(np.median(est_centers[:, 1]))
        err_sift = math.hypot(med_center_x - gt_x, med_center_y - gt_y)

        print(f"SIFT RANSAC Estimated Center: ({med_center_x:.2f}, {med_center_y:.2f}) | Error to GT: {err_sift:.2f} px")
