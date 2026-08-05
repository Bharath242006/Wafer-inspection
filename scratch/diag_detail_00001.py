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

# 1. Dynamic Lattice Period Estimation from 2D Autocorrelation of Reference Image
def estimate_lattice_period(img: np.ndarray) -> float:
    # Downsample img to ~100x100 if large
    if img.shape[0] > 200:
        img_s = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        img_s = img.copy()

    img_f = img_s.astype(np.float32) - np.mean(img_s)
    # Compute 2D Autocorrelation via FFT
    f = np.fft.fft2(img_f)
    power = np.abs(f)**2
    autocorr = np.real(np.fft.ifft2(power))
    autocorr = np.fft.fftshift(autocorr)
    
    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    # Zero out central 5x5 region
    autocorr[cy-3:cy+4, cx-3:cx+4] = 0.0

    # Find secondary peak distance in autocorr
    _, max_val, _, max_loc = cv2.minMaxLoc(autocorr)
    peak_dx = max_loc[0] - cx
    peak_dy = max_loc[1] - cy
    period_in_template = math.hypot(peak_dx, peak_dy)

    # Scale back up to 1000x1000 search image scale (where template is 100x100)
    # If period in 100x100 template is period_in_template, in 1000x1000 search it is period_in_template * 10
    period_search = period_in_template * (img.shape[0] / img_s.shape[0]) * 10.0
    return float(period_search)

estimated_period = estimate_lattice_period(ref_raw)
print(f"Dynamically Estimated Lattice Period for 00001.png: {estimated_period:.2f} px")

# 2. Extract Candidates Across Multi-Scale Templates
scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]

search_grad = cv2.magnitude(cv2.Sobel(search_gray_f, cv2.CV_32F, 1, 0), cv2.Sobel(search_gray_f, cv2.CV_32F, 0, 1))
cv2.normalize(search_grad, search_grad, 0.0, 1.0, cv2.NORM_MINMAX)

ref_grad = cv2.magnitude(cv2.Sobel(ref_gray_f, cv2.CV_32F, 1, 0), cv2.Sobel(ref_gray_f, cv2.CV_32F, 0, 1))
cv2.normalize(ref_grad, ref_grad, 0.0, 1.0, cv2.NORM_MINMAX)

search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)
ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)

target_candidates_info = []

for s in scales:
    sw = int(round(ref_raw.shape[1] * s))
    sh = int(round(ref_raw.shape[0] * s))

    s_ref_gray = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)
    s_ref_grad = cv2.resize(ref_grad, (sw, sh), cv2.INTER_AREA)
    s_ref_log = cv2.resize(ref_log, (sw, sh), cv2.INTER_AREA)

    res_g = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
    res_d = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)
    res_l = cv2.matchTemplate(search_log, s_ref_log, cv2.TM_CCOEFF_NORMED)

    # Check match scores specifically for targets of interest:
    # Target 1: (642, 676) -> True Candidate
    # Target 2: (573, 624) -> Current Selected (-67 px)
    # Target 3: (703, 674) -> Alias Candidate (+67 px)
    # Target 4: (691, 688) -> Alias Candidate (+55 px)

    test_targets = [
        ("True-Near (642, 676)", 642.0, 676.0),
        ("Current Selected (573, 624)", 573.0, 624.0),
        ("Alias +67px (703, 674)", 703.0, 674.0),
        ("Alias +55px (691, 688)", 691.0, 688.0)
    ]

    for label, tx, ty in test_targets:
        tl_x = int(round(tx - sw / 2.0))
        tl_y = int(round(ty - sh / 2.0))

        if 0 <= tl_x < res_g.shape[1] and 0 <= tl_y < res_g.shape[0]:
            sg = float(res_g[tl_y, tl_x])
            sd = float(res_d[tl_y, tl_x])
            sl = float(res_l[tl_y, tl_x])

            # ZMUV Correlation on patch vs s_ref_gray
            patch_g = search_gray_f[tl_y:tl_y+sh, tl_x:tl_x+sw]
            patch_f = patch_g - np.mean(patch_g)
            ref_f = s_ref_gray - np.mean(s_ref_gray)
            zmuv_score = float(np.mean(patch_f * ref_f) / (np.std(patch_f) * np.std(ref_f) + 1e-5))

            dist_gt = math.hypot(tx - gt_x, ty - gt_y)

            target_candidates_info.append({
                'label': label,
                'scale': s,
                'center_x': tx,
                'center_y': ty,
                'top_left': (tl_x, tl_y),
                'score_gray': sg,
                'score_grad': sd,
                'score_log': sl,
                'score_zmuv': zmuv_score,
                'dist_gt': dist_gt
            })

print("\n" + "=" * 110)
print("     DETAILED SCORE BREAKDOWN FOR KEY CANDIDATE TARGETS ON 00001.PNG")
print("=" * 110)
print(f"{'Target Label':<28} {'Scale':<7} {'Gray Match':<12} {'Grad Match':<12} {'LoG Match':<12} {'ZMUV NCC':<12} {'Dist GT':<10}")
print("-" * 110)
for info in target_candidates_info:
    print(f"{info['label']:<28} {info['scale']:.3f}   {info['score_gray']:<12.4f} {info['score_grad']:<12.4f} {info['score_log']:<12.4f} {info['score_zmuv']:<12.4f} {info['dist_gt']:.2f} px")
print("=" * 110)
