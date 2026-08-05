import sys, os, math, time
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

start_t = time.perf_counter()

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

ref_gray_f = ref_raw.astype(np.float32)
search_gray_f = search_raw.astype(np.float32)

# Compute ZMUV Sobel Gradient Magnitude
def get_zmuv_sobel(img_f):
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_zero = mag - np.mean(mag)
    std_val = np.std(mag_zero)
    if std_val > 1e-5:
        mag_zero /= std_val
    return mag_zero

search_zgrad = get_zmuv_sobel(search_gray_f)
ref_zgrad = get_zmuv_sobel(ref_gray_f)

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]

res_zgrad_maps = {}
cand_peaks = []

for s in scales:
    sw = int(round(ref_raw.shape[1] * s))
    sh = int(round(ref_raw.shape[0] * s))

    s_ref_zg = cv2.resize(ref_zgrad, (sw, sh), cv2.INTER_AREA)
    s_ref_gray = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)

    rzg = cv2.matchTemplate(search_zgrad, s_ref_zg, cv2.TM_CCOEFF_NORMED)
    res_zgrad_maps[s] = (rzg, sw, sh, s_ref_zg, s_ref_gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(rzg, kernel)
    peaks = (rzg == dilated) & (rzg > 0.02)
    py, px = np.where(peaks)

    for x, y in zip(px, py):
        cx = x + sw / 2.0
        cy = y + sh / 2.0
        cand_peaks.append((cx, cy, s, x, y, float(rzg[y, x])))

# Dense Spatial Clustering to form spatial candidates
cand_peaks.sort(key=lambda c: c[5], reverse=True)

top_candidates = []
for c in cand_peaks:
    cx, cy, s, x, y, score = c
    too_close = False
    for k in top_candidates:
        if math.hypot(cx - k['center_x'], cy - k['center_y']) < 12.0:
            too_close = True
            k['scale_hits'] += 1
            k['multi_scale_zgrad'] += score
            break
    if not too_close:
        top_candidates.append({
            'center_x': cx,
            'center_y': cy,
            'scale_hits': 1,
            'multi_scale_zgrad': score,
            'dist_gt': math.hypot(cx - gt_x, cy - gt_y)
        })
    if len(top_candidates) >= 40:
        break

# Compute Phase Correlation & Scale Stability Score for top candidates
for cand in top_candidates:
    cx, cy = cand['center_x'], cand['center_y']

    phase_scores = []
    for s in scales:
        rzg, sw, sh, s_ref_zg, s_ref_g = res_zgrad_maps[s]
        tl_x = int(round(cx - sw / 2.0))
        tl_y = int(round(cy - sh / 2.0))

        if 0 <= tl_x and 0 <= tl_y and tl_x + sw <= search_w and tl_y + sh <= search_h:
            patch_g = search_gray_f[tl_y:tl_y+sh, tl_x:tl_x+sw]
            shift, phase_resp = cv2.phaseCorrelate(patch_g, s_ref_g)
            phase_scores.append(float(phase_resp))

    cand['avg_phase_resp'] = float(np.mean(phase_scores)) if phase_scores else 0.0
    cand['scale_stability'] = cand['scale_hits'] / float(len(scales))
    cand['avg_zgrad'] = cand['multi_scale_zgrad'] / float(cand['scale_hits'])

    cand['final_score'] = float(
        0.50 * cand['avg_zgrad'] +
        0.30 * cand['scale_stability'] +
        0.20 * max(0.0, cand['avg_phase_resp'])
    )

top_candidates.sort(key=lambda c: c['final_score'], reverse=True)
elapsed = time.perf_counter() - start_t

print(f"Execution Time: {elapsed*1000.0:.2f} ms ({elapsed:.4f} s)")

print("\n" + "=" * 110)
print("     ZMUV SOBEL GRADIENT + PHASE CORRELATION CANDIDATE RANKING DIAGNOSTIC")
print("=" * 110)
print(f"{'Rank':<6} {'Candidate Center (x,y)':<24} {'Avg ZGrad':<14} {'Scale Stability':<18} {'Avg Phase Resp':<16} {'Final Score':<12} {'Dist GT (px)':<12}")
print("-" * 110)
for idx, c in enumerate(top_candidates[:10], start=1):
    print(f"#{idx:<5} ({c['center_x']:.2f}, {c['center_y']:.2f})           {c['avg_zgrad']:.4f}         {c['scale_stability']:.4f}             {c['avg_phase_resp']:.4f}           {c['final_score']:.4f}        {c['dist_gt']:.2f} px")
print("=" * 110)
