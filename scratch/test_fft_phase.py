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

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]
raw_candidates = []

for s in scales:
    scaled_w = int(round(ref_raw.shape[1] * s))
    scaled_h = int(round(ref_raw.shape[0] * s))

    s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    res_gray = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(res_gray, kernel)
    peaks = (res_gray == dilated) & (res_gray > 0.05)
    py, px = np.where(peaks)

    for tl_x, tl_y in zip(px, py):
        if tl_x < 0 or tl_y < 0 or tl_x + scaled_w > search_w or tl_y + scaled_h > search_h:
            continue

        cx = tl_x + (scaled_w / 2.0)
        cy = tl_y + (scaled_h / 2.0)

        score_g = float(res_gray[tl_y, tl_x])
        dist_gt = math.hypot(cx - gt_x, cy - gt_y)

        raw_candidates.append({
            'center_x': cx,
            'center_y': cy,
            'top_left': (tl_x, tl_y),
            'scaled_w': scaled_w,
            'scaled_h': scaled_h,
            'scale': s,
            's_ref_gray': s_ref_gray,
            'score_gray': score_g,
            'dist_gt': dist_gt
        })

# Apply NMS to get top 30 candidates
raw_candidates.sort(key=lambda c: c['score_gray'], reverse=True)

top_nms = []
for c in raw_candidates:
    too_close = False
    for k in top_nms:
        if math.hypot(c['center_x'] - k['center_x'], c['center_y'] - k['center_y']) < 15.0:
            too_close = True
            break
    if not too_close:
        top_nms.append(c)
    if len(top_nms) >= 30:
        break

# Compute FFT Phase Correlation ONLY on top 30 NMS candidates
for cand in top_nms:
    tl_x, tl_y = cand['top_left']
    sw, sh = cand['scaled_w'], cand['scaled_h']
    s_ref = cand['s_ref_gray']
    patch_g = search_gray_f[tl_y:tl_y+sh, tl_x:tl_x+sw]

    shift, phase_resp = cv2.phaseCorrelate(patch_g, s_ref)
    cand['phase_resp'] = float(phase_resp)
    cand['combined_score'] = float(0.5 * cand['score_gray'] + 0.5 * max(0.0, float(phase_resp)))

top_nms.sort(key=lambda c: c['combined_score'], reverse=True)
elapsed = time.perf_counter() - start_t

print(f"Execution Time: {elapsed*1000.0:.2f} ms ({elapsed:.4f} s)")
print("\nTop 10 Candidates Ranked by FFT Phase-Correlation Enhanced Score:")
for idx, c in enumerate(top_nms[:10], start=1):
    print(f"  Rank {idx:02d}: Center=({c['center_x']:.2f}, {c['center_y']:.2f}) | Dist to GT={c['dist_gt']:.2f}px | Scale={c['scale']:.3f} | Match={c['score_gray']:.4f} | PhaseResp={c['phase_resp']:.4f} | Combined={c['combined_score']:.4f}")
