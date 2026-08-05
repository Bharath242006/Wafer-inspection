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

def compute_sobel_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    cv2.normalize(mag, mag, 0.0, 1.0, cv2.NORM_MINMAX)
    return mag

ref_grad = compute_sobel_gradient(ref_gray_f)
search_grad = compute_sobel_gradient(search_gray_f)

ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)
search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]

cand_peaks = []

def extract_local_peaks(response_map, window_size=5, min_thresh=0.01, top_k=50):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dilated = cv2.dilate(response_map, kernel)
    peaks = (response_map == dilated) & (response_map >= min_thresh)
    py, px = np.where(peaks)
    scores = response_map[py, px]
    if len(scores) == 0:
        return []
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(px[i]), int(py[i]), float(scores[i])) for i in top_idx]

for s in scales:
    sw = int(round(ref_raw.shape[1] * s))
    sh = int(round(ref_raw.shape[0] * s))

    s_ref_g = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)
    s_ref_d = cv2.resize(ref_grad, (sw, sh), cv2.INTER_AREA)
    s_ref_l = cv2.resize(ref_log, (sw, sh), cv2.INTER_AREA)

    rg = cv2.matchTemplate(search_gray_f, s_ref_g, cv2.TM_CCOEFF_NORMED)
    rd = cv2.matchTemplate(search_grad, s_ref_d, cv2.TM_CCOEFF_NORMED)
    rl = cv2.matchTemplate(search_log, s_ref_l, cv2.TM_CCOEFF_NORMED)

    peaks_g = extract_local_peaks(rg, window_size=5, min_thresh=0.01, top_k=50)
    peaks_d = extract_local_peaks(rd, window_size=5, min_thresh=0.01, top_k=50)
    peaks_l = extract_local_peaks(rl, window_size=5, min_thresh=0.01, top_k=50)

    peak_locs = set([(x, y) for x, y, _ in peaks_g] + [(x, y) for x, y, _ in peaks_d] + [(x, y) for x, y, _ in peaks_l])

    for tl_x, tl_y in peak_locs:
        cx = tl_x + sw / 2.0
        cy = tl_y + sh / 2.0

        if cx < 50.0 or cy < 50.0 or cx > (search_w - 50.0) or cy > (search_h - 50.0):
            continue

        sg = float(rg[tl_y, tl_x]) if 0 <= tl_y < rg.shape[0] and 0 <= tl_x < rg.shape[1] else 0.0
        sd = float(rd[tl_y, tl_x]) if 0 <= tl_y < rd.shape[0] and 0 <= tl_x < rd.shape[1] else 0.0
        sl = float(rl[tl_y, tl_x]) if 0 <= tl_y < rl.shape[0] and 0 <= tl_x < rl.shape[1] else 0.0

        score = 0.40 * sg + 0.40 * sd + 0.20 * sl
        cand_peaks.append((cx, cy, s, score, sg, sd, sl))

cand_peaks.sort(key=lambda c: c[3], reverse=True)

top_candidates = []
for c in cand_peaks:
    cx, cy, s, score, sg, sd, sl = c
    too_close = False
    for k in top_candidates:
        if math.hypot(cx - k['cx'], cy - k['cy']) < 12.0:
            too_close = True
            break
    if not too_close:
        top_candidates.append({'cx': cx, 'cy': cy, 's': s, 'score': score, 'sg': sg, 'sd': sd, 'sl': sl, 'd_gt': math.hypot(cx - gt_x, cy - gt_y)})
    if len(top_candidates) >= 60:
        break

print("=" * 105)
print("     STAGE 1 EXTRACTED CANDIDATE POOL FOR 00001.PNG")
print("=" * 105)
print(f"{'Rank':<6} {'Center (x,y)':<24} {'Scale':<7} {'Raw Score':<12} {'Dist GT (px)':<14}")
print("-" * 105)

for rank, c in enumerate(top_candidates, start=1):
    is_gt = " <--- TRUE CELL CANDIDATE!" if c['d_gt'] < 15.0 else ""
    print(f"#{rank:<5} ({c['cx']:.2f}, {c['cy']:.2f}){is_gt:<28} {c['s']:.3f}   {c['score']:.4f}        {c['d_gt']:.2f} px")

print("=" * 105)
