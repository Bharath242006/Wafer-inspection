import sys, os, math, time
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

start_time = time.perf_counter()

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

ref_gray_f = ref_raw.astype(np.float32)
search_gray_f = search_raw.astype(np.float32)

def compute_sobel_gradient(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    sobelx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    cv2.normalize(mag, mag, 0.0, 1.0, cv2.NORM_MINMAX)
    return mag

ref_grad = compute_sobel_gradient(ref_raw)
search_grad = compute_sobel_gradient(search_raw)

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]
raw_candidates = []

for s in scales:
    scaled_w = int(round(ref_raw.shape[1] * s))
    scaled_h = int(round(ref_raw.shape[0] * s))

    s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    res_gray = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
    res_grad = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(res_gray, kernel)
    peaks = (res_gray == dilated) & (res_gray > 0.05)
    py, px = np.where(peaks)

    ref_mean = float(np.mean(s_ref_gray))
    ref_std = float(np.std(s_ref_gray))

    for tl_x, tl_y in zip(px, py):
        cx = tl_x + (scaled_w / 2.0)
        cy = tl_y + (scaled_h / 2.0)

        score_g = float(res_gray[tl_y, tl_x])
        score_d = float(res_grad[tl_y, tl_x]) if 0 <= tl_y < res_grad.shape[0] and 0 <= tl_x < res_grad.shape[1] else 0.0

        patch_g = search_gray_f[tl_y:tl_y+scaled_h, tl_x:tl_x+scaled_w]
        patch_mean = float(np.mean(patch_g))
        patch_std = float(np.std(patch_g))

        std_match = (2.0 * ref_std * patch_std) / (ref_std**2 + patch_std**2 + 1e-5)
        mean_match = max(0.0, 1.0 - abs(patch_mean - ref_mean) / 255.0)

        raw_match = 0.5 * score_g + 0.5 * score_d
        dist_gt = math.hypot(cx - gt_x, cy - gt_y)

        raw_candidates.append({
            'center_x': cx,
            'center_y': cy,
            'top_left': (tl_x, tl_y),
            'scaled_w': scaled_w,
            'scaled_h': scaled_h,
            'scale': s,
            'raw_match': float(raw_match),
            'std_match': float(std_match),
            'mean_match': float(mean_match),
            'score_gray': score_g,
            'score_grad': score_d,
            'dist_gt': dist_gt
        })

# Initial NMS to reduce to top 30 candidate locations
raw_candidates.sort(key=lambda c: c['raw_match'], reverse=True)

top_nms = []
for c in raw_candidates:
    too_close = False
    for k in top_nms:
        if math.hypot(c['center_x'] - k['center_x'], c['center_y'] - k['center_y']) < 15.0:
            too_close = True
            break
    if not too_close:
        top_nms.append(c)
    if len(top_nms) >= 35:
        break

# Compute context macro anchor for top 35 NMS candidates
for cand in top_nms:
    cx, cy = cand['center_x'], cand['center_y']
    sw, sh = cand['scaled_w'], cand['scaled_h']
    ctx_w = min(search_w, int(round(sw * 2.5)))
    ctx_h = min(search_h, int(round(sh * 2.5)))
    x1_c = max(0, int(round(cx - ctx_w / 2.0)))
    y1_c = max(0, int(round(cy - ctx_h / 2.0)))
    x2_c = min(search_w, int(round(cx + ctx_w / 2.0)))
    y2_c = min(search_h, int(round(cy + ctx_h / 2.0)))

    s_ctx = search_gray_f[y1_c:y2_c, x1_c:x2_c]
    r_ctx = cv2.resize(ref_gray_f, (x2_c - x1_c, y2_c - y1_c), cv2.INTER_AREA)

    ctx_ncc = float(max(0.0, cv2.matchTemplate(s_ctx, r_ctx, cv2.TM_CCOEFF_NORMED)[0, 0]))
    cand['macro_anchor'] = float(0.5 * ctx_ncc + 0.3 * cand['std_match'] + 0.2 * cand['mean_match'])
    cand['combined_score'] = float(0.5 * cand['raw_match'] + 0.5 * cand['macro_anchor'])

# Periodic Lattice Alias Disambiguation
def is_lattice_alias(c1, c2, lattice_period=67.0, tol=12.0):
    dist = math.hypot(c1['center_x'] - c2['center_x'], c1['center_y'] - c2['center_y'])
    if dist < 15.0:
        return True
    k = round(dist / lattice_period)
    return k >= 1 and abs(dist - k * lattice_period) <= tol

# Sort candidates by combined score
top_nms.sort(key=lambda c: c['combined_score'], reverse=True)

disambiguated = []
for cand in top_nms:
    # Check if cand is a periodic lattice alias of an ALREADY KEPT higher-scoring candidate
    # BUT only suppress cand if kept candidate has a STRONGER or EQUAL macro anchor score!
    alias_suppressed = False
    for kept in disambiguated:
        if is_lattice_alias(cand, kept):
            alias_suppressed = True
            break
    if not alias_suppressed:
        disambiguated.append(cand)

elapsed = time.perf_counter() - start_time

print(f"Execution Time: {elapsed*1000.0:.2f} ms ({elapsed:.4f} s)")
print("\nTop 10 Candidates BEFORE Alias Suppression:")
for idx, c in enumerate(top_nms[:10], start=1):
    print(f"  Before Rank {idx:02d}: Center=({c['center_x']:.2f}, {c['center_y']:.2f}) | Dist to GT={c['dist_gt']:.2f}px | Scale={c['scale']:.3f} | RawMatch={c['raw_match']:.4f} | Anchor={c['macro_anchor']:.4f} | Combined={c['combined_score']:.4f}")

print("\nTop 10 Candidates AFTER Periodic Alias Disambiguation:")
for idx, c in enumerate(disambiguated[:10], start=1):
    print(f"  After Rank {idx:02d}:  Center=({c['center_x']:.2f}, {c['center_y']:.2f}) | Dist to GT={c['dist_gt']:.2f}px | Scale={c['scale']:.3f} | RawMatch={c['raw_match']:.4f} | Anchor={c['macro_anchor']:.4f} | Combined={c['combined_score']:.4f}")
