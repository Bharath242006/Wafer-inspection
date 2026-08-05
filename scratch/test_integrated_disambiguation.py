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

# 1. DYNAMIC LATTICE PERIOD ESTIMATION
def estimate_dynamic_lattice_period(ref_img: np.ndarray) -> float:
    ref_s = cv2.resize(ref_img, (100, 100), cv2.INTER_AREA).astype(np.float32)
    ref_s -= np.mean(ref_s)
    f = np.fft.fft2(ref_s)
    autocorr = np.real(np.fft.ifft2(np.abs(f)**2))
    autocorr = np.fft.fftshift(autocorr)
    
    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[cy-2:cy+3, cx-2:cx+3] = 0.0

    _, _, _, max_loc = cv2.minMaxLoc(autocorr)
    p_dx = max_loc[0] - cx
    p_dy = max_loc[1] - cy
    period_tmpl = math.hypot(p_dx, p_dy)

    # In 1000x1000 search space with 100x100 template, period is period_tmpl * 10
    period_search = period_tmpl * 10.0
    return period_search if period_search > 20.0 else 67.0

dyn_period = estimate_dynamic_lattice_period(ref_raw)
print(f"Dynamically Estimated Lattice Period: {dyn_period:.2f} px")

# 2. MULTI-SCALE CORRELATION MAP GENERATION
scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]

search_grad = cv2.magnitude(cv2.Sobel(search_gray_f, cv2.CV_32F, 1, 0), cv2.Sobel(search_gray_f, cv2.CV_32F, 0, 1))
cv2.normalize(search_grad, search_grad, 0.0, 1.0, cv2.NORM_MINMAX)

ref_grad = cv2.magnitude(cv2.Sobel(ref_gray_f, cv2.CV_32F, 1, 0), cv2.Sobel(ref_gray_f, cv2.CV_32F, 0, 1))
cv2.normalize(ref_grad, ref_grad, 0.0, 1.0, cv2.NORM_MINMAX)

res_gray_maps = {}
res_grad_maps = {}
cand_peaks = []

for s in scales:
    sw = int(round(ref_raw.shape[1] * s))
    sh = int(round(ref_raw.shape[0] * s))

    s_ref_g = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)
    s_ref_d = cv2.resize(ref_grad, (sw, sh), cv2.INTER_AREA)

    rg = cv2.matchTemplate(search_gray_f, s_ref_g, cv2.TM_CCOEFF_NORMED)
    rd = cv2.matchTemplate(search_grad, s_ref_d, cv2.TM_CCOEFF_NORMED)

    res_gray_maps[s] = (rg, sw, sh, s_ref_g)
    res_grad_maps[s] = (rd, sw, sh, s_ref_d)

    # Extract peaks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(rg, kernel)
    peaks = (rg == dilated) & (rg > 0.02)
    py, px = np.where(peaks)

    for x, y in zip(px, py):
        cx = x + sw / 2.0
        cy = y + sh / 2.0
        cand_peaks.append((cx, cy, s, x, y, float(rg[y, x])))

# 3. INITIAL DENSE NMS TO EXTRACT TOP CANDIDATE LOCATIONS
# Filter candidates to top 40 unique spatial locations
cand_peaks.sort(key=lambda c: c[5], reverse=True)

top_candidates = []
for c in cand_peaks:
    cx, cy, s, x, y, score = c
    too_close = False
    for k in top_candidates:
        if math.hypot(cx - k['center_x'], cy - k['center_y']) < 12.0:
            too_close = True
            break
    if not too_close:
        top_candidates.append({
            'center_x': cx,
            'center_y': cy,
            'primary_scale': s,
            'primary_score': score,
            'dist_gt': math.hypot(cx - gt_x, cy - gt_y)
        })
    if len(top_candidates) >= 40:
        break

# 4. COMPUTE MULTI-SCALE INTEGRATED PYRAMID SCORE FOR EACH CANDIDATE
for cand in top_candidates:
    cx, cy = cand['center_x'], cand['center_y']

    integrated_gray_score = 0.0
    integrated_grad_score = 0.0
    valid_scale_cnt = 0

    for s in scales:
        rg, sw, sh, s_ref_g = res_gray_maps[s]
        rd, _, _, s_ref_d = res_grad_maps[s]

        tl_x = int(round(cx - sw / 2.0))
        tl_y = int(round(cy - sh / 2.0))

        if 0 <= tl_x < rg.shape[1] and 0 <= tl_y < rg.shape[0]:
            sg = float(rg[tl_y, tl_x])
            sd = float(rd[tl_y, tl_x])
            integrated_gray_score += sg
            integrated_grad_score += sd
            valid_scale_cnt += 1

    integrated_gray_score /= max(1, valid_scale_cnt)
    integrated_grad_score /= max(1, valid_scale_cnt)

    # Compute surrounding macro pyramid context (300x300 window)
    sw_01 = int(round(ref_raw.shape[1] * 0.10))
    sh_01 = int(round(ref_raw.shape[0] * 0.10))
    ctx_w = min(search_w, sw_01 * 3)
    ctx_h = min(search_h, sh_01 * 3)
    x1_c = max(0, int(round(cx - ctx_w / 2.0)))
    y1_c = max(0, int(round(cy - ctx_h / 2.0)))
    x2_c = min(search_w, int(round(cx + ctx_w / 2.0)))
    y2_c = min(search_h, int(round(cy + ctx_h / 2.0)))

    s_ctx = search_gray_f[y1_c:y2_c, x1_c:x2_c]
    r_ctx = cv2.resize(ref_gray_f, (x2_c - x1_c, y2_c - y1_c), cv2.INTER_AREA)

    s_ctx_p = cv2.resize(s_ctx, (30, 30), cv2.INTER_AREA)
    r_ctx_p = cv2.resize(r_ctx, (30, 30), cv2.INTER_AREA)

    macro_ctx_score = float(max(0.0, cv2.matchTemplate(s_ctx_p, r_ctx_p, cv2.TM_CCOEFF_NORMED)[0, 0]))

    # Final Combined Structural Score
    multi_scale_score = 0.60 * integrated_gray_score + 0.40 * integrated_grad_score
    cand['integrated_gray'] = integrated_gray_score
    cand['integrated_grad'] = integrated_grad_score
    cand['multi_scale_score'] = multi_scale_score
    cand['macro_ctx_score'] = macro_ctx_score
    cand['final_score'] = float(0.70 * multi_scale_score + 0.30 * macro_ctx_score)

# 5. PERIODIC ALIAS GROUP FORMATION & INTRA-GROUP DISAMBIGUATION
def is_lattice_alias(c1, c2, period=dyn_period, tol=12.0):
    dist = math.hypot(c1['center_x'] - c2['center_x'], c1['center_y'] - c2['center_y'])
    if dist < 12.0:
        return True
    k = round(dist / period)
    return 1 <= k <= 4 and abs(dist - k * period) <= tol

alias_groups = []
visited = set()

# Sort by final_score descending to pick group winners
top_candidates.sort(key=lambda c: c['final_score'], reverse=True)

for i, c in enumerate(top_candidates):
    if i in visited:
        continue
    group = [c]
    visited.add(i)
    for j in range(i + 1, len(top_candidates)):
        if j in visited:
            continue
        c_other = top_candidates[j]
        if any(is_lattice_alias(c_other, member) for member in group):
            group.append(c_other)
            visited.add(j)
    alias_groups.append(group)

# Intra-group winner selection
disambiguated_groups = []
for g_idx, group in enumerate(alias_groups, start=1):
    group.sort(key=lambda c: c['final_score'], reverse=True)
    winner = group[0]
    disambiguated_groups.append((g_idx, winner, group))

disambiguated_groups.sort(key=lambda item: item[1]['final_score'], reverse=True)
elapsed = time.perf_counter() - start_t

print(f"Execution Time: {elapsed*1000.0:.2f} ms ({elapsed:.4f} s)")

print("\n" + "=" * 105)
print("              INTEGRATED MULTI-SCALE PERIODIC-ALIAS DISAMBIGUATION REPORT")
print("=" * 105)

for g_idx, winner, group in disambiguated_groups[:4]:
    print(f"\nAlias Group #{g_idx} ({len(group)} candidates in group):")
    print(f"  {'Candidate Center (x,y)':<24} {'Multi-Scale Score':<20} {'Macro Ctx Score':<18} {'Final Score':<14} {'Dist to GT':<14}")
    print("  " + "-" * 100)
    for cand in group:
        is_win = " [WINNER]" if cand == winner else ""
        print(f"  ({cand['center_x']:.2f}, {cand['center_y']:.2f}){is_win:<9} {cand['multi_scale_score']:.4f}               {cand['macro_ctx_score']:.4f}             {cand['final_score']:.4f}         {cand['dist_gt']:.2f} px")

selected_coarse = disambiguated_groups[0][1]
print("\n" + "=" * 105)
print(f"SELECTED COARSE CANDIDATE: ({selected_coarse['center_x']:.2f}, {selected_coarse['center_y']:.2f})")
print(f"Ground Truth Coordinate:  ({gt_x:.2f}, {gt_y:.2f})")
print(f"Coarse Pixel Error:       {selected_coarse['dist_gt']:.2f} px")
print("=" * 105)
