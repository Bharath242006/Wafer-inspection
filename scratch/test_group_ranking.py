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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(res_gray, kernel)
    peaks = (res_gray == dilated) & (res_gray > 0.02)
    py, px = np.where(peaks)

    ref_mean = float(np.mean(s_ref_gray))
    ref_std = float(np.std(s_ref_gray))

    for tl_x, tl_y in zip(px, py):
        if tl_x < 0 or tl_y < 0 or tl_x + scaled_w > search_w or tl_y + scaled_h > search_h:
            continue

        cx = tl_x + (scaled_w / 2.0)
        cy = tl_y + (scaled_h / 2.0)

        score_g = float(res_gray[tl_y, tl_x])
        patch_g = search_gray_f[tl_y:tl_y+scaled_h, tl_x:tl_x+scaled_w]

        patch_mean = float(np.mean(patch_g))
        patch_std = float(np.std(patch_g))

        std_match = (2.0 * ref_std * patch_std) / (ref_std**2 + patch_std**2 + 1e-5)
        mean_match = max(0.0, 1.0 - abs(patch_mean - ref_mean) / 255.0)

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
            'std_match': std_match,
            'mean_match': mean_match,
            'dist_gt': dist_gt
        })

# Initial NMS
raw_candidates.sort(key=lambda c: c['score_gray'], reverse=True)
top_candidates = []
for c in raw_candidates:
    too_close = False
    for k in top_candidates:
        if math.hypot(c['center_x'] - k['center_x'], c['center_y'] - k['center_y']) < 15.0:
            too_close = True
            break
    if not too_close:
        top_candidates.append(c)
    if len(top_candidates) >= 40:
        break

# Compute surrounding context and ZMUV score for top 40 candidates
for cand in top_candidates:
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

    s_ctx_p = cv2.resize(s_ctx, (25, 25), cv2.INTER_AREA)
    r_ctx_p = cv2.resize(r_ctx, (25, 25), cv2.INTER_AREA)

    pyramid_ncc = float(max(0.0, cv2.matchTemplate(s_ctx_p, r_ctx_p, cv2.TM_CCOEFF_NORMED)[0, 0]))

    cand['pyramid_ncc'] = pyramid_ncc
    cand['macro_score'] = float(0.60 * cand['score_gray'] + 0.25 * pyramid_ncc + 0.15 * cand['std_match'])

# Form Periodic Alias Groups (~67 px lattice period)
def is_lattice_alias(c1, c2, lattice_period=67.0, tol=12.0):
    dist = math.hypot(c1['center_x'] - c2['center_x'], c1['center_y'] - c2['center_y'])
    if dist < 15.0:
        return True
    k = round(dist / lattice_period)
    return 1 <= k <= 4 and abs(dist - k * lattice_period) <= tol

alias_groups = []
visited = set()
top_candidates.sort(key=lambda c: c['macro_score'], reverse=True)

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

disambiguated_winners = []
for g_idx, group in enumerate(alias_groups, start=1):
    group.sort(key=lambda c: c['macro_score'], reverse=True)
    winner = group[0]
    disambiguated_winners.append((g_idx, winner, group))

disambiguated_winners.sort(key=lambda item: item[1]['macro_score'], reverse=True)

elapsed = time.perf_counter() - start_t
print(f"Execution Time: {elapsed*1000.0:.2f} ms ({elapsed:.4f} s)")

print("\n" + "=" * 90)
print("              PERIODIC-ALIAS GROUP RANKING DISAMBIGUATION REPORT")
print("=" * 90)

for g_idx, winner, group in disambiguated_winners[:3]:
    print(f"\nGroup #{g_idx} Candidates ({len(group)} periodic alias candidates in group):")
    print(f"  {'Candidate Center (x,y)':<24} {'Scale':<7} {'Match Score':<12} {'Pyramid NCC':<14} {'Std Match':<12} {'Macro Score':<12} {'Dist to GT':<14}")
    print("  " + "-" * 95)
    for cand in group:
        is_win = " [WINNER]" if cand == winner else ""
        print(f"  ({cand['center_x']:.2f}, {cand['center_y']:.2f}){is_win:<9} {cand['scale']:.3f}   {cand['score_gray']:.4f}       {cand['pyramid_ncc']:.4f}         {cand['std_match']:.4f}       {cand['macro_score']:.4f}         {cand['dist_gt']:.2f} px")

selected_coarse = disambiguated_winners[0][1]
print("\n" + "=" * 90)
print(f"SELECTED COARSE CANDIDATE: ({selected_coarse['center_x']:.2f}, {selected_coarse['center_y']:.2f})")
print(f"Ground Truth Coordinate:  ({gt_x:.2f}, {gt_y:.2f})")
print(f"Coarse Pixel Error:       {selected_coarse['dist_gt']:.2f} px")
print("=" * 90)
