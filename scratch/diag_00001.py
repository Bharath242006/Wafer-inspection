import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np
from localization.hierarchical_localizer import (
    compute_sobel_gradient, compute_canny_edge, extract_local_peaks,
    is_lattice_alias
)

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"

ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

gt_x, gt_y = 636.26, 676.77

ref_gray_f = ref_raw.astype(np.float32)
search_gray_f = search_raw.astype(np.float32)

ref_grad = compute_sobel_gradient(ref_raw)
search_grad = compute_sobel_gradient(search_raw)

ref_edge = compute_canny_edge(ref_raw)
search_edge = compute_canny_edge(search_raw)

ref_blur = cv2.GaussianBlur(ref_gray_f, (21, 21), 5.0)
search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)

scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
search_h, search_w = search_raw.shape[:2]
all_candidates = []

for s in scales:
    scaled_w = int(round(ref_raw.shape[1] * s))
    scaled_h = int(round(ref_raw.shape[0] * s))

    s_ref_gray = cv2.resize(ref_gray_f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    s_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    s_ref_blur = cv2.resize(ref_blur, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    res_gray = cv2.matchTemplate(search_gray_f, s_ref_gray, cv2.TM_CCOEFF_NORMED)
    res_grad = cv2.matchTemplate(search_grad, s_ref_grad, cv2.TM_CCOEFF_NORMED)
    res_blur = cv2.matchTemplate(search_blur, s_ref_blur, cv2.TM_CCOEFF_NORMED)

    peaks_g = extract_local_peaks(res_gray, window_size=9, min_thresh=0.01, top_k=50)
    peaks_d = extract_local_peaks(res_grad, window_size=9, min_thresh=0.01, top_k=50)

    peak_locs = set([(x, y) for x, y, _ in peaks_g] + [(x, y) for x, y, _ in peaks_d])

    ref_mean = float(np.mean(s_ref_gray))
    ref_std = float(np.std(s_ref_gray))

    for tl_x, tl_y in peak_locs:
        if tl_x < 0 or tl_y < 0 or tl_x + scaled_w > search_w or tl_y + scaled_h > search_h:
            continue

        cx = tl_x + (scaled_w / 2.0)
        cy = tl_y + (scaled_h / 2.0)

        score_g = float(res_gray[tl_y, tl_x]) if 0 <= tl_y < res_gray.shape[0] and 0 <= tl_x < res_gray.shape[1] else 0.0
        score_d = float(res_grad[tl_y, tl_x]) if 0 <= tl_y < res_grad.shape[0] and 0 <= tl_x < res_grad.shape[1] else 0.0
        score_b = float(res_blur[tl_y, tl_x]) if 0 <= tl_y < res_blur.shape[0] and 0 <= tl_x < res_blur.shape[1] else 0.0

        patch_g = search_gray_f[tl_y:tl_y+scaled_h, tl_x:tl_x+scaled_w]
        patch_mean = float(np.mean(patch_g))
        patch_std = float(np.std(patch_g))

        std_match = (2.0 * ref_std * patch_std) / (ref_std**2 + patch_std**2 + 1e-5)

        ctx_w = min(search_w, scaled_w * 2)
        ctx_h = min(search_h, scaled_h * 2)
        x1_ctx = max(0, int(cx - ctx_w / 2.0))
        y1_ctx = max(0, int(cy - ctx_h / 2.0))
        x2_ctx = min(search_w, x1_ctx + ctx_w)
        y2_ctx = min(search_h, y1_ctx + ctx_h)

        search_ctx_patch = search_grad[y1_ctx:y2_ctx, x1_ctx:x2_ctx]
        anchor_density = float(np.mean(search_ctx_patch)) if search_ctx_patch.size > 0 else 0.0

        raw_coarse_score = (
            0.35 * score_g +
            0.35 * score_d +
            0.15 * score_b +
            0.15 * std_match
        )

        dist_gt = math.hypot(cx - gt_x, cy - gt_y)

        all_candidates.append({
            'center_x': cx,
            'center_y': cy,
            'top_left': (tl_x, tl_y),
            'scaled_w': scaled_w,
            'scaled_h': scaled_h,
            'scale': s,
            'raw_coarse_score': float(raw_coarse_score),
            'macro_anchor_score': float(anchor_density + 0.5 * score_b),
            'score_gray': score_g,
            'score_grad': score_d,
            'is_alias_rejected': False,
            'alias_penalty': 0.0,
            'coarse_score': float(raw_coarse_score),
            'dist_gt': dist_gt
        })

print(f"Total extracted raw candidates: {len(all_candidates)}")

# Print candidates sorted by distance to GT
sorted_by_gt = sorted(all_candidates, key=lambda c: c['dist_gt'])
print("\nTop 5 Candidates Closest to Ground Truth (636.26, 676.77):")
for idx, c in enumerate(sorted_by_gt[:5], start=1):
    print(f"  GT-Closest {idx}: Center=({c['center_x']:.2f}, {c['center_y']:.2f}) | Dist to GT={c['dist_gt']:.2f}px | Scale={c['scale']:.3f} | Raw Score={c['raw_coarse_score']:.4f} | Int={c['score_gray']:.4f} | Grad={c['score_grad']:.4f}")

# Sort candidates by raw_coarse_score (Before Disambiguation)
sorted_before = sorted(all_candidates, key=lambda c: c['raw_coarse_score'], reverse=True)
print("\nTop 10 Candidates BEFORE Periodic Disambiguation:")
for idx, c in enumerate(sorted_before[:10], start=1):
    print(f"  Before Rank {idx:02d}: Center=({c['center_x']:.2f}, {c['center_y']:.2f}) | Dist to GT={c['dist_gt']:.2f}px | Scale={c['scale']:.3f} | Raw Score={c['raw_coarse_score']:.4f} | Anchor={c['macro_anchor_score']:.4f}")

# Perform Periodic Disambiguation
for i in range(len(all_candidates)):
    c1 = all_candidates[i]
    if c1['is_alias_rejected']:
        continue
    for j in range(i + 1, len(all_candidates)):
        c2 = all_candidates[j]
        if is_lattice_alias(c1, c2, lattice_period=67.0, tolerance=12.0):
            if c1['macro_anchor_score'] >= c2['macro_anchor_score'] - 0.02:
                c2['alias_penalty'] += 0.25
                c2['coarse_score'] -= 0.25
                if c2['coarse_score'] < c1['coarse_score'] - 0.10:
                    c2['is_alias_rejected'] = True
            else:
                c1['alias_penalty'] += 0.25
                c1['coarse_score'] -= 0.25

# Sort candidates AFTER Periodic Disambiguation
sorted_after = sorted(all_candidates, key=lambda c: c['coarse_score'], reverse=True)
print("\nTop 10 Candidates AFTER Periodic Disambiguation:")
for idx, c in enumerate(sorted_after[:10], start=1):
    print(f"  After Rank {idx:02d}: Center=({c['center_x']:.2f}, {c['center_y']:.2f}) | Dist to GT={c['dist_gt']:.2f}px | Scale={c['scale']:.3f} | Initial Score={c['raw_coarse_score']:.4f} | Anchor={c['macro_anchor_score']:.4f} | Penalty={c['alias_penalty']:.2f} | Final Score={c['coarse_score']:.4f}")
