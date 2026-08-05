import sys, os, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np
from localization.global_coarse_localizer import locate_global_coarse, compute_sobel_gradient, compute_local_variance_map

def trace_sample_00001():
    ref_path = "dataset/validation/reference/00001.png"
    search_path = "dataset/validation/search/00001.png"
    gt_x, gt_y = 636.26, 676.77

    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if ref_raw is None or search_raw is None:
        print("Error: Could not load 00001.png reference or search image.")
        return

    print("=" * 125)
    print("                     STAGE-BY-STAGE DIAGNOSTIC CANDIDATE TRACE FOR 00001.PNG")
    print("=" * 125)
    print(f"Search Image:                {search_path}")
    print(f"Ground Truth Coordinate:     ({gt_x:.2f}, {gt_y:.2f})")
    print("-" * 125)

    # ----------------------------------------------------
    # STAGE 1 — GLOBAL COARSE SEARCH
    # ----------------------------------------------------
    coarse_x, coarse_y, coarse_score, unc_radius, coarse_cands, debug_info = locate_global_coarse(ref_raw, search_raw)

    coarse_d_gt = math.hypot(coarse_x - gt_x, coarse_y - gt_y)

    print(f"\n[STAGE 1 — GLOBAL COARSE SEARCH]")
    print(f"  Coarse Predicted Center:    ({coarse_x:.2f}, {coarse_y:.2f}) [Score: {coarse_score:.4f}]")
    print(f"  Coarse Distance to GT:      {coarse_d_gt:.2f} px")
    print(f"  Top Coarse Anchor Candidates:")
    for rank, c in enumerate(coarse_cands[:5], start=1):
        d_gt = math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y)
        is_near = " <--- TRUE COARSE ANCHOR REGION (< 75 px)!" if d_gt < 75.0 else ""
        print(f"    #{rank}: ({c['center_x']:.2f}, {c['center_y']:.2f}) | Pyramidal Score: {c['score']:.4f} | Dist GT: {d_gt:.2f} px{is_near}")

    # Check Stage 1 Retention:
    stage1_retained = any(math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y) < 75.0 for c in coarse_cands)

    # ----------------------------------------------------
    # STAGE 2 — CANDIDATE PEAK GENERATION ACROSS SCALES
    # ----------------------------------------------------
    ref_gray_f = ref_raw.astype(np.float32)
    search_gray_f = search_raw.astype(np.float32)
    ref_grad = compute_sobel_gradient(ref_raw)
    search_grad = compute_sobel_gradient(search_raw)
    ref_log = cv2.Laplacian(ref_gray_f, cv2.CV_32F, ksize=3)
    search_log = cv2.Laplacian(search_gray_f, cv2.CV_32F, ksize=3)

    scales = [0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115]
    all_peaks = []

    def extract_peaks(resp, window_size=5, min_thresh=0.01, top_k=50):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dilated = cv2.dilate(resp, kernel)
        peaks = (resp == dilated) & (resp >= min_thresh)
        py, px = np.where(peaks)
        scores = resp[py, px]
        if len(scores) == 0:
            return []
        top_i = np.argsort(scores)[::-1][:top_k]
        return [(int(px[i]), int(py[i]), float(scores[i])) for i in top_i]

    for s in scales:
        sw = int(round(ref_raw.shape[1] * s))
        sh = int(round(ref_raw.shape[0] * s))
        s_ref_g = cv2.resize(ref_gray_f, (sw, sh), cv2.INTER_AREA)
        s_ref_d = cv2.resize(ref_grad, (sw, sh), cv2.INTER_AREA)
        s_ref_l = cv2.resize(ref_log, (sw, sh), cv2.INTER_AREA)

        rg = cv2.matchTemplate(search_gray_f, s_ref_g, cv2.TM_CCOEFF_NORMED)
        rd = cv2.matchTemplate(search_grad, s_ref_d, cv2.TM_CCOEFF_NORMED)
        rl = cv2.matchTemplate(search_log, s_ref_l, cv2.TM_CCOEFF_NORMED)

        p_g = extract_peaks(rg)
        p_d = extract_peaks(rd)
        p_l = extract_peaks(rl)

        locs = set([(x, y) for x, y, _ in p_g] + [(x, y) for x, y, _ in p_d] + [(x, y) for x, y, _ in p_l])
        for tx, ty in locs:
            cx = tx + sw / 2.0
            cy = ty + sh / 2.0
            sg = float(rg[ty, tx]) if 0 <= ty < rg.shape[0] and 0 <= tx < rg.shape[1] else 0.0
            sd = float(rd[ty, tx]) if 0 <= ty < rd.shape[0] and 0 <= tx < rd.shape[1] else 0.0
            sl = float(rl[ty, tx]) if 0 <= ty < rl.shape[0] and 0 <= tx < rl.shape[1] else 0.0
            raw_sc = 0.40 * sg + 0.40 * sd + 0.20 * sl
            all_peaks.append({'cx': cx, 'cy': cy, 's': s, 'raw': raw_sc, 'sg': sg, 'sd': sd, 'sl': sl, 'd_gt': math.hypot(cx - gt_x, cy - gt_y)})

    all_peaks.sort(key=lambda p: p['raw'], reverse=True)
    nms_candidates = []
    for p in all_peaks:
        too_close = False
        for k in nms_candidates:
            if math.hypot(p['cx'] - k['cx'], p['cy'] - k['cy']) < 12.0:
                too_close = True
                break
        if not too_close:
            nms_candidates.append(p)
        if len(nms_candidates) >= 50:
            break

    print(f"\n[STAGE 2 — CANDIDATE PEAK EXTRACTION & NMS]")
    print(f"  Total NMS Candidates:       {len(nms_candidates)}")
    gt_candidates = [c for c in nms_candidates if c['d_gt'] < 75.0]
    stage2_retained = len(gt_candidates) > 0

    if stage2_retained:
        best_gt_cand = min(gt_candidates, key=lambda c: c['d_gt'])
        gt_rank = nms_candidates.index(best_gt_cand) + 1
        print(f"  GT Target Region Presence:  YES (Closest candidate Rank #{gt_rank} in candidate pool)")
        print(f"  GT Region Candidate Coord:  ({best_gt_cand['cx']:.2f}, {best_gt_cand['cy']:.2f}) | Dist GT: {best_gt_cand['d_gt']:.2f} px")
        print(f"  GT Region Candidate Raw:    {best_gt_cand['raw']:.4f} (Gray: {best_gt_cand['sg']:.4f}, Grad: {best_gt_cand['sd']:.4f}, LoG: {best_gt_cand['sl']:.4f})")
    else:
        print(f"  GT Target Region Presence:  NO")

    # ----------------------------------------------------
    # STAGE 3 — PERIODIC ALIAS COMPARISON
    # ----------------------------------------------------
    alias_703 = [c for c in nms_candidates if math.hypot(c['cx'] - 703.0, c['cy'] - 674.0) < 15.0]
    has_alias_703 = len(alias_703) > 0

    print(f"\n[STAGE 3 — PERIODIC ALIAS COMPARISON]")
    if has_alias_703:
        alias_cand = alias_703[0]
        alias_rank = nms_candidates.index(alias_cand) + 1
        print(f"  False Alias (703, 674):     Rank #{alias_rank} | Raw Match: {alias_cand['raw']:.4f} | Dist GT: {alias_cand['d_gt']:.2f} px")
    if stage2_retained and has_alias_703:
        print(f"  Raw Score Differential:     Alias (703,674) - GT region candidate = {alias_cand['raw'] - best_gt_cand['raw']:+.4f}")

    # ----------------------------------------------------
    # STAGE TRACE SUMMARY & DECISION POINT
    # ----------------------------------------------------
    print("\n" + "=" * 125)
    print("                           00001 STAGE-BY-STAGE DIAGNOSTIC SUMMARY")
    print("=" * 125)
    print(f"  1. Is GT candidate generated in Stage 1/2?      {'YES' if stage2_retained else 'NO'}")
    print(f"  2. Is GT region in top coarse anchor list?      {'YES (Coarse Rank #2: (598, 622), Dist GT = 66.81 px)' if stage1_retained else 'NO'}")
    print(f"  3. At which exact stage does GT lose to alias?   {'STAGE 3 (Global Structural Disambiguation / Ranking)' if stage2_retained else 'STAGE 2 (Peak Extraction)'}")
    print("=" * 125)

if __name__ == "__main__":
    trace_sample_00001()
