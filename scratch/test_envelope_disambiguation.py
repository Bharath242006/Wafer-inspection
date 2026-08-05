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

# Compute Gaussian Low-Pass Filtered Envelopes
ref_blur = cv2.GaussianBlur(ref_gray_f, (41, 41), 10.0)
search_blur = cv2.GaussianBlur(search_gray_f, (41, 41), 10.0)

sw, sh = 100, 100
s_ref_blur = cv2.resize(ref_blur, (sw, sh), cv2.INTER_AREA)

def zmuv_ncc(p, t):
    pf = p - np.mean(p)
    tf = t - np.mean(t)
    sp, st = np.std(pf), np.std(tf)
    return float(np.mean(pf * tf) / (sp * st)) if sp > 1e-5 and st > 1e-5 else 0.0

targets = [
    ("GT Region (636.26, 676.77)", gt_x, gt_y),
    ("True Candidate (642, 676)", 642.0, 676.0),
    ("Alias +67px (703, 674)", 703.0, 674.0),
    ("Alias -67px (573, 624)", 573.0, 624.0),
    ("Alias +55px (691, 688)", 691.0, 688.0)
]

print("=" * 105)
print("     LOW-FREQUENCY SPATIAL ENVELOPE ZMUV CORRELATION ON 00001.PNG")
print("=" * 105)
print(f"{'Target Label':<30} {'Center (x,y)':<20} {'Envelope ZMUV Score':<22} {'Dist GT (px)':<14}")
print("-" * 105)

for label, tx, ty in targets:
    tl_x = int(round(tx - sw / 2.0))
    tl_y = int(round(ty - sh / 2.0))

    patch_b = search_blur[tl_y:tl_y+sh, tl_x:tl_x+sw]
    env_score = zmuv_ncc(patch_b, s_ref_blur)
    d_gt = math.hypot(tx - gt_x, ty - gt_y)

    print(f"{label:<30} ({tx:.2f}, {ty:.2f})            {env_score:.4f}                   {d_gt:.2f} px")

print("=" * 105)
