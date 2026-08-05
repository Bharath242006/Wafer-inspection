import sys, os, math
sys.path.append(os.path.abspath("."))
from localization.hybrid_localizer import locate_reference_pattern

pred_x, pred_y, status, info = locate_reference_pattern(
    'dataset/validation/reference/00001.png',
    'dataset/validation/search/00001.png'
)

print(f"Total candidates in top_candidates: {len(info['top_candidates'])}")
for c in info['top_candidates']:
    err = math.hypot(c['center_x'] - 636.26, c['center_y'] - 676.77)
    print(
        f"Rank {c['rank']}: center=({c['center_x']:.2f}, {c['center_y']:.2f}), "
        f"dist_to_gt={err:.2f}px, final_score={c['final_score']:.4f}, "
        f"int={c['score_intensity']:.3f}, grad={c['score_grad']:.3f}, "
        f"edge={c['score_edge']:.3f}, scale={c['scale']:.3f}"
    )
