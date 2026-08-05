import sys, os, math
sys.path.append(os.path.abspath("."))
from localization.hybrid_localizer import locate_reference_pattern

pred, conf, status, info = locate_reference_pattern(
    'dataset/validation/reference/00001.png',
    'dataset/validation/search/00001.png'
)

top_b = info['top_cand_before_context']
top_a = info['top_cand_after_context']
fin_s = info['final_selected_cand']

print(f"Top before context: center=({top_b['center_x']:.2f}, {top_b['center_y']:.2f}), initial_score={top_b['initial_score']:.4f}")
print(f"Top after context:  center=({top_a['center_x']:.2f}, {top_a['center_y']:.2f}), final_score={top_a['final_score']:.4f}")
print(f"Final selected:     center=({fin_s['center_x']:.2f}, {fin_s['center_y']:.2f})")

print("\nTop 5 Candidates After Context Verification:")
for idx, c in enumerate(info['candidates'][:5], start=1):
    err = math.hypot(c['center_x'] - 636.26, c['center_y'] - 676.77)
    print(
        f"Rank {idx}: center=({c['center_x']:.2f}, {c['center_y']:.2f}), dist_to_gt={err:.2f}px | "
        f"initial={c['initial_score']:.4f}, ctx_cons={c['context_consistency']:.4f}, "
        f"ring_corr={c['ring_corr']:.4f}, final={c['final_score']:.4f}"
    )
