import sys, os, math
sys.path.append(os.path.abspath("."))
from localization.final_localizer import locate_reference_pattern_final

ref_path = "dataset/validation/reference/00001.png"
search_path = "dataset/validation/search/00001.png"
gt_x, gt_y = 636.26, 676.77

_, _, _, _, dbg = locate_reference_pattern_final(ref_path, search_path)
cands = dbg['all_candidates']

print(f"{'Rank':<5} | {'Coord (x, y)':<18} | {'Dist GT':<10} | {'FinalScore':<10} | {'LowFreq':<10} | {'LoG':<10} | {'Grad':<10} | {'Macro':<10} | {'MultiScale':<10}")
print("-" * 110)

for i, c in enumerate(cands[:15], start=1):
    d_gt = math.hypot(c['center_x'] - gt_x, c['center_y'] - gt_y)
    print(f"#{i:<4} | ({c['center_x']:.2f}, {c['center_y']:.2f}){'':<4} | {d_gt:<10.2f} | {c['final_score']:<10.4f} | {c['low_frequency']:<10.4f} | {c['log']:<10.4f} | {c['gradient']:<10.4f} | {c['macro']:<10.4f} | {c['multi_scale']:<10.4f}")
