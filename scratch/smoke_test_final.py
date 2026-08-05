import sys
sys.path.insert(0, '.')
from localization.final_localizer_hybrid import locate_target

result = locate_target(
    'dataset/validation/reference/00174.png',
    'dataset/validation/search/00174.png',
    gt_x=513.87, gt_y=783.43,
    draw_output_path='results/demo_prediction_00174.png'
)

print()
print('=' * 60)
print('  DriftSense-X Final Localizer -- Result')
print('=' * 60)
print('  Predicted X    :', round(result["predicted_x"], 2), 'px')
print('  Predicted Y    :', round(result["predicted_y"], 2), 'px')
if result["pixel_error"] is not None:
    print('  Pixel Error    :', round(result["pixel_error"], 2), 'px')
print('  Confidence     :', round(result["confidence"], 4))
print('  Candidate Rank :', result["candidate_rank"])
print('  Runtime        :', round(result["runtime_sec"] * 1000, 1), 'ms')
print('  Status         :', result["status"])
print('=' * 60)
print('Annotated image saved to: results/demo_prediction_00174.png')
