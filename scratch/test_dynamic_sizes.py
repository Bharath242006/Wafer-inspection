import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from localization.global_landmark_localizer import compute_global_landmark_heatmap, locate_global_landmark

test_sizes = [
    (1000, 1000),
    (500, 500),
    (800, 600),
    (1024, 1024),
    (400, 700),
    (300, 300)
]

for sh, sw in test_sizes:
    ref = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    search = np.random.randint(0, 255, (sh, sw), dtype=np.uint8)
    print(f"Testing search size {sw}x{sh}...")
    heatmap = compute_global_landmark_heatmap(ref, search)
    assert heatmap.shape == (sh, sw), f"Heatmap shape mismatch: {heatmap.shape} vs ({sh}, {sw})"
    pred_x, pred_y, rank, score, _ = locate_global_landmark(ref, search, top_k_cands=10)
    print(f"  OK! Predicted: ({pred_x:.1f}, {pred_y:.1f})")

print("\nAll dynamic image size tests PASSED cleanly!")
