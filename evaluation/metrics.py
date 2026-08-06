"""
evaluation/metrics.py

Performance & accuracy evaluation metrics (MAE, Precision@k, Success Rate @ IoU).
"""

import numpy as np


def compute_center_error(pred_x: float, pred_y: float, gt_x: float, gt_y: float) -> float:
    """Computes Euclidean center distance error in pixels."""
    return float(np.hypot(pred_x - gt_x, pred_y - gt_y))


def compute_iou(boxA: tuple, boxB: tuple) -> float:
    """
    Computes Intersection over Union (IoU) of two bounding boxes (x_min, y_min, x_max, y_max).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(np.clip(iou, 0.0, 1.0))
