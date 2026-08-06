"""
localization/visualization.py

Localization output visualization helper functions.
"""

import os
import cv2
import numpy as np


def draw_prediction_visualization(
    search_img: np.ndarray,
    pred_x: float,
    pred_y: float,
    gt_x: float = None,
    gt_y: float = None,
    target_w: int = 100,
    target_h: int = 100,
    status: str = "SUCCESS",
    output_path: str = None
) -> np.ndarray:
    """
    Renders prediction bounding box, crosshair, ground-truth error vector, and info labels.
    """
    vis = cv2.cvtColor(search_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    px, py = int(round(pred_x)), int(round(pred_y))
    tl_x = int(round(pred_x - target_w / 2.0))
    tl_y = int(round(pred_y - target_h / 2.0))

    box_color = (0, 255, 0) if status == "SUCCESS" else (0, 0, 255)
    cv2.rectangle(vis, (tl_x, tl_y), (tl_x + target_w, tl_y + target_h), box_color, 2)
    cv2.circle(vis, (px, py), 4, (0, 255, 255), -1)

    if gt_x is not None and gt_y is not None:
        gx, gy = int(round(gt_x)), int(round(gt_y))
        cv2.circle(vis, (gx, gy), 4, (0, 0, 255), -1)
        cv2.line(vis, (px, py), (gx, gy), (255, 0, 255), 2)
        err = np.hypot(pred_x - gt_x, pred_y - gt_y)
        cv2.putText(vis, f"Error: {err:.2f} px", (tl_x, max(20, tl_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis)

    return vis
