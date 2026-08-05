"""
localization/final_localizer_hybrid.py

DriftSense-X — Final Localizer Application.

Uses the BEST validated method: Global Landmark Localizer (431.93 px mean error).

Evaluation verdict:
  - Hybrid Ranker evaluated (534.74 px) — did NOT improve over baseline (431.93 px).
  - Root cause: periodic aliasing features don't generalize across wafer pattern types.
  - Global Landmark remains the best non-oracle method.

Returns a structured result dict with all required fields:
    predicted_x    : float
    predicted_y    : float
    pixel_error    : float | None   (requires gt_x, gt_y)
    confidence     : float [0, 1]
    candidate_rank : int
    runtime_sec    : float
    status         : "SUCCESS" | "FAILED"

Optional: draws crosshair + bounding box on search image.

CLI usage:
    python localization/final_localizer_hybrid.py REF SEARCH [--gt_x X] [--gt_y Y] [--output PATH]
"""

import math
import os
import time
import cv2
import numpy as np

from localization.global_landmark_localizer import locate_global_landmark


def draw_prediction(
    search_img: np.ndarray,
    pred_x: float,
    pred_y: float,
    ref_img: np.ndarray,
    scale: float = 0.10,
    color_bgr: tuple = (0, 255, 0),
    gt_x: float = None,
    gt_y: float = None,
) -> np.ndarray:
    """
    Draws prediction crosshair + bounding box on the search image.

    Args:
        search_img:  Grayscale or BGR search image.
        pred_x/y:    Predicted center coordinates (pixels).
        ref_img:     Reference image (used to determine bounding-box size).
        scale:       Template scale factor (default 0.10).
        color_bgr:   BGR colour for predicted marker (default green).
        gt_x/y:      Optional ground-truth center (draws red cross marker).

    Returns:
        BGR annotated image (uint8).
    """
    vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR) if search_img.ndim == 2 else search_img.copy()
    sw = int(round(ref_img.shape[1] * scale))
    sh = int(round(ref_img.shape[0] * scale))

    cx, cy = int(round(pred_x)), int(round(pred_y))
    H, W = vis.shape[:2]
    tl = (max(0, cx - sw // 2), max(0, cy - sh // 2))
    br = (min(W - 1, cx + sw // 2), min(H - 1, cy + sh // 2))

    # Bounding box
    cv2.rectangle(vis, tl, br, color_bgr, 2)
    # Crosshair
    arm = 15
    cv2.line(vis, (max(0, cx - arm), cy), (min(W - 1, cx + arm), cy), color_bgr, 2)
    cv2.line(vis, (cx, max(0, cy - arm)), (cx, min(H - 1, cy + arm)), color_bgr, 2)
    cv2.putText(vis, f"Pred ({cx},{cy})", (cx + arm + 2, max(12, cy - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

    # Ground-truth marker (red)
    if gt_x is not None and gt_y is not None:
        gx, gy = int(round(gt_x)), int(round(gt_y))
        cv2.drawMarker(vis, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, f"GT ({gx},{gy})", (gx + 8, max(12, gy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        if math.hypot(pred_x - gt_x, pred_y - gt_y) < 300:
            cv2.line(vis, (cx, cy), (gx, gy), (0, 165, 255), 1)

    return vis


def locate_target(
    ref_path: str,
    search_path: str,
    gt_x: float = None,
    gt_y: float = None,
    draw_output_path: str = None,
) -> dict:
    """
    Full DriftSense-X localization pipeline.

    Uses Global Landmark method — best validated: 431.93 px mean error on 40 held-out images.

    Args:
        ref_path:         Path to reference image (grayscale, 1000x1000 px).
        search_path:      Path to search image (grayscale, 1000x1000 px).
        gt_x, gt_y:       Optional ground-truth center for pixel_error reporting.
        draw_output_path: If set, saves annotated BGR image to this path.

    Returns:
        dict with keys:
            predicted_x    : float  - predicted center X (pixels)
            predicted_y    : float  - predicted center Y (pixels)
            pixel_error    : float | None - Euclidean error to GT (if GT provided)
            confidence     : float  - alignment confidence [0, 1]
            candidate_rank : int    - winner rank in top-500 candidate pool
            runtime_sec    : float  - total runtime in seconds
            status         : str    - "SUCCESS" or "FAILED"
    """
    t_start = time.perf_counter()

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot load reference image: '{ref_path}'")

    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_img is None:
        raise FileNotFoundError(f"Cannot load search image: '{search_path}'")

    # Global Landmark Localization — best validated method (431.93 px)
    pred_x, pred_y, cand_rank, alignment_score, ranked_cands = locate_global_landmark(
        ref_img, search_img, top_k_cands=500
    )

    runtime = time.perf_counter() - t_start

    confidence = float(np.clip(alignment_score, 0.0, 1.0))
    status = "SUCCESS" if confidence >= 0.12 else "FAILED"

    pixel_error = (
        math.hypot(pred_x - gt_x, pred_y - gt_y)
        if (gt_x is not None and gt_y is not None)
        else None
    )

    if draw_output_path is not None:
        vis = draw_prediction(search_img, pred_x, pred_y, ref_img, gt_x=gt_x, gt_y=gt_y)
        cv2.imwrite(draw_output_path, vis)

    return {
        "predicted_x":    float(pred_x),
        "predicted_y":    float(pred_y),
        "pixel_error":    float(pixel_error) if pixel_error is not None else None,
        "confidence":     float(confidence),
        "candidate_rank": int(cand_rank),
        "runtime_sec":    float(runtime),
        "status":         status,
    }


# ── CLI demo interface ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="DriftSense-X Final Localizer — locate reference pattern in search image."
    )
    parser.add_argument("ref",    help="Path to reference image (1000x1000 px, grayscale)")
    parser.add_argument("search", help="Path to search image (1000x1000 px, grayscale)")
    parser.add_argument("--gt_x", type=float, default=None, help="Ground-truth X (optional)")
    parser.add_argument("--gt_y", type=float, default=None, help="Ground-truth Y (optional)")
    parser.add_argument("--output", default=None,
                        help="Save annotated search image to this path (optional)")
    args = parser.parse_args()

    result = locate_target(
        args.ref, args.search,
        gt_x=args.gt_x, gt_y=args.gt_y,
        draw_output_path=args.output,
    )

    print("\n" + "=" * 60)
    print("  DriftSense-X Final Localizer — Result")
    print("=" * 60)
    print(f"  Predicted X    : {result['predicted_x']:.2f} px")
    print(f"  Predicted Y    : {result['predicted_y']:.2f} px")
    if result['pixel_error'] is not None:
        print(f"  Pixel Error    : {result['pixel_error']:.2f} px")
    print(f"  Confidence     : {result['confidence']:.4f}")
    print(f"  Candidate Rank : {result['candidate_rank']}")
    print(f"  Runtime        : {result['runtime_sec'] * 1000:.1f} ms")
    print(f"  Status         : {result['status']}")
    if args.output:
        print(f"  Annotated img  : {args.output}")
    print("=" * 60)
