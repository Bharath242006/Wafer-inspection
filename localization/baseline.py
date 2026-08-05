"""
baseline.py

Baseline template-matching localization algorithm for the DriftSense-X challenge.

Loads a 1000x1000 high-magnification reference image and a 1000x1000 low-magnification
search image, downscales the reference image by ~0.10x (to match its ~100x100 representation
in the search image), and runs normalized cross-correlation template matching (cv2.TM_CCOEFF_NORMED).

If multiple candidate correlation peaks exist within a tolerance threshold of the maximum score,
it selects the candidate closest to the search image center (500, 500), per competition rules.
"""

import argparse
import os
import sys
import numpy as np
import cv2


def locate_reference_pattern(
    ref_path: str,
    search_path: str,
    scale_factor: float = 0.10,
    peak_threshold_ratio: float = 0.95
) -> tuple:
    """
    Locates the center coordinate (x, y) of the reference pattern within the search image.

    Args:
        ref_path (str): Path to the high-res 1000x1000 reference image.
        search_path (str): Path to the 1000x1000 search image.
        scale_factor (float): Downscaling ratio for the reference image (default: 0.10).
        peak_threshold_ratio (float): Ratio of peak correlation score to consider candidate peaks (default: 0.95).

    Returns:
        tuple: (predicted_center_x, predicted_center_y, max_val, scaled_ref, search_img, best_top_left)
    """
    # 1. Load reference and search images in grayscale
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_img is None:
        raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    search_h, search_w = search_img.shape[:2]
    search_center_x = search_w / 2.0  # 500.0 for 1000x1000 image
    search_center_y = search_h / 2.0  # 500.0 for 1000x1000 image

    # 2. Downscale the reference image by scale_factor (~0.10x)
    # 1000x1000 reference image becomes ~100x100 pixels
    scaled_w = int(round(ref_img.shape[1] * scale_factor))
    scaled_h = int(round(ref_img.shape[0] * scale_factor))
    scaled_ref = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # 3. Perform Normalized Template Matching (cv2.TM_CCOEFF_NORMED)
    res = cv2.matchTemplate(search_img, scaled_ref, cv2.TM_CCOEFF_NORMED)

    # 4. Find maximum correlation score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 5. Extract candidate peaks above (max_val * peak_threshold_ratio)
    # To handle repeating pattern candidates, evaluate all strong local peaks
    threshold = max_val * peak_threshold_ratio
    loc_y, loc_x = np.where(res >= threshold)

    best_top_left = max_loc
    min_dist_to_center = float('inf')

    for x, y in zip(loc_x, loc_y):
        # Calculate center of scaled reference bounding box for this candidate
        cand_center_x = x + (scaled_w / 2.0)
        cand_center_y = y + (scaled_h / 2.0)

        # Distance from search image center (500, 500)
        dist = np.hypot(cand_center_x - search_center_x, cand_center_y - search_center_y)

        if dist < min_dist_to_center:
            min_dist_to_center = dist
            best_top_left = (int(x), int(y))

    # 6. Convert best match top-left location into bounding-box CENTER (x, y)
    best_x, best_y = best_top_left
    predicted_center_x = float(best_x + (scaled_w / 2.0))
    predicted_center_y = float(best_y + (scaled_h / 2.0))

    return (predicted_center_x, predicted_center_y, max_val, scaled_ref, search_img, best_top_left)


def save_debug_visualization(
    search_img: np.ndarray,
    top_left: tuple,
    scaled_w: int,
    scaled_h: int,
    center_x: float,
    center_y: float,
    output_path: str
):
    """
    Renders and saves a visual overlay showing search image, bounding box, and center prediction.
    """
    # Convert grayscale to BGR for colorful overlay rendering
    vis_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    x1, y1 = top_left
    x2, y2 = x1 + scaled_w, y1 + scaled_h

    # Draw predicted bounding box (Green rectangle)
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw predicted center point (Red dot & crosshair)
    cx, cy = int(round(center_x)), int(round(center_y))
    cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.drawMarker(vis_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # Add text annotation
    label = f"Pred Center: ({center_x:.2f}, {center_y:.2f})"
    cv2.putText(vis_img, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline localization algorithm for DriftSense-X")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image (1000x1000)")
    parser.add_argument("--search", type=str, required=True, help="Path to search image (1000x1000)")
    parser.add_argument("--scale", type=float, default=0.10, help="Downscale factor for reference pattern (default: 0.10)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and save visualization")
    parser.add_argument("--vis-path", type=str, default="baseline_debug.png", help="Output path for debug visualization")

    args = parser.parse_args()

    pred_x, pred_y, max_val, scaled_ref, search_img, top_left = locate_reference_pattern(
        ref_path=args.reference,
        search_path=args.search,
        scale_factor=args.scale
    )

    # Print exact required output line format
    print(f"Predicted center: ({pred_x:.2f}, {pred_y:.2f})")

    if args.debug:
        save_debug_visualization(
            search_img=search_img,
            top_left=top_left,
            scaled_w=scaled_ref.shape[1],
            scaled_h=scaled_ref.shape[0],
            center_x=pred_x,
            center_y=pred_y,
            output_path=args.vis_path
        )


if __name__ == "__main__":
    main()
