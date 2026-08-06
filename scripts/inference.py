"""
scripts/inference.py

Unified CLI runner for single pair or batch image inference.
"""

import sys
import os
import argparse
import cv2

sys.path.append(os.path.abspath("."))
from localization.final_localizer import locate_target_final
from localization.visualization import draw_prediction_visualization


def main():
    parser = argparse.ArgumentParser(description="DriftSense-X Inference CLI")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image (1000x1000)")
    parser.add_argument("--search", type=str, required=True, help="Path to search image (1000x1000)")
    parser.add_argument("--output_vis", type=str, default="outputs/predictions/prediction_result.png", help="Path to save output visualization")
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"Reference image not found at '{args.reference}'")
        return
    if not os.path.exists(args.search):
        print(f"Search image not found at '{args.search}'")
        return

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    print(f"Running DriftSense-X inference on {args.reference} and {args.search}...")
    pred_x, pred_y, score, status, details = locate_target_final(ref_img, search_img)

    print(f"Predicted Location: Center ({pred_x:.2f}, {pred_y:.2f}) | Confidence: {score:.4f} | Status: {status}")

    draw_prediction_visualization(search_img, pred_x, pred_y, status=status, output_path=args.output_vis)
    print(f"Saved visualization to '{args.output_vis}'.")


if __name__ == "__main__":
    main()
