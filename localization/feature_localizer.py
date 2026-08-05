"""
feature_localizer.py

Feature-based localization algorithm for DriftSense-X using SIFT keypoints,
descriptor matching, Lowe's ratio test, and RANSAC homography estimation.
"""

import argparse
import os
import cv2
import numpy as np


def locate_reference_pattern(
    ref_path: str,
    search_path: str,
    scale_factor: float = 0.10,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 5.0,
    min_inliers: int = 4
) -> tuple:
    """
    Locates the center coordinate (x, y) of the reference pattern in the search image using SIFT + RANSAC.

    Args:
        ref_path (str): Path to 1000x1000 reference image.
        search_path (str): Path to 1000x1000 search image.
        scale_factor (float): Downscale factor for reference image (default: 0.10).
        ratio_thresh (float): Lowe's ratio test threshold (default: 0.75).
        ransac_reproj_thresh (float): RANSAC reprojection error threshold (default: 5.0).
        min_inliers (int): Minimum required RANSAC inliers for success (default: 4).

    Returns:
        tuple: (pred_x, pred_y, metrics_dict, search_img, debug_data_dict)
    """
    # 1. Load reference and search images in grayscale
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_img is None:
        raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    # 2. Downscale reference image to ~100x100 representation (~0.10 scale)
    scaled_w = int(round(ref_img.shape[1] * scale_factor))
    scaled_h = int(round(ref_img.shape[0] * scale_factor))
    scaled_ref = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # 3. & 4. SIFT detector and descriptor computation
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(scaled_ref, None)
    kp_search, des_search = sift.detectAndCompute(search_img, None)

    num_kp_ref = len(kp_ref) if kp_ref is not None else 0
    num_kp_search = len(kp_search) if kp_search is not None else 0

    metrics = {
        "num_kp_ref": num_kp_ref,
        "num_kp_search": num_kp_search,
        "num_descriptor_matches": 0,
        "num_good_matches": 0,
        "num_inliers": 0,
        "status": "SUCCESS"
    }

    debug_data = {
        "scaled_ref": scaled_ref,
        "kp_ref": kp_ref,
        "kp_search": kp_search,
        "good_matches": [],
        "inlier_matches": [],
        "homography": None,
        "bounding_box": None
    }

    if des_ref is None or des_search is None or len(des_ref) < 2 or len(des_search) < 2:
        metrics["status"] = "TOO_FEW_KEYPOINTS"
        return None, None, metrics, search_img, debug_data

    # 5. KNN matching using BFMatcher (L2 distance for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_ref, des_search, k=2)
    metrics["num_descriptor_matches"] = len(raw_matches)

    # 6. Apply Lowe's ratio test
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    metrics["num_good_matches"] = len(good_matches)
    debug_data["good_matches"] = good_matches

    if len(good_matches) < min_inliers:
        metrics["status"] = "TOO_FEW_GOOD_MATCHES"
        return None, None, metrics, search_img, debug_data

    # 7. RANSAC Homography Estimation
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_search[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)

    if H is None or mask is None:
        metrics["status"] = "HOMOGRAPHY_FAILED"
        return None, None, metrics, search_img, debug_data

    inlier_mask = mask.ravel().tolist()
    inlier_matches = [m for m, is_inlier in zip(good_matches, inlier_mask) if is_inlier == 1]
    num_inliers = len(inlier_matches)
    metrics["num_inliers"] = num_inliers
    debug_data["inlier_matches"] = inlier_matches
    debug_data["homography"] = H

    if num_inliers < min_inliers:
        metrics["status"] = "TOO_FEW_INLIERS"
        return None, None, metrics, search_img, debug_data

    # 8. & 9. Estimate target location (project reference corners and center)
    ref_corners = np.float32([
        [0, 0],
        [scaled_w, 0],
        [scaled_w, scaled_h],
        [0, scaled_h]
    ]).reshape(-1, 1, 2)

    dst_corners = cv2.perspectiveTransform(ref_corners, H)
    debug_data["bounding_box"] = dst_corners

    # Project center point (scaled_w/2, scaled_h/2) through Homography
    ref_center = np.float32([[[scaled_w / 2.0, scaled_h / 2.0]]])
    dst_center = cv2.perspectiveTransform(ref_center, H)

    pred_x = float(dst_center[0][0][0])
    pred_y = float(dst_center[0][0][1])

    return pred_x, pred_y, metrics, search_img, debug_data


def save_debug_visualization(
    search_img: np.ndarray,
    debug_data: dict,
    pred_x: float,
    pred_y: float,
    output_path: str
):
    """Saves visual overlay showing keypoint matches, inliers, and predicted center."""
    scaled_ref = debug_data["scaled_ref"]
    kp_ref = debug_data["kp_ref"]
    kp_search = debug_data["kp_search"]
    inlier_matches = debug_data["inlier_matches"]
    good_matches = debug_data["good_matches"]

    # Draw matches image (Inliers in green, outliers in red)
    if inlier_matches and kp_ref and kp_search:
        vis_matches = cv2.drawMatches(
            scaled_ref, kp_ref,
            search_img, kp_search,
            inlier_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    else:
        vis_matches = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    # Draw predicted target box and center on search image view
    vis_search = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    if debug_data["bounding_box"] is not None:
        pts = np.int32(debug_data["bounding_box"])
        cv2.polylines(vis_search, [pts], True, (0, 255, 0), 2)

    if pred_x is not None and pred_y is not None:
        cx, cy = int(round(pred_x)), int(round(pred_y))
        cv2.circle(vis_search, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawMarker(vis_search, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        label = f"Pred: ({pred_x:.2f}, {pred_y:.2f})"
        cv2.putText(vis_search, label, (max(10, cx - 80), max(25, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_search)
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature-based SIFT localizer for DriftSense-X")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to search image")
    parser.add_argument("--scale", type=float, default=0.10, help="Downscale factor for reference image (default: 0.10)")
    parser.add_argument("--ratio-thresh", type=float, default=0.75, help="Lowe's ratio test threshold (default: 0.75)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and save visualization")
    parser.add_argument("--vis-path", type=str, default="feature_debug.png", help="Output path for debug visualization")

    args = parser.parse_args()

    pred_x, pred_y, metrics, search_img, debug_data = locate_reference_pattern(
        ref_path=args.reference,
        search_path=args.search,
        scale_factor=args.scale,
        ratio_thresh=args.ratio_thresh
    )

    print(f"Status: {metrics['status']}")
    print(f"Reference Keypoints: {metrics['num_kp_ref']}")
    print(f"Search Keypoints: {metrics['num_kp_search']}")
    print(f"Descriptor Matches: {metrics['num_descriptor_matches']}")
    print(f"Good Matches (Ratio Test): {metrics['num_good_matches']}")
    print(f"RANSAC Inliers: {metrics['num_inliers']}")

    if pred_x is not None and pred_y is not None:
        print(f"Predicted center: ({pred_x:.2f}, {pred_y:.2f})")
    else:
        print("Predicted center: None (Localization Failed)")

    if args.debug:
        save_debug_visualization(
            search_img=search_img,
            debug_data=debug_data,
            pred_x=pred_x,
            pred_y=pred_y,
            output_path=args.vis_path
        )


if __name__ == "__main__":
    main()
