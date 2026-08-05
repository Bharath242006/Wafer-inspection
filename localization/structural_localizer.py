"""
structural_localizer.py

First improved structural localization method for DriftSense-X.
Uses multi-scale template matching combined with edge/gradient features,
intensity correlation, distance-to-center penalties, and local uniqueness scoring.
"""

import argparse
import os
import cv2
import numpy as np


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Computes normalized Sobel gradient magnitude image."""
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def extract_top_k_candidates(response_map: np.ndarray, k: int = 20, window_size: int = 5) -> list:
    """
    Extracts top-k candidate peak locations (top_left_x, top_left_y, score) from a response map
    using morphological dilation to find local maxima.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dilated = cv2.dilate(response_map, kernel)
    local_peaks = (response_map == dilated) & (response_map > 0)

    peak_y, peak_x = np.where(local_peaks)
    scores = response_map[peak_y, peak_x]

    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:k]
    candidates = []
    for idx in top_indices:
        candidates.append((int(peak_x[idx]), int(peak_y[idx]), float(scores[idx])))
    return candidates


def remove_duplicate_candidates(candidates: list, distance_threshold: float = 15.0) -> list:
    """
    Suppresses near-duplicate candidate centers based on spatial distance threshold.
    """
    candidates_sorted = sorted(candidates, key=lambda c: c['raw_score'], reverse=True)
    kept = []

    for cand in candidates_sorted:
        cx, cy = cand['center_x'], cand['center_y']
        too_close = False
        for k in kept:
            dist = np.hypot(cx - k['center_x'], cy - k['center_y'])
            if dist < distance_threshold:
                too_close = True
                break
        if not too_close:
            kept.append(cand)

    return kept


def locate_reference_pattern(
    ref_path: str,
    search_path: str,
    scale_min: float = 0.08,
    scale_max: float = 0.12,
    num_scales: int = 9,
    candidates_per_scale: int = 20
) -> tuple:
    """
    Locates the center coordinate (x, y) of the reference pattern within the search image.

    Args:
        ref_path (str): Path to reference image (1000x1000).
        search_path (str): Path to search image (1000x1000).
        scale_min (float): Minimum reference downscale factor (default: 0.08).
        scale_max (float): Maximum reference downscale factor (default: 0.12).
        num_scales (int): Number of scale steps to test (default: 9).
        candidates_per_scale (int): Number of top peaks per scale (default: 20).

    Returns:
        tuple: (pred_x, pred_y, best_candidate_dict, search_img, top_left, scaled_w, scaled_h)
    """
    # 1. Load reference and search images in grayscale
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_img is None:
        raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    search_h, search_w = search_img.shape[:2]
    center_search_x = search_w / 2.0
    center_search_y = search_h / 2.0
    max_center_dist = np.hypot(center_search_x, center_search_y)

    # 2. & 3. Edge / gradient representations using OpenCV Sobel
    ref_grad = compute_gradient_magnitude(ref_img)
    search_grad = compute_gradient_magnitude(search_img)

    # 4. Test multiple template scales from 0.08 to 0.12
    scales = np.linspace(scale_min, scale_max, num_scales)
    all_candidates = []

    for s in scales:
        scaled_w = int(round(ref_img.shape[1] * s))
        scaled_h = int(round(ref_img.shape[0] * s))

        scaled_ref_int = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        scaled_ref_grad = cv2.resize(ref_grad, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Template matching on intensity and gradient representations
        res_int = cv2.matchTemplate(search_img, scaled_ref_int, cv2.TM_CCOEFF_NORMED)
        res_grad = cv2.matchTemplate(search_grad, scaled_ref_grad, cv2.TM_CCOEFF_NORMED)

        # 5. Extract top candidate locations per scale
        top_int = extract_top_k_candidates(res_int, k=candidates_per_scale)
        top_grad = extract_top_k_candidates(res_grad, k=candidates_per_scale)

        loc_set = set([(x, y) for x, y, _ in top_int] + [(x, y) for x, y, _ in top_grad])

        for (tl_x, tl_y) in loc_set:
            cx = tl_x + (scaled_w / 2.0)
            cy = tl_y + (scaled_h / 2.0)

            score_int = float(res_int[tl_y, tl_x]) if 0 <= tl_y < res_int.shape[0] and 0 <= tl_x < res_int.shape[1] else 0.0
            score_grad = float(res_grad[tl_y, tl_x]) if 0 <= tl_y < res_grad.shape[0] and 0 <= tl_x < res_grad.shape[1] else 0.0

            # Calculate local uniqueness (peak minus mean of surrounding neighborhood)
            y_min = max(0, tl_y - 10)
            y_max = min(res_int.shape[0], tl_y + 11)
            x_min = max(0, tl_x - 10)
            x_max = min(res_int.shape[1], tl_x + 11)

            patch_int = res_int[y_min:y_max, x_min:x_max]
            local_mean_int = float(np.mean(patch_int)) if patch_int.size > 0 else score_int
            uniqueness_int = score_int - local_mean_int

            dist_to_center = float(np.hypot(cx - center_search_x, cy - center_search_y))
            raw_score = 0.5 * score_int + 0.5 * score_grad

            all_candidates.append({
                'top_left': (tl_x, tl_y),
                'scaled_w': scaled_w,
                'scaled_h': scaled_h,
                'scale': s,
                'center_x': cx,
                'center_y': cy,
                'score_intensity': score_int,
                'score_edge': score_grad,
                'dist_to_center': dist_to_center,
                'uniqueness': uniqueness_int,
                'raw_score': raw_score
            })

    # 6. Remove duplicate/near-identical candidate locations
    filtered_candidates = remove_duplicate_candidates(all_candidates, distance_threshold=15.0)

    # 7. & 8. Score candidates combining edge similarity, intensity similarity, center distance, and local uniqueness
    best_candidate = None
    best_combined_score = -float('inf')

    for cand in filtered_candidates:
        s_edge = cand['score_edge']
        s_int = cand['score_intensity']
        norm_dist = cand['dist_to_center'] / max_center_dist
        s_uniq = cand['uniqueness']

        # Combined scoring formula
        combined_score = 0.40 * s_edge + 0.40 * s_int + 0.15 * s_uniq - 0.05 * norm_dist
        cand['combined_score'] = combined_score

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_candidate = cand

    pred_x = float(best_candidate['center_x'])
    pred_y = float(best_candidate['center_y'])

    return (
        pred_x,
        pred_y,
        best_candidate,
        search_img,
        best_candidate['top_left'],
        best_candidate['scaled_w'],
        best_candidate['scaled_h']
    )


def save_debug_visualization(
    search_img: np.ndarray,
    top_left: tuple,
    scaled_w: int,
    scaled_h: int,
    center_x: float,
    center_y: float,
    output_path: str
):
    """Saves visual overlay showing search image, bounding box, and center prediction."""
    vis_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    x1, y1 = top_left
    x2, y2 = x1 + scaled_w, y1 + scaled_h

    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cx, cy = int(round(center_x)), int(round(center_y))
    cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.drawMarker(vis_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    label = f"Pred: ({center_x:.2f}, {center_y:.2f})"
    cv2.putText(vis_img, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Structural localizer for DriftSense-X")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to search image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and save visualization")
    parser.add_argument("--vis-path", type=str, default="structural_debug.png", help="Output path for debug visualization")

    args = parser.parse_args()

    pred_x, pred_y, cand, search_img, top_left, scaled_w, scaled_h = locate_reference_pattern(
        ref_path=args.reference,
        search_path=args.search
    )

    print(f"Predicted center: ({pred_x:.2f}, {pred_y:.2f})")

    if args.debug:
        save_debug_visualization(
            search_img=search_img,
            top_left=top_left,
            scaled_w=scaled_w,
            scaled_h=scaled_h,
            center_x=pred_x,
            center_y=pred_y,
            output_path=args.vis_path
        )


if __name__ == "__main__":
    main()
