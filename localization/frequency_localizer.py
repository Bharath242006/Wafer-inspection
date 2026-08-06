"""
frequency_localizer.py

Frequency-domain and phase correlation localization method for DriftSense-X.
Uses Sobel gradient preprocessing, 2D FFT correlation, sub-window phase correlation
(cv2.phaseCorrelate), multi-feature candidate scoring, and safety thresholding.
"""

import argparse
import os
import cv2
import numpy as np


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Computes normalized Sobel gradient magnitude image in float32."""
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    cv2.normalize(mag, mag, 0, 1, cv2.NORM_MINMAX)
    return mag.astype(np.float32)


def extract_candidate_regions(res_map: np.ndarray, num_candidates: int = 15, window_size: int = 7) -> list:
    """Extracts top candidate top-left coordinates from correlation response map."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dilated = cv2.dilate(res_map, kernel)
    local_peaks = (res_map == dilated) & (res_map > 0)

    peak_y, peak_x = np.where(local_peaks)
    scores = res_map[peak_y, peak_x]

    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:num_candidates]
    candidates = []
    for idx in top_indices:
        candidates.append((int(peak_x[idx]), int(peak_y[idx]), float(scores[idx])))
    return candidates


def locate_reference_pattern(
    ref_path: str,
    search_path: str,
    scale_factor: float = 0.10,
    min_phase_response: float = 0.03,
    num_candidates: int = 15
) -> tuple:
    """
    Locates reference pattern in search image using FFT phase correlation.

    Args:
        ref_path (str): Path to 1000x1000 reference image.
        search_path (str): Path to 1000x1000 search image.
        scale_factor (float): Downscale ratio for reference (default: 0.10).
        min_phase_response (float): Minimum phase correlation response threshold (default: 0.03).
        num_candidates (int): Number of candidate search regions (default: 15).

    Returns:
        tuple: (pred_x, pred_y, metrics_dict, search_img, debug_data)
    """
    # 1. Load images in grayscale float32
    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_raw is None:
        raise FileNotFoundError(f"Could not load reference image at '{ref_path}'")

    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search_raw is None:
        raise FileNotFoundError(f"Could not load search image at '{search_path}'")

    ref_img = ref_raw.astype(np.float32)
    search_img = search_raw.astype(np.float32)

    # 2. Downscale reference image to ~100x100 representation (~0.10 scale)
    scaled_w = int(round(ref_img.shape[1] * scale_factor))
    scaled_h = int(round(ref_img.shape[0] * scale_factor))
    scaled_ref = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # 3. & 4. Edge / gradient preprocessing
    ref_grad = compute_gradient_magnitude(scaled_ref)
    search_grad = compute_gradient_magnitude(search_img)

    # 5. & 6. FFT correlation & Candidate Generation
    res_grad = cv2.matchTemplate(search_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    res_int = cv2.matchTemplate(search_img, scaled_ref, cv2.TM_CCOEFF_NORMED)

    # Combine correlation maps
    res_combined = 0.5 * res_grad + 0.5 * res_int

    candidate_locs = extract_candidate_regions(res_combined, num_candidates=num_candidates)

    metrics = {
        "num_candidates": len(candidate_locs),
        "best_freq_score": 0.0,
        "phase_response": 0.0,
        "status": "FAILED"
    }

    debug_data = {
        "candidate_boxes": [],
        "best_box": None,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h
    }

    if len(candidate_locs) == 0:
        metrics["status"] = "NO_CANDIDATES"
        return None, None, metrics, search_raw, debug_data

    # Create Hanning window for phaseCorrelate
    hann_window = cv2.createHanningWindow((scaled_w, scaled_h), cv2.CV_32F)

    evaluated_candidates = []

    # 7. Sub-window cv2.phaseCorrelate evaluation
    for (tl_x, tl_y, raw_score) in candidate_locs:
        # Extract search crop corresponding to candidate bounding box
        crop_search = search_grad[tl_y:tl_y + scaled_h, tl_x:tl_x + scaled_w]

        if crop_search.shape != (scaled_h, scaled_w):
            continue

        # Phase correlation between template gradient and crop gradient
        (shift_x, shift_y), phase_resp = cv2.phaseCorrelate(ref_grad, crop_search, hann_window)

        # Refined top-left and center
        refined_tl_x = tl_x + shift_x
        refined_tl_y = tl_y + shift_y
        refined_cx = refined_tl_x + (scaled_w / 2.0)
        refined_cy = refined_tl_y + (scaled_h / 2.0)

        # Gradient / edge similarity
        edge_sim = float(res_grad[tl_y, tl_x]) if 0 <= tl_y < res_grad.shape[0] and 0 <= tl_x < res_grad.shape[1] else 0.0
        freq_sim = float(res_int[tl_y, tl_x]) if 0 <= tl_y < res_int.shape[0] and 0 <= tl_x < res_int.shape[1] else 0.0

        # Distance penalty from search image center (500, 500)
        dist_to_center = float(np.hypot(refined_cx - 500.0, refined_cy - 500.0)) / 707.1

        # 8. & 9. Combined score ranking
        combined_score = 0.40 * phase_resp + 0.40 * freq_sim + 0.20 * edge_sim - 0.05 * dist_to_center

        cand_info = {
            "top_left": (tl_x, tl_y),
            "refined_center": (refined_cx, refined_cy),
            "phase_response": float(phase_resp),
            "freq_sim": freq_sim,
            "edge_sim": edge_sim,
            "combined_score": combined_score
        }
        evaluated_candidates.append(cand_info)
        debug_data["candidate_boxes"].append((tl_x, tl_y, scaled_w, scaled_h))

    if not evaluated_candidates:
        metrics["status"] = "INVALID_CROPS"
        return None, None, metrics, search_raw, debug_data

    # Rank candidates by combined score
    evaluated_candidates.sort(key=lambda c: c["combined_score"], reverse=True)
    best_cand = evaluated_candidates[0]

    metrics["best_freq_score"] = best_cand["freq_sim"]
    metrics["phase_response"] = best_cand["phase_response"]

    # 11. Safety Check: Return None if phase response or combined score is unreliable
    if best_cand["phase_response"] < min_phase_response or best_cand["combined_score"] <= 0.0:
        metrics["status"] = "FAILED_UNRELIABLE"
        return None, None, metrics, search_raw, debug_data

    metrics["status"] = "SUCCESS"
    pred_x, pred_y = best_cand["refined_center"]
    debug_data["best_box"] = (best_cand["top_left"][0], best_cand["top_left"][1], scaled_w, scaled_h)

    return pred_x, pred_y, metrics, search_raw, debug_data


def save_debug_visualization(
    search_img: np.ndarray,
    debug_data: dict,
    pred_x: float,
    pred_y: float,
    output_path: str
):
    """Saves visual overlay showing search image, candidate locations, best location, and predicted center."""
    vis_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    scaled_w = debug_data["scaled_w"]
    scaled_h = debug_data["scaled_h"]

    # Draw all candidate bounding boxes in blue
    for (cx, cy, w, h) in debug_data["candidate_boxes"]:
        cv2.rectangle(vis_img, (cx, cy), (cx + w, cy + h), (255, 100, 0), 1)

    # Draw best predicted box in green and center marker in red
    if debug_data["best_box"] is not None:
        bx, by, bw, bh = debug_data["best_box"]
        cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    if pred_x is not None and pred_y is not None:
        cx, cy = int(round(pred_x)), int(round(pred_y))
        cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawMarker(vis_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        label = f"Pred Center: ({pred_x:.2f}, {pred_y:.2f})"
        cv2.putText(vis_img, label, (max(10, cx - 80), max(25, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Frequency-domain & phase correlation localizer for DriftSense-X")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to search image")
    parser.add_argument("--scale", type=float, default=0.10, help="Downscale factor for reference image (default: 0.10)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and save visualization")
    parser.add_argument("--vis-path", type=str, default="frequency_debug.png", help="Output path for debug visualization")

    args = parser.parse_args()

    pred_x, pred_y, metrics, search_img, debug_data = locate_reference_pattern(
        ref_path=args.reference,
        search_path=args.search,
        scale_factor=args.scale
    )

    print(f"Localization Status: {metrics['status']}")
    print(f"Number of Candidates: {metrics['num_candidates']}")
    print(f"Best Frequency Similarity Score: {metrics['best_freq_score']:.4f}")
    print(f"Phase Correlation Response: {metrics['phase_response']:.4f}")

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

locate_reference_pattern_frequency = locate_reference_pattern


if __name__ == "__main__":
    main()
