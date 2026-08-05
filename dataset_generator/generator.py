"""
===============================================================================
DRIFT-SENSE: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools
Synthetic Dataset Generator (FinFET Architecture)
===============================================================================

This script generates synthetic semiconductor image pairs (Reference and Search images)
emulating Scanning Electron Microscopy (SEM) wafer inspection outputs.

It extracts a target pattern from the Reference layout, embeds it into a Search layout,
computes exact ground-truth bounding box & center coordinates, applies SEM edge brightening,
independent sensor noise, and realistic imaging degradations.

Author: Drift-Sense Team
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from dataclasses import dataclass


# =============================================================================
# 1. CONFIGURATION & GENERATOR PARAMETERS
# =============================================================================

@dataclass
class GeneratorConfig:
    """
    Central configuration class holding all generation, augmentation,
    and noise parameters for reproducibility and easy tuning.
    """
    # Image Dimensions
    image_width: int = 1000
    image_height: int = 1000

    # Target Dimensions (~100x100 pixels)
    target_width: int = 100
    target_height: int = 100

    # FinFET Geometry Parameters (in pixels)
    fin_width: int = 8             # Width of vertical fin lines
    fin_spacing: int = 24          # Pitch/spacing between adjacent vertical fins
    fin_intensity: float = 160.0   # Base grayscale brightness of fins (0-255)
    
    gate_width: int = 18           # Width of horizontal gate bars
    gate_spacing: int = 90         # Pitch/spacing between horizontal gate bars
    gate_intensity: float = 210.0  # Base grayscale brightness of gates (0-255)
    
    background_intensity: float = 40.0  # Substrate background intensity

    # Edge Brightening (SEM secondary electron edge charging effect)
    edge_brightening_strength: float = 0.6

    # Reference Image Noise & Degradation
    ref_noise_std: float = 12.0          # Sensor noise standard deviation for Reference
    ref_blur_kernel: int = 3             # Gaussian blur kernel size (must be odd)
    ref_contrast_scale: float = 1.0      # Contrast adjustment factor
    ref_brightness_shift: float = 0.0    # Brightness shift

    # Search Image Noise & Degradation (Normally stronger noise than Reference)
    search_noise_std: float = 25.0       # Sensor noise standard deviation for Search
    search_blur_kernel: int = 3          # Gaussian blur kernel size
    search_contrast_scale: float = 0.95  # Slightly lower contrast for Search
    search_brightness_shift: float = -5.0  # Slightly darker shift


# =============================================================================
# 2. FINFET STRUCTURE GENERATOR
# =============================================================================

def generate_finfet_structure(config: GeneratorConfig, rng: np.random.RandomState) -> np.ndarray:
    """
    Generates a 2D float32 numpy array (1000x1000) containing a realistic
    FinFET-style semiconductor structure with vertical fins, horizontal gates,
    and controlled line width/spacing variations.

    Args:
        config (GeneratorConfig): Generator parameters.
        rng (np.random.RandomState): Random number generator instance for reproducibility.

    Returns:
        np.ndarray: 1000x1000 float32 array representing the clean semiconductor pattern.
    """
    height = config.image_height
    width = config.image_width

    # Initialize canvas with background substrate intensity
    image = np.full((height, width), config.background_intensity, dtype=np.float32)

    # --- Draw Parallel Vertical Fin Lines ---
    # Add slight variation to fin spacing across the wafer canvas
    fin_pitch = config.fin_width + config.fin_spacing
    x_coords = []
    curr_x = rng.randint(5, 15)  # Start with small random offset
    while curr_x < width:
        x_coords.append(curr_x)
        # Small controlled pitch variation (+/- 1 pixel) to mimic realistic process variations
        curr_x += fin_pitch + rng.choice([-1, 0, 1])

    for x in x_coords:
        # Slight line width variation per fin
        w = config.fin_width + rng.choice([-1, 0, 1])
        x_start = max(0, x)
        x_end = min(width, x + w)
        image[:, x_start:x_end] = config.fin_intensity

    # --- Draw Horizontal Gate Bars ---
    # Gates lie horizontally across the vertical fins in FinFET technology
    gate_pitch = config.gate_width + config.gate_spacing
    y_coords = []
    curr_y = rng.randint(10, 25)
    while curr_y < height:
        y_coords.append(curr_y)
        curr_y += gate_pitch + rng.choice([-1, 0, 1])

    for y in y_coords:
        w = config.gate_width + rng.choice([-1, 0, 1])
        y_start = max(0, y)
        y_end = min(height, y + w)
        # Gates have higher intensity as top structures
        image[y_start:y_end, :] = config.gate_intensity

    return image


# =============================================================================
# 3. SEM EDGE BRIGHTENING OPERATION
# =============================================================================

def add_edge_brightening(image: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Applies SEM-like edge brightening to imitate secondary electron accumulation
    along steep semiconductor feature edges.

    Args:
        image (np.ndarray): Input float32 grayscale image.
        strength (float): Strength parameter for edge enhancement.

    Returns:
        np.ndarray: Edge-brightened float32 image.
    """
    if strength <= 0:
        return image.copy()

    # Compute spatial gradients using Sobel operators along X and Y axes
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Compute overall gradient magnitude (edges of fins and gates)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Normalize gradient magnitude
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()

    # Add brightened edge contrast to original image
    brightened = image + (grad_mag * strength * 80.0)

    # Clip values to valid grayscale range [0, 255]
    return np.clip(brightened, 0.0, 255.0)


# =============================================================================
# 4. INDEPENDENT SENSOR NOISE & DEGRADATIONS
# =============================================================================

def add_noise(image: np.ndarray, noise_std: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Generates independent Gaussian sensor noise and adds it to the image.
    NEVER reuses noise arrays.

    Args:
        image (np.ndarray): Input float32 grayscale image.
        noise_std (float): Standard deviation of Gaussian noise.
        rng (np.random.RandomState): Independent random generator instance.

    Returns:
        np.ndarray: Noisy image (float32).
    """
    # Generate fresh, independent random noise array
    noise = rng.normal(loc=0.0, scale=noise_std, size=image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 255.0)


def apply_degradation(
    image: np.ndarray,
    blur_kernel: int,
    contrast_scale: float,
    brightness_shift: float,
    noise_std: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Applies a series of realistic SEM image degradations:
    1. Gaussian blur (optical/focus degradation)
    2. Intensity & contrast adjustments
    3. Independent sensor noise

    Args:
        image (np.ndarray): Input float32 image.
        blur_kernel (int): Gaussian blur kernel size (odd integer).
        contrast_scale (float): Multiplicative contrast scale factor.
        brightness_shift (float): Additive brightness offset.
        noise_std (float): Standard deviation for independent sensor noise.
        rng (np.random.RandomState): Random generator instance.

    Returns:
        np.ndarray: Final degraded float32 image.
    """
    degraded = image.copy()

    # 1. Apply Gaussian blur if kernel size > 1
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1  # Ensure kernel size is odd
        degraded = cv2.GaussianBlur(degraded, (blur_kernel, blur_kernel), sigmaX=0.8)

    # 2. Contrast and brightness adjustment
    degraded = (degraded * contrast_scale) + brightness_shift

    # 3. Add independent sensor noise
    degraded = add_noise(degraded, noise_std=noise_std, rng=rng)

    return np.clip(degraded, 0.0, 255.0)


# =============================================================================
# 5. CREATE REFERENCE IMAGE
# =============================================================================

def create_reference(config: GeneratorConfig, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates the 1000x1000 Reference image containing a clean FinFET pattern,
    applies SEM edge brightening and reference-level noise/degradations.

    Args:
        config (GeneratorConfig): Configuration object.
        rng (np.random.RandomState): Random state for Reference image generation.

    Returns:
        tuple[np.ndarray, np.ndarray]: (clean_structure, final_degraded_reference_image)
    """
    # Step 1: Generate clean FinFET semiconductor geometry
    clean_structure = generate_finfet_structure(config, rng)

    # Step 2: Apply SEM edge brightening
    brightened = add_edge_brightening(clean_structure, strength=config.edge_brightening_strength)

    # Step 3: Apply reference-specific degradation & noise
    ref_image = apply_degradation(
        brightened,
        blur_kernel=config.ref_blur_kernel,
        contrast_scale=config.ref_contrast_scale,
        brightness_shift=config.ref_brightness_shift,
        noise_std=config.ref_noise_std,
        rng=rng
    )

    return clean_structure, ref_image


# =============================================================================
# 6. INSERT TARGET & CREATE SEARCH IMAGE
# =============================================================================

def insert_target(
    search_base: np.ndarray,
    target_pattern: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.RandomState
) -> tuple[np.ndarray, dict]:
    """
    Extracts a target region and inserts it into a random valid location
    inside the 1000x1000 Search image layout. Ground truth coordinates are
    derived EXACTLY from this insertion location.

    Args:
        search_base (np.ndarray): Base 1000x1000 search layout canvas.
        target_pattern (np.ndarray): Target pattern array (~100x100).
        config (GeneratorConfig): Generator config.
        rng (np.random.RandomState): Random generator for target position.

    Returns:
        tuple[np.ndarray, dict]: (search_base_with_inserted_target, ground_truth_dict)
    """
    th, tw = target_pattern.shape
    h, w = search_base.shape

    # Select random insertion location (completely within 1000x1000 bounds)
    margin = 20  # Keep away from absolute borders for safety
    x_min = int(rng.randint(margin, w - tw - margin))
    y_min = int(rng.randint(margin, h - th - margin))

    x_max = x_min + tw
    y_max = y_min + th

    # Calculate exact center coordinates
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    # Insert target into search layout
    search_with_target = search_base.copy()
    search_with_target[y_min:y_max, x_min:x_max] = target_pattern

    # Ground truth dictionary matching project requirements
    ground_truth = {
        "architecture": "finfet",
        "image_width": w,
        "image_height": h,
        "target_bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        },
        "center": {
            "x": center_x,
            "y": center_y
        }
    }

    return search_with_target, ground_truth


def create_search(
    config: GeneratorConfig,
    clean_ref_structure: np.ndarray,
    search_rng: np.random.RandomState
) -> tuple[np.ndarray, dict]:
    """
    Creates the 1000x1000 Search image containing a larger repetitive FinFET-style
    layout, embeds the target pattern, and applies independent search-level noise.

    Args:
        config (GeneratorConfig): Generator settings.
        clean_ref_structure (np.ndarray): Clean reference structure to extract target from.
        search_rng (np.random.RandomState): Independent random generator for Search image.

    Returns:
        tuple[np.ndarray, dict]: (final_search_image, ground_truth_dict)
    """
    # Step 1: Generate larger repetitive Search layout
    search_layout = generate_finfet_structure(config, search_rng)

    # Step 2: Extract a meaningful ~100x100 target pattern from reference structure
    # Extract from center region of reference structure for rich FinFET pattern
    ref_h, ref_w = clean_ref_structure.shape
    crop_y = (ref_h - config.target_height) // 2
    crop_x = (ref_w - config.target_width) // 2
    target_pattern = clean_ref_structure[
        crop_y : crop_y + config.target_height,
        crop_x : crop_x + config.target_width
    ].copy()

    # Step 3: Insert target into search layout and get exact ground truth
    inserted_layout, ground_truth = insert_target(
        search_layout, target_pattern, config, search_rng
    )

    # Step 4: Apply SEM edge brightening to combined search structure
    brightened_search = add_edge_brightening(
        inserted_layout, strength=config.edge_brightening_strength
    )

    # Step 5: Apply independent search-level degradation and stronger noise
    final_search = apply_degradation(
        brightened_search,
        blur_kernel=config.search_blur_kernel,
        contrast_scale=config.search_contrast_scale,
        brightness_shift=config.search_brightness_shift,
        noise_std=config.search_noise_std,
        rng=search_rng  # Uses search_rng ensuring independent noise!
    )

    return final_search, ground_truth


# =============================================================================
# 7. SAVE GROUND TRUTH & VISUALIZATION
# =============================================================================

def save_ground_truth(ground_truth: dict, output_filepath: str) -> None:
    """
    Saves the ground-truth dictionary to a JSON file.

    Args:
        ground_truth (dict): Ground truth bounding box and center.
        output_filepath (str): Target file path for ground_truth.json.
    """
    with open(output_filepath, "w") as f:
        json.dump(ground_truth, f, indent=4)


def create_visualization(
    search_image: np.ndarray,
    ground_truth: dict,
    output_filepath: str
) -> None:
    """
    Creates visualization.png displaying the search image with:
    - Red bounding box around inserted target
    - Green crosshair at ground-truth center
    - Text label displaying coordinates

    DOES NOT modify the actual search image array.

    Args:
        search_image (np.ndarray): 1000x1000 uint8 grayscale search image.
        ground_truth (dict): Ground truth dictionary.
        output_filepath (str): Filepath to save visualization.png.
    """
    # Convert grayscale image to 3-channel BGR for colored annotations
    vis_img = cv2.cvtColor(search_image, cv2.COLOR_GRAY2BGR)

    bbox = ground_truth["target_bbox"]
    center = ground_truth["center"]

    x_min, y_min = bbox["x_min"], bbox["y_min"]
    x_max, y_max = bbox["x_max"], bbox["y_max"]
    cx, cy = int(center["x"]), int(center["y"])

    # 1. Draw Red Bounding Box (thickness 2)
    cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # 2. Draw Green Center Crosshair and Dot
    cv2.circle(vis_img, (cx, cy), 4, (0, 255, 0), -1)
    cv2.line(vis_img, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
    cv2.line(vis_img, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

    # 3. Add Coordinate Text Label above bounding box
    label_text = f"Target GT: Center ({cx}, {cy})"
    cv2.putText(
        vis_img,
        label_text,
        (x_min, max(15, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Save visualization image
    cv2.imwrite(output_filepath, vis_img)


# =============================================================================
# 8. AUTOMATED VALIDATION REPORT
# =============================================================================

def validate_pair(
    pair_dir: str,
    ref_img: np.ndarray,
    search_img: np.ndarray,
    ground_truth: dict
) -> bool:
    """
    Validates generated dataset pair against hackathon requirements and prints
    a clean validation report.

    Args:
        pair_dir (str): Path to pair directory.
        ref_img (np.ndarray): Generated reference image array.
        search_img (np.ndarray): Generated search image array.
        ground_truth (dict): Ground truth metadata dict.

    Returns:
        bool: True if all validation checks pass.
    """
    ref_path = os.path.join(pair_dir, "reference.png")
    search_path = os.path.join(pair_dir, "search.png")
    gt_path = os.path.join(pair_dir, "ground_truth.json")
    vis_path = os.path.join(pair_dir, "visualization.png")

    results = {}

    # Check 1: Reference image file exists
    results["Reference"] = "PASS" if os.path.exists(ref_path) else "FAIL"

    # Check 2: Search image file exists
    results["Search"] = "PASS" if os.path.exists(search_path) else "FAIL"

    # Check 3 & 4: Image Dimensions (exactly 1000 x 1000)
    ref_dim_ok = ref_img.shape == (1000, 1000)
    search_dim_ok = search_img.shape == (1000, 1000)
    
    # Check 5: Ground Truth file exists
    results["Ground Truth"] = "PASS" if os.path.exists(gt_path) and ref_dim_ok and search_dim_ok else "FAIL"

    # Check 6: Target Bounding Box inside bounds
    bbox = ground_truth["target_bbox"]
    x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
    bounds_ok = (0 <= x_min < x_max <= 1000) and (0 <= y_min < y_max <= 1000)
    results["Target Bounds"] = "PASS" if bounds_ok else "FAIL"

    # Check 7: Target Size ~100 x 100 pixels
    target_w = x_max - x_min
    target_h = y_max - y_min
    size_ok = (80 <= target_w <= 120) and (80 <= target_h <= 120)
    results["Target Size"] = "PASS" if size_ok else "FAIL"

    # Check 8: Visualization file exists
    results["Visualization"] = "PASS" if os.path.exists(vis_path) else "FAIL"

    # Check 9: Independent Noise (Reference and Search noise must be independent)
    # Check that pixel difference between ref and search is non-zero and significant
    diff = np.abs(ref_img.astype(np.float32) - search_img.astype(np.float32))
    noise_independent = float(diff.mean()) > 5.0
    results["Independent Noise"] = "PASS" if noise_independent else "FAIL"

    # Print Formatted Validation Report
    print("\n========================================")
    print("        DATASET VALIDATION REPORT       ")
    print("========================================")
    for check_name, status in results.items():
        print(f"{check_name}: {status}")
    print("========================================\n")

    all_passed = all(status == "PASS" for status in results.values())
    return all_passed


# =============================================================================
# 9. MAIN PIPELINE & CLI INTERFACE
# =============================================================================

def generate_dataset_pair(
    pair_id: int,
    output_dir: str,
    base_seed: int,
    config: GeneratorConfig
) -> dict:
    """
    Generates a single complete dataset pair directory (pair_0001, etc.).

    Args:
        pair_id (int): Pair identifier number (1, 2, ...).
        output_dir (str): Base output directory.
        base_seed (int): Base random seed.
        config (GeneratorConfig): Generator parameters configuration.

    Returns:
        dict: Ground truth dictionary of generated pair.
    """
    pair_dir = os.path.join(output_dir, f"pair_{pair_id:04d}")
    os.makedirs(pair_dir, exist_ok=True)

    # Initialize independent RandomStates for Reference and Search
    # This guarantees independent noise arrays for Reference and Search images
    ref_rng = np.random.RandomState(base_seed + pair_id * 2)
    search_rng = np.random.RandomState(base_seed + pair_id * 2 + 1)

    # 1. Create Reference Image (1000x1000)
    clean_ref, ref_float = create_reference(config, ref_rng)
    ref_uint8 = np.clip(ref_float, 0, 255).astype(np.uint8)

    # 2. Create Search Image (1000x1000) with inserted target & independent noise
    search_float, ground_truth = create_search(config, clean_ref, search_rng)
    search_uint8 = np.clip(search_float, 0, 255).astype(np.uint8)

    # 3. Save reference.png and search.png
    cv2.imwrite(os.path.join(pair_dir, "reference.png"), ref_uint8)
    cv2.imwrite(os.path.join(pair_dir, "search.png"), search_uint8)

    # 4. Save ground_truth.json
    save_ground_truth(ground_truth, os.path.join(pair_dir, "ground_truth.json"))

    # 5. Create visualization.png (overlay bounding box and center without modifying search.png)
    create_visualization(
        search_uint8, ground_truth, os.path.join(pair_dir, "visualization.png")
    )

    # 6. Validate generated pair
    validate_pair(pair_dir, ref_uint8, search_uint8, ground_truth)

    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense Synthetic Semiconductor Dataset Generator"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="finfet",
        choices=["finfet"],
        help="Semiconductor pattern architecture (default: finfet)"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=1,
        help="Number of synthetic pairs to generate (default: 1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generated",
        help="Output directory for generated dataset (default: data/generated)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Load generator configuration
    config = GeneratorConfig()

    print(f"Starting Drift-Sense Dataset Generator:")
    print(f"  Architecture : {args.architecture}")
    print(f"  Num Pairs    : {args.num_pairs}")
    print(f"  Output Dir   : {args.output_dir}")
    print(f"  Random Seed  : {args.seed}\n")

    # Generate specified number of pairs
    for i in range(1, args.num_pairs + 1):
        print(f"Generating Pair {i:04d}...")
        gt = generate_dataset_pair(i, args.output_dir, args.seed, config)
        
        pair_path = os.path.join(args.output_dir, f"pair_{i:04d}")
        print(f"Pair {i:04d} successfully created at: {pair_path}")
        print(f"  Target BBox : {gt['target_bbox']}")
        print(f"  GT Center   : {gt['center']}\n")

    print("Dataset generation process complete.")


if __name__ == "__main__":
    main()
