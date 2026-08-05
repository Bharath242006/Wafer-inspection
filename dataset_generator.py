"""
dataset_generator.py

Master synthetic dataset generator script for the Applied Materials Drift-Sense challenge.

Generates a complete dataset of paired 100x high-resolution Reference images and 10x
downsampled Search images with exact ground-truth center coordinates under spatial warp
and SEM physical augmentations.

Dataset Splits:
- Train: 1,000 pairs
- Validation: 200 pairs
- Test: 200 pairs
- Visualizations: 20 sample overlays

Run command:
python dataset_generator.py
"""

import os
import time
import numpy as np
import cv2

# Import project modules
from utils import set_seed, ensure_dir, save_image_pillow, write_labels_csv
from generate_dram import generate_dram_layout, get_random_dram_params
from generate_finfet import generate_finfet_layout, get_random_finfet_params
from augmentations import apply_full_augmentation_pipeline
from visualize import draw_visualization


def generate_split(
    split_name: str,
    num_samples: int,
    output_dir: str,
    rng: np.random.RandomState,
    save_vis_count: int = 0,
    vis_dir: str = None
) -> None:
    """
    Generates synthetic image pairs (Reference and Search) for a specific dataset split.

    Args:
        split_name (str): Split identifier ("train", "validation", or "test").
        num_samples (int): Number of image pairs to generate.
        output_dir (str): Base output directory (e.g. "dataset/").
        rng (np.random.RandomState): Master random state.
        save_vis_count (int): Number of visual overlays to render (for first N samples).
        vis_dir (str, optional): Directory to save visual overlay images.
    """
    print(f"\n==================================================")
    print(f"Generating Split: '{split_name.upper()}' ({num_samples} pairs)")
    print(f"==================================================")

    # 1. Define folder paths
    split_dir = os.path.join(output_dir, split_name)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    csv_path = os.path.join(split_dir, "labels.csv")

    ensure_dir(ref_dir)
    ensure_dir(search_dir)
    if vis_dir:
        ensure_dir(vis_dir)

    labels = []
    start_time = time.time()

    for idx in range(1, num_samples + 1):
        filename = f"{idx:05d}.png"
        
        # 2. Randomly select layout style (50/50 balance between DRAM and FinFET)
        style = "DRAM" if rng.rand() < 0.5 else "FinFET"

        # 3. Generate 10000x10000 high-resolution wafer layout
        if style == "DRAM":
            dram_params = get_random_dram_params(rng)
            layout, _ = generate_dram_layout(width=10000, height=10000, rng=rng, params=dram_params)
        else:
            finfet_params = get_random_finfet_params(rng)
            layout, _ = generate_finfet_layout(width=10000, height=10000, rng=rng, params=finfet_params)

        # 4. Crop Reference Image (1000x1000 from high-res layout)
        # Random crop top-left coordinate (leave margin to ensure full 1000x1000 box)
        crop_x = rng.randint(500, 8500)
        crop_y = rng.randint(500, 8500)
        ref_crop_base = layout[crop_y : crop_y + 1000, crop_x : crop_x + 1000].copy()

        # 5. Create Search Image Base (Downsample 10000x10000 high-res layout to 1000x1000)
        # Downsample factor = 10x
        search_base = cv2.resize(layout, (1000, 1000), interpolation=cv2.INTER_AREA)

        # Calculate exact center coordinate of reference inside downsampled search image
        # High-res reference center = (crop_x + 500, crop_y + 500)
        # Downsampled search center = (crop_x + 500) * 0.1, (crop_y + 500) * 0.1
        base_center_x = (crop_x + 500.0) * 0.1
        base_center_y = (crop_y + 500.0) * 0.1

        # Free high-res layout memory
        del layout

        # 6. Apply Independent SEM Augmentations & Geometric Warping
        # Reference Image Augmentation
        augmented_ref, _, _ = apply_full_augmentation_pipeline(
            ref_crop_base, rng=rng, center_coord=None, is_search=False
        )

        # Search Image Augmentation (Tracks ground-truth center under geometric rotation/scale/translation)
        augmented_search, (gt_x, gt_y), _ = apply_full_augmentation_pipeline(
            search_base, rng=rng, center_coord=(base_center_x, base_center_y), is_search=True
        )

        # 7. Save Reference & Search Images via Pillow
        ref_save_path = os.path.join(ref_dir, filename)
        search_save_path = os.path.join(search_dir, filename)
        
        save_image_pillow(augmented_ref, ref_save_path)
        save_image_pillow(augmented_search, search_save_path)

        # 8. Record Ground-Truth Label (filename, x, y, style)
        labels.append((filename, gt_x, gt_y, style))

        # 9. Render & Save Visualization Overlay for first N samples
        if idx <= save_vis_count and vis_dir:
            vis_save_path = os.path.join(vis_dir, f"sample_{idx:04d}.png")
            draw_visualization(
                ref_image=augmented_ref,
                search_image=augmented_search,
                center_x=gt_x,
                center_y=gt_y,
                style_name=style,
                box_size=100.0,
                save_path=vis_save_path
            )

        # Log progress periodically
        if idx % 100 == 0 or idx == num_samples:
            elapsed = time.time() - start_time
            print(f"[{split_name.capitalize()}] Completed {idx}/{num_samples} pairs ({elapsed:.1f}s) | Last: {filename} -> GT: ({gt_x:.2f}, {gt_y:.2f}), Style: {style}")

    # 10. Save labels.csv
    write_labels_csv(csv_path, labels)
    print(f"--> Saved '{csv_path}' with {len(labels)} ground-truth records.")


def main():
    """
    Main entry point for generating the complete Drift-Sense synthetic wafer dataset.
    """
    print("\n=======================================================================")
    print(" Applied Materials Drift-Sense Synthetic Wafer Dataset Generator")
    print("=======================================================================\n")

    # Master Configuration
    MASTER_SEED = 42
    OUTPUT_DIR = "dataset"
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

    SPLIT_COUNTS = {
        "train": 1000,
        "validation": 200,
        "test": 200
    }

    # Initialize reproducible master random state
    rng = set_seed(MASTER_SEED)
    total_start_time = time.time()

    # Generate Train, Validation, and Test splits
    for split_name, count in SPLIT_COUNTS.items():
        # Save visualizations for the first 20 samples of train split
        vis_count = 20 if split_name == "train" else 0
        generate_split(
            split_name=split_name,
            num_samples=count,
            output_dir=OUTPUT_DIR,
            rng=rng,
            save_vis_count=vis_count,
            vis_dir=VISUALIZATION_DIR if vis_count > 0 else None
        )

    total_elapsed = time.time() - total_start_time
    print("\n=======================================================================")
    print(f" SUCCESS: Complete dataset successfully generated in {total_elapsed:.1f} seconds!")
    print(f" Root Directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f" Visualizations: {os.path.abspath(VISUALIZATION_DIR)}")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()
