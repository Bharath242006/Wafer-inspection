"""
visualize.py

Visualization utility for the Applied Materials Drift-Sense synthetic dataset generator.
Generates side-by-side comparison images for Reference and Search image pairs with:
- Ground-truth bounding box
- Ground-truth center point crosshair
- Annotations for coordinates and layout style
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple


def draw_visualization(
    ref_image: np.ndarray,
    search_image: np.ndarray,
    center_x: float,
    center_y: float,
    style_name: str,
    box_size: float = 100.0,
    save_path: str = None
) -> np.ndarray:
    """
    Renders a high-quality visual overlay comparing Reference and Search images.

    Args:
        ref_image (np.ndarray): Reference image 1000x1000 array (uint8 or float32).
        search_image (np.ndarray): Search image 1000x1000 array (uint8 or float32).
        center_x (float): Ground-truth center X coordinate in search image.
        center_y (float): Ground-truth center Y coordinate in search image.
        style_name (str): Semiconductor layout style ("DRAM" or "FinFET").
        box_size (float): Ground-truth target box size in search image (default 100.0).
        save_path (str, optional): Destination file path for saving PNG visualization.

    Returns:
        np.ndarray: Combined side-by-side RGB visualization array (1000, 2000, 3).
    """
    # 1. Convert grayscale 2D arrays to 3-channel BGR image arrays
    ref_bgr = cv2.cvtColor(np.clip(ref_image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    search_bgr = cv2.cvtColor(np.clip(search_image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # 2. Draw Ground-Truth Bounding Box on Search Image
    half_box = box_size / 2.0
    x1 = int(round(center_x - half_box))
    y1 = int(round(center_y - half_box))
    x2 = int(round(center_x + half_box))
    y2 = int(round(center_y + half_box))
    
    # Bright Green Bounding Box (thickness 3)
    cv2.rectangle(search_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 3. Draw Ground-Truth Center Crosshair & Point
    cx, cy = int(round(center_x)), int(round(center_y))
    # Red center circle
    cv2.circle(search_bgr, (cx, cy), 6, (0, 0, 255), -1)
    # White inner dot
    cv2.circle(search_bgr, (cx, cy), 2, (255, 255, 255), -1)
    # Red crosshair lines
    cv2.line(search_bgr, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
    cv2.line(search_bgr, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

    # 4. Add Text Header Panels
    # Reference Panel Header
    cv2.rectangle(ref_bgr, (0, 0), (1000, 45), (30, 30, 30), -1)
    cv2.putText(
        ref_bgr, f"REFERENCE IMAGE (100x SEM) | Style: {style_name}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )

    # Search Panel Header & Annotations
    cv2.rectangle(search_bgr, (0, 0), (1000, 45), (30, 30, 30), -1)
    cv2.putText(
        search_bgr, f"SEARCH IMAGE (10x SEM) | GT Center: ({center_x:.1f}, {center_y:.1f})",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
    )

    # Combine images side-by-side (1000x2000x3)
    canvas = np.hstack([ref_bgr, search_bgr])

    # Save via Pillow if path provided
    if save_path:
        # Convert BGR to RGB for Pillow
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_canvas)
        pil_img.save(save_path)

    return canvas
