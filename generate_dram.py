"""
generate_dram.py

DRAM semiconductor layout generator for the Applied Materials Drift-Sense challenge.
Generates large high-resolution 10000x10000 DRAM wafer layouts featuring:
- Periodic horizontal word lines
- Vertical bit lines
- Contact / via dots at array grid intersections
- Process variations (pitch, thickness, intensity micro-variations)
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


def get_random_dram_params(rng: np.random.RandomState) -> Dict[str, Any]:
    """
    Randomizes DRAM feature parameters for synthetic dataset variety.

    Args:
        rng (np.random.RandomState): Random state generator.

    Returns:
        Dict[str, Any]: Dictionary of randomized DRAM structural parameters.
    """
    return {
        # Background substrate intensity (30.0 - 60.0)
        'bg_intensity': float(rng.uniform(30.0, 60.0)),
        
        # Word lines (Horizontal)
        'word_line_spacing': int(rng.randint(40, 80)),       # Pitch between word lines
        'word_line_thickness': int(rng.randint(10, 24)),     # Width of word lines
        'word_line_intensity': float(rng.uniform(140.0, 190.0)),
        
        # Bit lines (Vertical)
        'bit_line_spacing': int(rng.randint(30, 65)),        # Pitch between bit lines
        'bit_line_thickness': int(rng.randint(8, 20)),       # Width of bit lines
        'bit_line_intensity': float(rng.uniform(120.0, 175.0)),
        
        # Contact / Via dots
        'contact_diameter': int(rng.randint(10, 22)),        # Via dot diameter
        'contact_intensity': float(rng.uniform(200.0, 250.0)), # Bright contact dots
    }


def generate_dram_layout(
    width: int = 10000,
    height: int = 10000,
    rng: np.random.RandomState = None,
    params: Dict[str, Any] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates a 10000x10000 high-resolution float32 DRAM semiconductor wafer layout.

    Args:
        width (int): Canvas width in pixels (default 10000).
        height (int): Canvas height in pixels (default 10000).
        rng (np.random.RandomState): Random state for reproducible randomization.
        params (Dict[str, Any], optional): Pre-defined or overridden layout parameters.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - 2D float32 numpy array (height, width) representing wafer layout.
            - Dictionary of structural parameters used for generation.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    if params is None:
        params = get_random_dram_params(rng)

    # Initialize layout canvas with background substrate intensity
    layout = np.full((height, width), params['bg_intensity'], dtype=np.float32)

    # --- 1. Draw Periodic Horizontal Word Lines ---
    wl_pitch = params['word_line_spacing'] + params['word_line_thickness']
    wl_y_coords = []
    curr_y = rng.randint(10, 30)
    
    while curr_y < height:
        wl_y_coords.append(curr_y)
        # Small process variation (+/- 1 px) across wafer
        curr_y += wl_pitch + rng.choice([-1, 0, 1])

    for y in wl_y_coords:
        thick = params['word_line_thickness'] + rng.choice([-1, 0, 1])
        thick = max(2, thick)
        y_end = min(height, y + thick)
        layout[y:y_end, :] = params['word_line_intensity']

    # --- 2. Draw Vertical Bit Lines ---
    bl_pitch = params['bit_line_spacing'] + params['bit_line_thickness']
    bl_x_coords = []
    curr_x = rng.randint(10, 30)
    
    while curr_x < width:
        bl_x_coords.append(curr_x)
        curr_x += bl_pitch + rng.choice([-1, 0, 1])

    for x in bl_x_coords:
        thick = params['bit_line_thickness'] + rng.choice([-1, 0, 1])
        thick = max(2, thick)
        x_end = min(width, x + thick)
        # Blend bit line intensity with underlying word lines
        layout[:, x:x_end] = np.maximum(layout[:, x:x_end], params['bit_line_intensity'])

    # --- 3. Draw Contact / Via Dots at Intersections ---
    # Create contact via grid at intersections of word and bit lines
    contact_r = max(2, params['contact_diameter'] // 2)
    contact_val = params['contact_intensity']

    # Vectorized fast dot drawing using meshgrid & boolean masking or cv2 batch points
    # Select grid points (subsampled to mimic realistic DRAM memory cell array layout)
    grid_y = np.array(wl_y_coords[::1], dtype=np.int32) + params['word_line_thickness'] // 2
    grid_x = np.array(bl_x_coords[::1], dtype=np.int32) + params['bit_line_thickness'] // 2

    # Draw circles at intersection centers
    for cy in grid_y:
        if cy - contact_r < 0 or cy + contact_r >= height:
            continue
        y_slice = slice(max(0, cy - contact_r), min(height, cy + contact_r + 1))
        
        for cx in grid_x:
            if cx - contact_r < 0 or cx + contact_r >= width:
                continue
            x_slice = slice(max(0, cx - contact_r), min(width, cx + contact_r + 1))
            
            # Apply contact intensity
            layout[y_slice, x_slice] = np.maximum(layout[y_slice, x_slice], contact_val)

    return layout, params
