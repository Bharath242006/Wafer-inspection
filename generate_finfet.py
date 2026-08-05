"""
generate_finfet.py

FinFET semiconductor layout generator for the Applied Materials Drift-Sense challenge.
Generates large high-resolution 10000x10000 FinFET wafer layouts featuring:
- Dense vertical fins (silicon fin channels)
- Horizontal gate bars (poly/metal gate electrodes crossing vertical fins)
- Contact pads and process variations (fin/gate pitch and thickness jitter)
"""

import numpy as np
from typing import Dict, Any, Tuple


def get_random_finfet_params(rng: np.random.RandomState) -> Dict[str, Any]:
    """
    Randomizes FinFET feature parameters for synthetic dataset variety.

    Args:
        rng (np.random.RandomState): Random state generator.

    Returns:
        Dict[str, Any]: Dictionary of randomized FinFET structural parameters.
    """
    return {
        # Substrate background intensity (25.0 - 55.0)
        'bg_intensity': float(rng.uniform(25.0, 55.0)),
        
        # Dense Vertical Fins
        'fin_spacing': int(rng.randint(14, 32)),         # Pitch between vertical fins
        'fin_thickness': int(rng.randint(6, 16)),        # Width of vertical fins
        'fin_intensity': float(rng.uniform(130.0, 185.0)),
        
        # Horizontal Gate Bars
        'gate_spacing': int(rng.randint(55, 110)),       # Pitch between gate bars
        'gate_thickness': int(rng.randint(16, 32)),      # Width of gate bars
        'gate_intensity': float(rng.uniform(185.0, 240.0)),
        
        # Gate cuts / contact pads probability
        'pad_probability': float(rng.uniform(0.05, 0.15)),
    }


def generate_finfet_layout(
    width: int = 10000,
    height: int = 10000,
    rng: np.random.RandomState = None,
    params: Dict[str, Any] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates a 10000x10000 high-resolution float32 FinFET semiconductor wafer layout.

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
        params = get_random_finfet_params(rng)

    # Initialize layout canvas with background substrate intensity
    layout = np.full((height, width), params['bg_intensity'], dtype=np.float32)

    # --- 1. Draw Dense Vertical Fin Lines ---
    fin_pitch = params['fin_spacing'] + params['fin_thickness']
    fin_x_coords = []
    curr_x = rng.randint(5, 20)
    
    while curr_x < width:
        fin_x_coords.append(curr_x)
        # Pitch variation across wafer (+/- 1 px)
        curr_x += fin_pitch + rng.choice([-1, 0, 1])

    for x in fin_x_coords:
        thick = params['fin_thickness'] + rng.choice([-1, 0, 1])
        thick = max(2, thick)
        x_end = min(width, x + thick)
        layout[:, x:x_end] = params['fin_intensity']

    # --- 2. Draw Horizontal Gate Bars (Over Fins) ---
    gate_pitch = params['gate_spacing'] + params['gate_thickness']
    gate_y_coords = []
    curr_y = rng.randint(10, 30)
    
    while curr_y < height:
        gate_y_coords.append(curr_y)
        curr_y += gate_pitch + rng.choice([-1, 0, 1])

    for y in gate_y_coords:
        thick = params['gate_thickness'] + rng.choice([-1, 0, 1])
        thick = max(4, thick)
        y_end = min(height, y + thick)
        
        # Gates are top structures, overwriting fin regions underneath
        layout[y:y_end, :] = params['gate_intensity']

    # --- 3. Add Occasional Contact Pads / Interconnect Vias ---
    if params['pad_probability'] > 0:
        pad_size = int(params['gate_thickness'] * 1.5)
        for y in gate_y_coords[::2]:
            for x in fin_x_coords[::4]:
                if rng.rand() < params['pad_probability']:
                    y1 = max(0, y - pad_size // 4)
                    y2 = min(height, y + pad_size)
                    x1 = max(0, x - pad_size // 4)
                    x2 = min(width, x + pad_size)
                    layout[y1:y2, x1:x2] = min(255.0, params['gate_intensity'] + 20.0)

    return layout, params
