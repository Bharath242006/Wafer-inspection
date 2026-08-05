"""
utils.py

Utility functions for directory creation, random seeding, image saving via Pillow,
and CSV label writing for the Drift-Sense synthetic wafer dataset generator.
"""

import os
import csv
import numpy as np
from PIL import Image


def set_seed(seed: int = 42) -> np.random.RandomState:
    """
    Sets global seeds for numpy and returns a dedicated RandomState instance
    for reproducible random generation across script runs.

    Args:
        seed (int): The master random seed.

    Returns:
        np.random.RandomState: An initialized random state object.
    """
    np.random.seed(seed)
    return np.random.RandomState(seed)


def ensure_dir(dir_path: str) -> None:
    """
    Ensures that a target directory path exists. Creates all missing parent directories.

    Args:
        dir_path (str): Absolute or relative directory path to create.
    """
    os.makedirs(dir_path, exist_ok=True)


def save_image_pillow(image: np.ndarray, file_path: str) -> None:
    """
    Saves a 2D numpy array (float32 or uint8) as a grayscale 8-bit image using Pillow.

    Args:
        image (np.ndarray): Input 2D grayscale array.
        file_path (str): Destination file path (e.g., .png).
    """
    # Clip intensity range to valid 8-bit limits [0, 255]
    img_clipped = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert to Pillow Image object and save
    pil_img = Image.fromarray(img_clipped, mode='L')
    
    # Ensure output folder exists before saving
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pil_img.save(file_path)


def write_labels_csv(csv_path: str, labels: list) -> None:
    """
    Writes ground-truth annotations to a CSV file with the schema:
    image,x,y,style

    Args:
        csv_path (str): Path to the destination CSV file.
        labels (list): List of tuples or dicts containing (image_name, x, y, style).
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write exact required header: image,x,y,style
        writer.writerow(['image', 'x', 'y', 'style'])
        
        for item in labels:
            if isinstance(item, (list, tuple)):
                img_name, x_val, y_val, style_name = item
            elif isinstance(item, dict):
                img_name = item['image']
                x_val = item['x']
                y_val = item['y']
                style_name = item['style']
            else:
                continue
            
            # Format coordinates to 2 decimal places for clean, accurate ground-truth labels
            writer.writerow([img_name, f"{float(x_val):.2f}", f"{float(y_val):.2f}", style_name])
