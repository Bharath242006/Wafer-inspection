"""
localization/features/structural_features.py

Structural signature extraction and multi-scale signature vectors.
"""

import cv2
import numpy as np


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Computes normalized Sobel gradient magnitude image."""
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)
