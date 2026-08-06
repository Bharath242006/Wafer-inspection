"""
localization/features/edge_features.py

Sobel gradient magnitude and Canny edge map feature extraction.
"""

import cv2
import numpy as np


def compute_sobel_gradient(img: np.ndarray) -> np.ndarray:
    """
    Computes spatial gradient magnitude using 3x3 Sobel operator.
    """
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def compute_canny_edge(img: np.ndarray) -> np.ndarray:
    """
    Computes Canny edge map normalized to float32 range [0, 1].
    """
    edges = cv2.Canny(img, 50, 150)
    return edges.astype(np.float32) / 255.0
