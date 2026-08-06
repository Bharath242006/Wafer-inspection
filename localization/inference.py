"""
localization/inference.py

Production-ready inference wrapper for single image pair or directory batch localization.
"""

import cv2
import numpy as np
from localization.final_localizer import locate_target_final


class WaferLocalizerInference:
    """
    Inference engine for DriftSense-X semiconductor wafer pattern localization.
    """

    def __init__(self, use_hybrid_ranker: bool = True):
        self.use_hybrid_ranker = use_hybrid_ranker

    def predict_pair(self, ref_img: np.ndarray, search_img: np.ndarray) -> dict:
        """
        Runs localization on a single reference and search image pair.

        Returns:
            dict: {pred_x, pred_y, confidence, status, execution_time_ms}
        """
        pred_x, pred_y, score, status, details = locate_target_final(ref_img, search_img)
        return {
            'pred_x': float(pred_x),
            'pred_y': float(pred_y),
            'confidence': float(score),
            'status': status,
            'details': details
        }
