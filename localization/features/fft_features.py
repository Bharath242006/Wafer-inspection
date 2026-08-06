"""
localization/features/fft_features.py

2D FFT, Power Spectrum, Autocorrelation, and Lattice Period Estimation features.
"""

import cv2
import numpy as np
from typing import Tuple


def estimate_lattice_period_2d(ref_img: np.ndarray) -> Tuple[float, float]:
    """
    Dynamically estimates 2D semiconductor lattice periods lambda_x, lambda_y in search image coordinates.
    """
    if ref_img.shape[0] > 200:
        ref_s = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_s = ref_img.copy()

    ref_f = ref_s.astype(np.float32) - np.mean(ref_s)
    f = np.fft.fft2(ref_f)
    power = np.abs(f) ** 2
    autocorr = np.real(np.fft.ifft2(power))
    autocorr = np.fft.fftshift(autocorr)

    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[max(0, cy - 2):min(autocorr.shape[0], cy + 3), max(0, cx - 2):min(autocorr.shape[1], cx + 3)] = 0.0

    _, _, _, max_loc = cv2.minMaxLoc(autocorr)
    p_dx = max_loc[0] - cx
    p_dy = max_loc[1] - cy

    scale_fac = (ref_img.shape[0] / ref_s.shape[0]) * 10.0
    lx = abs(p_dx) * scale_fac if abs(p_dx) > 2 else 67.0
    ly = abs(p_dy) * scale_fac if abs(p_dy) > 2 else 67.0

    lx = float(np.clip(lx, 30.0, 150.0))
    ly = float(np.clip(ly, 30.0, 150.0))
    return lx, ly
