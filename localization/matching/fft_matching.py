"""
localization/matching/fft_matching.py

FFT phase correlation matching and spectral similarity scoring.
"""

import cv2
import numpy as np


def fft_phase_correlation_score(patch: np.ndarray, ref_tmpl: np.ndarray) -> float:
    """
    Computes FFT phase-correlation peak height between a search patch and reference template.
    """
    if patch.size == 0 or ref_tmpl.size == 0:
        return 0.0

    h = min(patch.shape[0], ref_tmpl.shape[0], 64)
    w = min(patch.shape[1], ref_tmpl.shape[1], 64)
    if h < 4 or w < 4:
        return 0.0

    p = cv2.resize(patch.astype(np.float32), (w, h), cv2.INTER_AREA)
    r = cv2.resize(ref_tmpl.astype(np.float32), (w, h), cv2.INTER_AREA)

    p -= np.mean(p)
    r -= np.mean(r)

    f_p = np.fft.fft2(p)
    f_r = np.fft.fft2(r)

    cross_power = (f_p * np.conj(f_r)) / (np.abs(f_p * np.conj(f_r)) + 1e-6)
    phase_corr = np.real(np.fft.ifft2(cross_power))

    max_val = np.max(phase_corr)
    return float(np.clip(max_val, 0.0, 1.0))
