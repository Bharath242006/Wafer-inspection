"""
tests/test_localization.py

Unit tests for localization stages, peak detection, and subpixel peak refinement.
"""

import unittest
import numpy as np

from localization.candidate_generation import extract_local_peaks
from localization.fine_localization import refine_subpixel_peak
from localization.features.fft_features import estimate_lattice_period_2d


class TestLocalization(unittest.TestCase):

    def test_peak_extraction(self):
        resp = np.zeros((50, 50), dtype=np.float32)
        resp[25, 25] = 0.95
        peaks = extract_local_peaks(resp, window_size=5, min_thresh=0.1, top_k=5)
        self.assertGreaterEqual(len(peaks), 1)
        self.assertEqual(peaks[0][0], 25)
        self.assertEqual(peaks[0][1], 25)

    def test_subpixel_refinement(self):
        resp = np.zeros((10, 10), dtype=np.float32)
        resp[5, 5] = 1.0
        resp[5, 4] = 0.8
        resp[5, 6] = 0.6
        resp[4, 5] = 0.8
        resp[6, 5] = 0.8

        rx, ry = refine_subpixel_peak(resp, 5, 5)
        self.assertIsInstance(rx, float)
        self.assertIsInstance(ry, float)
        self.assertLess(abs(rx - 5.0), 0.5)

    def test_lattice_estimation(self):
        ref = np.zeros((100, 100), dtype=np.uint8)
        ref[::20, :] = 200
        lx, ly = estimate_lattice_period_2d(ref)
        self.assertGreater(lx, 0.0)
        self.assertGreater(ly, 0.0)


if __name__ == '__main__':
    unittest.main()
