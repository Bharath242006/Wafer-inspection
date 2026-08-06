"""
tests/test_dataset.py

Unit tests for dataset generation, layouts, SEM augmentations, and label formatting.
"""

import unittest
import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.finfet_generator import generate_finfet_layout
from dataset_generator.dram_generator import generate_dram_layout
from dataset_generator.edge_brightening import apply_edge_brightening
from dataset_generator.sem_noise import add_gaussian_noise, add_poisson_noise
from dataset_generator.labels import compute_bounding_box


class TestDatasetGenerator(unittest.TestCase):

    def setUp(self):
        self.config = GeneratorConfig()
        self.rng = np.random.RandomState(42)

    def test_finfet_generation_shape(self):
        layout, params = generate_finfet_layout(100, 100, self.rng)
        self.assertEqual(layout.shape, (100, 100))
        self.assertTrue(np.all(layout >= 0.0))

    def test_dram_generation_shape(self):
        layout, params = generate_dram_layout(100, 100, self.rng)
        self.assertEqual(layout.shape, (100, 100))

    def test_edge_brightening(self):
        img = np.full((50, 50), 100.0, dtype=np.float32)
        img[20:30, 20:30] = 200.0
        brightened = apply_edge_brightening(img, strength=0.5)
        self.assertEqual(brightened.shape, img.shape)
        self.assertTrue(np.max(brightened) >= np.max(img))

    def test_bounding_box_computation(self):
        bbox = compute_bounding_box(10, 20, 100, 100)
        self.assertEqual(bbox['center_x'], 60.0)
        self.assertEqual(bbox['center_y'], 70.0)


if __name__ == '__main__':
    unittest.main()
