"""
tests/test_models.py

Unit tests for PyTorch neural ranking model forward passes and output shapes.
"""

import unittest
import torch

from models.cnn import SiameseNet, CNNEncoder
from models.hybrid_model import HybridRankerNet
from models.coordinate_model import CoordinateAwareRankerNet
from models.context_model import ContextRankerNet


class TestModels(unittest.TestCase):

    def test_siamese_cnn_forward(self):
        model = SiameseNet(embedding_dim=32)
        x_ref = torch.randn(4, 1, 100, 100)
        x_cand = torch.randn(4, 1, 100, 100)
        sim = model(x_ref, x_cand)
        self.assertEqual(sim.shape, (4,))

    def test_hybrid_ranker_forward(self):
        model = HybridRankerNet(input_dim=56)
        x = torch.randn(8, 56)
        score = model(x)
        self.assertEqual(score.shape, (8,))

    def test_coordinate_ranker_forward(self):
        model = CoordinateAwareRankerNet(input_dim=44)
        x = torch.randn(8, 44)
        score = model(x)
        self.assertEqual(score.shape, (8,))

    def test_context_ranker_forward(self):
        model = ContextRankerNet(embedding_dim=32)
        ref_l = torch.randn(2, 1, 100, 100)
        ref_m = torch.randn(2, 1, 100, 100)
        ref_g = torch.randn(2, 1, 100, 100)
        cand_l = torch.randn(2, 1, 100, 100)
        cand_m = torch.randn(2, 1, 100, 100)
        cand_g = torch.randn(2, 1, 100, 100)

        score = model.forward_pair_similarity(ref_l, ref_m, ref_g, cand_l, cand_m, cand_g)
        self.assertEqual(score.shape, (2,))


if __name__ == '__main__':
    unittest.main()
