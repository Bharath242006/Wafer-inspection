"""
tests/test_training.py

Unit tests for training utilities, loss functions, and optimizers.
"""

import unittest
import torch
import torch.nn as nn

from models.losses import TripletMarginRankingLoss, ContrastiveLoss
from training.optimizer import get_optimizer
from training.scheduler import get_scheduler


class TestTraining(unittest.TestCase):

    def test_contrastive_loss(self):
        loss_fn = ContrastiveLoss(margin=0.2)
        sim = torch.tensor([0.9, 0.1])
        label = torch.tensor([1.0, 0.0])
        loss = loss_fn(sim, label)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_triplet_loss(self):
        loss_fn = TripletMarginRankingLoss(margin=0.2)
        pos = torch.tensor([0.8, 0.9])
        neg = torch.tensor([0.2, 0.3])
        loss = loss_fn(pos, neg)
        self.assertEqual(loss.item(), 0.0)

    def test_optimizer_builder(self):
        model = nn.Linear(10, 2)
        opt = get_optimizer(model, opt_type="adamw", lr=1e-3)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_scheduler_builder(self):
        model = nn.Linear(10, 2)
        opt = get_optimizer(model)
        sched = get_scheduler(opt, scheduler_type="cosine", epochs=10)
        self.assertIsNotNone(sched)


if __name__ == '__main__':
    unittest.main()
