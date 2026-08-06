"""
training/trainer.py

Universal PyTorch Neural Network Trainer loop engine.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model_utils import get_device
from training.optimizer import get_optimizer
from training.scheduler import get_scheduler
from training.callbacks import ModelCheckpoint


class Trainer:
    """Universal Neural Network model training loop runner."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        criterion: nn.Module = None,
        lr: float = 1e-3,
        checkpoint_path: str = "weights/checkpoints/model.pt"
    ):
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = get_optimizer(self.model, lr=lr)
        self.checkpoint = ModelCheckpoint(checkpoint_path)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x_ref, x_crop, label = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x_ref, x_crop)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    def run(self, epochs: int = 10):
        print(f"Starting model training for {epochs} epochs on device '{self.device}'...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}")
            self.checkpoint(train_loss, self.model)
        print("Training complete!")
