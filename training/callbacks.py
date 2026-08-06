"""
training/callbacks.py

Model checkpointing, early stopping, and metric logging callbacks.
"""

import os
import torch
import torch.nn as nn


class ModelCheckpoint:
    """Saves best model weights during training based on validation loss or accuracy."""

    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_val = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, val_metric: float, model: nn.Module) -> bool:
        improved = (val_metric < self.best_val) if self.mode == 'min' else (val_metric > self.best_val)
        if improved:
            self.best_val = val_metric
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
            print(f"[ModelCheckpoint] Metric improved to {val_metric:.4f}. Checkpoint saved to '{self.filepath}'.")
            return True
        return False
