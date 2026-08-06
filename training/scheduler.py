"""
training/scheduler.py

Learning rate scheduler builders.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str = "cosine", epochs: int = 20):
    """Returns requested PyTorch learning rate scheduler."""
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    return None
