"""
training/optimizer.py

PyTorch Optimizer builders (Adam, AdamW, SGD).
"""

import torch.nn as nn
import torch.optim as optim


def get_optimizer(model: nn.Module, opt_type: str = "adamw", lr: float = 1e-3, weight_decay: float = 1e-4):
    """Returns requested PyTorch optimizer."""
    if opt_type.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optim.AdamW(model.parameters(), lr=lr)
