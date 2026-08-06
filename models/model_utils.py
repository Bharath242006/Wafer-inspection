"""
models/model_utils.py

Model persistence utilities, device selection, and weight loading helpers.
"""

import os
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: nn.Module, filepath: str) -> None:
    """Saves model weights to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_checkpoint(model: nn.Module, filepath: str, device: torch.device = None) -> bool:
    """
    Loads model state_dict from disk if file exists.

    Returns:
        bool: True if weights loaded successfully, False if file not found.
    """
    if device is None:
        device = get_device()

    if os.path.exists(filepath):
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict)
        return True
    return False
