"""
localization/ranking/context_ranker.py

Context Transformer Ranker evaluator wrapper.
"""

import os
import cv2
import numpy as np
import torch

from models.context_model import ContextRankerNet


_context_model_cache = None


def load_trained_context_model(checkpoint_path: str = "weights/checkpoints/context_ranker.pt") -> ContextRankerNet:
    """Loads trained Multi-Branch Context Ranker model checkpoint."""
    global _context_model_cache
    if _context_model_cache is not None:
        return _context_model_cache

    if not os.path.exists(checkpoint_path) and os.path.exists("checkpoints/context_ranker.pt"):
        checkpoint_path = "checkpoints/context_ranker.pt"

    model = ContextRankerNet(embedding_dim=32)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    _context_model_cache = model
    return model
