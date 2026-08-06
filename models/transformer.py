"""
models/transformer.py

Transformer Self-Attention and Cross-Attention modules for context feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionBlock(nn.Module):
    """
    Spatial Attention Block for feature map weighting.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_map = self.conv_attn(x)
        return x * attn_map
