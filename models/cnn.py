"""
models/cnn.py

Siamese Convolutional Neural Network architecture for candidate pattern matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Shared Convolutional Neural Network Encoder.
    Extracts L2-normalized 32-D feature embeddings.
    """

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.bn1(self.conv1(x)))
        p1 = self.pool(h1)

        h2 = F.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool(h2)

        h3 = F.relu(self.bn3(self.conv3(p2)))
        g = self.gap(h3)

        flat = torch.flatten(g, 1)
        emb = self.fc(flat)
        norm_emb = F.normalize(emb, p=2, dim=1)
        return norm_emb


class SiameseNet(nn.Module):
    """
    Siamese Neural Network comparing reference patch and candidate search crop.
    """

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.encoder = CNNEncoder(embedding_dim=embedding_dim)

    def forward(self, x_ref: torch.Tensor, x_cand: torch.Tensor) -> torch.Tensor:
        emb_ref = self.encoder(x_ref)
        emb_cand = self.encoder(x_cand)
        similarity = torch.sum(emb_ref * emb_cand, dim=1)
        return similarity
