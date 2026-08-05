"""
localization/cnn_candidate_ranker.py

Siamese PyTorch CNN Candidate Ranker for DriftSense-X.

Architecture:
- Shared Convolutional Encoder (16 -> 32 -> 64 channels, BatchNorm, ReLU, AdaptiveAvgPool).
- 32-D L2-normalized embedding representation.
- Cosine similarity matching between reference and candidate search crops.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Shared Convolutional Neural Network Encoder."""
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
        # Input shape: (B, 1, 100, 100)
        h1 = F.relu(self.bn1(self.conv1(x)))
        p1 = self.pool(h1)  # (B, 16, 50, 50)

        h2 = F.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool(h2)  # (B, 32, 25, 25)

        h3 = F.relu(self.bn3(self.conv3(p2)))
        g = self.gap(h3)    # (B, 64, 1, 1)

        flat = torch.flatten(g, 1)  # (B, 64)
        emb = self.fc(flat)         # (B, 32)

        norm_emb = F.normalize(emb, p=2, dim=1)
        return norm_emb


class SiameseNet(nn.Module):
    """Siamese Neural Network comparing reference and candidate image patches."""
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.encoder = CNNEncoder(embedding_dim=embedding_dim)

    def forward(self, x_ref: torch.Tensor, x_cand: torch.Tensor) -> torch.Tensor:
        emb_ref = self.encoder(x_ref)
        emb_cand = self.encoder(x_cand)

        # Cosine similarity (dot product of L2-normalized embeddings)
        similarity = torch.sum(emb_ref * emb_cand, dim=1)
        return similarity


_model_cache = None


def load_trained_siamese_model(checkpoint_path: str = "checkpoints/siamese_cnn.pt") -> SiameseNet:
    """Loads trained Siamese CNN model checkpoint."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = SiameseNet(embedding_dim=32)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[SiameseNet] Successfully loaded trained weights from '{checkpoint_path}'.")
    else:
        print(f"[SiameseNet] Warning: Checkpoint '{checkpoint_path}' not found! Using random initialization.")

    _model_cache = model
    return model


def compute_cnn_similarity_scores(ref_img: np.ndarray, search_img: np.ndarray, candidates: list, checkpoint_path: str = "checkpoints/siamese_cnn.pt") -> list:
    """
    Calculates Siamese CNN similarity scores for candidate pool.

    Args:
        ref_img (np.ndarray): 1000x1000 reference image.
        search_img (np.ndarray): 1000x1000 search image.
        candidates (list): Candidate pool dictionaries.

    Returns:
        list: Float similarity scores in range [-1.0, 1.0].
    """
    model = load_trained_siamese_model(checkpoint_path)

    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA).astype(np.float32)
    ref_norm = (ref_100 - np.mean(ref_100)) / (np.std(ref_100) + 1e-5)
    t_ref = torch.tensor(ref_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pad = 60
    search_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    sh, sw = search_img.shape[:2]

    scores = []
    model.eval()

    with torch.no_grad():
        emb_ref = model.encoder(t_ref)

        for cand in candidates:
            cx = cand['center_x']
            cy = cand['center_y']
            s = cand.get('primary_scale', 0.10)
            cw = int(round(ref_img.shape[1] * s))
            ch = int(round(ref_img.shape[0] * s))

            tl_x_pad = int(round(cx + pad - cw / 2.0))
            tl_y_pad = int(round(cy + pad - ch / 2.0))

            crop = search_pad[tl_y_pad:tl_y_pad+ch, tl_x_pad:tl_x_pad+cw]
            if crop.shape[0] != 100 or crop.shape[1] != 100:
                crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)

            crop_f = crop.astype(np.float32)
            crop_norm = (crop_f - np.mean(crop_f)) / (np.std(crop_f) + 1e-5)
            t_cand = torch.tensor(crop_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            emb_cand = model.encoder(t_cand)
            sim = float(torch.sum(emb_ref * emb_cand).item())
            scores.append(float(np.clip(sim, -1.0, 1.0)))

    return scores
