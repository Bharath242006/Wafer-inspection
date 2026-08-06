"""
localization/coordinate_aware_ranker.py

Industrial Coordinate-Aware Candidate Ranker for DriftSense-X.

Architecture:
- Shared ResNet BasicBlock Backbone for feature extraction from reference and candidate crops.
- Multi-modal visual fusion: Feature Difference (|ref - cand|), Element-wise Product (ref * cand), Concatenation [ref, cand].
- Spatial coordinate fusion: Normalized candidate X, Y, and Scale.
- 3-Layer MLP Classifier with Sigmoid output.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Lightweight Residual Block for fast feature extraction."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """Lightweight 4-stage Residual CNN Backbone generating 64-D embeddings."""
    def __init__(self, in_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = BasicBlock(32, 32, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 50x50

        self.layer2 = BasicBlock(32, 64, stride=2)  # 25x25
        self.layer3 = BasicBlock(64, 64, stride=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool1(self.layer1(h))
        h = self.layer3(self.layer2(h))
        g = self.gap(h)
        flat = torch.flatten(g, 1)
        emb = self.fc(flat)
        norm_emb = F.normalize(emb, p=2, dim=1)
        return norm_emb


class CoordinateAwareRanker(nn.Module):
    """
    Industrial Coordinate-Aware Candidate Ranker for DriftSense-X.
    Combines visual embeddings, difference/product features, and spatial coordinates (norm_x, norm_y, scale).
    """
    def __init__(self, embedding_dim: int = 64, spatial_dim: int = 3):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels=1, embedding_dim=embedding_dim)

        # Visual fusion dimension: |ref - cand| (64) + ref * cand (64) + concat[ref, cand] (128) = 256
        visual_fusion_dim = embedding_dim * 4

        total_input_dim = visual_fusion_dim + spatial_dim  # 256 + 3 = 259

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        x_ref: torch.Tensor,
        x_cand: torch.Tensor,
        spatial_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        x_ref: (B, 1, 100, 100)
        x_cand: (B, 1, 100, 100)
        spatial_feats: (B, 3) -> [norm_x, norm_y, scale]

        Returns:
            logit tensor of shape (B, 1)
        """
        emb_ref = self.backbone(x_ref)
        emb_cand = self.backbone(x_cand)

        f_diff = torch.abs(emb_ref - emb_cand)
        f_mult = emb_ref * emb_cand
        f_concat = torch.cat([emb_ref, emb_cand], dim=1)

        f_visual = torch.cat([f_diff, f_mult, f_concat], dim=1)  # (B, 256)
        f_all = torch.cat([f_visual, spatial_feats], dim=1)        # (B, 259)

        logits = self.mlp(f_all)
        return logits


_model_cache = None


def load_coordinate_aware_ranker(
    checkpoint_path: str = "checkpoints/coordinate_ranker.pt"
) -> CoordinateAwareRanker:
    """Loads trained CoordinateAwareRanker model checkpoint."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = CoordinateAwareRanker(embedding_dim=64, spatial_dim=3)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[CoordinateAwareRanker] Loaded weights from '{checkpoint_path}'.")
    else:
        # Fallback to debug checkpoint if main checkpoint doesn't exist yet
        debug_path = "checkpoints/coordinate_ranker_debug.pt"
        if os.path.exists(debug_path):
            state_dict = torch.load(debug_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"[CoordinateAwareRanker] Loaded debug weights from '{debug_path}'.")
        else:
            print(f"[CoordinateAwareRanker] Warning: No checkpoint found! Using random initialization.")

    _model_cache = model
    return model


def compute_coordinate_ranker_scores(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    checkpoint_path: str = "checkpoints/coordinate_ranker.pt"
) -> list:
    """
    Computes ranking scores for candidate pool using CoordinateAwareRanker.

    Args:
        ref_img: 1000x1000 grayscale reference image.
        search_img: 1000x1000 grayscale search image.
        candidates: List of candidate dictionaries from candidate generation.

    Returns:
        list of float: Ranking scores in range [0.0, 1.0].
    """
    if not candidates:
        return []

    model = load_coordinate_aware_ranker(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    sh, sw = search_img.shape[:2]
    ref_h, ref_w = ref_img.shape[:2]

    # Pre-process reference patch
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA).astype(np.float32)
    ref_norm = (ref_100 - np.mean(ref_100)) / (np.std(ref_100) + 1e-5)
    t_ref = torch.tensor(ref_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    pad = 60
    search_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    cand_tensors = []
    spatial_tensors = []

    for cand in candidates:
        cx = float(cand.get('center_x', cand.get('cx')))
        cy = float(cand.get('center_y', cand.get('cy')))
        s = float(cand.get('scale', cand.get('primary_scale', 0.10)))

        # Normalized coordinates in [0, 1]
        norm_x = float(np.clip(cx / sw, 0.0, 1.0))
        norm_y = float(np.clip(cy / sh, 0.0, 1.0))
        spatial_tensors.append([norm_x, norm_y, s])

        cw = max(4, int(round(ref_w * s)))
        ch = max(4, int(round(ref_h * s)))

        tl_x_pad = int(round(cx + pad - cw / 2.0))
        tl_y_pad = int(round(cy + pad - ch / 2.0))

        crop = search_pad[tl_y_pad:tl_y_pad+ch, tl_x_pad:tl_x_pad+cw]
        if crop.shape[0] != 100 or crop.shape[1] != 100:
            crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)

        crop_f = crop.astype(np.float32)
        crop_norm = (crop_f - np.mean(crop_f)) / (np.std(crop_f) + 1e-5)
        cand_tensors.append(torch.tensor(crop_norm, dtype=torch.float32).unsqueeze(0))

    t_cands = torch.stack(cand_tensors, dim=0).to(device)         # (N, 1, 100, 100)
    t_spatial = torch.tensor(spatial_tensors, dtype=torch.float32).to(device) # (N, 3)

    # Expand t_ref to match batch size N
    t_ref_batch = t_ref.expand(len(candidates), -1, -1, -1)

    with torch.no_grad():
        logits = model(t_ref_batch, t_cands, t_spatial)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    return [float(p) for p in probs]
