"""
localization/context_ranker.py

Multi-Branch Context-Aware Candidate Ranker for DriftSense-X.

Architecture:
- Multi-Branch Shared Encoder for 3 spatial context fields:
  1. Local (100x100)
  2. Medium (250x250 -> 100x100)
  3. Large (500x500 -> 100x100)
- Computes cosine similarities across all 3 context resolutions.
- Passes similarity vector [S_local, S_med, S_large] into an MLP Ranking Head to output a score in [0, 1].
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiContextCNNEncoder(nn.Module):
    """Shared Convolutional Neural Network Encoder for Multi-Context Crops."""
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
        return F.normalize(emb, p=2, dim=1)


class ContextRankerNet(nn.Module):
    """Multi-Branch Context-Aware Siamese Neural Network."""
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.encoder = MultiContextCNNEncoder(embedding_dim=embedding_dim)

        # MLP Ranking Head operating on 3-scale cosine similarities [S_local, S_med, S_large]
        self.mlp_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward_pair_similarity(self, t_ref_loc: torch.Tensor, t_ref_med: torch.Tensor, t_ref_lrg: torch.Tensor,
                                t_cand_loc: torch.Tensor, t_cand_med: torch.Tensor, t_cand_lrg: torch.Tensor) -> torch.Tensor:
        e_ref_l = self.encoder(t_ref_loc)
        e_ref_m = self.encoder(t_ref_med)
        e_ref_g = self.encoder(t_ref_lrg)

        e_cand_l = self.encoder(t_cand_loc)
        e_cand_m = self.encoder(t_cand_med)
        e_cand_g = self.encoder(t_cand_lrg)

        s_loc = torch.sum(e_ref_l * e_cand_l, dim=1, keepdim=True)
        s_med = torch.sum(e_ref_m * e_cand_m, dim=1, keepdim=True)
        s_lrg = torch.sum(e_ref_g * e_cand_g, dim=1, keepdim=True)

        sim_vec = torch.cat([s_loc, s_med, s_lrg], dim=1)  # (B, 3)
        score = self.mlp_head(sim_vec).squeeze(1)          # (B,)
        return score


_context_model_cache = None


def load_trained_context_model(checkpoint_path: str = "checkpoints/context_ranker.pt") -> ContextRankerNet:
    """Loads trained Multi-Branch Context Ranker model checkpoint."""
    global _context_model_cache
    if _context_model_cache is not None:
        return _context_model_cache

    model = ContextRankerNet(embedding_dim=32)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[ContextRankerNet] Successfully loaded trained weights from '{checkpoint_path}'.")
    else:
        print(f"[ContextRankerNet] Warning: Checkpoint '{checkpoint_path}' not found! Using random initialization.")

    _context_model_cache = model
    return model


def compute_context_ranker_scores(ref_img: np.ndarray, search_img: np.ndarray, candidates: list, checkpoint_path: str = "checkpoints/context_ranker.pt") -> list:
    """
    Calculates multi-context ranking scores for candidate pool.

    Args:
        ref_img (np.ndarray): 1000x1000 reference image.
        search_img (np.ndarray): 1000x1000 search image.
        candidates (list): Candidate pool dictionaries.

    Returns:
        list: Float ranking scores in range [0.0, 1.0].
    """
    model = load_trained_context_model(checkpoint_path)

    # Reference multi-context crops around reference center (500, 500)
    pad = 300
    ref_pad = cv2.copyMakeBorder(ref_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    sch_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    cx_r, cy_r = 500.0 + pad, 500.0 + pad

    ref_l = ref_pad[int(cy_r-50):int(cy_r+50), int(cx_r-50):int(cx_r+50)]
    ref_m = cv2.resize(ref_pad[int(cy_r-125):int(cy_r+125), int(cx_r-125):int(cx_r+125)], (100, 100), cv2.INTER_AREA)
    ref_g = cv2.resize(ref_pad[int(cy_r-250):int(cy_r+250), int(cx_r-250):int(cx_r+250)], (100, 100), cv2.INTER_AREA)

    def norm_t(patch):
        pf = patch.astype(np.float32)
        m = (pf - np.mean(pf)) / (np.std(pf) + 1e-5)
        return torch.tensor(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    t_ref_l = norm_t(ref_l)
    t_ref_m = norm_t(ref_m)
    t_ref_g = norm_t(ref_g)

    scores = []
    model.eval()

    with torch.no_grad():
        for cand in candidates:
            cx_s = cand['cx'] + pad
            cy_s = cand['cy'] + pad

            cand_l = sch_pad[int(cy_s-50):int(cy_s+50), int(cx_s-50):int(cx_s+50)]
            if cand_l.shape[0] != 100 or cand_l.shape[1] != 100:
                cand_l = cv2.resize(cand_l, (100, 100), cv2.INTER_AREA)

            cand_m = cv2.resize(sch_pad[int(cy_s-125):int(cy_s+125), int(cx_s-125):int(cx_s+125)], (100, 100), cv2.INTER_AREA)
            cand_g = cv2.resize(sch_pad[int(cy_s-250):int(cy_s+250), int(cx_s-250):int(cx_s+250)], (100, 100), cv2.INTER_AREA)

            t_cand_l = norm_t(cand_l)
            t_cand_m = norm_t(cand_m)
            t_cand_g = norm_t(cand_g)

            score = model.forward_pair_similarity(t_ref_l, t_ref_m, t_ref_g, t_cand_l, t_cand_m, t_cand_g)
            scores.append(float(score.item()))

    return scores
