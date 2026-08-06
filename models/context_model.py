"""
models/context_model.py

Multi-Branch Context-Aware Candidate Ranker architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiContextCNNEncoder(nn.Module):
    """
    Shared Convolutional Neural Network Encoder for Multi-Context Crops.
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
        return F.normalize(emb, p=2, dim=1)


class ContextRankerNet(nn.Module):
    """
    Multi-Branch Context-Aware Siamese Neural Network.
    """

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.encoder = MultiContextCNNEncoder(embedding_dim=embedding_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward_pair_similarity(
        self,
        t_ref_loc: torch.Tensor,
        t_ref_med: torch.Tensor,
        t_ref_lrg: torch.Tensor,
        t_cand_loc: torch.Tensor,
        t_cand_med: torch.Tensor,
        t_cand_lrg: torch.Tensor
    ) -> torch.Tensor:
        e_ref_l = self.encoder(t_ref_loc)
        e_ref_m = self.encoder(t_ref_med)
        e_ref_g = self.encoder(t_ref_lrg)

        e_cand_l = self.encoder(t_cand_loc)
        e_cand_m = self.encoder(t_cand_med)
        e_cand_g = self.encoder(t_cand_lrg)

        s_loc = torch.sum(e_ref_l * e_cand_l, dim=1, keepdim=True)
        s_med = torch.sum(e_ref_m * e_cand_m, dim=1, keepdim=True)
        s_lrg = torch.sum(e_ref_g * e_cand_g, dim=1, keepdim=True)

        sim_vec = torch.cat([s_loc, s_med, s_lrg], dim=1)
        score = self.mlp_head(sim_vec).squeeze(1)
        return score
