"""
localization/context_aware_ranker_v2.py

Context-Aware Ranker V2 for DriftSense-X.

Inputs:
- Reference Local Patch (100x100)
- Reference Context Patch (300x300 downsampled to 100x100)
- Candidate Local Patch (100x100)
- Relative spatial offset (relative_dx, relative_dy)
- Candidate Generator score (cand_score)

Network Architecture:
- Shared 4-stage ResNet Backbone generating 64-D normalized embeddings.
- Multi-field Feature Fusion:
  * Local Visual Fusion (|ref_loc - cand|, ref_loc * cand, concat[ref_loc, cand]) -> 256-D
  * Context Visual Fusion (|ref_ctx - cand|, ref_ctx * cand, concat[ref_ctx, cand]) -> 256-D
- Attention Layer:
  * Spatial-Gated Cross-Attention mechanism weighting local vs contextual features dynamically
    based on relative dx/dy spatial offsets and candidate generator confidence score.
- 3-Layer MLP Head producing scalar ranking score.
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


class SharedResNetBackbone(nn.Module):
    """Lightweight 4-stage Residual CNN Backbone generating 64-D normalized embeddings."""
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


class ContextAttentionFusion(nn.Module):
    """
    Spatial & Generator Gated Cross-Attention module.
    Dynamically re-weights local vs context visual representations based on relative spatial offset and candidate generator score.
    """
    def __init__(self, visual_dim: int = 256, spatial_dim: int = 3):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.proj_fused = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU()
        )

    def forward(
        self,
        f_local: torch.Tensor,
        f_context: torch.Tensor,
        spatial_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_local: (B, 256)
            f_context: (B, 256)
            spatial_feats: (B, 3) -> [relative_dx, relative_dy, cand_score]

        Returns:
            f_attended: (B, 256)
        """
        attn_weights = self.attn_mlp(spatial_feats)  # (B, 2) -> [w_local, w_context]
        w_local = attn_weights[:, 0:1]
        w_context = attn_weights[:, 1:2]

        f_weighted = w_local * f_local + w_context * f_context
        f_attended = self.proj_fused(f_weighted)
        return f_attended


class ContextAwareRankerV2(nn.Module):
    """
    Context-Aware Candidate Ranker V2 for DriftSense-X.
    Combines:
    - Reference Local Patch (100x100)
    - Reference Context Patch (300x300 downsampled to 100x100)
    - Candidate Local Patch (100x100)
    - Relative Spatial Offset (relative_dx, relative_dy)
    - Candidate Generator Confidence Score
    - Spatial-Gated Attention Layer
    - 3-Layer MLP Head
    """
    def __init__(self, embedding_dim: int = 64, spatial_dim: int = 3):
        super().__init__()
        self.backbone = SharedResNetBackbone(in_channels=1, embedding_dim=embedding_dim)

        # Visual fusion dim per branch: |ref - cand| (64) + ref * cand (64) + concat[ref, cand] (128) = 256
        visual_dim = embedding_dim * 4
        self.attention_fusion = ContextAttentionFusion(visual_dim=visual_dim, spatial_dim=spatial_dim)

        total_input_dim = visual_dim + spatial_dim  # 256 + 3 = 259

        self.mlp_head = nn.Sequential(
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
        x_ref_loc: torch.Tensor,
        x_ref_ctx: torch.Tensor,
        x_cand: torch.Tensor,
        spatial_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        x_ref_loc: (B, 1, 100, 100)
        x_ref_ctx: (B, 1, 100, 100)
        x_cand: (B, 1, 100, 100)
        spatial_feats: (B, 3) -> [relative_dx, relative_dy, cand_score]

        Returns:
            logit tensor of shape (B, 1)
        """
        emb_ref_loc = self.backbone(x_ref_loc)
        emb_ref_ctx = self.backbone(x_ref_ctx)
        emb_cand = self.backbone(x_cand)

        # 1. Local visual alignment
        f_loc_diff = torch.abs(emb_ref_loc - emb_cand)
        f_loc_mult = emb_ref_loc * emb_cand
        f_loc_concat = torch.cat([emb_ref_loc, emb_cand], dim=1)
        f_local = torch.cat([f_loc_diff, f_loc_mult, f_loc_concat], dim=1)  # (B, 256)

        # 2. Context visual alignment
        f_ctx_diff = torch.abs(emb_ref_ctx - emb_cand)
        f_ctx_mult = emb_ref_ctx * emb_cand
        f_ctx_concat = torch.cat([emb_ref_ctx, emb_cand], dim=1)
        f_context = torch.cat([f_ctx_diff, f_ctx_mult, f_ctx_concat], dim=1)  # (B, 256)

        # 3. Spatial & Generator Gated Attention Fusion
        f_visual_attn = self.attention_fusion(f_local, f_context, spatial_feats)  # (B, 256)

        # 4. Final multi-modal fusion & MLP Head
        f_all = torch.cat([f_visual_attn, spatial_feats], dim=1)  # (B, 259)
        scores = self.mlp_head(f_all)
        return scores


_model_cache_v2 = None


def load_context_aware_ranker_v2(
    checkpoint_path: str = "checkpoints/context_aware_ranker_v2.pt"
) -> ContextAwareRankerV2:
    """Loads trained ContextAwareRankerV2 model checkpoint."""
    global _model_cache_v2
    if _model_cache_v2 is not None:
        return _model_cache_v2

    model = ContextAwareRankerV2(embedding_dim=64, spatial_dim=3)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[ContextAwareRankerV2] Loaded weights from '{checkpoint_path}'.")
    else:
        debug_path = "checkpoints/context_aware_ranker_v2_debug.pt"
        if os.path.exists(debug_path):
            state_dict = torch.load(debug_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"[ContextAwareRankerV2] Loaded debug weights from '{debug_path}'.")
        else:
            print(f"[ContextAwareRankerV2] Warning: No checkpoint found! Using random initialization.")

    _model_cache_v2 = model
    return model


def compute_context_aware_v2_scores(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    checkpoint_path: str = "checkpoints/context_aware_ranker_v2.pt"
) -> list:
    """
    Computes ranking scores for candidate pool using ContextAwareRankerV2.

    Args:
        ref_img: 1000x1000 grayscale reference image.
        search_img: 1000x1000 grayscale search image.
        candidates: List of candidate dictionaries from candidate generation.

    Returns:
        list of float: Ranking scores.
    """
    if not candidates:
        return []

    model = load_context_aware_ranker_v2(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    sh, sw = search_img.shape[:2]
    ref_h, ref_w = ref_img.shape[:2]

    # Pre-process reference patches (Local: 100x100 around center, Context: 300x300 downsampled to 100x100)
    pad = 300
    ref_pad = cv2.copyMakeBorder(ref_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    search_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    rcx, rcy = ref_w / 2.0 + pad, ref_h / 2.0 + pad

    # Reference Local Patch (100x100)
    ref_loc_crop = ref_pad[int(rcy-50):int(rcy+50), int(rcx-50):int(rcx+50)]
    if ref_loc_crop.shape[0] != 100 or ref_loc_crop.shape[1] != 100:
        ref_loc_crop = cv2.resize(ref_loc_crop, (100, 100), cv2.INTER_AREA)
    ref_loc_f = ref_loc_crop.astype(np.float32)
    ref_loc_norm = (ref_loc_f - np.mean(ref_loc_f)) / (np.std(ref_loc_f) + 1e-5)
    t_ref_loc = torch.tensor(ref_loc_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Reference Context Patch (300x300 -> 100x100)
    ref_ctx_crop = cv2.resize(ref_pad[int(rcy-150):int(rcy+150), int(rcx-150):int(rcx+150)], (100, 100), cv2.INTER_AREA)
    ref_ctx_f = ref_ctx_crop.astype(np.float32)
    ref_ctx_norm = (ref_ctx_f - np.mean(ref_ctx_f)) / (np.std(ref_ctx_f) + 1e-5)
    t_ref_ctx = torch.tensor(ref_ctx_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    ref_center_x, ref_center_y = ref_w / 2.0, ref_h / 2.0

    cand_tensors = []
    spatial_tensors = []

    for cand in candidates:
        cx = float(cand.get('center_x', cand.get('cx')))
        cy = float(cand.get('center_y', cand.get('cy')))
        s = float(cand.get('scale', cand.get('primary_scale', 0.10)))
        cand_score = float(cand.get('rank_score', cand.get('score', 0.0)))

        # Relative spatial offsets normalized by image dimensions
        rel_dx = float((cx - ref_center_x) / sw)
        rel_dy = float((cy - ref_center_y) / sh)
        spatial_tensors.append([rel_dx, rel_dy, cand_score])

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

    # Expand reference tensors to match candidate batch size N
    t_ref_loc_batch = t_ref_loc.expand(len(candidates), -1, -1, -1)
    t_ref_ctx_batch = t_ref_ctx.expand(len(candidates), -1, -1, -1)

    with torch.no_grad():
        logits = model(t_ref_loc_batch, t_ref_ctx_batch, t_cands, t_spatial)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    return [float(p) for p in probs]
