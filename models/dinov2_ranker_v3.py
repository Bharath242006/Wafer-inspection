"""
models/dinov2_ranker_v3.py

Production-ready DINOv2 (ViT-B/14) ListMLE Candidate Ranker V3 for Semiconductor Wafer Pattern Matching.

Architecture Overview:
1. Feature Extractor: Vision Transformer backbone (vit_base_patch14_dinov2) producing 768-D L2-normalized embeddings.
2. Pairwise Interaction Head: Computes 3072-D feature representations from reference and candidate embeddings:
   - Absolute difference: |e_ref - e_cand| (768-D)
   - Element-wise product: e_ref * e_cand (768-D)
   - Concatenation: [e_ref, e_cand] (1536-D)
3. Ranking MLP: 3072 -> 1024 -> 256 -> 1 with LayerNorm, GELU, and Dropout(0.2).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv2RankerV3(nn.Module):
    """
    Experimental V3 Candidate Ranker leveraging DINOv2 ViT-B/14 backbone and deep MLP ranking head.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        embedding_dim: int = 768,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initialize DINOv2 Ranker V3 model.

        Args:
            model_name (str): Backbone model identifier from timm. Default: 'vit_base_patch14_dinov2'.
            pretrained (bool): Whether to load pretrained weights. Default: True.
            freeze_backbone (bool): Whether to freeze backbone weights. Default: True.
            embedding_dim (int): Feature dimension output by ViT backbone. Default: 768.
            dropout_rate (float): Dropout probability for MLP head. Default: 0.2.
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze_backbone_flag = freeze_backbone

        # Load backbone via timm
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
            )
        except Exception as e:
            print(f"Warning: Primary model load error ({e}). Attempting fallback to vit_base_patch14_dinov2.lvd142m...")
            self.backbone = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                num_classes=0,
            )

        # Freeze backbone parameters if requested
        if freeze_backbone:
            self.set_freeze_backbone(True)

        # Total input dimension to MLP head: 768 (diff) + 768 (product) + 1536 (concat) = 3072
        head_in_dim = embedding_dim * 4

        # Ranking MLP Head: 3072 -> 1024 -> 256 -> 1
        self.ranker_head = nn.Sequential(
            nn.Linear(head_in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

    def set_freeze_backbone(self, freeze: bool = True) -> None:
        """Helper method to freeze or unfreeze backbone weights dynamically."""
        self.freeze_backbone_flag = freeze
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv2RankerV3":
        """Overridden train mode to keep backbone in eval mode if frozen."""
        super().train(mode)
        if self.freeze_backbone_flag:
            self.backbone.eval()
        return self

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized 768-D embeddings from backbone.

        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).

        Returns:
            torch.Tensor: L2-normalized feature embeddings of shape (N, 768).
        """
        if self.freeze_backbone_flag:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)

        # Ensure 2D tensor
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)

        norm_feats = F.normalize(feats, p=2, dim=-1)
        return norm_feats

    def score_pair_embeddings(self, ref_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute candidate similarity scores from pre-computed reference and candidate embeddings.

        Args:
            ref_emb (torch.Tensor): Reference embeddings of shape (M, 768).
            cand_emb (torch.Tensor): Candidate embeddings of shape (M, 768).

        Returns:
            torch.Tensor: Candidate scores of shape (M, 1).
        """
        abs_diff = torch.abs(ref_emb - cand_emb)
        elem_prod = ref_emb * cand_emb
        concat_feats = torch.cat([ref_emb, cand_emb], dim=-1)

        head_input = torch.cat([abs_diff, elem_prod, concat_feats], dim=-1)
        scores = self.ranker_head(head_input)
        return scores

    def forward(
        self,
        ref_patch: torch.Tensor,
        cand_patches: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for scoring candidates relative to a reference image.

        Args:
            ref_patch (torch.Tensor): Reference image patches of shape (B, 3, H, W).
            cand_patches (torch.Tensor): Candidate patches of shape (B, K, 3, H, W) or (B, 3, H, W).
            return_embeddings (bool): If True, return (scores, (ref_emb, cand_emb)). Default: False.

        Returns:
            torch.Tensor or Tuple:
                - scores: Candidate scores of shape (B, K) if cand_patches is 5D, or (B, 1) if 4D.
                - Optional embeddings tuple if return_embeddings is True.
        """
        # Extract reference embeddings
        ref_emb = self.extract_embedding(ref_patch)  # (B, 768)

        if cand_patches.ndim == 5:
            # Batch of K candidates per reference: (B, K, 3, H, W)
            batch_size, k_candidates, channels, height, width = cand_patches.shape
            cand_flat = cand_patches.view(batch_size * k_candidates, channels, height, width)

            # Extract candidate embeddings
            cand_emb_flat = self.extract_embedding(cand_flat)  # (B * K, 768)

            # Expand reference embeddings to match candidate count
            ref_emb_expanded = ref_emb.unsqueeze(1).expand(batch_size, k_candidates, self.embedding_dim)
            ref_emb_flat = ref_emb_expanded.reshape(batch_size * k_candidates, self.embedding_dim)

            # Compute scores via ranker MLP head
            scores_flat = self.score_pair_embeddings(ref_emb_flat, cand_emb_flat)  # (B * K, 1)
            scores = scores_flat.view(batch_size, k_candidates)  # (B, K)

            if return_embeddings:
                cand_emb = cand_emb_flat.view(batch_size, k_candidates, self.embedding_dim)
                return scores, (ref_emb, cand_emb)
            return scores

        elif cand_patches.ndim == 4:
            # 1-to-1 candidate patches: (B, 3, H, W)
            cand_emb = self.extract_embedding(cand_patches)  # (B, 768)
            scores = self.score_pair_embeddings(ref_emb, cand_emb)  # (B, 1)

            if return_embeddings:
                return scores, (ref_emb, cand_emb)
            return scores
        else:
            raise ValueError(f"Invalid cand_patches shape {cand_patches.shape}. Expected 4D or 5D tensor.")
