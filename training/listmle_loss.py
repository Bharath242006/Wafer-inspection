"""
training/listmle_loss.py

Numerically Stable ListMLE (Listwise Maximum Likelihood Estimation) Loss for Candidate Ranking.

Mathematical Formulation:
Given a batch of predicted scores s = [s_1, s_2, ..., s_K] and ground-truth permutation indices pi = [pi_1, pi_2, ..., pi_K],
ListMLE computes the negative log-likelihood under the Plackett-Luce ranking probability model:

    L_ListMLE(s, pi) = - sum_{i=1}^{K} ( s_{pi_i} - log( sum_{j=i}^{K} exp(s_{pi_j}) ) )

Numerical Stability:
To prevent numerical overflow/underflow, the log-sum-exp term is computed using vectorized suffix log-sum-exp via torch.logcumsumexp on flipped score tensors.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ListMLELoss(nn.Module):
    """
    ListMLE Loss module for ranking tasks.
    Supports batch training and arbitrary candidate list length K.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-12) -> None:
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'. Default: 'mean'.
            eps (float): Small epsilon constant for numerical safety. Default: 1e-12.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'. Supported: 'mean', 'sum', 'none'.")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the ListMLE loss for a batch of candidate score lists.

        Args:
            y_pred (torch.Tensor): Predicted candidate scores of shape (B, K).
                                  Higher predicted score indicates higher ranking.
            y_true (torch.Tensor): Target permutation indices of shape (B, K) (long tensor),
                                  where y_true[:, i] is the candidate index at ground-truth rank position i.
                                  Alternatively, if y_true contains floating point distances/relevance scores,
                                  candidates are sorted ascendingly (smaller distance = better rank).

        Returns:
            torch.Tensor: Scalar loss tensor if reduction is 'mean' or 'sum', or 1D tensor of shape (B,) if 'none'.
        """
        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must be a 2D tensor of shape (B, K), got shape {y_pred.shape}")

        batch_size, num_candidates = y_pred.shape

        if num_candidates < 2:
            raise ValueError(f"ListMLE requires at least 2 candidates per list, got K = {num_candidates}")

        # Resolve permutation indices
        if torch.is_floating_point(y_true):
            # Sort ascendingly: candidate with smallest ground-truth distance gets rank index 0
            target_perm = torch.argsort(y_true, dim=-1, descending=False)
        else:
            target_perm = y_true.long()

        if target_perm.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch between target_perm {target_perm.shape} and y_pred {y_pred.shape}")

        # Re-order predicted scores according to target permutation:
        # ordered_scores[:, i] is the predicted score of the item at ground-truth rank position i
        ordered_scores = torch.gather(y_pred, dim=1, index=target_perm)  # (B, K)

        # Vectorized Suffix Log-Sum-Exp computation:
        # For rank position i, we need log( sum_{j=i}^{K-1} exp(ordered_scores[:, j]) )
        # Flipping along candidate dimension transforms suffix sum into prefix sum:
        flipped_scores = torch.flip(ordered_scores, dims=[1])  # (B, K)

        # LogCumSumExp on flipped tensor computes numerically stable cumulative log-sum-exp
        log_denom_flipped = torch.logcumsumexp(flipped_scores, dim=1)  # (B, K)

        # Flip back to restore original suffix alignment
        log_denom = torch.flip(log_denom_flipped, dims=[1])  # (B, K)

        # Plackett-Luce negative log likelihood per item: - (s_{pi_i} - log_denom_i)
        loss_per_item = log_denom - ordered_scores  # (B, K)

        # Sum negative log likelihood across all rank positions in list
        loss_per_seq = torch.sum(loss_per_item, dim=1)  # (B,)

        if self.reduction == "mean":
            return loss_per_seq.mean()
        elif self.reduction == "sum":
            return loss_per_seq.sum()
        else:
            return loss_per_seq
