"""
training/train_dinov2_v3.py

Training Pipeline for DINOv2 ListMLE Candidate Ranker V3.

Features:
- Argparse CLI interface
- PyTorch AMP mixed-precision training
- AdamW optimizer & CosineAnnealingLR scheduler
- Gradient clipping
- Per-epoch validation computing Validation Mean Rank metric
- Early stopping with patience = 5
- Automatic checkpointing saving best_model.pt and last_model.pt
- Structured logging with tqdm progress bars
"""

import argparse
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root directory to path for seamless imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dinov2_ranker_v3 import DINOv2RankerV3
from training.dataset_listmle_v3 import ListMLEDatasetV3
from training.listmle_loss import ListMLELoss


def seed_everything(seed: int = 42) -> None:
    """Sets random seed across python, numpy, and torch for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for V3 ranker training."""
    parser = argparse.ArgumentParser(description="Train DINOv2 ListMLE Candidate Ranker V3")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--k_candidates", type=int, default=32, help="Number of candidates K per sample")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze DINOv2 backbone weights")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false", help="Unfreeze DINOv2 backbone weights")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset_root", type=str, default="dataset_small", help="Root directory of dataset_small")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers count")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Trains model for one epoch using ListMLE loss and AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        ref_patch = batch["reference_patch"].to(device, non_blocking=True)  # (B, 3, 224, 224)
        cand_patches = batch["candidate_patches"].to(device, non_blocking=True)  # (B, K, 3, 224, 224)
        target_rank = batch["target_rank"].to(device, non_blocking=True)  # (B, K)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                scores = model(ref_patch, cand_patches)  # (B, K)
                loss = criterion(scores, target_rank)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            scores = model(ref_patch, cand_patches)  # (B, K)
            loss = criterion(scores, target_rank)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluates validation set and computes Mean Rank of the ground-truth top candidate.

    Returns:
        float: Mean rank of ground-truth candidate across validation set (lower is better, 1.0 is perfect).
    """
    model.eval()
    all_ranks = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        ref_patch = batch["reference_patch"].to(device, non_blocking=True)
        cand_patches = batch["candidate_patches"].to(device, non_blocking=True)
        target_rank = batch["target_rank"].to(device, non_blocking=True)  # (B, K)

        scores = model(ref_patch, cand_patches)  # (B, K)
        batch_size, k_candidates = scores.shape

        for b in range(batch_size):
            b_scores = scores[b]  # (K,)
            b_targets = target_rank[b]  # (K,)

            # Candidate index 0 in target_rank is the top ground-truth candidate
            gt_cand_index = b_targets[0].item()

            # Rank predicted scores descendingly (higher score = better rank)
            sorted_pred_indices = torch.argsort(b_scores, descending=True).cpu().tolist()

            # Find 1-based rank position of ground-truth top candidate
            rank_position = sorted_pred_indices.index(gt_cand_index) + 1
            all_ranks.append(rank_position)

    mean_rank = float(np.mean(all_ranks)) if len(all_ranks) > 0 else float("inf")
    return mean_rank


def run_training(args: argparse.Namespace) -> None:
    """Main training routine."""
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare checkpoint save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pt"
    last_model_path = save_dir / "last_model.pt"

    # Initialize Datasets and DataLoaders
    print(f"Loading dataset from '{args.dataset_root}'...")
    train_dataset = ListMLEDatasetV3(
        root_dir=args.dataset_root,
        split="train",
        num_candidates=args.k_candidates,
        seed=args.seed,
    )
    val_dataset = ListMLEDatasetV3(
        root_dir=args.dataset_root,
        split="validation",
        num_candidates=args.k_candidates,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Initialize Model
    print(f"Initializing DINOv2RankerV3 (freeze_backbone={args.freeze_backbone})...")
    model = DINOv2RankerV3(
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # Initialize Loss, Optimizer, Scheduler, and AMP Scaler
    criterion = ListMLELoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Early stopping and performance tracking
    best_val_mean_rank = float("inf")
    best_epoch = -1
    patience = 5
    patience_counter = 0

    print("\nStarting Training Loop...")
    print("=" * 90)
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Mean Rank':<18}{'Learning Rate':<16}{'Epoch Time':<14}{'Best Epoch':<10}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=(device.type == "cuda"),
        )

        val_mean_rank = evaluate_validation(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_rank": val_mean_rank,
                "train_loss": train_loss,
            },
            last_model_path,
        )

        # Check for best validation metric
        is_best = val_mean_rank < best_val_mean_rank
        if is_best:
            best_val_mean_rank = val_mean_rank
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mean_rank": val_mean_rank,
                    "train_loss": train_loss,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        print(
            f"{epoch:<8}{train_loss:<15.4f}{val_mean_rank:<18.4f}{current_lr:<16.6e}{epoch_time:<14.2f}s{best_epoch:<10}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered: No validation improvement for {patience} consecutive epochs.")
            break

    print("=" * 90)
    print(f"Training completed! Best Validation Mean Rank: {best_val_mean_rank:.4f} at Epoch {best_epoch}")
    print(f"Best model saved to: {best_model_path.resolve()}")
    print(f"Last model saved to: {last_model_path.resolve()}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_training(cli_args)
