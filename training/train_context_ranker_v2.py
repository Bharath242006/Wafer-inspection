"""
training/train_context_ranker_v2.py

Industrial Fast Training Script for ContextAwareRankerV2.
Trains ResNet backbone + Attention Fusion + MLP head using Pairwise Margin Ranking Loss
and Hard Negative Mining from Top500 candidate pool.

Validation-Based Early Stopping:
    Every `val_every` epochs, the model is evaluated on a held-out validation set.
    Checkpoints are saved only when GT Mean Rank improves — not when train loss drops.
    This prevents score-saturation overfitting (train loss=0 while val performance regresses).
"""

import os
import sys
import csv
import math
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from localization.context_aware_ranker_v2 import ContextAwareRankerV2
from localization.candidate_generation import generate_candidate_pool_multi
from localization.context_aware_ranker_v2 import compute_context_aware_v2_scores
from training.dataset_context_ranker_v2 import ContextRankerDatasetV2


def run_val_epoch(model, val_records, ref_dir: str, search_dir: str,
                  checkpoint_path: str, device) -> float:
    """
    Evaluate the current model on val_records and return GT Mean Rank.
    Saves a temporary checkpoint, runs inference via compute_context_aware_v2_scores,
    then returns the mean GT rank across all val images.
    """
    # Save current weights to a temp file for the inference helper
    tmp_ckpt = checkpoint_path + ".tmp_val.pt"
    torch.save(model.state_dict(), tmp_ckpt)

    gt_ranks = []
    model.eval()
    with torch.no_grad():
        for item in val_records:
            img_name = item["image"]
            gt_x = float(item["x"])
            gt_y = float(item["y"])

            ref_path  = os.path.join(ref_dir, img_name)
            srch_path = os.path.join(search_dir, img_name)
            ref_img   = cv2.imread(ref_path,  cv2.IMREAD_GRAYSCALE)
            srch_img  = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)

            if ref_img is None or srch_img is None:
                continue

            cands = generate_candidate_pool_multi(ref_img, srch_img, max_pool_size=500)
            if not cands:
                gt_ranks.append(500)
                continue

            scores = compute_context_aware_v2_scores(
                ref_img, srch_img, cands, checkpoint_path=tmp_ckpt
            )
            for c, sc in zip(cands, scores):
                cg_sc = float(c.get('final_score', c.get('score', 0.0)))
                c['rank_score'] = 0.60 * sc + 0.40 * cg_sc

            cands_sorted = sorted(cands, key=lambda c: c['rank_score'], reverse=True)
            dists = [math.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in cands_sorted]
            gt_rank = int(np.argmin(dists)) + 1
            gt_ranks.append(gt_rank)

    # Clean up temp file
    if os.path.exists(tmp_ckpt):
        os.remove(tmp_ckpt)

    return float(np.mean(gt_ranks)) if gt_ranks else 500.0


def train_context_ranker_v2(
    phase: str = "debug",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    margin: float = 0.2,
    max_samples: int = None,
    val_every: int = 5,
    val_samples: int = 20
):
    print("=" * 100, flush=True)
    print(f"      STARTING CONTEXT-AWARE RANKER V2 TRAINING ({phase.upper()} PHASE)", flush=True)
    print("=" * 100 + "\n", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}", flush=True)

    val_csv    = os.path.join("dataset", "validation", "labels.csv")
    val_ref    = os.path.join("dataset", "validation", "reference")
    val_search = os.path.join("dataset", "validation", "search")

    # Load held-out validation records (skip first 100 which are used for final benchmarking)
    with open(val_csv, "r", encoding="utf-8") as f:
        all_val = list(csv.DictReader(f))
    val_records = all_val[100 : 100 + val_samples]  # images 101-120 as held-out val
    print(f"[Validation] Held-out val set: {len(val_records)} images (records 101-{100+len(val_records)})", flush=True)

    train_csv = os.path.join("dataset", "train", "labels.csv")
    train_ref = os.path.join("dataset", "train", "reference")
    train_search = os.path.join("dataset", "train", "search")

    if not os.path.exists(train_csv):
        train_csv = val_csv
        train_ref = val_ref
        train_search = val_search

    if max_samples is None:
        max_train_samples = 100 if phase == "debug" else 500
    else:
        max_train_samples = max_samples

    checkpoint_name = "context_aware_ranker_v2_debug.pt" if phase == "debug" else "context_aware_ranker_v2.pt"

    train_dataset = ContextRankerDatasetV2(
        csv_path=train_csv,
        ref_dir=train_ref,
        search_dir=train_search,
        max_samples=max_train_samples,
        top_k_hard_negs=10
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ContextAwareRankerV2(embedding_dim=64, spatial_dim=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MarginRankingLoss(margin=margin)

    total_pairs = len(train_dataset)
    total_batches = math.ceil(total_pairs / batch_size)
    print(f"\nStarting Pairwise Margin Ranking Training Loop ({total_pairs} pairs)...", flush=True)
    print(f"Total Epochs  : {epochs}", flush=True)
    print(f"Total Batches : {total_batches} per epoch", flush=True)
    print(f"Batch Size    : {batch_size}", flush=True)
    print(f"Val Every     : every {val_every} epochs on {val_samples} held-out images", flush=True)
    print(f"Checkpoint On : best Val GT Mean Rank (not train loss)", flush=True)

    best_val_rank = 1e9   # lower is better — checkpoint when this improves
    checkpoint_path = os.path.join("checkpoints", "context_aware_ranker_v2.pt")

    for epoch in range(1, epochs + 1):
        # ── Epoch header ──────────────────────────────────────────────────────
        print(f"\n{'=' * 50}", flush=True)
        print(f"  Epoch {epoch} / {epochs}", flush=True)
        print(f"{'=' * 50}", flush=True)

        model.train()
        running_loss = 0.0
        batch_cnt = 0
        epoch_start = time.time()

        # ── tqdm batch progress bar ───────────────────────────────────────────
        pbar = tqdm(
            train_loader,
            total=total_batches,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True
        )

        for t_ref_loc, t_ref_ctx, t_cand_pos, t_spatial_pos, t_cand_neg, t_spatial_neg in pbar:
            t_ref_loc = t_ref_loc.to(device)
            t_ref_ctx = t_ref_ctx.to(device)
            t_cand_pos = t_cand_pos.to(device)
            t_spatial_pos = t_spatial_pos.to(device)
            t_cand_neg = t_cand_neg.to(device)
            t_spatial_neg = t_spatial_neg.to(device)

            optimizer.zero_grad()

            # Forward pass for positive candidate & negative candidate
            score_pos = model(t_ref_loc, t_ref_ctx, t_cand_pos, t_spatial_pos)
            score_neg = model(t_ref_loc, t_ref_ctx, t_cand_neg, t_spatial_neg)

            # Pairwise Margin Loss: score_pos should be higher than score_neg by at least margin
            target = torch.ones_like(score_pos).to(device)
            loss = criterion(score_pos, score_neg, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            batch_cnt += 1

            # Update tqdm postfix with current batch loss and running average
            avg_so_far = running_loss / batch_cnt
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_so_far:.4f}")

        pbar.close()
        scheduler.step()

        epoch_loss = running_loss / max(1, batch_cnt)
        epoch_elapsed = time.time() - epoch_start

        # ── Post-epoch summary ────────────────────────────────────────────────
        print(f"\nEpoch {epoch}/{epochs} Completed", flush=True)
        print(f"  Average Loss : {epoch_loss:.4f}", flush=True)
        print(f"  Time         : {epoch_elapsed:.1f} seconds", flush=True)

        # ── Validation-Based Early Stopping ───────────────────────────────────
        if epoch % val_every == 0 or epoch == epochs:
            print(f"\n  [Validation] Running GT Rank eval on {len(val_records)} held-out images...", flush=True)
            val_start = time.time()
            val_gt_rank = run_val_epoch(
                model, val_records, val_ref, val_search, checkpoint_path, device
            )
            val_elapsed = time.time() - val_start
            is_best = val_gt_rank < best_val_rank

            if is_best:
                best_val_rank = val_gt_rank
                ckpt_p1 = os.path.join("checkpoints", "context_aware_ranker_v2_debug.pt")
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), ckpt_p1)
                status_str = "Checkpoint Saved [BEST VAL]"
            else:
                status_str = "No improvement"

            print(f"  [Validation] Val GT Mean Rank : {val_gt_rank:.1f} / 500", flush=True)
            print(f"  [Validation] Best So Far      : {best_val_rank:.1f} / 500", flush=True)
            print(f"  [Validation] Status           : {status_str}", flush=True)
            print(f"  [Validation] Time             : {val_elapsed:.1f} seconds", flush=True)
        else:
            print(f"  Status       : Trained (next val at epoch {((epoch // val_every) + 1) * val_every})", flush=True)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  TRAINING COMPLETE", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  Best Val GT Mean Rank : {best_val_rank:.1f} / 500", flush=True)
    print(f"  Checkpoints saved to  : checkpoints/context_aware_ranker_v2.pt", flush=True)
    print(f"{'=' * 100}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",       type=str,   default="debug", choices=["debug", "final"])
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--val_every",   type=int,   default=5,
                        help="Run validation every N epochs and checkpoint on best GT Mean Rank.")
    parser.add_argument("--val_samples", type=int,   default=20,
                        help="Number of held-out validation images to use for early stopping.")
    args = parser.parse_args()

    train_context_ranker_v2(
        phase=args.phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        val_every=args.val_every,
        val_samples=args.val_samples
    )

