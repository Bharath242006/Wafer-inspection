"""
training/train_cnn.py

Fast-debug and scalable training script for the Siamese CNN Candidate Ranker.
"""

import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))

from models.cnn import SiameseNet
from models.losses import ContrastiveLoss
from training.dataset_siamese import SiameseDataset

# ==============================================================================
# FAST DEBUG / SCALABLE CONFIGURATION
# Set TRAIN_IMAGE_LIMIT = None and VALIDATION_IMAGE_LIMIT = None for full dataset
# ==============================================================================
TRAIN_IMAGE_LIMIT = 100
VALIDATION_IMAGE_LIMIT = 20
EPOCHS = 10
BATCH_SIZE = 64


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_siamese_network():
    set_seed(42)

    train_dir = os.path.join("dataset", "train")
    val_dir = os.path.join("dataset", "validation")

    os.makedirs("weights/checkpoints", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("weights", "checkpoints", "siamese_cnn.pt")

    cuda_available = torch.cuda.is_available()
    device_name = "CUDA" if cuda_available else "CPU"
    device = torch.device("cuda" if cuda_available else "cpu")

    # Load Training and Validation Datasets
    train_dataset = SiameseDataset(data_dir=train_dir, max_images=TRAIN_IMAGE_LIMIT)
    val_dataset = SiameseDataset(data_dir=val_dir, max_images=VALIDATION_IMAGE_LIMIT)

    print("====================================")
    print(f"Training Images Used : {train_dataset.num_images}")
    print(f"Validation Images Used : {val_dataset.num_images}")
    print(f"Epochs : {EPOCHS}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"Device : {device_name}")
    print("====================================")

    if train_dataset.num_images == 0:
        print("[Warning] No training images found in dataset/train/. Exiting.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    ) if val_dataset.num_images > 0 else None

    model = SiameseNet(embedding_dim=32).to(device)
    criterion = ContrastiveLoss(margin=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            epoch1_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"Epoch 1 start time: {epoch1_start_str}")

        model.train()
        train_loss = 0.0
        start_t = time.time()

        for x_ref, x_cand, label in train_loader:
            x_ref = x_ref.to(device, non_blocking=True)
            x_cand = x_cand.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            sim = model(x_ref, x_cand)
            loss = criterion(sim, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label)

        avg_train_loss = train_loss / len(train_dataset)

        avg_val_loss = 0.0
        if val_loader is not None and len(val_dataset) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_ref, x_cand, label in val_loader:
                    x_ref = x_ref.to(device, non_blocking=True)
                    x_cand = x_cand.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    sim = model(x_ref, x_cand)
                    loss = criterion(sim, label)
                    val_loss += loss.item() * len(label)
            avg_val_loss = val_loss / len(val_dataset)

        elapsed = time.time() - start_t
        if val_loader is not None and len(val_dataset) > 0:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.2f}s")
        else:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), "checkpoints/siamese_cnn.pt")
    print(f"[Success] Saved trained model checkpoint to '{checkpoint_path}'.")


if __name__ == "__main__":
    train_siamese_network()
