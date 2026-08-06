"""
localization/ranking/cnn_ranker.py

Siamese CNN Candidate Ranker evaluator wrapper.
"""

import os
import cv2
import numpy as np
import torch

from models.cnn import SiameseNet


_model_cache = None


def load_trained_siamese_model(checkpoint_path: str = "weights/checkpoints/siamese_cnn.pt") -> SiameseNet:
    """Loads trained Siamese CNN model checkpoint."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(checkpoint_path) and os.path.exists("checkpoints/siamese_cnn.pt"):
        checkpoint_path = "checkpoints/siamese_cnn.pt"

    model = SiameseNet(embedding_dim=32)
    model.eval()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    
    _model_cache = model
    return model


def compute_cnn_similarity_scores(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list,
    checkpoint_path: str = "weights/checkpoints/siamese_cnn.pt"
) -> list:
    """
    Calculates Siamese CNN similarity scores for candidate pool.
    """
    model = load_trained_siamese_model(checkpoint_path)

    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA).astype(np.float32)
    ref_norm = (ref_100 - np.mean(ref_100)) / (np.std(ref_100) + 1e-5)
    t_ref = torch.tensor(ref_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pad = 60
    search_pad = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

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
