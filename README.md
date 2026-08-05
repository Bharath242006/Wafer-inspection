# DriftSense-X

> AI-Powered Navigation-Error Recovery for Wafer Inspection

---

## Overview

DriftSense-X is a visual localization system for semiconductor wafer inspection. Given a reference image (the target pattern) and a search image (the full wafer field), it predicts the **center coordinates (X, Y)** of the target site within the search image.

The system uses a deterministic **Global Landmark** localization pipeline with no runtime dependencies on GPU or neural network inference.

---

## Problem

Charged-particle beam inspection tools (SEM, FIB) can experience navigation drift. If the stage drifts, the tool may not return to the correct target site. DriftSense-X recovers the correct coordinates from image evidence alone.

The core challenge is **periodic aliasing**: wafer cell arrays repeat at lattice intervals (~67 px), making it visually impossible to distinguish the correct cell from its aliases using local appearance alone.

---

## Solution

DriftSense-X uses a two-stage pipeline:

1. **Candidate Generation** — Multi-scale template matching extracts up to 500 candidate locations.
2. **Global Landmark Scoring** — A macro-resolution heatmap (low-frequency Gaussian + edge density) gives each candidate a global consistency score, resolving periodic aliases at the coarse level.

---

## Architecture

```
Reference Image (1000×1000 px)
    │
    ├─► Multi-scale template matching (7 scales, grayscale + gradient + LoG + blur)
    │       → Top-500 candidate pool (100% recall within 100 px)
    │
    ├─► Global Landmark Heatmap
    │       (Gaussian blur correlation + Canny edge density matching at 100×100 macro scale)
    │
    └─► Candidate Scoring (0.60 × heatmap + 0.40 × peak score)
            → Winner selection
            → Fine sub-pixel refinement (±50 px, 100×100 template)
            → Final (X, Y) coordinates
```

**Production method**: Global Landmark  
**Hybrid Ranker**: Evaluated and rejected (see Benchmark section).

---

## Installation

```bash
cd D:\DriftSense-X
python -m pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application |
| `opencv-python` | Image processing, template matching |
| `numpy` | Numerical computation |
| `Pillow` | Image I/O |
| `torch` | (Optional) ranker model inference |

---

## Running the Application

```bash
cd D:\DriftSense-X
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Programmatic API

```python
import sys
sys.path.insert(0, r"D:\DriftSense-X")
from localization.final_localizer_hybrid import locate_target

result = locate_target(
    ref_path="path/to/reference.png",
    search_path="path/to/search.png",
    gt_x=513.87,   # optional ground truth
    gt_y=783.43,
    draw_output_path="annotated.png",  # optional visualization
)

print(result["predicted_x"])    # float — X coordinate (px)
print(result["predicted_y"])    # float — Y coordinate (px)
print(result["pixel_error"])    # float | None
print(result["confidence"])     # float [0, 1]
print(result["candidate_rank"]) # int
print(result["runtime_sec"])    # float
print(result["status"])         # "SUCCESS" | "FAILED"
```

---

## Input

| Field | Format | Notes |
|-------|--------|-------|
| Reference image | PNG / JPG grayscale | Typically 1000×1000 px SEM crop |
| Search image | PNG / JPG grayscale | Typically 1000×1000 px field image |
| Ground truth X/Y | float (pixels) | Optional — enables pixel error reporting |

---

## Output

| Field | Type | Description |
|-------|------|-------------|
| `predicted_x` | float | Predicted target center X (pixels) |
| `predicted_y` | float | Predicted target center Y (pixels) |
| `pixel_error` | float \| None | Euclidean error to GT (if GT provided) |
| `confidence` | float | Global landmark alignment score [0, 1] |
| `candidate_rank` | int | Rank of winning candidate in Top-500 pool |
| `runtime_sec` | float | Total pipeline runtime (seconds) |
| `status` | str | "SUCCESS" or "FAILED" |

---

## Benchmark

Evaluated on 40 held-out validation images (00161–00200), never seen during training.

| Method | Mean Error | Median | ≤100 px Acc. | Notes |
|--------|-----------|--------|-------------|-------|
| Oracle Top-500 | **21.53 px** | 22.58 px | 100.0% | Best possible (pool upper bound) |
| **Global Landmark** | **431.93 px** | 437.33 px | 10.0% | **Production method** |
| Hybrid Ranker (56-D MLP) | 534.74 px | 544.57 px | 2.5% | Evaluated and **rejected** |
| Siamese CNN | 457.75 px | — | 5.0% | Experimental only |
| Context CNN | 506.02 px | — | 5.0% | Experimental only |
| Global Lattice | 472.30 px | — | 7.5% | Experimental only |

> **The Hybrid Ranker was evaluated but rejected** because its held-out mean error was
> **534.74 px**, which is worse than the Global Landmark baseline of **431.93 px**.
> The Hybrid Ranker files (`localization/hybrid_ranker.py`, `checkpoints/hybrid_ranker.pt`,
> `results/hybrid_ranker_report.md`) are preserved for documentation.

### Oracle Gap Analysis

The 410 px gap between oracle (21.53 px) and production (431.93 px) is due to **periodic aliasing**: candidate cells at integer lattice periods (~67 px apart) are visually indistinguishable. Global structural disambiguation is needed to close this gap.

---

## Project Structure

```
DriftSense-X/
├── app.py                              # Streamlit web application
├── requirements.txt
├── README.md
├── localization/
│   ├── final_localizer_hybrid.py       # locate_target() — production API
│   ├── global_landmark_localizer.py    # Core Global Landmark algorithm
│   ├── final_localizer.py              # Stage pipeline utilities
│   ├── global_coarse_localizer.py      # Stage 1 coarse search
│   ├── hybrid_ranker.py                # Hybrid Ranker (experimental, not used)
│   └── coordinate_aware_ranker.py      # Coordinate ranker (experimental)
├── training/
│   ├── train_hybrid_ranker.py
│   └── dataset_hybrid_ranker.py
├── checkpoints/
│   ├── siamese_cnn.pt
│   ├── context_ranker.pt
│   ├── global_lattice_ranker.pt
│   ├── coordinate_aware_ranker.pt
│   └── hybrid_ranker.pt                # Trained but not used in production
├── results/
│   ├── hybrid_ranker_report.md
│   ├── hybrid_ranker_validation.csv
│   └── global_landmark_report.md
└── dataset/
    └── validation/
        ├── reference/    # 00001.png – 00200.png
        ├── search/
        └── labels.csv
```

---

## Limitations

- Runtime: ~1 second per image pair (CPU, no GPU required).
- Accuracy is constrained by periodic aliasing (see Oracle Gap Analysis).
- Images should be grayscale SEM crops at approximately 1000×1000 px.
- The system does not claim nanometre accuracy.
- Validated mean error of 431.93 px applies to the specific benchmark dataset.

---

## Citation / License

Internal research project — DriftSense-X, 2026.
