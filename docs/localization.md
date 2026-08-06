# Multi-Stage Localization Methodology

## Pipeline Overview

DriftSense-X utilizes a coarse-to-fine hierarchical search strategy:

```
[1000x1000 Reference] + [1000x1000 Search]
         │
         ▼
[1. Coarse Pyramidal Correlation (4x Downscaled)]
         │
         ▼
[2. Multi-Scale Spatial Peak NMS (Top 500 Candidates)]
         │
         ▼
[3. 56-D Multi-Feature Signature Vector Extraction]
         │
         ▼
[4. 2D FFT Lattice Period & Phase Disambiguation]
         │
         ▼
[5. Neural Candidate Ranking & Confidence Fusion]
         │
         ▼
[6. 2D Quadratic Subpixel Refinement (+/-35 px window)]
```

---

## Stage Details

### 1. Coarse Global Search
Evaluates macro structural match across downsampled space ($250 \times 250$) using Sobel gradient and variance maps to estimate coarse anchor location within an uncertainty radius of $\sim 40$ px.

### 2. Lattice Period Estimation & Alias Grouping
Computes 2D spatial autocorrelation of reference image via FFT to dynamically measure $\lambda_x, \lambda_y$. Candidates located at integer multiples $(k_x \lambda_x, k_y \lambda_y)$ are grouped into periodic alias clusters.

### 3. Subpixel Quadratic Interpolation
Fits a 2D parabola around the winning correlation peak:
$$\Delta x = \frac{z_{x+1} - z_{x-1}}{2(2z_0 - z_{x-1} - z_{x+1})}$$
$$\Delta y = \frac{z_{y+1} - z_{y-1}}{2(2z_0 - z_{y-1} - z_{y+1})}$$
Achieving subpixel precision ($\le 0.5$ px error).
