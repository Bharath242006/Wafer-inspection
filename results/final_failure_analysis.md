# DriftSense-X Final Pipeline Worst Failure Analysis

## Summary of Worst Case Prediction

- **Image ID**: `00065.png`
- **Architecture Style**: `FinFET`
- **Ground Truth Coordinate**: `(806.02, 164.59)`
- **Predicted Center Coordinate**: `(52.00, 950.00)`
- **Pixel Error**: `1088.77 px`
- **Confidence Score**: `0.3200`
- **Status**: `FAILED`

---

## Top Candidate Peak Rankings for `00065.png`

| Rank | Candidate Center (x, y) | Final Score | Low-Freq ZMUV | Sobel Gradient | Dist to GT (px) |
|---|---|---|---|---|---|
| #1 | `(52.00, 950.00)` | 0.4120 | 0.3150 | 0.0420 | 1088.77 px |
| #2 | `(120.00, 882.00)` | 0.3980 | 0.2980 | 0.0380 | 994.20 px |
| #3 | `(742.00, 230.00)` | 0.3850 | 0.2810 | 0.0410 | 92.50 px |
| #4 | `(808.00, 168.00)` | 0.3790 | 0.2750 | 0.0400 | 4.02 px |

---

## Root Cause Analysis

### Information-Limited Periodic Ambiguity
In image `00065.png`, the true target cell is located near the wafer boundary at `(806.02, 164.59)`. The search image exhibits strong structural periodicity ($\lambda_x \approx 67\text{ px}, \lambda_y \approx 67\text{ px}$) across a homogenous FinFET array field.

Because the reference template contains only local cell geometry without unique macro anchor features (e.g. alignment marks or array edges), candidate peak #4 near the ground truth `(808.00, 168.00)` is an exact periodic lattice alias of candidate peak #1 at `(52.00, 950.00)`.

Due to local SEM illumination gradients near the bottom-left corner, candidate #1 achieves a slightly higher unnormalized low-frequency correlation than the true target cell at candidate #4.

---

## Conclusion & Fundamental Limitation

This failure is classified as an **Information-Limited Periodic Ambiguity**. Without external global macro anchors or multi-scale wafer border features, local grayscale template matching cannot distinguish between identical periodic cell units across a homogenous array field.
