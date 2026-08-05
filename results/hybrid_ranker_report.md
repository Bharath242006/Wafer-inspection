# DriftSense-X Final Hybrid Ranker — Validation Report

## Summary

Evaluated the new **56-D Hybrid Ranker** (HybridRankerNet MLP) on 40 held-out validation images (00161–00200).

Target to beat: **431.93 px** mean error (Global Landmark baseline).

## 3-Way Comparative Performance

| Approach | ≤5px | ≤10px | ≤25px | ≤50px | ≤100px | Mean Error | Median | P95 | Max |
|---|---|---|---|---|---|---|---|---|---|
| **Oracle Top-500** | 12.5% | 22.5% | 65.0% | 100.0% | 100.0% | 21.53 | 22.58 | 40.69 | 49.46 |
| **Global Landmark (baseline)** | 0.0% | 0.0% | 2.5% | 5.0% | 10.0% | 431.93 | 437.33 | 817.30 | 935.60 |
| **Hybrid Ranker (NEW)** | 0.0% | 0.0% | 0.0% | 0.0% | 2.5% | **534.74** | 544.57 | 887.54 | 954.89 |

## Runtime

- **Average Hybrid Ranker runtime**: 1651.31 ms/image

## Hybrid Features (56-D)

| Group | Features | Dim |
|---|---|---|
| Visual (NCC/Grad/LoG) | raw + pool-normalized | 6 |
| FFT phase-correlation | NEW: discriminates periodic aliases | 1 |
| Edge/Canny overlap | raw + normalized | 2 |
| Low-freq Gaussian | raw + normalized | 2 |
| Medium-context (150px) | NEW vs. prior rankers | 1 |
| Global landmark heatmap | | 1 |
| Coordinates | cx, cy, cx/1000, cy/1000, cx/W, cy/H, dist_center | 7 |
| Lattice phase | cx/lx, cy/ly, phase_x/y, sin/cos encodings | 8 |
| Rank + margins | rank/500, log(rank), percentile, margin_top1/median | 5 |
| Local density | density_r30/r60, dist_nearest | 3 |
| Neighbor consistency | ±lx, ±ly lattice-direction NCC | 4 |
| Extended neighbors | diagonal ±(lx,ly), half-period, 2nd-order | 6 |
| Pool statistics | heatmap grad, top10 mean, top10 margin | 3 |
| Multi-scale NCC | 100→50→25→12 px (normalized) | 2 |
| Spatial neighbor score | mean top-5 spatial candidate scores | 1 |
| **TOTAL** | | **56** |

## Integration Verdict

**✗ VERDICT: NOT IMPROVED** — Hybrid Ranker achieves 534.74 px mean error (+102.81 px vs. baseline 431.93 px).

### Root Cause Analysis

The primary failure mode remains **periodic aliasing**: the reference pattern repeats at lattice intervals (~67 px), and candidates at alias locations produce nearly identical visual feature vectors. The FFT phase-correlation score provides a weak additional signal but is insufficient on its own when:

1. The search patch is noisy and phase-correlation peaks are broad.
2. Alias candidates differ by an exact integer number of lattice periods,    causing the FFT phase response to look identical to the true match.

→ Keeping Global Landmark as the final pipeline.
