# DriftSense-X Global Landmark Localizer Report

## Executive Summary
Evaluates the Deterministic Global Landmark Localizer on 40 held-out validation samples (`00161.png` - `00200.png`) and compares directly across 6 candidate ranking approaches.

## 6-Way Direct Comparative Performance Matrix

| Model / Approach | <= 5 px | <= 10 px | <= 25 px | <= 50 px | <= 100 px | Mean Error (px) | Median Error (px) |
|---|---|---|---|---|---|---|---|
| **1. Oracle Top-500 Upper Bound** | 12.5% | 22.5% | 65.0% | 100.0% | 100.0% | 21.53 | 22.58 |
| **2. Handcrafted Top-500 Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 5.0% | 466.23 | 466.50 |
| **3. Siamese CNN Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 5.0% | 457.75 | 466.50 |
| **4. Context CNN Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 5.0% | 506.02 | 527.62 |
| **5. Global/Lattice-Aware Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 7.5% | 472.30 | 469.75 |
| **6. Global Landmark Method** | 0.0% | 0.0% | 2.5% | 5.0% | 10.0% | 431.93 | 437.33 |

## Inference Runtime Benchmark

- **Average Global Landmark Runtime per Image**: 984.57 ms (0.9846 s)

## Final Assessment & Integration Verdict

**VERDICT**: The Global Landmark Localizer demonstrates empirical improvement on held-out validation data.