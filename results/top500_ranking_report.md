# DriftSense-X Top-500 Structural Candidate Ranking Report

## Executive Summary
Evaluates handcrafted structural score ranking across all 200 validation samples using the Top-500 candidate pool.

## Candidate Recall vs Handcrafted Ranking Accuracy

| Tolerance | Oracle Candidate Recall (%) | Handcrafted Ranking Accuracy (%) |
|---|---|---|
| $\le 5\text{ px}$ | 12.5% | 0.0%
| $\le 10\text{ px}$ | 24.5% | 0.0%
| $\le 25\text{ px}$ | 64.5% | 0.5%
| $\le 50\text{ px}$ | 95.0% | 1.0%
| $\le 75\text{ px}$ | 99.5% | 1.5%
| $\le 100\text{ px}$ | 100.0% | 2.0%

## Ranked Error Statistics

- **Mean Pixel Error**: 485.39 px
- **Median Pixel Error**: 479.96 px
- **P95 Pixel Error**: 829.72 px
- **Maximum Pixel Error**: 1086.00 px
- **Average Computation Runtime**: 1919.74 ms (1.9197 s)

## Diagnostic Finding

While Top-500 candidate generation achieves **100.0% Oracle Recall** within 100 px, handcrafted structural ranking achieves **2.0% Ranking Accuracy**. This confirms that unnormalized intensity correlation shifts winner selection to periodic cell neighbors.