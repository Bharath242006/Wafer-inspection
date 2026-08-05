# DriftSense-X Trained Siamese CNN Evaluation Report

## Executive Summary
Evaluated on 40 held-out validation samples (`00161.png` - `00200.png`) using Contrastive Loss trained Siamese Neural Network weights (`checkpoints/siamese_cnn.pt`).

## Candidate Recall vs CNN Ranking Accuracy

- **Candidate Recall (before CNN, <= 50 px)**: 32.5%
- **Candidate Recall (before CNN, <= 100 px)**: 67.5%
- **CNN Ranking Accuracy (<= 50 px)**: 0.0%
- **CNN Ranking Accuracy (<= 100 px)**: 0.0%

## Held-Out Validation Error Statistics

- **Mean Error**: 473.78 px
- **Median Error**: 496.68 px
- **P95 Error**: 863.06 px
- **Maximum Error**: 885.18 px

| Accuracy Threshold | Percentage (%) | Count |
|---|---|---|
| $\le 5\text{ px}$ | 0.0% | 0/40 |
| $\le 10\text{ px}$ | 0.0% | 0/40 |
| $\le 25\text{ px}$ | 0.0% | 0/40 |
| $\le 50\text{ px}$ | 0.0% | 0/40 |
| $\le 100\text{ px}$ | 0.0% | 0/40 |

## Inference Runtime Benchmark

- **Average CNN Scoring Time per Image**: 96.24 ms (0.0962 s)
- **Average Total Pipeline Time per Image**: 963.85 ms (0.9638 s)
