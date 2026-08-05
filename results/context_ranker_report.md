# DriftSense-X Multi-Branch Context-Aware Candidate Ranker Report

## Executive Summary
Evaluates the Multi-Branch Context-Aware Candidate Ranker on 40 held-out validation samples (`00161.png` - `00200.png`) and compares directly against Handcrafted, Siamese CNN, and Oracle Top-500 bounds.

## 4-Way Direct Comparative Performance Matrix

| Model / Approach | <= 5 px | <= 10 px | <= 25 px | <= 50 px | <= 100 px | Mean Error (px) | Median Error (px) |
|---|---|---|---|---|---|---|---|
| **D. Oracle Top-500 Upper Bound** | 12.5% | 22.5% | 65.0% | 100.0% | 100.0% | 21.53 | 22.58 |
| **A. Handcrafted Top-500 Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 5.0% | 466.23 | 466.50 |
| **B. Siamese CNN Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 5.0% | 457.75 | 466.50 |
| **C. Context-Aware Ranker** | 0.0% | 0.0% | 2.5% | 5.0% | 5.0% | 506.02 | 527.62 |

## Inference Runtime Benchmark

- **Average Context Scoring Runtime per Image**: 3072.64 ms (3.0726 s)

## Decision & Verdict

**VERDICT**: The Context-Aware Candidate Ranker does NOT demonstrate sufficient held-out validation improvement over handcrafted ranking. Per instructions, `final_localizer.py` will NOT be modified.