# DriftSense-X Diagnostic Oracle-Ranking & Candidate Recall Report

## Executive Summary
This diagnostic study evaluates the candidate generation pool across all 200 validation samples under an **Oracle Ranking** assumption (i.e. if an oracle selector always chose the candidate closest to Ground Truth).

## Candidate-Generation Recall Statistics

- **Total Validation Samples**: 200
- **Average Candidates per Pool**: 50.0
- **Median Candidates per Pool**: 50.0
- **Worst-Case Candidate Distance to GT**: 427.54 px

| Distance Tolerance | Oracle Recall (%) | Count |
|---|---|---|
| $\le 5\text{ px}$ | 5.0% | 10/200 |
| $\le 10\text{ px}$ | 6.5% | 13/200 |
| $\le 25\text{ px}$ | 15.5% | 31/200 |
| $\le 50\text{ px}$ | 30.5% | 61/200 |
| $\le 75\text{ px}$ | 49.5% | 99/200 |
| $\le 100\text{ px}$ | 65.0% | 130/200 |

## Architecture Recall Breakdown (DRAM vs FinFET)

| Architecture Style | Sample Count | Recall $\le 50\text{ px}$ | Recall $\le 100\text{ px}$ |
|---|---|---|---|
| **DRAM** | 97 | 30.9% | 69.1%
| **FinFET** | 103 | 30.1% | 61.2%

## Error Taxonomy Breakdown

- **A. Candidate Generation Failures** (`min_dist > 100 px`): **70 samples (35.0%)**
- **B. Candidate Ranking Failures** (`min_dist <= 50 px`, but prediction error `> 100 px`): **56 samples (28.0%)**
- **C. Fine Search Failures** (`min_dist <= 10 px`, but prediction error `> 10 px`): **2 samples (1.0%)**

### A. Samples with Candidate Generation Failure (No Candidate within 100 px)

| Sample Image | Architecture | Nearest Candidate Dist (px) |
|---|---|---|
| `00002.png` | DRAM | 220.67 px |
| `00004.png` | DRAM | 189.68 px |
| `00005.png` | DRAM | 128.93 px |
| `00006.png` | FinFET | 178.73 px |
| `00007.png` | DRAM | 235.81 px |
| `00013.png` | FinFET | 146.58 px |
| `00014.png` | FinFET | 358.44 px |
| `00016.png` | FinFET | 135.55 px |
| `00034.png` | FinFET | 158.90 px |
| `00036.png` | DRAM | 128.47 px |
| `00039.png` | FinFET | 178.00 px |
| `00040.png` | DRAM | 102.83 px |
| `00043.png` | DRAM | 139.91 px |
| `00045.png` | FinFET | 173.93 px |
| `00047.png` | DRAM | 222.65 px |
| `00051.png` | FinFET | 167.59 px |
| `00052.png` | DRAM | 140.01 px |
| `00056.png` | FinFET | 125.11 px |
| `00058.png` | FinFET | 102.60 px |
| `00060.png` | DRAM | 109.21 px |
| `00067.png` | FinFET | 427.54 px |
| `00068.png` | FinFET | 139.61 px |
| `00070.png` | FinFET | 109.87 px |
| `00071.png` | FinFET | 107.19 px |
| `00075.png` | DRAM | 133.63 px |
| `00078.png` | FinFET | 134.54 px |
| `00082.png` | DRAM | 125.13 px |
| `00085.png` | FinFET | 106.05 px |
| `00086.png` | DRAM | 117.97 px |
| `00087.png` | FinFET | 163.80 px |
| `00088.png` | FinFET | 125.07 px |
| `00089.png` | FinFET | 172.64 px |
| `00090.png` | FinFET | 195.14 px |
| `00091.png` | DRAM | 103.97 px |
| `00097.png` | DRAM | 126.82 px |
| `00100.png` | DRAM | 126.35 px |
| `00102.png` | DRAM | 250.64 px |
| `00103.png` | FinFET | 132.66 px |
| `00112.png` | DRAM | 170.21 px |
| `00114.png` | DRAM | 144.73 px |
| `00115.png` | DRAM | 365.05 px |
| `00117.png` | FinFET | 134.76 px |
| `00119.png` | FinFET | 188.58 px |
| `00122.png` | FinFET | 116.00 px |
| `00125.png` | FinFET | 185.36 px |
| `00128.png` | FinFET | 118.50 px |
| `00131.png` | FinFET | 145.84 px |
| `00133.png` | FinFET | 127.60 px |
| `00137.png` | DRAM | 102.58 px |
| `00139.png` | FinFET | 114.14 px |
| `00147.png` | DRAM | 117.83 px |
| `00151.png` | DRAM | 106.80 px |
| `00154.png` | FinFET | 236.43 px |
| `00156.png` | FinFET | 141.28 px |
| `00157.png` | FinFET | 203.96 px |
| `00158.png` | DRAM | 120.85 px |
| `00159.png` | DRAM | 133.49 px |
| `00161.png` | FinFET | 154.03 px |
| `00162.png` | FinFET | 223.17 px |
| `00166.png` | FinFET | 197.31 px |
| `00168.png` | FinFET | 143.40 px |
| `00169.png` | DRAM | 132.40 px |
| `00171.png` | DRAM | 267.99 px |
| `00173.png` | FinFET | 106.01 px |
| `00174.png` | FinFET | 247.53 px |
| `00175.png` | FinFET | 208.09 px |
| `00177.png` | DRAM | 143.16 px |
| `00184.png` | DRAM | 232.10 px |
| `00185.png` | FinFET | 120.69 px |
| `00198.png` | DRAM | 104.60 px |

### B. Candidate Ranking Failures (GT Candidate Exists, but False Periodic Alias Winner Selected)

| Sample Image | Architecture | Nearest GT Candidate Dist (px) | Final Prediction Error (px) |
|---|---|---|---|
| `00010.png` | DRAM | 27.23 px | 498.15 px |
| `00017.png` | FinFET | 30.71 px | 498.45 px |
| `00018.png` | FinFET | 26.22 px | 705.67 px |
| `00019.png` | DRAM | 30.39 px | 482.89 px |
| `00023.png` | FinFET | 0.82 px | 420.79 px |
| `00028.png` | DRAM | 14.72 px | 879.59 px |
| `00029.png` | DRAM | 40.76 px | 554.12 px |
| `00037.png` | DRAM | 28.19 px | 870.01 px |
| `00041.png` | FinFET | 12.88 px | 230.94 px |
| `00046.png` | DRAM | 31.71 px | 166.19 px |
| `00054.png` | FinFET | 0.70 px | 330.89 px |
| `00055.png` | DRAM | 32.52 px | 300.61 px |
| `00059.png` | FinFET | 3.59 px | 189.19 px |
| `00063.png` | FinFET | 10.44 px | 260.98 px |
| `00069.png` | DRAM | 27.99 px | 557.59 px |
| `00074.png` | FinFET | 2.17 px | 660.17 px |
| `00077.png` | DRAM | 35.40 px | 447.08 px |
| `00079.png` | DRAM | 13.52 px | 389.94 px |
| `00083.png` | FinFET | 42.37 px | 598.65 px |
| `00084.png` | DRAM | 26.89 px | 498.37 px |
*... and 36 more samples.*
