# Synthetic Wafer Dataset Specification

## Overview

The synthetic dataset generation pipeline emulates Scanning Electron Microscopy (SEM) images captured during semiconductor wafer defect inspection and tool stage alignment.

---

## Dataset Splits

| Split | Reference Images | Search Images | Labels CSV | Target Resolution |
|---|---|---|---|---|
| Train | 1,000 | 1,000 | `labels.csv` | 1000x1000 |
| Validation | 200 | 200 | `labels.csv` | 1000x1000 |
| Test | 200 | 200 | `labels.csv` | 1000x1000 |

---

## Semiconductor Architecture Variations

1. **FinFET Layout**:
   - Vertical Silicon Fin Lines (Width: 6-16 px, Pitch: 14-32 px)
   - Horizontal Poly/Metal Gate Bars (Width: 16-32 px, Pitch: 55-110 px)
   - Interconnect Contact Pads

2. **DRAM Array Layout**:
   - Horizontal Word Lines (Width: 10-24 px, Pitch: 40-80 px)
   - Vertical Bit Lines (Width: 8-20 px, Pitch: 30-65 px)
   - Contact / Via Array Dots (Diameter: 10-22 px)

---

## Physics-Based SEM Noise Simulation

- **Secondary Electron Edge Brightening**: Spatial Sobel gradient magnitude enhancement simulating charging along steep feature edges.
- **Defocusing Blur**: Spatial Gaussian blur simulating electron beam defocusing.
- **Shot Noise**: Poisson random process simulating electron landing fluctuations.
- **Sensor Noise**: Additive Gaussian noise $\mathcal{N}(0, \sigma^2)$.
- **Stage Drift**: Rotation ($\pm 3^\circ$), isotropic scale ($0.95-1.05$), translation ($\pm 10$ px).
