# Model Training Guide

## Training Neural Rankers

DriftSense-X includes four PyTorch neural candidate rankers:

1. **Siamese CNN** (`training/train_cnn.py`):
   - Contrastive Margin Loss ($m = 0.2$)
   - Adam Optimizer ($\text{lr} = 10^{-3}$)
   - Batch Size: 32

2. **Hybrid Candidate Ranker** (`training/train_hybrid.py`):
   - Triplet Margin Ranking Loss
   - Inputs: 56-D feature vectors
   - Architecture: $56 \to 128 \to 64 \to 32 \to 1$

3. **Coordinate-Aware Ranker** (`training/train_coordinate.py`):
   - Absolute + Normalized Coordinate Encodings
   - Inputs: 44-D feature vectors

4. **Context Transformer** (`training/train_context.py`):
   - Multi-branch 3-field spatial context encoder

---

## Execution Commands

```bash
# Train Siamese CNN model
python main.py train --model cnn

# Train Hybrid Ranker model
python main.py train --model hybrid
```
