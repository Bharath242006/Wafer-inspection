# Evaluation & Benchmark Methodology

## Evaluation Metrics

1. **Mean Absolute Error (MAE)**: Mean Euclidean distance between predicted target center $(\hat{x}, \hat{y})$ and ground-truth center $(x^*, y^*)$ in search image space.
2. **Median Error**: Median center error across test dataset.
3. **P95 Error**: 95th percentile center error.
4. **Accuracy Thresholds**: Percentage of test samples with error $\le 1$ px, $\le 2$ px, $\le 5$ px, $\le 10$ px, $\le 25$ px, $\le 50$ px, $\le 100$ px.
5. **Success Rate @ IoU 0.5**: Percentage of predictions with bounding box Intersection over Union $\ge 0.50$.
6. **Throughput (FPS)**: Inferences per second on 1000x1000 wafer images.

---

## Running Evaluation Suites

```bash
# Evaluate final integrated pipeline
python main.py evaluate --pipeline final

# Evaluate baseline normalized cross-correlation
python main.py evaluate --pipeline baseline
```
