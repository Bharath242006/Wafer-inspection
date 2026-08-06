"""
evaluation/benchmark.py

Speed, FPS, latency, and memory performance benchmarking suite.
"""

import time
import numpy as np


class PerformanceBenchmark:
    """Benchmark class for measuring runtime latency and throughput."""

    def __init__(self, name: str = "InferenceBenchmark"):
        self.name = name
        self.latencies_ms = []

    def record(self, latency_sec: float):
        self.latencies_ms.append(latency_sec * 1000.0)

    def summary() -> dict:
        if not self.latencies_ms:
            return {}
        arr = np.array(self.latencies_ms)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else 0.0
        }
