"""
scripts/evaluate.py

Unified CLI runner for running evaluation suites.
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath("."))
from evaluation.evaluate_final import main as eval_final
from evaluation.evaluate_baseline import main as eval_baseline
from evaluation.evaluate_global_landmark import main as eval_landmark
from evaluation.evaluate_frequency import main as eval_freq
from evaluation.evaluate_hierarchical import main as eval_hier
from evaluation.evaluate_hybrid import main as eval_hybrid


def main():
    parser = argparse.ArgumentParser(description="DriftSense-X Evaluation Suite CLI")
    parser.add_argument("--pipeline", type=str, default="final", choices=["final", "baseline", "landmark", "frequency", "hierarchical", "hybrid"], help="Pipeline to evaluate")
    args = parser.parse_args()

    if args.pipeline == "final":
        eval_final()
    elif args.pipeline == "baseline":
        eval_baseline()
    elif args.pipeline == "landmark":
        eval_landmark()
    elif args.pipeline == "frequency":
        eval_freq()
    elif args.pipeline == "hierarchical":
        eval_hier()
    elif args.pipeline == "hybrid":
        eval_hybrid()


if __name__ == "__main__":
    main()
