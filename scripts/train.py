"""
scripts/train.py

Unified CLI runner for training neural ranking models.
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath("."))
from training.train_cnn import train_siamese_network
from training.train_hybrid import train_hybrid_ranker
from training.train_coordinate import train_coordinate_ranker
from training.train_context import train_context_ranker


def main():
    parser = argparse.ArgumentParser(description="DriftSense-X Model Training CLI")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "hybrid", "coordinate", "context"], help="Model to train")
    args = parser.parse_args()

    if args.model == "cnn":
        train_siamese_network()
    elif args.model == "hybrid":
        train_hybrid_ranker()
    elif args.model == "coordinate":
        train_coordinate_ranker()
    elif args.model == "context":
        train_context_ranker()


if __name__ == "__main__":
    main()
