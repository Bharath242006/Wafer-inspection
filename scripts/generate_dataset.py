"""
scripts/generate_dataset.py

Unified CLI runner for generating synthetic semiconductor wafer inspection datasets.
"""

import sys
import os

sys.path.append(os.path.abspath("."))
from dataset_generator.generate_dataset import main

if __name__ == "__main__":
    main()
