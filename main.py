"""
main.py

DriftSense-X Master CLI Entrypoint for Applied Materials Wafer Inspection AI Research Repository.

Supports subcommands:
- generate : Synthetic dataset generation
- train    : Neural ranker model training
- evaluate : Evaluation suite benchmark execution
- inference: Standalone image pair inference
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="DriftSense-X: Physics-Aware AI Navigation Error Recovery for Semiconductor Wafer Inspection"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Generate Subcommand
    parser_gen = subparsers.add_parser("generate", help="Generate synthetic wafer dataset")
    parser_gen.add_argument("--num_pairs", type=int, default=10000, help="Total number of pairs (80/10/10 split)")
    parser_gen.add_argument("--output_dir", type=str, default="dataset", help="Output dataset directory")

    # 2. Train Subcommand
    parser_train = subparsers.add_parser("train", help="Train neural ranking models")
    parser_train.add_argument("--model", type=str, default="cnn", choices=["cnn", "hybrid", "coordinate", "context"])

    # 3. Evaluate Subcommand
    parser_eval = subparsers.add_parser("evaluate", help="Run evaluation benchmarks")
    parser_eval.add_argument("--pipeline", type=str, default="final", choices=["final", "baseline", "landmark", "frequency", "hierarchical", "hybrid"])

    # 4. Inference Subcommand
    parser_inf = subparsers.add_parser("inference", help="Run single image pair inference")
    parser_inf.add_argument("--reference", type=str, required=True, help="Reference image path")
    parser_inf.add_argument("--search", type=str, required=True, help="Search image path")
    parser_inf.add_argument("--output_vis", type=str, default="outputs/predictions/prediction_result.png")

    args = parser.parse_args()

    if args.command == "generate":
        from dataset_generator.generate_dataset import generate_dataset
        generate_dataset(num_pairs=args.num_pairs, output_dir=args.output_dir)
    elif args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0], "--model", args.model]
        train_main()
    elif args.command == "evaluate":
        from scripts.evaluate import main as eval_main
        sys.argv = [sys.argv[0], "--pipeline", args.pipeline]
        eval_main()
    elif args.command == "inference":
        from scripts.inference import main as inf_main
        sys.argv = [sys.argv[0], "--reference", args.reference, "--search", args.search, "--output_vis", args.output_vis]
        inf_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
