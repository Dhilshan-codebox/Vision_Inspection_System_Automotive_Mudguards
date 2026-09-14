#!/usr/bin/env python3
"""
CLI Entry Point: Run Mudguard Backbone Model Training / Fine-Tuning.
Usage:
    python run_training.py --config configs/baseline_rgb.yaml --data-dir data/mudguard_dataset --epochs 5
    python run_training.py --demo --mock-run
"""

import sys
import os
import argparse
import yaml
import json
from pathlib import Path

from src.training.trainer import MudguardTrainer
from src.data.discovery import DatasetDiscovery


def validate_training_dataset(data_dir: str, is_demo: bool):
    """
    Validates dataset layout for training.
    Fails if train split is missing or if val/test splits contain unseen classes without --demo.
    """
    path = Path(data_dir)
    if not path.exists():
        if is_demo:
            return
        print(f"Error: Training dataset directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    discovery = DatasetDiscovery(path)
    records = discovery.discover_all()
    if not records:
        if is_demo:
            return
        print(f"Error: No images found in dataset directory '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    splits = {r.split for r in records}
    if "train" not in splits and not is_demo:
        print(
            f"Error: Training requires a 'train' split directory. Available splits: {splits}. "
            f"Use --demo flag to permit unsplit training data.",
            file=sys.stderr,
        )
        sys.exit(1)

    if ("val" not in splits or "test" not in splits) and not is_demo:
        print(
            f"Error: Standard training requires both 'val' and 'test' splits for proper evaluation. "
            f"Available splits: {splits}. Use --demo flag to permit partial splits.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_classes = {r.label for r in records if r.split == "train"}
    eval_classes = {r.label for r in records if r.split in ("val", "test")}
    unseen_in_train = eval_classes - train_classes

    if unseen_in_train and not is_demo:
        print(
            f"Error: Validation/test splits contain classes absent from training split: {unseen_in_train}. "
            f"Use --demo flag to override.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Mudguard Multi-Head Backbone Training CLI")
    parser.add_argument("--config", type=str, default="configs/baseline_rgb.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--data-dir", type=str, default="data/mudguard_dataset", help="Root directory of training data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Directory to store model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--demo", action="store_true", help="Explicit demo flag allowing mock runs or unsplit data")
    parser.add_argument("--mock-run", action="store_true", help="Execute dry-run simulation pass")

    args = parser.parse_args()

    if args.mock_run and not args.demo:
        print("Error: --mock-run requires the --demo flag.", file=sys.stderr)
        sys.exit(1)

    if not args.mock_run:
        validate_training_dataset(args.data_dir, is_demo=args.demo)

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    trainer = MudguardTrainer(
        config=config,
        data_root=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        is_demo=args.demo,
    )

    print("=" * 60)
    print(f"       STARTING MUDGUARD MULTI-HEAD BACKBONE TRAINING")
    if args.demo:
        print("       [DEMO MODE - MOCK / UNVALIDATED RUN]")
    print("=" * 60)
    print(f"Config File : {args.config}")
    print(f"Data Root   : {args.data_dir}")
    print(f"Epochs      : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Output Dir  : {args.output_dir}")
    print("-" * 60)

    summary = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("=" * 60)
    print(f"Training Status : {summary.get('status')}")
    print(f"Best Macro-F1   : {summary.get('best_macro_f1', 0.0):.4f}")
    if args.demo:
        print("Note: Demo mode metrics are for verification only and not held-out evaluation claims.")
    print("=" * 60)


if __name__ == "__main__":
    main()
