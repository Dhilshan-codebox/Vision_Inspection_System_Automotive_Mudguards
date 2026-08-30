#!/usr/bin/env python3
"""
CLI Entry Point: Run Mudguard Backbone Model Training / Fine-Tuning.
Usage:
    python run_training.py --config configs/baseline_rgb.yaml --epochs 10 --batch-size 8
"""

import sys
import os
import argparse
import yaml
import json

from src.training.trainer import MudguardTrainer


def main():
    parser = argparse.ArgumentParser(description="Mudguard Multi-Head Backbone Training CLI")
    parser.add_argument("--config", type=str, default="configs/baseline_rgb.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory of training data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Directory to store model checkpoints")
    parser.add_argument("--mock-run", action="store_true", help="Execute dry-run simulation pass")

    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    trainer = MudguardTrainer(
        config=config,
        data_root=args.data_dir,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"       STARTING MUDGUARD MULTI-HEAD BACKBONE TRAINING")
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
    print("=" * 60)


if __name__ == "__main__":
    main()
