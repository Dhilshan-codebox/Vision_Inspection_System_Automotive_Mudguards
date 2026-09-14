#!/usr/bin/env python3
"""
CLI Entry Point: Run Mudguard AI Multi-Perspective Quality Inspection.
Usage:
    python run_inspection.py --image data/mudguard_dataset/test/scratch/part_300_cam_top_scratch_low_01.png
    python run_inspection.py --demo --mock-scratch
"""

import sys
import os
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np

from src.pipeline.inspection_pipeline import InspectionPipeline
from src.data.contracts import ImageRecord
from src.data.discovery import DatasetDiscovery


def create_synthetic_image(defect_type: str = "good") -> np.ndarray:
    """Create a synthetic test mudguard image for CLI demonstration."""
    img = np.full((400, 400, 3), 160, dtype=np.uint8)
    if defect_type == "scratch":
        img[180:190, 80:320, :] = [20, 20, 20]
    elif defect_type == "dent":
        y, x = np.ogrid[:400, :400]
        mask = (x - 200)**2 + (y - 200)**2 <= 60**2
        img[mask] = (img[mask] * 0.5).astype(np.uint8)
    return img


def validate_dataset_protocol(data_root: str, is_demo: bool):
    """
    Validates dataset layout for non-demo execution.
    Fails if missing train, val, or test splits or missing classes in training.
    """
    path = Path(data_root)
    if not path.exists():
        if is_demo:
            return
        print(f"Error: Dataset directory '{data_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    discovery = DatasetDiscovery(path)
    records = discovery.discover_all()
    if not records:
        if is_demo:
            return
        print(f"Error: No images found in dataset '{data_root}'.", file=sys.stderr)
        sys.exit(1)

    splits = {r.split for r in records}
    required_splits = {"train", "val", "test"}
    missing_splits = required_splits - splits

    if missing_splits and not is_demo:
        print(
            f"Error: Normal inspection execution requires all dataset splits {required_splits}. "
            f"Missing splits: {missing_splits}. "
            f"Use --demo flag to permit unsplit or incomplete datasets.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_classes = {r.label for r in records if r.split == "train"}
    eval_classes = {r.label for r in records if r.split in ("val", "test")}
    unseen_in_train = eval_classes - train_classes

    if unseen_in_train and not is_demo:
        print(
            f"Error: Evaluation splits contain classes absent from training split: {unseen_in_train}. "
            f"Use --demo flag to permit unbalanced or unrepresented classes.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Mudguard AI Quality Intelligence - Multi-Perspective Inspection CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input RGB image file")
    parser.add_argument("--dataset-dir", type=str, default="data/mudguard_dataset", help="Path to dataset root")
    parser.add_argument("--depth", type=str, default=None, help="Optional path to 3D depth map (.npy)")
    parser.add_argument("--view", type=str, default="cam_top", help="Camera / Viewpoint identifier")
    parser.add_argument("--part-id", type=str, default="part_001", help="Physical mudguard part identifier")
    parser.add_argument("--config", type=str, default="configs/baseline_rgb.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON inspection report")
    parser.add_argument("--demo", action="store_true", help="Explicit demo flag allowing synthetic/unsplit data")
    parser.add_argument("--mock-good", action="store_true", help="Run with synthetic clean normal mudguard image")
    parser.add_argument("--mock-scratch", action="store_true", help="Run with synthetic scratched mudguard image")

    args = parser.parse_args()

    # Enforce demo flag if synthetic flags are set
    if (args.mock_good or args.mock_scratch) and not args.demo:
        print("Error: Synthetic mock runs require the --demo flag.", file=sys.stderr)
        sys.exit(1)

    if not args.image and not args.mock_good and not args.mock_scratch:
        # Check dataset protocol if checking dataset
        validate_dataset_protocol(args.dataset_dir, is_demo=args.demo)

    config_path = args.config if os.path.exists(args.config) else "configs/baseline_rgb.yaml"
    pipeline = InspectionPipeline.from_config_file(config_path) if os.path.exists(config_path) else InspectionPipeline()

    if args.mock_scratch:
        image_np = create_synthetic_image("scratch")
        image_path = "demo://synthetic_scratch.png"
    elif args.mock_good:
        image_np = create_synthetic_image("good")
        image_path = "demo://synthetic_clean.png"
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        pil_img = Image.open(args.image).convert("RGB")
        image_np = np.array(pil_img)
        image_path = args.image
    else:
        print("Error: Specify --image <path> or use --demo --mock-good / --mock-scratch.", file=sys.stderr)
        sys.exit(1)

    depth_np = None
    if args.depth and os.path.exists(args.depth):
        depth_np = np.load(args.depth)

    metadata = {
        "view_id": args.view,
        "part_id": args.part_id,
        "file_path": image_path,
        "is_demo": args.demo,
    }
    result = pipeline.inspect_image(image=image_np, depth_map=depth_np, metadata=metadata)
    result.is_demo = args.demo

    print("=" * 65)
    print(f"       MUDGUARD AI QUALITY INTELLIGENCE - INSPECTION REPORT")
    if args.demo:
        print("       [DEMO MODE - SYNTHETIC / UNVALIDATED DATASET]")
    print("=" * 65)
    print(f"Inspection ID  : {result.inspection_id}")
    print(f"Decision       : {result.decision.value}")
    if result.prediction or result.primary_prediction:
        pred = result.prediction or result.primary_prediction
        print(f"Predicted Class: {pred.label} (Confidence: {pred.confidence:.2%})")

    graph = result.__dict__.get("evidence_graph")
    if graph:
        print(f"Severity Level : {graph.severity_level.upper()}")
        print(f"Fired Rule     : {graph.rule_fired}")
        print("-" * 65)
        print("Reasoning Chain:")
        for step in graph.reasoning_chain:
            print(f"  * {step}")

    print("=" * 65)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        dump_data = result.model_dump()
        if graph:
            dump_data["evidence_graph"] = graph.to_dict()
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, default=str)
        print(f"Saved full JSON inspection report to: {args.output}")


if __name__ == "__main__":
    main()
