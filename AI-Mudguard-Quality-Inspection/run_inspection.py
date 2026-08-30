#!/usr/bin/env python3
"""
CLI Entry Point: Run Mudguard AI Multi-Perspective Quality Inspection.
Usage:
    python run_inspection.py --image path/to/mudguard.jpg [--depth path/to/depth.npy] [--view cam_top]
    python run_inspection.py --mock-good
    python run_inspection.py --mock-scratch
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


def create_synthetic_image(defect_type: str = "good") -> np.ndarray:
    """Create a synthetic test mudguard image for CLI demonstration."""
    img = np.full((400, 400, 3), 160, dtype=np.uint8)
    if defect_type == "scratch":
        # Draw dark sharp scratch line
        img[180:190, 80:320, :] = [20, 20, 20]
    elif defect_type == "dent":
        # Draw smooth shadow gradient
        y, x = np.ogrid[:400, :400]
        mask = (x - 200)**2 + (y - 200)**2 <= 60**2
        img[mask] = (img[mask] * 0.5).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Mudguard AI Quality Intelligence - Multi-Perspective Inspection CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input RGB image file")
    parser.add_argument("--depth", type=str, default=None, help="Optional path to 3D depth map (.npy)")
    parser.add_argument("--view", type=str, default="cam_top", help="Camera / Viewpoint identifier (e.g. cam_top, view_front)")
    parser.add_argument("--part-id", type=str, default="part_001", help="Physical mudguard part identifier")
    parser.add_argument("--config", type=str, default="configs/baseline_rgb.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON inspection report")
    parser.add_argument("--mock-good", action="store_true", help="Run with synthetic clean normal mudguard image")
    parser.add_argument("--mock-scratch", action="store_true", help="Run with synthetic scratched mudguard image")

    args = parser.parse_args()

    # Load configuration
    config_path = args.config if os.path.exists(args.config) else "configs/baseline_rgb.yaml"
    pipeline = InspectionPipeline.from_config_file(config_path) if os.path.exists(config_path) else InspectionPipeline()

    # Load or generate image
    if args.mock_scratch:
        image_np = create_synthetic_image("scratch")
        image_path = "mock://synthetic_scratch.png"
    elif args.mock_good or args.image is None:
        image_np = create_synthetic_image("good")
        image_path = "mock://synthetic_clean.png"
    else:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        pil_img = Image.open(args.image).convert("RGB")
        image_np = np.array(pil_img)
        image_path = args.image

    # Load optional depth map
    depth_np = None
    if args.depth and os.path.exists(args.depth):
        depth_np = np.load(args.depth)

    # Run full inspection
    metadata = {
        "view_id": args.view,
        "part_id": args.part_id,
        "file_path": image_path,
    }
    result = pipeline.inspect_image(image=image_np, depth_map=depth_np, metadata=metadata)

    # Print Formatted Report
    print("=" * 65)
    print(f"       MUDGUARD AI QUALITY INTELLIGENCE - INSPECTION REPORT")
    print("=" * 65)
    print(f"Inspection ID  : {result.inspection_id}")
    print(f"Decision       : {result.decision.value}")
    if result.prediction:
        print(f"Predicted Class: {result.prediction.label} (Confidence: {result.prediction.confidence:.2%})")

    # Extract EvidenceGraph and SeverityEstimate if present
    graph = result.__dict__.get("evidence_graph")
    if graph:
        print(f"Severity Level : {graph.severity_level.upper()}")
        print(f"Fired Rule     : {graph.rule_fired}")
        print("-" * 65)
        print("Reasoning Chain:")
        for step in graph.reasoning_chain:
            print(f"  * {step}")
        print("-" * 65)
        print("Perspective Evidences:")
        for name, node in graph.nodes.items():
            sc = f"{node.score:.3f}" if node.score is not None else "  N/A"
            print(f"  * {name:<15} : score={sc}  conf={node.confidence:.2f}  [{node.status}]")

    print("=" * 65)

    # Save to JSON if requested
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
