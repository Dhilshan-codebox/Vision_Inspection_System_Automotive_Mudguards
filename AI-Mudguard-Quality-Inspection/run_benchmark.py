#!/usr/bin/env python3
"""
CLI Entry Point: Run Edge AI Deployment Benchmark.
Usage:
    python run_benchmark.py --runs 50 --export-onnx
"""

import sys
import os
import argparse
import json

from src.evaluation.edge_benchmark import EdgeBenchmark
from src.pipeline.inspection_pipeline import InspectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Mudguard AI Edge Deployment Benchmark CLI")
    parser.add_argument("--runs", type=int, default=30, help="Number of inference benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations (excluded from timing)")
    parser.add_argument("--export-onnx", action="store_true", help="Export backbone model to ONNX format")
    parser.add_argument("--onnx-path", type=str, default="models/onnx/mudguard_backbone.onnx", help="ONNX output path")
    parser.add_argument("--output", type=str, default="outputs/metrics/benchmark_report.json", help="Path to save benchmark JSON report")
    parser.add_argument("--config", type=str, default="configs/baseline_rgb.yaml", help="Path to YAML pipeline config")

    args = parser.parse_args()

    # Load pipeline
    if os.path.exists(args.config):
        pipeline = InspectionPipeline.from_config_file(args.config)
    else:
        pipeline = InspectionPipeline()

    benchmarker = EdgeBenchmark(pipeline=pipeline)

    print("=" * 65)
    print(f"       MUDGUARD AI - EDGE DEPLOYMENT LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Iterations   : {args.runs} (+ {args.warmup} warmup)")
    print(f"ONNX Export  : {'Yes' if args.export_onnx else 'No'}")
    print("-" * 65)

    profile = benchmarker.run_benchmark(
        num_runs=args.runs,
        warmup_runs=args.warmup,
        export_onnx=args.export_onnx,
        onnx_output_path=args.onnx_path,
    )

    print(f"Average Latency : {profile.avg_latency_ms:.2f} ms")
    print(f"p50 Latency     : {profile.p50_latency_ms:.2f} ms")
    print(f"p95 Latency     : {profile.p95_latency_ms:.2f} ms")
    print(f"p99 Latency     : {profile.p99_latency_ms:.2f} ms")
    print(f"Min Latency     : {profile.min_latency_ms:.2f} ms")
    print(f"Max Latency     : {profile.max_latency_ms:.2f} ms")
    print(f"Throughput      : {profile.fps:.1f} FPS")
    if profile.onnx_exported:
        print(f"ONNX Model Size : {profile.model_size_mb:.2f} MB  -> {profile.onnx_path}")
    print("=" * 65)

    # Save JSON report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    print(f"Benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
