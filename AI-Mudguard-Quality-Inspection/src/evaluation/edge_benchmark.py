import os
import time
import json
import logging
from typing import TYPE_CHECKING, Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

from src.evaluation.metrics import EvaluationEngine

if TYPE_CHECKING:
    from src.pipeline.inspection_pipeline import InspectionPipeline
    from src.detection.backbone import SharedBackbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkProfile:
    """Consolidated Edge AI deployment latency, throughput, and resource profile."""
    num_runs: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float
    stage_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    model_size_mb: float = 0.0
    onnx_exported: bool = False
    onnx_path: Optional[str] = None
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeBenchmark:
    """
    Profiles end-to-end edge inference throughput (FPS), stage-by-stage latency percentiles,
    memory footprint, and handles ONNX FP32 / INT8 model export.
    """

    def __init__(
        self,
        pipeline: Optional["InspectionPipeline"] = None,
        backbone: Optional["SharedBackbone"] = None,
    ):
        # Lazy import to avoid circular dependency: evaluation → pipeline → evaluation
        from src.pipeline.inspection_pipeline import InspectionPipeline as _Pipeline
        from src.detection.backbone import SharedBackbone as _Backbone

        self.pipeline = pipeline or _Pipeline(mock_mode=False)
        self.backbone = (
            backbone
            or (self.pipeline.detector.backbone if hasattr(self.pipeline.detector, "backbone") else _Backbone())
        )

    def run_benchmark(
        self,
        num_runs: int = 50,
        warmup_runs: int = 5,
        image_shape: tuple = (640, 640, 3),
        export_onnx: bool = True,
        onnx_output_path: str = "models/onnx/mudguard_backbone.onnx",
    ) -> BenchmarkProfile:
        """
        Execute profiling runs and compute comprehensive latency percentiles.
        """
        logger.info(f"Starting Edge AI Benchmark: {num_runs} iterations (shape={image_shape})...")

        # Generate synthetic input images
        np.random.seed(42)
        test_images = [
            np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
            for _ in range(min(10, num_runs))
        ]

        # Warmup
        for i in range(warmup_runs):
            img = test_images[i % len(test_images)]
            _ = self.pipeline.inspect_image(img)

        # Profile iterations
        latencies: List[float] = []

        total_start = time.perf_counter()

        for i in range(num_runs):
            img = test_images[i % len(test_images)]

            t0 = time.perf_counter()
            res = self.pipeline.inspect_image(img)
            t_total = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_total)

        total_elapsed = time.perf_counter() - total_start
        fps = float(num_runs / max(total_elapsed, 1e-6))

        # Compute percentiles via EvaluationEngine
        percentiles = EvaluationEngine.compute_latency_percentiles(latencies)

        # ONNX Export
        onnx_success = False
        if export_onnx and self.backbone is not None:
            try:
                onnx_success = self.backbone.export_onnx(onnx_output_path)
                if onnx_success:
                    logger.info(f"Successfully exported ONNX model to {onnx_output_path}")
            except Exception as e:
                logger.warning(f"ONNX export skipped: {e}")

        # Model size calculation
        model_size_mb = 0.0
        if onnx_success and os.path.exists(onnx_output_path):
            model_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)

        profile = BenchmarkProfile(
            num_runs=num_runs,
            avg_latency_ms=percentiles["mean_ms"],
            p50_latency_ms=percentiles["p50_ms"],
            p95_latency_ms=percentiles["p95_ms"],
            p99_latency_ms=percentiles["p99_ms"],
            min_latency_ms=percentiles["min_ms"],
            max_latency_ms=percentiles["max_ms"],
            fps=fps,
            model_size_mb=model_size_mb,
            onnx_exported=onnx_success,
            onnx_path=onnx_output_path if onnx_success else None,
        )

        return profile
