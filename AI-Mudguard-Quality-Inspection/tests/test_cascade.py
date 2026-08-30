"""
Tests for CascadeStageEngine and Early Exit Execution.
"""

import pytest
import numpy as np

from src.pipeline.cascade import CascadeStageEngine, CascadeStage, CascadeExitResult
from src.data.contracts import ImageQualityAssessment, QualityAction, Prediction, Evidence, EvidenceStatus
from src.detection.detector_protocol import DetectionOutput


def test_cascade_stage0_quality_gate_exit():
    """Verify that Stage 0 triggers early exit when quality gate fails."""
    engine = CascadeStageEngine(enable_early_exits=True)

    failed_quality = ImageQualityAssessment(
        is_acceptable=False,
        recommended_action=QualityAction.REQUEST_RECAPTURE,
        blur_score=15.0,
        is_blurry=True,
        saturation_score=0.01,
        is_overexposed=False,
        is_underexposed=False,
        coverage_score=0.80,
        has_missing_region=False,
        reasons=["Image blur score 15.0 below threshold 100.0"],
        metrics={},
    )

    exit_res = engine.evaluate_stage0_quality(failed_quality)
    assert exit_res is not None
    assert exit_res.exited_early is True
    assert exit_res.exit_stage == CascadeStage.STAGE0_QUALITY_GATE
    assert "Quality gate failure" in exit_res.reason


def test_cascade_stage1_screening_clean_pass_exit():
    """Verify that Stage 1 triggers early exit for clear 'Good' parts."""
    engine = CascadeStageEngine(
        enable_early_exits=True,
        screening_pass_threshold=0.90,
        screening_max_persp_score=0.25,
    )

    det_out = DetectionOutput(
        primary_prediction=Prediction(label="Good", confidence=0.95),
    )

    initial_evidences = [
        Evidence(
            perspective="appearance",
            status=EvidenceStatus.AVAILABLE,
            confidence=0.90,
            score=0.05,
        ),
        Evidence(
            perspective="texture",
            status=EvidenceStatus.AVAILABLE,
            confidence=0.88,
            score=0.08,
        ),
    ]

    exit_res = engine.evaluate_stage1_screening(
        detection_output=det_out,
        initial_evidences=initial_evidences,
        novelty_score=0.05,
    )

    assert exit_res is not None
    assert exit_res.exited_early is True
    assert exit_res.exit_stage == CascadeStage.STAGE1_SCREENING
    assert "High-confidence clean pass" in exit_res.reason


def test_cascade_geometry_stage_activation_conditions():
    """Verify geometry stage is only invoked when depth data is available and condition is met."""
    engine = CascadeStageEngine()

    # No depth data -> should NOT execute geometry stage
    det_out = DetectionOutput(primary_prediction=Prediction(label="Dent", confidence=0.85))
    evidences = [Evidence(perspective="appearance", status=EvidenceStatus.AVAILABLE, confidence=0.9, score=0.4)]
    assert engine.should_execute_geometry_stage(det_out, evidences, has_depth_data=False) is False

    # Depth data + Dent label -> should execute
    assert engine.should_execute_geometry_stage(det_out, evidences, has_depth_data=True) is True

    # Depth data + Good label with low scores -> should NOT execute
    det_good = DetectionOutput(primary_prediction=Prediction(label="Good", confidence=0.95))
    ev_good = [Evidence(perspective="appearance", status=EvidenceStatus.AVAILABLE, confidence=0.9, score=0.05)]
    assert engine.should_execute_geometry_stage(det_good, ev_good, has_depth_data=True) is False
