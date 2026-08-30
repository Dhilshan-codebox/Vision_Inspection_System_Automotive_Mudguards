"""
Tests for InspectionPipeline Orchestrator and Full 9-Step Reasoning.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.pipeline.inspection_pipeline import InspectionPipeline
from src.data.contracts import Decision, ImageRecord, SeverityLevel


def test_inspection_pipeline_clean_good_mudguard():
    """Verify that a clean nominal image is routed to PASS with clean evidence nodes."""
    pipeline = InspectionPipeline(mock_mode=False)

    h, w = 640, 640
    # Clean homogeneous mudguard surface
    clean_img = np.full((h, w, 3), 120, dtype=np.uint8)

    rec = ImageRecord(
        image_id="test_clean_001",
        file_path="memory://clean.png",
        timestamp=datetime.now(timezone.utc),
        camera_id="view_top",
        part_id="part_001",
    )

    res = pipeline.inspect_image(clean_img, image_record=rec)
    assert res.decision == Decision.PASS
    assert res.prediction is not None
    assert res.prediction.label.lower() in {"good", "normal", "pass"}

    graph = pipeline.get_last_evidence_graph()
    assert graph is not None
    assert graph.decision == "PASS"
    assert "Rule6" in graph.rule_fired or "Early" in graph.rule_fired
    assert len(graph.reasoning_chain) >= 1


def test_inspection_pipeline_scratch_defect_fails():
    """Verify that an image with a high-contrast sharp scratch is routed to FAIL."""
    pipeline = InspectionPipeline(mock_mode=False)

    h, w = 640, 640
    scratch_img = np.full((h, w, 3), 70, dtype=np.uint8)
    # Add scratch lines
    scratch_img[300:304, 150:500] = 255
    scratch_img[320:322, 180:420] = 255

    rec = ImageRecord(
        image_id="test_scratch_001",
        file_path="memory://scratch.png",
        timestamp=datetime.now(timezone.utc),
        camera_id="view_top",
        part_id="part_002",
    )

    res = pipeline.inspect_image(scratch_img, image_record=rec)
    assert res.decision == Decision.FAIL
    assert res.prediction is not None

    graph = pipeline.get_last_evidence_graph()
    assert graph is not None
    assert graph.decision == "FAIL"
    assert graph.severity_level in {"low", "medium", "high", "critical"}


def test_inspection_pipeline_dent_with_depth_map():
    """Verify that a dent candidate confirmed by 3D depth sensor is routed to FAIL."""
    pipeline = InspectionPipeline(mock_mode=False)

    h, w = 640, 640
    dent_img = np.full((h, w, 3), 90, dtype=np.uint8)
    # Optical shading
    dent_img[280:360, 280:360] = 50

    # 3D Depth Map with 3.5mm depression
    depth_map = np.zeros((h, w), dtype=np.float32)
    depth_map[280:360, 280:360] = 3.5

    rec = ImageRecord(
        image_id="test_dent_001",
        file_path="memory://dent.png",
        timestamp=datetime.now(timezone.utc),
        camera_id="view_top",
        part_id="part_003",
    )

    res = pipeline.inspect_image(dent_img, image_record=rec, depth_map=depth_map)
    assert res.decision == Decision.FAIL

    graph = pipeline.get_last_evidence_graph()
    assert graph is not None
    assert graph.decision == "FAIL"
    assert "geometry" in graph.nodes


def test_inspection_pipeline_blurry_image_routes_to_review():
    """Verify that an unacceptably blurry image is flagged by the quality gate and routes to REVIEW."""
    pipeline = InspectionPipeline(mock_mode=False)

    # Completely flat/blurred image with zero high frequency gradients
    blurry_img = np.full((640, 640, 3), 128, dtype=np.uint8)

    rec = ImageRecord(
        image_id="test_blurry_001",
        file_path="memory://blurry.png",
        timestamp=datetime.now(timezone.utc),
        camera_id="view_top",
        part_id="part_004",
    )

    res = pipeline.inspect_image(blurry_img, image_record=rec)
    assert res.decision == Decision.REVIEW

    graph = pipeline.get_last_evidence_graph()
    assert graph is not None
    assert graph.decision == "REVIEW"
    assert "Quality" in graph.rule_fired or "Rule0" in graph.rule_fired
