import pytest
import numpy as np

from src.perspectives import (
    PerspectiveInterface,
    AppearancePerspective,
    TexturePerspective,
    GeometryPerspective,
    ContextPerspective,
    UncertaintyPerspective,
    ContradictionDetector,
)
from src.data.contracts import Evidence, EvidenceStatus


@pytest.fixture
def sample_image() -> np.ndarray:
    """Synthetic test image with sharp defect in center."""
    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    # Scratch line
    img[90:110, 50:150, :] = [20, 20, 20]
    return img


def test_appearance_perspective_execution(sample_image: np.ndarray):
    """Test appearance perspective feature extraction, region detection, and timing."""
    perspective = AppearancePerspective()
    evidence = perspective.extract(sample_image)

    assert evidence.perspective == "appearance"
    assert evidence.status == EvidenceStatus.AVAILABLE
    assert 0.0 <= evidence.score <= 1.0
    assert 0.0 <= evidence.confidence <= 1.0
    assert evidence.runtime_ms >= 0.0
    assert evidence.region is not None
    assert len(evidence.region) == 4


def test_texture_perspective_execution(sample_image: np.ndarray):
    """Test texture perspective surface roughness evaluation."""
    perspective = TexturePerspective()
    evidence = perspective.extract(sample_image)

    assert evidence.perspective == "texture"
    assert evidence.status == EvidenceStatus.AVAILABLE
    assert 0.0 <= evidence.score <= 1.0
    assert 0.0 <= evidence.confidence <= 1.0
    assert evidence.runtime_ms >= 0.0
    assert "mean_roughness" in evidence.metadata


def test_geometry_perspective_missing_depth_data(sample_image: np.ndarray):
    """Verify that GeometryPerspective explicitly returns UNAVAILABLE when depth map is absent."""
    perspective = GeometryPerspective()
    # Call without depth_map metadata
    evidence = perspective.extract(sample_image, metadata={})

    assert evidence.perspective == "geometry"
    assert evidence.status == EvidenceStatus.UNAVAILABLE
    assert evidence.score == 0.0
    assert evidence.confidence == 0.0
    assert evidence.runtime_ms >= 0.0
    assert "No 3D/depth sensor data provided" in evidence.metadata.get("reason", "")


def test_geometry_perspective_with_depth_data(sample_image: np.ndarray):
    """Verify that GeometryPerspective analyzes dent depression when depth is provided."""
    perspective = GeometryPerspective(dent_depth_threshold_mm=3.0)

    # Create synthetic depth map with a 4mm dent in center
    depth_map = np.full((200, 200), 50.0, dtype=np.float32)
    depth_map[80:120, 80:120] = 54.0  # 4mm depression

    evidence = perspective.extract(sample_image, metadata={"depth_map": depth_map})

    assert evidence.perspective == "geometry"
    assert evidence.status == EvidenceStatus.AVAILABLE
    assert evidence.score > 0.80  # 4mm > 3mm threshold
    assert evidence.confidence >= 0.90
    assert evidence.region is not None


def test_context_perspective(sample_image: np.ndarray):
    """Test context evaluation for known vs unknown viewpoints."""
    perspective = ContextPerspective()

    # Known viewpoint
    ev_known = perspective.extract(sample_image, metadata={"part_id": "part_123", "view_id": "cam_top"})
    assert ev_known.status == EvidenceStatus.AVAILABLE
    assert ev_known.score < 0.20  # No penalty

    # Unknown viewpoint
    ev_unknown = perspective.extract(sample_image, metadata={"part_id": "part_123", "view_id": "random_angle"})
    assert ev_unknown.score >= 0.25  # Viewpoint penalty applied


def test_uncertainty_perspective(sample_image: np.ndarray):
    """Test uncertainty estimation based on cross-perspective variances."""
    ev_app = Evidence(perspective="appearance", status=EvidenceStatus.AVAILABLE, confidence=0.9, score=0.9)
    ev_geo = Evidence(perspective="geometry", status=EvidenceStatus.AVAILABLE, confidence=0.8, score=0.1)

    perspective = UncertaintyPerspective()
    ev_unc = perspective.extract(sample_image, metadata={"evidences": [ev_app, ev_geo]})

    assert ev_unc.perspective == "uncertainty"
    assert ev_unc.status == EvidenceStatus.AVAILABLE
    assert ev_unc.score > 0.20  # High uncertainty due to strong disagreement
    assert ev_unc.runtime_ms >= 0.0


def test_contradiction_detection_optical_shadow():
    """Test contradiction resolution when optical shadow produces false dent in 2D."""
    ev_app = Evidence(
        perspective="appearance",
        status=EvidenceStatus.AVAILABLE,
        confidence=0.85,
        score=0.85,
        region=[50.0, 50.0, 100.0, 100.0]
    )
    ev_geo = Evidence(
        perspective="geometry",
        status=EvidenceStatus.AVAILABLE,
        confidence=0.90,
        score=0.05,
        region=None
    )

    detector = ContradictionDetector()
    findings = detector.analyze([ev_app, ev_geo])

    assert len(findings) == 1
    assert findings[0].dispute_type == "OPTICAL_SHADOW_VS_SURFACE_NOMINAL"
    assert findings[0].severity == "HIGH"

    # Test status marking
    updated_evidences = detector.resolve_and_mark_contradictions([ev_app, ev_geo])
    statuses = {e.perspective: e.status for e in updated_evidences}
    assert statuses["appearance"] == EvidenceStatus.CONTRADICTORY
    assert statuses["geometry"] == EvidenceStatus.CONTRADICTORY
