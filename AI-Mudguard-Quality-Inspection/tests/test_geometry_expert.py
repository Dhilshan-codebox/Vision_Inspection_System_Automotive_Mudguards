import pytest
import numpy as np
from src.geometry.surface_expert import DentGeometryExpert
from src.geometry.calibration import DepthCalibrationMetadata, assess_depth_quality
from src.data.contracts import EvidenceStatus


@pytest.fixture
def simple_depth_map() -> np.ndarray:
    """Create a uniform depth map (no dent) with valid values."""
    return np.full((200, 200), 50.0, dtype=np.float32)


@pytest.fixture
def dent_depth_map() -> np.ndarray:
    """Create a depth map with a 4mm dent in the center (threshold 3.0mm)."""
    depth = np.full((200, 200), 50.0, dtype=np.float32)
    depth[80:120, 80:120] = 54.0  # 4mm depression
    return depth


def test_assess_depth_quality_valid(simple_depth_map):
    quality = assess_depth_quality(simple_depth_map)
    assert quality.is_usable
    assert quality.surface_coverage > 0.9
    assert quality.noise_level_mm < 5.0
    assert quality.calibration_valid


def test_assess_depth_quality_missing():
    quality = assess_depth_quality(None)
    assert not quality.is_usable
    assert "missing" in " ".join(quality.reasons).lower()


def test_dent_geometry_expert_detects_dent(dent_depth_map):
    expert = DentGeometryExpert(dent_depth_threshold_mm=3.0)
    _, metrics, _ = expert.analyze_surface(dent_depth_map)
    assert metrics is not None
    assert metrics.has_dent
    # Max dent depth should be close to 4mm (allow small tolerance)
    assert abs(metrics.max_dent_depth_mm - 4.0) < 0.2
    assert metrics.dent_area_mm2 > 0
    assert metrics.dent_volume_mm3 > 0


def test_dent_geometry_expert_no_dent(simple_depth_map):
    expert = DentGeometryExpert(dent_depth_threshold_mm=3.0)
    _, metrics, _ = expert.analyze_surface(simple_depth_map)
    assert metrics is not None
    assert not metrics.has_dent
    assert metrics.max_dent_depth_mm < 0.5


def test_optical_shadow_dispute_detection(dent_depth_map):
    # Simulate a high appearance confidence (shadow) but shallow depth -> dispute
    expert = DentGeometryExpert(dent_depth_threshold_mm=5.0)
    # Use a shallow dent (2mm) to trigger dispute logic
    shallow = np.full((200, 200), 50.0, dtype=np.float32)
    shallow[80:120, 80:120] = 52.0  # 2mm depression
    evidence = expert.evaluate_to_evidence(
        depth_map=shallow,
        calibration=None,
        rgb_appearance_score=0.85,
    )
    assert evidence.perspective == "geometry"
    assert evidence.status == EvidenceStatus.AVAILABLE
    # Dispute flag should be True because appearance score is high and depth score low
    assert evidence.metadata.get("optical_shadow_dispute") is True


def test_expert_returns_unavailable_when_no_depth():
    expert = DentGeometryExpert()
    evidence = expert.evaluate_to_evidence(depth_map=None)
    assert evidence.status == EvidenceStatus.UNAVAILABLE
    assert evidence.metadata.get("usable") is False
