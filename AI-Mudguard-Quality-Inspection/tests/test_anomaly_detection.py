import pytest
import numpy as np

from src.anomaly import (
    AnomalyDetectorProtocol,
    AnomalyResult,
    NoveltyType,
    NoveltyAnomalyDetector,
    AnomalyEvaluationProtocol,
)
from src.data.contracts import Evidence, EvidenceStatus


@pytest.fixture
def normal_images() -> list[np.ndarray]:
    """Generates a batch of clean normal mudguard images."""
    images = []
    for _ in range(5):
        img = np.full((128, 128, 3), 140, dtype=np.uint8)
        # Subtle texture noise
        noise = np.random.randint(-5, 5, (128, 128, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
    return images


@pytest.fixture
def unknown_defect_image() -> np.ndarray:
    """Generates an image containing an unseen novel defect (e.g. circular burn mark)."""
    img = np.full((128, 128, 3), 140, dtype=np.uint8)
    # Heavy novel irregular blob in the corner
    img[30:60, 30:60, :] = [20, 20, 20]
    return img


@pytest.fixture
def lighting_shifted_image(normal_images: list[np.ndarray]) -> np.ndarray:
    """Generates an image with severe global lighting drift while retaining texture."""
    base = normal_images[0].astype(np.int16) + 65
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture
def blank_image() -> np.ndarray:
    """Generates a blank / sensor dropout image."""
    return np.zeros((128, 128, 3), dtype=np.uint8)


def test_anomaly_detector_protocol_conformance():
    detector = NoveltyAnomalyDetector()
    assert isinstance(detector, AnomalyDetectorProtocol)


def test_fitting_nominal_representation(normal_images: list[np.ndarray]):
    detector = NoveltyAnomalyDetector(patch_size=16)
    assert not detector.is_fitted
    detector.fit(normal_images)
    assert detector.is_fitted
    assert detector.nominal_mean_features is not None


def test_calibrate_thresholds_validation_data():
    detector = NoveltyAnomalyDetector()
    val_normal_scores = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    calibrated = detector.calibrate_thresholds(val_normal_scores, target_fpr=0.10)
    assert 0.20 <= calibrated <= 0.35
    assert detector.anomaly_threshold == calibrated


def test_detect_unknown_defect_and_localization(
    normal_images: list[np.ndarray],
    unknown_defect_image: np.ndarray
):
    detector = NoveltyAnomalyDetector(patch_size=16, anomaly_threshold=0.35)
    detector.fit(normal_images)

    result = detector.detect(unknown_defect_image)
    assert isinstance(result, AnomalyResult)
    assert result.is_anomalous
    assert result.novelty_type == NoveltyType.UNKNOWN_DEFECT
    assert result.anomaly_region is not None
    assert len(result.anomaly_region) == 4

    # Verify to_evidence conversion
    evidence = result.to_evidence()
    assert isinstance(evidence, Evidence)
    assert evidence.perspective == "anomaly"
    assert evidence.status == EvidenceStatus.AVAILABLE
    assert evidence.score == result.anomaly_score
    assert evidence.metadata["novelty_type"] == "UNKNOWN_DEFECT"


def test_detect_lighting_shift(
    normal_images: list[np.ndarray],
    lighting_shifted_image: np.ndarray
):
    detector = NoveltyAnomalyDetector(patch_size=16)
    detector.fit(normal_images)

    result = detector.detect(lighting_shifted_image)
    assert result.novelty_type == NoveltyType.LIGHTING_SHIFT


def test_detect_blank_invalid_frame(blank_image: np.ndarray):
    detector = NoveltyAnomalyDetector()
    result = detector.detect(blank_image)
    assert result.is_anomalous
    assert result.novelty_type == NoveltyType.BLANK_OR_INVALID


def test_anomaly_evaluation_protocol(
    normal_images: list[np.ndarray],
    unknown_defect_image: np.ndarray,
    lighting_shifted_image: np.ndarray,
    blank_image: np.ndarray
):
    detector = NoveltyAnomalyDetector(patch_size=16, anomaly_threshold=0.35)
    detector.fit(normal_images)

    # Known defect image
    known_img = np.full((128, 128, 3), 140, dtype=np.uint8)
    known_img[60:70, 20:100, :] = [10, 10, 10]

    eval_report = AnomalyEvaluationProtocol.evaluate(
        detector=detector,
        normal_images=normal_images,
        known_defect_images=[known_img],
        unknown_defect_images=[unknown_defect_image],
        lighting_shift_images=[lighting_shifted_image],
        blank_invalid_images=[blank_image],
    )

    assert eval_report["unknown_defects_detection_rate"] == 1.0
    assert eval_report["blank_invalid_detection_rate"] == 1.0
    assert eval_report["auroc"] >= 0.80
    assert eval_report["false_positive_rate_on_normal"] <= 0.20
