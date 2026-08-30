import pytest
from datetime import datetime, timezone
import numpy as np

from src.detection import (
    DetectorProtocol,
    DetectionOutput,
    DetectionItem,
    RGBBaselineDetector,
)
from src.fusion import LateFusionEngine
from src.evaluation import EvaluationEngine
from src.data.contracts import (
    ImageRecord,
    Prediction,
    Evidence,
    EvidenceStatus,
    Decision,
    ImageQualityAssessment,
    QualityAction,
)


@pytest.fixture
def sample_image_record() -> ImageRecord:
    return ImageRecord(
        image_id="img_test_001",
        file_path="data/train/Scratch/sample_01.png",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam_top",
        part_id="part_42",
    )


@pytest.fixture
def mock_scratched_image() -> np.ndarray:
    img = np.full((300, 300, 3), 180, dtype=np.uint8)
    # Intense scratch line
    img[140:160, 50:250, :] = [10, 10, 10]
    return img


def test_rgb_detector_protocol_conformance():
    """Verify that RGBBaselineDetector adheres strictly to DetectorProtocol."""
    detector = RGBBaselineDetector()
    assert isinstance(detector, DetectorProtocol)
    meta = detector.get_model_metadata()
    assert meta.name == "RGB_Baseline_Detector"
    assert "confidence_threshold" in meta.thresholds


def test_rgb_detector_prediction_and_localization(mock_scratched_image: np.ndarray):
    """Test detector inference, class probability outputs, and bounding box localization."""
    detector = RGBBaselineDetector(confidence_threshold=0.30)
    output = detector.predict(mock_scratched_image)

    assert isinstance(output, DetectionOutput)
    assert output.primary_prediction.label in detector.classes
    assert 0.0 <= output.primary_prediction.confidence <= 1.0
    assert len(output.class_probabilities) == len(detector.classes)
    assert output.latency_ms >= 0.0

    if output.primary_prediction.label != "Good":
        assert len(output.detections) >= 1
        bbox = output.detections[0].bbox
        assert len(bbox) == 4
        assert bbox[2] > bbox[0]  # x2 > x1
        assert bbox[3] > bbox[1]  # y2 > y1


def test_late_fusion_high_confidence_defect(sample_image_record: ImageRecord):
    """Test that a high-confidence defect produces a FAIL decision."""
    engine = LateFusionEngine(fail_defect_threshold=0.60)
    det_output = DetectionOutput(
        primary_prediction=Prediction(label="Scratch", confidence=0.88),
        class_probabilities={"Scratch": 0.88, "Good": 0.12},
    )
    ev_app = Evidence(perspective="appearance", status=EvidenceStatus.AVAILABLE, confidence=0.90, score=0.85)

    result = engine.fuse(
        image_record=sample_image_record,
        detection_output=det_output,
        evidences=[ev_app],
    )

    assert result.decision == Decision.FAIL
    assert result.prediction.label == "Scratch"


def test_late_fusion_clean_pass(sample_image_record: ImageRecord):
    """Test that a clean normal part produces a PASS decision."""
    engine = LateFusionEngine()
    det_output = DetectionOutput(
        primary_prediction=Prediction(label="Good", confidence=0.95),
        class_probabilities={"Good": 0.95, "Scratch": 0.05},
    )
    ev_app = Evidence(perspective="appearance", status=EvidenceStatus.AVAILABLE, confidence=0.92, score=0.05)
    ev_tex = Evidence(perspective="texture", status=EvidenceStatus.AVAILABLE, confidence=0.88, score=0.04)

    result = engine.fuse(
        image_record=sample_image_record,
        detection_output=det_output,
        evidences=[ev_app, ev_tex],
    )

    assert result.decision == Decision.PASS


def test_late_fusion_quality_failure_routes_to_review(sample_image_record: ImageRecord):
    """Verify that quality gate failure never silently returns PASS."""
    engine = LateFusionEngine()
    det_output = DetectionOutput(
        primary_prediction=Prediction(label="Good", confidence=0.99),
        class_probabilities={"Good": 0.99},
    )
    # Severe blur / quality gate failure
    quality = ImageQualityAssessment(
        is_acceptable=False,
        recommended_action=QualityAction.FLAG_FOR_REVIEW,
        blur_score=15.0,
        is_blurry=True,
        saturation_score=0.0,
        is_overexposed=False,
        is_underexposed=False,
        coverage_score=0.95,
        has_missing_region=False,
        reasons=["Severe blur"],
    )

    result = engine.fuse(
        image_record=sample_image_record,
        detection_output=det_output,
        evidences=[],
        quality_assessment=quality,
    )

    assert result.decision == Decision.REVIEW


def test_late_fusion_contradiction_routes_to_review(sample_image_record: ImageRecord):
    """Verify that contradictory evidence safely routes to REVIEW."""
    engine = LateFusionEngine()
    det_output = DetectionOutput(
        primary_prediction=Prediction(label="Dent", confidence=0.80),
    )
    ev_disputed = Evidence(
        perspective="appearance",
        status=EvidenceStatus.CONTRADICTORY,
        confidence=0.85,
        score=0.85,
    )

    result = engine.fuse(
        image_record=sample_image_record,
        detection_output=det_output,
        evidences=[ev_disputed],
    )

    assert result.decision == Decision.REVIEW


def test_evaluation_metrics_precision_recall_confusion_matrix():
    """Test calculation of per-class precision, recall, F1, and confusion matrix."""
    y_true = ["Scratch", "Scratch", "Dent", "Good", "Good"]
    y_pred = ["Scratch", "Dent", "Dent", "Good", "Good"]

    metrics = EvaluationEngine.compute_classification_metrics(y_true, y_pred)
    assert metrics["per_class"]["Scratch"]["precision"] == 1.0
    assert metrics["per_class"]["Scratch"]["recall"] == 0.5
    assert metrics["per_class"]["Dent"]["precision"] == 0.5
    assert metrics["per_class"]["Good"]["precision"] == 1.0
    assert metrics["macro_avg"]["f1_score"] > 0.60

    cm = EvaluationEngine.compute_confusion_matrix(y_true, y_pred)
    assert cm["matrix"]["Scratch"]["Dent"] == 1
    assert cm["matrix"]["Good"]["Good"] == 2


def test_evaluation_metrics_iou_and_calibration():
    """Test IoU, localization accuracy, and ECE confidence calibration."""
    # IoU
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [0.0, 0.0, 10.0, 10.0]  # Exact match
    box3 = [5.0, 5.0, 15.0, 15.0]  # Partial overlap

    assert EvaluationEngine.compute_iou(box1, box2) == 1.0
    assert 0.10 < EvaluationEngine.compute_iou(box1, box3) < 0.20

    loc = EvaluationEngine.compute_localization_quality([box1, box1], [box2, box3], iou_threshold=0.5)
    assert loc["mean_iou"] > 0.50
    assert loc["localization_accuracy"] == 0.50

    # Calibration
    y_true_binary = [1, 1, 0, 1]
    y_conf = [0.9, 0.85, 0.8, 0.4]
    cal = EvaluationEngine.compute_confidence_calibration(y_true_binary, y_conf, num_bins=5)
    assert 0.0 <= cal["ece"] <= 1.0


def test_evaluation_metrics_latency_percentiles():
    """Test p50 and p95 latency percentiles."""
    latencies = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 50.0, 100.0]
    stats = EvaluationEngine.compute_latency_percentiles(latencies)

    assert stats["sample_count"] == 10
    assert stats["p50"] <= stats["p95"]
    assert stats["min"] == 10.0
    assert stats["max"] == 100.0
    assert 15.0 <= stats["p50"] <= 25.0
    assert stats["p95"] >= 50.0
