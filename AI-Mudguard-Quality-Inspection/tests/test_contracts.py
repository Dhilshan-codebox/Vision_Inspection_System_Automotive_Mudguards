import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.data.contracts import (
    ImageRecord,
    Evidence,
    EvidenceStatus,
    Prediction,
    ModelMetadata,
    InspectionResult,
    Decision,
    SeverityLevel,
    ManifestRecord,
    DatasetManifest,
    AuditFinding,
    DatasetAuditReport,
)

def test_image_record_valid():
    record = ImageRecord(
        image_id="img_001",
        file_path="/data/images/img_001.jpg",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam_top",
        part_id="mudguard_front_left"
    )
    assert record.image_id == "img_001"
    
    # Test serialization
    dump = record.model_dump_json()
    assert "img_001" in dump
    
    # Test deserialization
    record_loaded = ImageRecord.model_validate_json(dump)
    assert record_loaded.image_id == record.image_id

def test_image_record_invalid():
    with pytest.raises(ValidationError):
        # Missing required fields
        ImageRecord(
            image_id="img_001"
        )

def test_evidence_valid():
    evidence = Evidence(
        perspective="appearance",
        status=EvidenceStatus.AVAILABLE,
        confidence=0.95,
        score=0.8,
        region=[10.0, 20.0, 100.0, 200.0]
    )
    assert evidence.perspective == "appearance"

def test_evidence_invalid_confidence():
    with pytest.raises(ValidationError):
        # Confidence must be <= 1.0
        Evidence(
            perspective="appearance",
            status=EvidenceStatus.AVAILABLE,
            confidence=1.5
        )

def test_prediction_valid():
    pred = Prediction(label="scratch", confidence=0.88)
    assert pred.label == "scratch"

def test_model_metadata_valid():
    meta = ModelMetadata(
        version="v1.0.0",
        name="RGB_Detector",
        thresholds={"scratch_conf": 0.5, "dent_conf": 0.6}
    )
    assert meta.name == "RGB_Detector"

def test_inspection_result_valid():
    record = ImageRecord(
        image_id="img_001",
        file_path="/data/images/img_001.jpg",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam_top"
    )
    
    evidence = Evidence(
        perspective="appearance",
        status=EvidenceStatus.AVAILABLE,
        confidence=0.9
    )
    
    meta = ModelMetadata(
        version="1.0",
        name="ensemble",
        thresholds={"confidence": 0.8}
    )
    
    result = InspectionResult(
        inspection_id="ins_001",
        decision=Decision.PASS,
        image_record=record,
        evidence=[evidence],
        model_metadata=meta
    )
    
    dump = result.model_dump_json()
    assert "ins_001" in dump
    
    result_loaded = InspectionResult.model_validate_json(dump)
    assert result_loaded.decision == Decision.PASS
    assert len(result_loaded.evidence) == 1
    assert result_loaded.model_metadata.version == "1.0"

def test_manifest_record_contracts():
    rec = ManifestRecord(
        path="train/Scratch/part01_cam01.jpg",
        label="Scratch",
        split="train",
        part_id="part_01",
        view_id="view_01",
        severity=SeverityLevel.HIGH,
        image_hash="a1b2c3d4e5f6",
        width=1920,
        height=1080,
        channels=3,
        is_valid=True
    )
    dump = rec.model_dump_json()
    loaded = ManifestRecord.model_validate_json(dump)
    assert loaded.path == rec.path
    assert loaded.severity == SeverityLevel.HIGH

    # Test invalid record missing required path or label
    with pytest.raises(ValidationError):
        ManifestRecord(path="train/img.png")

def test_dataset_manifest_and_audit_report_contracts():
    rec = ManifestRecord(
        path="train/Dent/img.jpg",
        label="Dent",
        split="train",
        severity=SeverityLevel.MEDIUM
    )
    manifest = DatasetManifest(
        dataset_name="mudguard_v1",
        version="1.0.0",
        records=[rec],
        total_images=1,
        classes_found=["Dent"]
    )
    dump_manifest = manifest.model_dump_json()
    loaded_manifest = DatasetManifest.model_validate_json(dump_manifest)
    assert loaded_manifest.total_images == 1

    finding = AuditFinding(
        category="corruption",
        severity="ERROR",
        message="Corrupted image file",
        affected_files=["train/Dent/corrupt.jpg"]
    )
    report = DatasetAuditReport(
        dataset_root="/data",
        total_images=1,
        valid_images=0,
        corrupted_images=1,
        findings=[finding],
        passed_quality_gate=False
    )
    dump_report = report.model_dump_json()
    loaded_report = DatasetAuditReport.model_validate_json(dump_report)
    assert not loaded_report.passed_quality_gate
    assert loaded_report.findings[0].category == "corruption"

