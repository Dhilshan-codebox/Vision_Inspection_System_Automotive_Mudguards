"""
Core Data Contracts and Pydantic Schemas for AI Mudguard Quality Inspection.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class Decision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    RECAPTURE = "RECAPTURE"
    REQUEST_RECAPTURE = "REQUEST_RECAPTURE"


class SeverityLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"
    CONTRADICTORY = "CONTRADICTORY"
    INCONCLUSIVE = "INCONCLUSIVE"


class QualityAction(str, Enum):
    ACCEPT = "ACCEPT"
    PROCEED_TO_INFERENCE = "ACCEPT"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    REQUEST_RECAPTURE = "REQUEST_RECAPTURE"
    REJECT = "REJECT"


class ImageRecord(BaseModel):
    image_id: str
    file_path: str
    label: Optional[str] = None
    split: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: str = "cam_main"
    part_id: Optional[str] = None
    view_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = 3
    annotation_reference: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    perspective: str
    status: EvidenceStatus = EvidenceStatus.AVAILABLE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    region: Optional[List[float]] = None  # [x1, y1, x2, y2]
    reference: Optional[str] = None
    runtime_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None

    @field_validator("score", "confidence")
    @classmethod
    def validate_probabilities(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0")
        return v


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    candidate_scores: Dict[str, float] = Field(default_factory=dict)
    distance_or_probability_evidence: Optional[Dict[str, float]] = None
    model_version: str = "2.0.0"
    bbox: Optional[List[float]] = None
    severity: Optional[SeverityLevel] = None


class ModelMetadata(BaseModel):
    version: str = "2.0.0"
    name: str = "mudguard_multi_perspective_inspector"
    thresholds: Dict[str, float] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    artifact_hash: Optional[str] = None


class ImageQualityAssessment(BaseModel):
    is_acceptable: bool
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    action: QualityAction = QualityAction.ACCEPT
    recommended_action: Optional[QualityAction] = None
    blur_score: float = 0.0
    is_blurry: bool = False
    exposure_score: float = 0.0
    saturation_score: float = 0.0
    is_overexposed: bool = False
    is_underexposed: bool = False
    coverage_score: float = 1.0
    has_missing_region: bool = False
    reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class InspectionResult(BaseModel):
    inspection_id: str
    decision: Decision
    image_record: ImageRecord
    evidence: List[Evidence] = Field(default_factory=list)
    primary_prediction: Optional[Prediction] = None
    prediction: Optional[Prediction] = None
    anomaly_score: float = 0.0
    severity: SeverityLevel = SeverityLevel.NONE
    reason: str = ""
    model_metadata: Optional[ModelMetadata] = None
    is_demo: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ManifestRecord(BaseModel):
    path: str
    label: str
    split: str = "train"
    part_id: Optional[str] = None
    view_id: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.NONE
    image_hash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = 3
    is_valid: bool = True


class DatasetManifest(BaseModel):
    dataset_name: str = "mudguard_dataset"
    version: str = "1.0.0"
    records: List[ManifestRecord] = Field(default_factory=list)
    total_images: int = 0
    classes_found: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditFinding(BaseModel):
    category: str
    severity: str = "WARNING"
    message: str
    affected_files: List[str] = Field(default_factory=list)


class DatasetAuditReport(BaseModel):
    dataset_root: str
    total_images: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    duplicate_images: int = 0
    split_leakage_count: int = 0
    class_counts: Dict[str, int] = Field(default_factory=dict)
    split_counts: Dict[str, int] = Field(default_factory=dict)
    dimension_stats: Dict[str, Any] = Field(default_factory=dict)
    findings: List[AuditFinding] = Field(default_factory=list)
    passed_quality_gate: bool = True


class EvaluationReport(BaseModel):
    split: str
    sample_count: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    per_class_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    calibration: Dict[str, Any] = Field(default_factory=dict)
    latency: Dict[str, float] = Field(default_factory=dict)
    artifact_hash: str = "unknown"
    is_demo: bool = False
    evaluation_claim: str = "HELD_OUT_EVALUATION"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
