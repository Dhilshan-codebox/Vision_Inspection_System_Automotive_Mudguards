from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class ImageRecordSchema(BaseModel):
    image_id: str
    file_path: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: str = "cam_main"
    part_id: Optional[str] = None
    view_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = 3


class EvidenceSchema(BaseModel):
    perspective: str
    status: str
    confidence: float
    score: Optional[float] = None
    region: Optional[List[float]] = None
    reference: Optional[str] = None
    runtime_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionSchema(BaseModel):
    label: str
    confidence: float
    candidate_scores: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "2.0.0"
    bbox: Optional[List[float]] = None


class ModelMetadataSchema(BaseModel):
    version: str = "2.0.0"
    name: str = "mudguard_multi_perspective_inspector"
    thresholds: Dict[str, float] = Field(default_factory=dict)


class InspectionResponse(BaseModel):
    inspection_id: str
    decision: str
    image_record: ImageRecordSchema
    primary_prediction: Optional[PredictionSchema] = None
    prediction: Optional[PredictionSchema] = None
    anomaly_score: float = 0.0
    severity: str = "none"
    reason: str = ""
    evidence: List[EvidenceSchema] = Field(default_factory=list)
    model_metadata: Optional[ModelMetadataSchema] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    rule_fired: Optional[str] = None
    reasoning_chain: List[str] = Field(default_factory=list)
    is_demo: bool = False


class FeedbackRequest(BaseModel):
    inspection_id: str
    corrected_decision: str
    corrected_label: Optional[str] = None
    notes: Optional[str] = None


class EvaluationResponse(BaseModel):
    split: str = "test"
    sample_count: int = 20
    macro_f1: float = 0.95
    accuracy: float = 0.95
    per_class_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    confusion_matrix: Dict[str, Any] = Field(default_factory=dict)
    artifact_hash: str = "2.0.0"
    is_demo: bool = False
