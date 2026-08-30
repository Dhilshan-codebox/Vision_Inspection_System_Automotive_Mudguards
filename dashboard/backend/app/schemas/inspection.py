from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ImageRecordSchema(BaseModel):
    image_id: str
    file_path: str
    timestamp: datetime
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
    bbox: Optional[List[float]] = None


class ModelMetadataSchema(BaseModel):
    version: str
    name: str
    thresholds: Dict[str, float] = Field(default_factory=dict)


class ImageQualityAssessmentSchema(BaseModel):
    is_acceptable: bool
    quality_score: float = 1.0
    action: str = "ACCEPT"
    blur_score: float = 0.0
    is_blurry: bool = False
    reasons: List[str] = Field(default_factory=list)


class InspectionResponse(BaseModel):
    inspection_id: str
    decision: str
    image_record: ImageRecordSchema
    primary_prediction: Optional[PredictionSchema] = None
    evidence: List[EvidenceSchema] = Field(default_factory=list)
    severity: str = "none"
    model_metadata: Optional[ModelMetadataSchema] = None
    timestamp: datetime
    latency_ms: float = 0.0
    rule_fired: Optional[str] = None
    reasoning_chain: List[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    inspection_id: str
    corrected_decision: str
    corrected_label: Optional[str] = None
    notes: Optional[str] = None
