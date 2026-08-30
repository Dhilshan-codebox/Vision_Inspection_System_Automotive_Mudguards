from typing import Protocol, runtime_checkable, Optional, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np

from src.data.contracts import Prediction, ModelMetadata


@dataclass
class DetectionItem:
    """Represents a localized defect detection instance."""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


@dataclass
class DetectionOutput:
    """Represents the complete inference output from an object detector/classifier."""
    primary_prediction: Prediction
    detections: List[DetectionItem] = field(default_factory=list)
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    features: Optional[np.ndarray] = None
    severity_level: Optional[str] = None
    severity_score: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    quality_scores: Optional[Dict[str, float]] = None
    geometry_residual: Optional[float] = None
    latency_ms: float = 0.0


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Protocol contract for all RGB defect detectors and backbone adapters.
    Ensures interchangeable model backends without coupling to specific ML frameworks.
    """

    def predict(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionOutput:
        """Run inference on the preprocessed image array."""
        ...

    def get_model_metadata(self) -> ModelMetadata:
        """Return version, name, and operational thresholds for this detector."""
        ...
