from typing import Protocol, runtime_checkable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus


class NoveltyType(str, Enum):
    NOMINAL = "NOMINAL"
    UNKNOWN_DEFECT = "UNKNOWN_DEFECT"
    LIGHTING_SHIFT = "LIGHTING_SHIFT"
    BLANK_OR_INVALID = "BLANK_OR_INVALID"


@dataclass
class AnomalyResult:
    """Represents the output from the anomaly & novelty detection guardrail."""
    anomaly_score: float  # [0.0 - 1.0], higher = more anomalous
    is_anomalous: bool
    novelty_type: NoveltyType
    confidence: float
    anomaly_region: Optional[List[float]] = None  # [x1, y1, x2, y2]
    heatmap_reference: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    runtime_ms: float = 0.0

    def to_evidence(self) -> Evidence:
        """Convert anomaly result into standardized Evidence contract."""
        return Evidence(
            perspective="anomaly",
            status=EvidenceStatus.AVAILABLE,
            confidence=self.confidence,
            score=self.anomaly_score,
            region=self.anomaly_region,
            reference=self.heatmap_reference or "anomaly_patch_map",
            runtime_ms=self.runtime_ms,
            metadata={
                "novelty_type": self.novelty_type.value,
                "is_anomalous": self.is_anomalous,
                **self.metrics,
            },
        )


@runtime_checkable
class AnomalyDetectorProtocol(Protocol):
    """
    Protocol for novelty and unknown anomaly detectors.
    """

    def fit(self, normal_images: List[np.ndarray]) -> None:
        """Learn nominal representation from normal/good images only."""
        ...

    def detect(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnomalyResult:
        """Evaluate an input image against the nominal baseline."""
        ...

    def calibrate_thresholds(
        self,
        val_normal_scores: List[float],
        val_anomaly_scores: Optional[List[float]] = None,
        target_fpr: float = 0.05,
    ) -> float:
        """Calibrate detection threshold strictly using validation data."""
        ...
