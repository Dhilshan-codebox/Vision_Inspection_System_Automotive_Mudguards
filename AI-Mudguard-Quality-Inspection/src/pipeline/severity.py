"""
Defect Severity Estimator for Automotive Mudguards.

Maps fused multi-perspective evidence, spatial defect footprints, and 3D surface depth
measurements into calibrated industrial severity levels (NONE, LOW, MEDIUM, HIGH, CRITICAL).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from src.data.contracts import SeverityLevel, Evidence, EvidenceStatus
from src.detection.detector_protocol import DetectionOutput


@dataclass
class SeverityAssessment:
    """Detailed severity classification and physical dimensional estimates."""
    level: SeverityLevel
    score: float  # [0.0 - 1.0]
    confidence: float = 0.90
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    estimated_area_mm2: Optional[float] = None
    estimated_depth_mm: Optional[float] = None
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


# Alias for compatibility
SeverityEstimate = SeverityAssessment


class SeverityEstimator:
    """
    Estimates defect severity by synthesizing primary backbone output,
    evidence scores from appearance/texture/geometry, and physical dimensions.
    """

    def __init__(
        self,
        default_scale_mm_per_pixel: float = 0.15,
        pixel_to_mm: Optional[float] = None,
        dent_depth_threshold_mm: float = 2.0,
    ):
        self.default_scale_mm_per_pixel = pixel_to_mm or default_scale_mm_per_pixel
        self.dent_depth_threshold_mm = dent_depth_threshold_mm

    def estimate(
        self,
        detection_output: DetectionOutput,
        evidences: List[Evidence],
        depth_map: Optional[np.ndarray] = None,
        scale_mm_per_pixel: Optional[float] = None,
    ) -> SeverityAssessment:
        """
        Compute severity assessment and physical defect dimensions.
        """
        scale = scale_mm_per_pixel or self.default_scale_mm_per_pixel
        pred = detection_output.primary_prediction
        pred_label = pred.label.lower()
        pred_conf = pred.confidence

        if pred_label in {"good", "normal", "pass"}:
            return SeverityAssessment(
                level=SeverityLevel.NONE,
                score=0.0,
                confidence=pred_conf,
                confidence_interval=(0.0, 0.05),
                estimated_area_mm2=0.0,
                estimated_depth_mm=0.0,
                contributing_factors={"normal_confidence": pred_conf},
                reasons=["Classified as nominal normal part"],
            )

        ev_map = {e.perspective: e for e in evidences}
        app_score = ev_map.get("appearance").score if ev_map.get("appearance") else 0.0
        tex_score = ev_map.get("texture").score if ev_map.get("texture") else 0.0
        geom_ev = ev_map.get("geometry")

        # Physical dimension estimation from bounding box
        area_mm2: Optional[float] = None
        if detection_output.detections:
            det = detection_output.detections[0]
            w_px = max(0.0, det.bbox[2] - det.bbox[0])
            h_px = max(0.0, det.bbox[3] - det.bbox[1])
            area_mm2 = float(w_px * h_px * (scale ** 2))

        # Depth estimation
        depth_mm: Optional[float] = None
        if geom_ev and geom_ev.status == EvidenceStatus.AVAILABLE and geom_ev.metadata:
            depth_mm = geom_ev.metadata.get("max_dent_depth_mm") or geom_ev.metadata.get("max_depth_mm") or geom_ev.metadata.get("dent_depth_mm")
        elif detection_output.geometry_residual is not None and detection_output.geometry_residual > 0:
            depth_mm = float(detection_output.geometry_residual)

        # Composite severity score
        factors: Dict[str, float] = {
            "prediction_confidence": pred_conf,
            "appearance_score": app_score or 0.0,
            "texture_score": tex_score or 0.0,
        }
        if depth_mm is not None:
            factors["depth_mm"] = depth_mm

        raw_sev: float = 0.0
        if pred_label == "scratch":
            raw_sev = float(0.40 * pred_conf + 0.35 * (app_score or 0.0) + 0.25 * (tex_score or 0.0))
        elif pred_label == "dent":
            depth_factor = min(depth_mm / 4.0, 1.0) if depth_mm is not None else 0.5
            raw_sev = float(0.30 * pred_conf + 0.50 * depth_factor + 0.20 * (app_score or 0.0))
        elif pred_label == "paint misalignment":
            raw_sev = float(0.50 * pred_conf + 0.50 * (app_score or 0.0))
        else:
            raw_sev = float(0.50 * pred_conf + 0.30 * (app_score or 0.0) + 0.20 * (tex_score or 0.0))

        # Map to severity level
        if raw_sev < 0.25:
            level = SeverityLevel.LOW
        elif raw_sev < 0.50:
            level = SeverityLevel.MEDIUM
        elif raw_sev < 0.75:
            level = SeverityLevel.HIGH
        else:
            level = SeverityLevel.CRITICAL

        conf = float(np.clip(pred_conf * 0.9 + 0.05, 0.5, 0.99))
        sev_score = float(np.clip(raw_sev, 0.0, 1.0))
        ci_low = float(np.clip(sev_score - 0.10, 0.0, 1.0))
        ci_high = float(np.clip(sev_score + 0.10, 0.0, 1.0))

        return SeverityAssessment(
            level=level,
            score=sev_score,
            confidence=conf,
            confidence_interval=(ci_low, ci_high),
            estimated_area_mm2=area_mm2,
            estimated_depth_mm=depth_mm,
            contributing_factors=factors,
            reasons=[f"Calculated severity level {level.value} with composite score {sev_score:.2f}"],
        )
