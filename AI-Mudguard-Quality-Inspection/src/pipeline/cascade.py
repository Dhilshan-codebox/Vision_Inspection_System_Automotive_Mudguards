"""
Coarse-to-Fine Cascade Engine with Early Exits.

Implements latency-optimized progressive inspection stages:
  - Stage 0: Quality Gate (Blur, Overexposure, Missing Region) -> Early Exit on Unusable Frame
  - Stage 1: Fast Full-Frame Screening (Backbone / Primary RGB Detector) -> Early Exit on Clean Nominal Part
  - Stage 2: High-Resolution Tile & Micro-Texture Analysis (Sub-pixel defects, LBP)
  - Stage 3: 3D Surface Geometry Analysis (Targeted only on dent candidates or when depth available)
  - Stage 4: Multi-Perspective Evidence Reasoning & Decision Justification
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import time
import numpy as np

from src.data.contracts import (
    Decision,
    ImageQualityAssessment,
    QualityAction,
    Evidence,
    EvidenceStatus,
    Prediction,
    InspectionResult,
)
from src.detection.detector_protocol import DetectionOutput


class CascadeStage(str, Enum):
    STAGE0_QUALITY_GATE = "STAGE0_QUALITY_GATE"
    STAGE1_SCREENING = "STAGE1_SCREENING"
    STAGE2_TEXTURE_TILE = "STAGE2_TEXTURE_TILE"
    STAGE3_GEOMETRY = "STAGE3_GEOMETRY"
    STAGE4_FUSION = "STAGE4_FUSION"
    # Aliases
    STAGE_0_QUALITY_GATE = "STAGE0_QUALITY_GATE"
    STAGE_1_PART_LOCALIZATION = "STAGE1_SCREENING"
    STAGE_2_SCREENING = "STAGE1_SCREENING"
    STAGE_3_HIGH_RES_TILES = "STAGE2_TEXTURE_TILE"
    STAGE_4_GEOMETRY_EXPERT = "STAGE3_GEOMETRY"
    STAGE_5_DECISION_ENGINE = "STAGE4_FUSION"


@dataclass
class CascadeExitResult:
    """Records the progress and early exit status of the cascade pipeline."""
    exited_early: bool
    exit_stage: CascadeStage
    decision: Optional[Decision] = None
    reason: str = ""
    stages_executed: List[CascadeStage] = field(default_factory=list)
    cumulative_latency_ms: float = 0.0


# Alias for backwards compatibility
CascadeResult = CascadeExitResult


class CascadeStageEngine:
    """
    Manages stage transitions and early termination criteria to minimize inspection latency
    while guaranteeing that defect checks are never bypassed on suspicious or low-quality parts.
    """

    def __init__(
        self,
        enable_early_exits: bool = True,
        screening_pass_threshold: float = 0.90,
        screening_max_persp_score: float = 0.25,
        require_depth_for_dents: bool = True,
        stage2_screening_pass_threshold: Optional[float] = None,
        stage2_screening_max_persp_score: Optional[float] = None,
        stage4_require_depth_for_dents: Optional[bool] = None,
    ):
        self.enable_early_exits = enable_early_exits
        self.screening_pass_threshold = stage2_screening_pass_threshold or screening_pass_threshold
        self.screening_max_persp_score = stage2_screening_max_persp_score or screening_max_persp_score
        self.require_depth_for_dents = require_depth_for_dents if stage4_require_depth_for_dents is None else stage4_require_depth_for_dents

    def evaluate_stage0_quality(
        self,
        quality_assessment: ImageQualityAssessment,
    ) -> Optional[CascadeExitResult]:
        """
        Stage 0: Image Quality Gate.
        Early exit to REVIEW / RECAPTURE if the frame is blurry, saturated, or part is truncated.
        """
        if not self.enable_early_exits:
            return None

        if not quality_assessment.is_acceptable:
            reasons_str = ", ".join(quality_assessment.reasons) if quality_assessment.reasons else "Quality below threshold"
            return CascadeExitResult(
                exited_early=True,
                exit_stage=CascadeStage.STAGE0_QUALITY_GATE,
                decision=Decision.REVIEW,
                reason=f"Early exit at Stage 0: Quality gate failure ({reasons_str})",
                stages_executed=[CascadeStage.STAGE0_QUALITY_GATE],
            )
        return None

    def evaluate_stage1_screening(
        self,
        detection_output: DetectionOutput,
        initial_evidences: List[Evidence],
        novelty_score: Optional[float] = None,
    ) -> Optional[CascadeExitResult]:
        """
        Stage 1: Fast Full-Frame Screening.
        Early exit to PASS if part is conclusively 'Good' across backbone and initial perspectives.
        """
        if not self.enable_early_exits:
            return None

        pred = detection_output.primary_prediction
        pred_label = pred.label.lower()
        pred_conf = pred.confidence

        # Check if classification is high-confidence Good
        is_clean_label = pred_label in {"good", "normal", "pass"} and pred_conf >= self.screening_pass_threshold

        # Check that no initial perspective flagged any defect
        max_persp_score = max(
            (e.score for e in initial_evidences if e.score is not None and e.status == EvidenceStatus.AVAILABLE and e.perspective != "uncertainty"),
            default=0.0
        )
        is_clean_perspectives = max_persp_score <= self.screening_max_persp_score

        # Check that novelty is low
        is_clean_novelty = (novelty_score is None) or (novelty_score < 0.35)

        if is_clean_label and is_clean_perspectives and is_clean_novelty:
            return CascadeExitResult(
                exited_early=True,
                exit_stage=CascadeStage.STAGE1_SCREENING,
                decision=Decision.PASS,
                reason=f"Early exit at Stage 1: High-confidence clean pass (Good={pred_conf:.2f}, max_persp={max_persp_score:.2f})",
                stages_executed=[CascadeStage.STAGE0_QUALITY_GATE, CascadeStage.STAGE1_SCREENING],
            )
        return None

    def should_exit_stage_0(self, quality: ImageQualityAssessment) -> Tuple[bool, str]:
        """Compatibility wrapper."""
        res = self.evaluate_stage0_quality(quality)
        if res is not None:
            return True, res.reason
        return False, "Quality acceptable"

    def should_exit_stage_2(
        self,
        detection_output: DetectionOutput,
        preliminary_evidences: List[Evidence],
    ) -> Tuple[bool, str]:
        """Compatibility wrapper."""
        res = self.evaluate_stage1_screening(detection_output, preliminary_evidences)
        if res is not None:
            return True, res.reason
        return False, "Proceeding to fine high-resolution analysis"

    def should_execute_geometry_stage(
        self,
        detection_output: DetectionOutput,
        evidences: List[Evidence],
        has_depth_data: bool,
    ) -> bool:
        """
        Determines whether Stage 3 (3D Surface Geometry Expert) should be invoked.
        Invoked if depth sensor is available AND (defect candidate is Dent OR high uncertainty).
        """
        if not has_depth_data:
            return False

        pred_label = detection_output.primary_prediction.label.lower()
        is_dent_candidate = "dent" in pred_label

        has_geometric_hint = any(
            e.metadata.get("defect_type_hint") == "dent_candidate" or (e.score and e.score > 0.40)
            for e in evidences
        )

        return is_dent_candidate or has_geometric_hint or (detection_output.geometry_residual is not None and detection_output.geometry_residual > 1.0)

    def should_run_geometry_stage_4(
        self,
        detection_output: DetectionOutput,
        evidences: List[Evidence],
        depth_map: Optional[np.ndarray],
    ) -> bool:
        """Compatibility wrapper."""
        return self.should_execute_geometry_stage(detection_output, evidences, has_depth_data=(depth_map is not None))
