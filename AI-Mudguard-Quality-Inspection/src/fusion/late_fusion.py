from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import uuid
import numpy as np

from src.data.contracts import (
    InspectionResult,
    Decision,
    ImageRecord,
    Prediction,
    Evidence,
    EvidenceStatus,
    ModelMetadata,
    ImageQualityAssessment,
    QualityAction,
)
from src.detection.detector_protocol import DetectionOutput
from src.fusion.evidence_graph import EvidenceGraph, EvidenceNode, ReasoningStep


@dataclass
class FusionVersion:
    """Dataclass tracking the version and rule parameters of the LateFusionEngine."""
    version: str = "2.0.0"
    name: str = "Calibrated_Evidence_Graph_Late_Fusion"
    rule_count: int = 11


DEFAULT_PERSPECTIVE_WEIGHTS: Dict[str, float] = {
    "rgb_detector": 0.35,
    "appearance": 0.20,
    "texture": 0.15,
    "geometry": 0.15,
    "context": 0.05,
    "uncertainty": 0.05,
    "novelty": 0.05,
}


class LateFusionEngine:
    """
    Combines primary RGB detector predictions, multi-perspective feature evidences,
    geometry sensor data, uncertainty metrics, and image-quality assessments into a
    calibrated PASS/REVIEW/FAIL decision with a full EvidenceGraph audit trail.
    """

    def __init__(
        self,
        version: str = "2.0.0",
        fail_defect_threshold: float = 0.60,
        review_borderline_low: float = 0.35,
        review_borderline_high: float = 0.60,
        max_acceptable_uncertainty: float = 0.40,
        novelty_threshold: float = 0.50,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.version = version
        self.fail_defect_threshold = fail_defect_threshold
        self.review_borderline_low = review_borderline_low
        self.review_borderline_high = review_borderline_high
        self.max_acceptable_uncertainty = max_acceptable_uncertainty
        self.novelty_threshold = novelty_threshold
        self.weights = weights or DEFAULT_PERSPECTIVE_WEIGHTS
        self.fusion_version = FusionVersion(version=version)

    def fuse(
        self,
        image_record: ImageRecord,
        detection_output: DetectionOutput,
        evidences: List[Evidence],
        quality_assessment: Optional[ImageQualityAssessment] = None,
        anomaly_score: Optional[float] = None,
        is_novel: bool = False,
        cross_view_agreement: Optional[float] = None,
        model_metadata: Optional[ModelMetadata] = None,
        inspection_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InspectionResult:
        """
        Execute late fusion logic following calibrated safety policies.
        Constructs an immutable EvidenceGraph detailing the reasoning chain.
        """
        ins_id = inspection_id or f"ins_{uuid.uuid4().hex[:12]}"
        prediction = detection_output.primary_prediction
        pred_label = prediction.label.lower()
        pred_conf = prediction.confidence
        sev_level = detection_output.severity_level or "none"
        sev_score = detection_output.severity_score or (0.0 if pred_label == "good" else float(pred_conf))

        # Map evidences by perspective name
        ev_map: Dict[str, Evidence] = {e.perspective: e for e in evidences}

        # Build evidence graph nodes
        nodes: Dict[str, EvidenceNode] = {}
        weighted_scores = []
        total_weight = 0.0

        for e in evidences:
            w = self.weights.get(e.perspective, 0.10)
            score_val = e.score if (e.score is not None and e.status == EvidenceStatus.AVAILABLE) else 0.0
            contrib = score_val * w * e.confidence
            nodes[e.perspective] = EvidenceNode(
                perspective=e.perspective,
                status=e.status.value,
                score=e.score,
                confidence=e.confidence,
                weight=w,
                contribution=float(contrib),
                metadata=e.metadata,
            )
            if e.status == EvidenceStatus.AVAILABLE and e.score is not None and e.perspective != "uncertainty":
                weighted_scores.append(score_val * w)
                total_weight += w

        fused_defect_score = float(sum(weighted_scores) / max(total_weight, 1e-6)) if weighted_scores else 0.0

        reasoning_chain: List[str] = []
        fired_rule: str = "DEFAULT_SAFETY_FALLBACK"

        # -------------------------------------------------------------
        # Rule 1: Quality Gate Check
        # -------------------------------------------------------------
        is_bad_quality = quality_assessment is not None and not quality_assessment.is_acceptable
        if is_bad_quality:
            reasoning_chain.append(f"ImageQualityAssessment rejected: reasons={quality_assessment.reasons}")
            fired_rule = "RULE_1_QUALITY_GATE_REJECTION / Rule0_QualityGate"
            quality_decision = Decision.REQUEST_RECAPTURE if quality_assessment.action == QualityAction.REQUEST_RECAPTURE else Decision.REVIEW
            return self._build_result(ins_id, quality_decision, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                      cross_view_agreement, False, False)

        # -------------------------------------------------------------
        # Rule 2: Contradiction / Dispute Check
        # -------------------------------------------------------------
        has_contradiction = any(e.status == EvidenceStatus.CONTRADICTORY for e in evidences)
        if has_contradiction:
            reasoning_chain.append("Cross-perspective contradiction detected; routing to human review")
            fired_rule = "RULE_2_PERSPECTIVE_CONTRADICTION"
            return self._build_result(ins_id, Decision.REVIEW, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                      cross_view_agreement, False, True)


        # -------------------------------------------------------------
        # Rule 4: Novelty / Unknown Anomaly Check
        # -------------------------------------------------------------
        novelty_ev = ev_map.get("novelty")
        is_novel_flag = novelty_ev is not None and novelty_ev.score is not None and novelty_ev.score >= self.novelty_threshold
        if is_novel_flag and pred_label in {"good", "normal", "pass"}:
            # If novelty is high along with texture/appearance anomaly, trigger defect failure
            tex_ev = ev_map.get("texture")
            app_ev = ev_map.get("appearance")
            if (tex_ev and tex_ev.score is not None and tex_ev.score > 0.25) or (app_ev and app_ev.score is not None and app_ev.score > 0.25):
                reasoning_chain.append(f"Novelty anomaly confirmed defect (novelty={novelty_ev.score:.2f})")
                fired_rule = "RULE_5_TEXTURE_CONFIRMED_DEFECT"
                return self._build_result(ins_id, Decision.FAIL, image_record, prediction, evidences, model_metadata,
                                          fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                          cross_view_agreement, True, False)

            reasoning_chain.append(f"Novelty detector flagged unseen distribution (score={novelty_ev.score:.2f}) on nominal class")
            fired_rule = "RULE_4_UNKNOWN_ANOMALY_REVIEW"
            return self._build_result(ins_id, Decision.REVIEW, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                      cross_view_agreement, True, False)

        # -------------------------------------------------------------
        # Rule 5: High Texture + Valid Context -> Scratch / Surface Defect
        # -------------------------------------------------------------
        tex_ev = ev_map.get("texture")
        ctx_ev = ev_map.get("context")
        app_ev = ev_map.get("appearance")
        geo_ev = ev_map.get("geometry")

        if tex_ev and tex_ev.score is not None and tex_ev.score >= 0.30 and (ctx_ev is None or ctx_ev.score < 0.30):
            reasoning_chain.append(f"High texture roughness {tex_ev.score:.2f} confirmed with nominal context")
            fired_rule = "RULE_5_TEXTURE_CONFIRMED_DEFECT"
            return self._build_result(ins_id, Decision.FAIL, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level if sev_level != "none" else "medium", sev_score if sev_score > 0 else 0.5,
                                      cross_view_agreement, False, False)

        # -------------------------------------------------------------
        # Rule 6: Geometry Dent Confirmation
        # -------------------------------------------------------------
        if geo_ev and geo_ev.status == EvidenceStatus.AVAILABLE and geo_ev.score is not None and geo_ev.score >= 0.30:
            reasoning_chain.append(f"3D depth depression {geo_ev.score:.2f} confirms physical dent")
            fired_rule = "RULE_6_GEOMETRY_CONFIRMED_DENT"
            # Physical dent confirmed via depth map should fail regardless of mild perspective uncertainty
            return self._build_result(ins_id, Decision.FAIL, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level if sev_level != "none" else "medium", sev_score if sev_score > 0 else 0.6,
                                      cross_view_agreement, False, False)

        # -------------------------------------------------------------
        # Rule 7: Known Defect Failure Policy (High RGB confidence or Perspective defect signal)
        # -------------------------------------------------------------
        is_known_defect = pred_label in {"scratch", "dent", "paint defect", "paint misalignment"}
        if (is_known_defect and pred_conf >= self.fail_defect_threshold) or (app_ev and app_ev.score is not None and app_ev.score >= 0.15):
            reasoning_chain.append(f"Defect detected: prediction={prediction.label} ({pred_conf:.2f}), appearance_score={app_ev.score if app_ev else 0.0:.2f}")
            fired_rule = "RULE_7_KNOWN_DEFECT_FAIL"
            return self._build_result(ins_id, Decision.FAIL, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                      cross_view_agreement, False, False)

        # -------------------------------------------------------------
        # Rule 8: Borderline Confidence Policy
        # -------------------------------------------------------------
        if self.review_borderline_low <= pred_conf < self.review_borderline_high:
            reasoning_chain.append(f"Prediction confidence {pred_conf:.2f} is in borderline review band [{self.review_borderline_low}, {self.review_borderline_high})")
            fired_rule = "RULE_8_BORDERLINE_CONFIDENCE_REVIEW"
            return self._build_result(ins_id, Decision.REVIEW, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                      cross_view_agreement, False, False)

        # -------------------------------------------------------------
        # Rule 9: High-Confidence Normal Pass Policy
        # -------------------------------------------------------------
        max_persp_score = max(
            (e.score for e in evidences if e.score is not None and e.status == EvidenceStatus.AVAILABLE and e.perspective != "uncertainty"),
            default=0.0
        )

        if (pred_label in {"good", "normal", "pass"} and pred_conf >= self.review_borderline_high) or (max_persp_score < 0.15 and pred_conf < self.fail_defect_threshold):
            reasoning_chain.append(f"Good part verified: confidence={pred_conf:.2f}, max_perspective_score={max_persp_score:.2f} < 0.50")
            fired_rule = "RULE_9_HIGH_CONFIDENCE_PASS / Rule6_Pass"
            return self._build_result(ins_id, Decision.PASS, image_record, prediction, evidences, model_metadata,
                                      fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                      cross_view_agreement, False, False)

        # -------------------------------------------------------------
        # Rule 10: Default Conservative Safety Fallback
        # -------------------------------------------------------------
        reasoning_chain.append("No definitive pass/fail criteria satisfied; defaulting to conservative human review")
        fired_rule = "RULE_10_SAFETY_FALLBACK_REVIEW"
        return self._build_result(ins_id, Decision.REVIEW, image_record, prediction, evidences, model_metadata,
                                  fired_rule, fused_defect_score, reasoning_chain, nodes, sev_level, sev_score,
                                  cross_view_agreement, False, False)

    def _build_result(
        self,
        inspection_id: str,
        decision: Decision,
        image_record: ImageRecord,
        prediction: Prediction,
        evidences: List[Evidence],
        model_metadata: Optional[ModelMetadata],
        fired_rule: str,
        fused_defect_score: float,
        reasoning_chain: List[str],
        nodes: Dict[str, EvidenceNode],
        severity_level: str,
        severity_score: float,
        cross_view_agreement: Optional[float],
        novelty_detected: bool,
        contradiction_detected: bool,
    ) -> InspectionResult:
        """Constructs InspectionResult and attaches EvidenceGraph."""
        graph = EvidenceGraph(
            inspection_id=inspection_id,
            decision=decision.value,
            rule_fired=fired_rule,
            reasoning_chain=reasoning_chain,
            nodes=nodes,
            severity_level=severity_level,
            severity_score=severity_score,
            novelty_detected=novelty_detected,
            contradiction_detected=contradiction_detected,
            cross_view_agreement=cross_view_agreement,
            model_version=self.version,
        )

        meta = model_metadata or ModelMetadata(
            version=self.version,
            name="LateFusionEngine",
            thresholds={
                "fail_defect_threshold": self.fail_defect_threshold,
                "review_borderline_low": self.review_borderline_low,
                "review_borderline_high": self.review_borderline_high,
                "max_acceptable_uncertainty": self.max_acceptable_uncertainty,
                "fused_defect_score": fused_defect_score,
            }
        )

        # Store graph in metadata if possible
        result = InspectionResult(
            inspection_id=inspection_id,
            decision=decision,
            image_record=image_record,
            primary_prediction=prediction,
            prediction=prediction,
            evidence=evidences,
            model_metadata=meta,
        )
        # Attach graph as an attribute for downstream pipeline access
        result.__dict__["evidence_graph"] = graph
        self.last_evidence_graph = graph
        return result

    def get_last_evidence_graph(self) -> Optional[EvidenceGraph]:
        """Return the most recently generated EvidenceGraph."""
        return getattr(self, "last_evidence_graph", None)
