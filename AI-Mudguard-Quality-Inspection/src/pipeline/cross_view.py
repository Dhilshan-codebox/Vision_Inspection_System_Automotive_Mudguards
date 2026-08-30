from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from src.data.contracts import InspectionResult, Decision, Evidence, EvidenceStatus, Prediction


@dataclass
class CrossViewReport:
    """Consolidated consistency report across multiple camera viewpoints for a single part."""
    part_id: str
    agreement_score: float  # [0.0, 1.0] (1.0 = unanimous consensus)
    view_count: int
    consensus_label: str
    consensus_decision: Decision = Decision.PASS
    is_consistent: bool = True
    has_contradiction: bool = False
    contradictions_found: List[str] = field(default_factory=list)
    supported_defects: List[Dict[str, Any]] = field(default_factory=list)
    view_results: Dict[str, InspectionResult] = field(default_factory=dict)
    per_defect_support: List[Dict[str, Any]] = field(default_factory=list)
    adjusted_results: List[InspectionResult] = field(default_factory=list)


class CrossViewConsistencyChecker:
    """
    Evaluates multi-view inspection batches for spatial and semantic consistency.
    Corroborates defect signals across angles, down-weights isolated single-view noise,
    and flags cross-view contradictions for operator review.
    """

    def __init__(
        self,
        spatial_iou_threshold: float = 0.30,
        single_view_penalty: float = 0.40,
    ):
        self.spatial_iou_threshold = spatial_iou_threshold
        self.single_view_penalty = single_view_penalty

    def evaluate_part_views(self, results: List[InspectionResult]) -> CrossViewReport:
        """Alias for check_consistency."""
        return self.check_consistency(results)

    def check_consistency(self, results: List[InspectionResult]) -> CrossViewReport:
        """
        Analyze a collection of InspectionResult objects from multiple camera views of the same part.
        """
        if not results:
            return CrossViewReport(
                part_id="unknown",
                agreement_score=1.0,
                view_count=0,
                consensus_label="Good",
                is_consistent=True,
                has_contradiction=False,
            )

        part_id = results[0].image_record.part_id or "part_unassigned"
        view_map: Dict[str, InspectionResult] = {}
        for r in results:
            cam_id = r.image_record.camera_id or f"cam_{len(view_map)}"
            view_map[cam_id] = r

        view_count = len(results)
        if view_count == 1:
            # Single view: no cross-view verification possible
            single_res = results[0]
            label = single_res.prediction.label if single_res.prediction else "Good"
            return CrossViewReport(
                part_id=part_id,
                agreement_score=1.0,
                view_count=1,
                consensus_label=label,
                is_consistent=True,
                has_contradiction=False,
                view_results=view_map,
                adjusted_results=results,
            )

        # 1. Label and Decision Consensus
        labels = [r.prediction.label for r in results if r.prediction is not None]
        decisions = [r.decision for r in results]

        unique_labels = set(labels)
        fail_count = sum(1 for d in decisions if d == Decision.FAIL)
        pass_count = sum(1 for d in decisions if d == Decision.PASS)
        review_count = sum(1 for d in decisions if d == Decision.REVIEW)

        # 2. Defect Corroboration & Disagreement
        defect_results = [r for r in results if r.prediction and r.prediction.label != "Good"]
        clean_results = [r for r in results if r.prediction and r.prediction.label == "Good"]

        has_contradiction = False
        per_defect_support: List[Dict[str, Any]] = []

        # Check for direct contradiction: high confidence defect in one view vs high confidence Pass in another
        if defect_results and clean_results:
            max_defect_conf = max(r.prediction.confidence for r in defect_results if r.prediction)
            max_clean_conf = max(r.prediction.confidence for r in clean_results if r.prediction)

            if max_defect_conf >= 0.50 and max_clean_conf >= 0.50:
                # View 1 sees defect, View 2 sees normal clearly -> contradiction
                has_contradiction = True

        # Check for conflicting defect classes across views
        defect_classes = {r.prediction.label for r in defect_results if r.prediction}
        if len(defect_classes) > 1:
            has_contradiction = True

        # Compute agreement score
        if len(unique_labels) == 1:
            agreement_score = 1.0
            consensus_label = labels[0] if labels else "Good"
            is_consistent = True
        elif len(unique_labels) == 0:
            agreement_score = 1.0
            consensus_label = "Good"
            is_consistent = True
        else:
            # Fraction of views agreeing on the dominant label
            from collections import Counter
            counts = Counter(labels)
            consensus_label, top_count = counts.most_common(1)[0]
            agreement_score = float(top_count / view_count)
            is_consistent = (agreement_score >= 0.65) and not has_contradiction

        # 3. Adjust Results based on cross-view consistency
        adjusted_results: List[InspectionResult] = []
        for r in results:
            r_pred = r.prediction
            if r_pred is None:
                adjusted_results.append(r)
                continue

            # If isolated single-view defect with other views seeing clean surface without contradiction
            if r_pred.label != "Good" and len(defect_results) == 1 and view_count >= 2 and not has_contradiction:
                # Down-weight single view confidence
                new_conf = float(np.clip(r_pred.confidence * (1.0 - self.single_view_penalty), 0.0, 1.0))
                new_decision = Decision.REVIEW if new_conf < 0.60 else r.decision
                adjusted_r = InspectionResult(
                    inspection_id=r.inspection_id,
                    decision=new_decision,
                    image_record=r.image_record,
                    prediction=Prediction(label=r_pred.label, confidence=new_conf),
                    evidence=r.evidence,
                    model_metadata=r.model_metadata,
                )
                adjusted_results.append(adjusted_r)
                per_defect_support.append({
                    "defect_label": r_pred.label,
                    "status": "isolated_single_view",
                    "penalty_applied": self.single_view_penalty,
                })
            elif has_contradiction:
                # Route to human review due to cross-view conflict
                adjusted_r = InspectionResult(
                    inspection_id=r.inspection_id,
                    decision=Decision.REVIEW,
                    image_record=r.image_record,
                    prediction=r.prediction,
                    evidence=r.evidence,
                    model_metadata=r.model_metadata,
                )
                adjusted_results.append(adjusted_r)
            else:
                adjusted_results.append(r)

        # Consensus decision
        if has_contradiction:
            consensus_decision = Decision.REVIEW
        elif fail_count > 0:
            consensus_decision = Decision.FAIL
        elif review_count > 0:
            consensus_decision = Decision.REVIEW
        else:
            consensus_decision = Decision.PASS

        supported_defects = [
            {"label": d_cls, "views_supporting": [r.image_record.camera_id for r in defect_results if r.prediction and r.prediction.label == d_cls]}
            for d_cls in defect_classes
        ]

        contradictions_found = ["Cross-view disagreement between defect and pass predictions"] if has_contradiction else []

        return CrossViewReport(
            part_id=part_id,
            agreement_score=agreement_score,
            view_count=view_count,
            consensus_label=consensus_label,
            consensus_decision=consensus_decision,
            is_consistent=is_consistent,
            has_contradiction=has_contradiction,
            contradictions_found=contradictions_found,
            supported_defects=supported_defects,
            view_results=view_map,
            per_defect_support=per_defect_support,
            adjusted_results=adjusted_results,
        )
