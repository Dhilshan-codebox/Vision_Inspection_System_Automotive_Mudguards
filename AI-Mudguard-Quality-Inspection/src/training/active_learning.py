"""
Active Learning Sampler & Priority Annotation Queue.

Identifies the most informative, uncertain, novel, or contradictory inspection samples
for prioritized human expert review and continuous model retraining.
"""

import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

from src.data.contracts import InspectionResult, EvidenceStatus, Decision
from src.fusion.evidence_graph import EvidenceGraph


@dataclass
class ActiveLearningSample:
    """Prioritized candidate sample for active learning annotation."""
    image_id: str
    file_path: str
    part_id: Optional[str] = None
    priority_score: float = 0.0  # [0.0, 1.0] (higher = more urgent for human annotation)
    priority_tier: str = "MEDIUM"  # "HIGH", "MEDIUM", "LOW"
    reasons: List[str] = field(default_factory=list)
    decision: str = "REVIEW"
    predicted_label: str = "Unknown"
    confidence: float = 0.0
    uncertainty_score: float = 0.0
    novelty_score: float = 0.0
    has_contradiction: bool = False
    cross_view_inconsistent: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for compatibility
ActiveLearningItem = ActiveLearningSample


class ActiveLearningSampler:
    """
    Identifies high-value samples for human labeling by prioritizing:
    1. High model / perspective uncertainty (> 0.35)
    2. Out-of-distribution / high novelty (> 0.50)
    3. Inter-perspective contradictions and disputes
    4. Cross-view inconsistencies across camera angles
    """

    def __init__(
        self,
        high_priority_uncertainty: float = 0.35,
        high_priority_novelty: float = 0.50,
        high_uncertainty_threshold: Optional[float] = None,
        high_novelty_threshold: Optional[float] = None,
        queue_path: str = "logs/active_learning_queue.jsonl",
    ):
        self.high_priority_uncertainty = high_uncertainty_threshold or high_priority_uncertainty
        self.high_priority_novelty = high_novelty_threshold or high_priority_novelty
        self.high_uncertainty_threshold = self.high_priority_uncertainty
        self.high_novelty_threshold = self.high_priority_novelty
        self.queue_path = queue_path
        self._in_memory_queue: List[ActiveLearningSample] = []

    def score_result(
        self,
        result: InspectionResult,
        evidence_graph: Optional[EvidenceGraph] = None,
        cross_view_inconsistent: bool = False,
    ) -> ActiveLearningSample:
        """
        Evaluate an InspectionResult and assign an active learning labeling priority score.
        """
        ev_map = {e.perspective: e for e in result.evidence}
        reasons: List[str] = []
        priority_accum = 0.0

        # 1. Contradiction Flag Check (Highest Priority)
        has_contradiction = any(e.status == EvidenceStatus.CONTRADICTORY for e in result.evidence)
        if has_contradiction:
            priority_accum += 0.40
            reasons.append("INTER_PERSPECTIVE_CONTRADICTION")

        # 2. Uncertainty Score Check
        unc_ev = ev_map.get("uncertainty")
        unc_score = unc_ev.score if (unc_ev and unc_ev.score is not None) else 0.0
        if unc_score >= self.high_priority_uncertainty:
            priority_accum += 0.35
            reasons.append(f"HIGH_UNCERTAINTY ({unc_score:.2f} >= {self.high_priority_uncertainty})")

        # 3. Novelty / OOD Check
        nov_score = 0.0
        if evidence_graph and evidence_graph.novelty_detected:
            nov_score = 0.80
        elif (nov_ev := ev_map.get("novelty")) and nov_ev.score is not None:
            nov_score = float(nov_ev.score)

        if nov_score >= self.high_priority_novelty:
            priority_accum += 0.30
            reasons.append(f"UNKNOWN_OUT_OF_DISTRIBUTION_DEFECT ({nov_score:.2f} >= {self.high_priority_novelty})")

        # 4. Cross-View Inconsistency
        if cross_view_inconsistent:
            priority_accum += 0.20
            reasons.append("CROSS_VIEW_INCONSISTENCY")

        # 5. Borderline Prediction Confidence
        pred_label = result.prediction.label if result.prediction else "Unknown"
        pred_conf = result.prediction.confidence if result.prediction else 0.0
        if 0.35 <= pred_conf < 0.60:
            priority_accum += 0.15
            reasons.append(f"BORDERLINE_CONFIDENCE ({pred_conf:.2f})")

        # 6. Routed to REVIEW
        if result.decision == Decision.REVIEW and not reasons:
            priority_accum += 0.15
            reasons.append("ROUTED_TO_REVIEW")

        priority_score = float(np.clip(priority_accum, 0.01, 1.0))

        if priority_score >= 0.60:
            tier = "HIGH"
        elif priority_score >= 0.30:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        sample = ActiveLearningSample(
            image_id=result.image_record.image_id,
            file_path=result.image_record.file_path,
            part_id=result.image_record.part_id,
            priority_score=priority_score,
            priority_tier=tier,
            reasons=reasons,
            decision=result.decision.value,
            predicted_label=pred_label,
            confidence=pred_conf,
            uncertainty_score=float(unc_score),
            novelty_score=float(nov_score),
            has_contradiction=has_contradiction,
            cross_view_inconsistent=cross_view_inconsistent,
        )

        return sample

    def sample_batch(
        self,
        results: List[InspectionResult],
        top_k: int = 20,
        evidence_graphs: Optional[List[Optional[EvidenceGraph]]] = None,
    ) -> List[ActiveLearningSample]:
        """
        Rank a batch of inspection results and return the top-k highest priority candidates.
        """
        graphs = evidence_graphs or [None for _ in results]
        items = [self.score_result(res, graph) for res, graph in zip(results, graphs)]
        items.sort(key=lambda x: x.priority_score, reverse=True)
        return items[:top_k]

    def add_to_queue(self, sample: ActiveLearningSample) -> None:
        """Add sample to in-memory queue and persist to JSONL log."""
        self._in_memory_queue.append(sample)
        self.export_queue([sample], output_path=self.queue_path)

    def export_queue(
        self,
        items: List[ActiveLearningSample],
        output_path: Optional[str] = None,
    ) -> str:
        """Appends active learning priority items to a JSONL file."""
        path = output_path or self.queue_path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict()) + "\n")
        return path

    def get_top_priority_batch(self, k: int = 20) -> List[ActiveLearningSample]:
        """Return the top k highest priority samples from the queue."""
        sorted_samples = sorted(self._in_memory_queue, key=lambda s: s.priority_score, reverse=True)
        return sorted_samples[:k]

    def load_queue_from_disk(self) -> List[ActiveLearningSample]:
        """Read all logged active learning samples from JSONL."""
        if not os.path.exists(self.queue_path):
            return []
        samples = []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(ActiveLearningSample(**data))
        self._in_memory_queue = samples
        return samples
