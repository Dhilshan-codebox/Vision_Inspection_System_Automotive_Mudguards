"""
Tests for ActiveLearningSampler and Priority Queuing.
"""

import pytest
import os
import json
from datetime import datetime, timezone

from src.training.active_learning import ActiveLearningSampler, ActiveLearningItem
from src.data.contracts import InspectionResult, Decision, ImageRecord, Prediction, Evidence, EvidenceStatus
from src.fusion.evidence_graph import EvidenceGraph, EvidenceGraphNode


def test_active_learning_scores_contradiction_as_highest_priority():
    """Verify samples with contradictory sensor evidence receive highest labeling priority."""
    sampler = ActiveLearningSampler()

    rec = ImageRecord(image_id="img_contradict", file_path="memory://c.png", timestamp=datetime.now(timezone.utc), camera_id="cam_0")
    evidences = [
        Evidence(perspective="appearance", status=EvidenceStatus.AVAILABLE, confidence=0.8, score=0.6),
        Evidence(perspective="geometry", status=EvidenceStatus.CONTRADICTORY, confidence=0.9, score=0.0),
    ]

    res = InspectionResult(
        inspection_id="ins_c",
        decision=Decision.REVIEW,
        image_record=rec,
        prediction=Prediction(label="Dent", confidence=0.65),
        evidence=evidences,
    )

    item = sampler.score_result(res)
    assert item.has_contradiction is True
    assert item.priority_score >= 0.40
    assert "INTER_PERSPECTIVE_CONTRADICTION" in item.reasons


def test_active_learning_batch_ranking():
    """Verify sampler ranks uncertain and contradictory items above clear passes."""
    sampler = ActiveLearningSampler()

    rec_pass = ImageRecord(image_id="img_pass", file_path="1", timestamp=datetime.now(timezone.utc), camera_id="c")
    res_pass = InspectionResult(
        inspection_id="i1",
        decision=Decision.PASS,
        image_record=rec_pass,
        prediction=Prediction(label="Good", confidence=0.95),
        evidence=[Evidence(perspective="uncertainty", status=EvidenceStatus.AVAILABLE, confidence=0.9, score=0.05)],
    )

    rec_unc = ImageRecord(image_id="img_uncertain", file_path="2", timestamp=datetime.now(timezone.utc), camera_id="c")
    res_unc = InspectionResult(
        inspection_id="i2",
        decision=Decision.REVIEW,
        image_record=rec_unc,
        prediction=Prediction(label="Scratch", confidence=0.45),
        evidence=[Evidence(perspective="uncertainty", status=EvidenceStatus.AVAILABLE, confidence=0.6, score=0.42)],
    )

    ranked = sampler.sample_batch([res_pass, res_unc], top_k=2)
    assert len(ranked) == 2
    assert ranked[0].image_id == "img_uncertain"
    assert ranked[0].priority_score > ranked[1].priority_score


def test_active_learning_queue_export(tmp_path):
    """Verify queue items can be exported to JSONL format."""
    queue_file = str(tmp_path / "test_queue.jsonl")
    sampler = ActiveLearningSampler(queue_path=queue_file)

    item = ActiveLearningItem(
        image_id="sample_1",
        file_path="/path/to/img.png",
        priority_score=0.75,
        reasons=["HIGH_UNCERTAINTY"],
        decision="REVIEW",
    )

    out_path = sampler.export_queue([item], output_path=queue_file)
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["image_id"] == "sample_1"
        assert data["priority_score"] == 0.75
