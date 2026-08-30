"""
Tests for CrossViewConsistencyChecker and Multi-Camera Consensus.
"""

import pytest
from datetime import datetime, timezone

from src.pipeline.cross_view import CrossViewConsistencyChecker, CrossViewReport
from src.data.contracts import InspectionResult, Decision, ImageRecord, Prediction


def test_cross_view_single_view():
    """Verify single view returns consistent baseline report."""
    checker = CrossViewConsistencyChecker()
    rec = ImageRecord(
        image_id="img_1",
        file_path="memory://1.png",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam_top",
        part_id="part_100",
    )
    res = InspectionResult(
        inspection_id="ins_1",
        decision=Decision.PASS,
        image_record=rec,
        prediction=Prediction(label="Good", confidence=0.92),
    )

    report = checker.evaluate_part_views([res])
    assert report.view_count == 1
    assert report.is_consistent is True
    assert report.consensus_decision == Decision.PASS


def test_cross_view_multi_view_confirmed_defect():
    """Verify multiple views seeing a defect result in FAIL consensus with high agreement."""
    checker = CrossViewConsistencyChecker()

    rec_top = ImageRecord(image_id="1", file_path="1", timestamp=datetime.now(timezone.utc), camera_id="cam_top", part_id="p1")
    rec_side = ImageRecord(image_id="2", file_path="2", timestamp=datetime.now(timezone.utc), camera_id="cam_side", part_id="p1")

    res1 = InspectionResult(inspection_id="i1", decision=Decision.FAIL, image_record=rec_top, prediction=Prediction(label="Scratch", confidence=0.88))
    res2 = InspectionResult(inspection_id="i2", decision=Decision.FAIL, image_record=rec_side, prediction=Prediction(label="Scratch", confidence=0.82))

    report = checker.evaluate_part_views([res1, res2])
    assert report.view_count == 2
    assert report.is_consistent is True
    assert report.consensus_decision == Decision.FAIL
    assert report.agreement_score >= 0.90
    assert len(report.supported_defects) == 1


def test_cross_view_conflict_routes_to_review():
    """Verify conflicting view decisions route consensus to REVIEW."""
    checker = CrossViewConsistencyChecker()

    rec_top = ImageRecord(image_id="1", file_path="1", timestamp=datetime.now(timezone.utc), camera_id="cam_top", part_id="p2")
    rec_side = ImageRecord(image_id="2", file_path="2", timestamp=datetime.now(timezone.utc), camera_id="cam_side", part_id="p2")

    res_fail = InspectionResult(inspection_id="i1", decision=Decision.FAIL, image_record=rec_top, prediction=Prediction(label="Scratch", confidence=0.55))
    res_pass = InspectionResult(inspection_id="i2", decision=Decision.PASS, image_record=rec_side, prediction=Prediction(label="Good", confidence=0.92))

    report = checker.evaluate_part_views([res_fail, res_pass])
    assert report.consensus_decision == Decision.REVIEW
    assert report.is_consistent is False
    assert len(report.contradictions_found) >= 1
