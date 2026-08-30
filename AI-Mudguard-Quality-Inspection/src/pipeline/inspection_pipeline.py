"""
Top-Level Inspection Pipeline Orchestrator for Automotive Mudguards.

Implements the complete 11-step multi-perspective inspection and evidence-reasoning process:
  1. Image Usability Verification (Quality Gate: blur, saturation, coverage)
  2. Part Localization & Primary Backbone Screening
  3. Appearance Perspective (Multi-scale gradients, RGB channels, boundary alignment)
  4. Texture Perspective (High-res tiling, local variance, LBP micro-texture)
  5. 3D Geometry Perspective (Surface depth fitting & dent profile verification)
  6. Operational & Environmental Context (Pose, lighting stability, viewpoint)
  7. Novelty & Anomaly Detection (Learned nominal representations, patch z-scores)
  8. Multi-Perspective Uncertainty Estimation (Epistemic + Aleatoric + OOD)
  9. Contradiction & Dispute Resolution
  10. Calibrated Late Fusion & Evidence Graph Generation
  11. Physical Severity Estimation
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

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
from src.detection.detector_protocol import DetectorProtocol, DetectionOutput
from src.detection.rgb_detector import BackboneDetector, RGBBaselineDetector, create_detector
from src.perspectives.appearance import AppearancePerspective
from src.perspectives.texture import TexturePerspective
from src.perspectives.geometry import GeometryPerspective
from src.perspectives.context import ContextPerspective
from src.perspectives.uncertainty import UncertaintyPerspective
from src.perspectives.contradiction import ContradictionDetector
from src.anomaly.novelty_detector import NoveltyAnomalyDetector
from src.fusion.late_fusion import LateFusionEngine
from src.fusion.evidence_graph import EvidenceGraph
from src.pipeline.cascade import CascadeStageEngine, CascadeStage, CascadeExitResult
from src.pipeline.cross_view import CrossViewConsistencyChecker, CrossViewReport
from src.pipeline.severity import SeverityEstimator, SeverityAssessment
from src.preprocessing.quality_gate import evaluate_image_quality


class InspectionPipeline:
    """
    Master Multi-Perspective Orchestrator for Automotive Mudguard Quality Inspection.
    Executes image quality gating, multi-head neural screening, coarse-to-fine tile experts,
    3D geometry verification, anomaly detection, contradiction resolution, cross-view consistency,
    and calibrated late fusion with an immutable EvidenceGraph audit trail.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        detector: Optional[DetectorProtocol] = None,
        fusion_engine: Optional[LateFusionEngine] = None,
        anomaly_detector: Optional[NoveltyAnomalyDetector] = None,
        cascade_engine: Optional[CascadeStageEngine] = None,
        severity_estimator: Optional[SeverityEstimator] = None,
        cross_view_checker: Optional[CrossViewConsistencyChecker] = None,
        mock_mode: bool = False,
    ):
        self.config = config or {}
        self.mock_mode = mock_mode

        # 1. Detector / Backbone
        if detector is not None:
            self.detector = detector
        else:
            self.detector = BackboneDetector(
                version=self.config.get("version", "2.0.0"),
                mock_mode=self.mock_mode,
            )

        # 2. Perspectives
        persp_cfg = self.config.get("perspectives", {})
        app_cfg = persp_cfg.get("appearance", {})
        self.appearance_perspective = AppearancePerspective(
            scratch_sensitivity=app_cfg.get("scratch_sensitivity", 0.50),
            mock_mode=self.mock_mode,
        )

        tex_cfg = persp_cfg.get("texture", {})
        self.texture_perspective = TexturePerspective(
            roughness_threshold=tex_cfg.get("roughness_threshold", 25.0),
            tile_size=tex_cfg.get("tile_size", 224),
            tile_stride=tex_cfg.get("tile_stride", 160),
            mock_mode=self.mock_mode,
        )

        geo_cfg = persp_cfg.get("geometry", {})
        self.geometry_perspective = GeometryPerspective(
            dent_depth_threshold_mm=geo_cfg.get("dent_depth_threshold_mm", 2.0),
            mock_mode=self.mock_mode,
        )

        ctx_cfg = persp_cfg.get("context", {})
        self.context_perspective = ContextPerspective(
            expected_viewpoints=ctx_cfg.get("expected_viewpoints"),
            mock_mode=self.mock_mode,
        )

        unc_cfg = persp_cfg.get("uncertainty", {})
        self.uncertainty_perspective = UncertaintyPerspective(
            epistemic_weight=unc_cfg.get("epistemic_weight", 0.60),
            aleatoric_weight=unc_cfg.get("aleatoric_weight", 0.40),
            mock_mode=self.mock_mode,
        )

        # 3. Anomaly / Novelty Detector
        nov_cfg = self.config.get("novelty", {})
        self.novelty_detector = anomaly_detector or NoveltyAnomalyDetector(
            patch_size=nov_cfg.get("patch_size", 16),
            anomaly_threshold=nov_cfg.get("anomaly_threshold", 0.50),
            mock_mode=self.mock_mode,
        )

        # 4. Contradiction Detector
        self.contradiction_detector = ContradictionDetector()

        # 5. Cross-View Checker
        cv_cfg = self.config.get("cross_view", {})
        self.cross_view_checker = cross_view_checker or CrossViewConsistencyChecker(
            spatial_iou_threshold=cv_cfg.get("spatial_iou_threshold", 0.30),
            single_view_penalty=cv_cfg.get("single_view_penalty", 0.40),
        )

        # 6. Cascade Engine
        casc_cfg = self.config.get("cascade", {})
        self.cascade_engine = cascade_engine or CascadeStageEngine(
            enable_early_exits=casc_cfg.get("enable_early_exits", True),
            stage2_screening_pass_threshold=casc_cfg.get("stage2_screening_pass_threshold", 0.90),
            stage2_screening_max_persp_score=casc_cfg.get("stage2_screening_max_persp_score", 0.25),
            stage4_require_depth_for_dents=casc_cfg.get("stage4_require_depth_for_dents", True),
        )

        # 7. Severity Estimator
        self.severity_estimator = severity_estimator or SeverityEstimator(
            dent_depth_threshold_mm=geo_cfg.get("dent_depth_threshold_mm", 2.0)
        )

        # 8. Late Fusion Engine
        lf_cfg = self.config.get("late_fusion", {})
        thresh_cfg = lf_cfg.get("thresholds", {})
        self.fusion_engine = fusion_engine or LateFusionEngine(
            version=lf_cfg.get("version", "2.0.0"),
            fail_defect_threshold=thresh_cfg.get("fail_defect_score", 0.60),
            review_borderline_low=thresh_cfg.get("review_borderline_low", 0.35),
            review_borderline_high=thresh_cfg.get("review_borderline_high", 0.60),
            max_acceptable_uncertainty=thresh_cfg.get("max_acceptable_uncertainty", 0.40),
            weights=lf_cfg.get("weights"),
        )

    @classmethod
    def from_config_file(cls, config_path: str | Path, mock_mode: bool = False) -> "InspectionPipeline":
        """Factory constructor loading settings from a YAML configuration file."""
        if yaml is None:
            return cls(mock_mode=mock_mode)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg, mock_mode=mock_mode)

    def inspect_image(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        image_record: Optional[ImageRecord] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bypass_cascade: bool = False,
        **kwargs: Any,
    ) -> InspectionResult:
        """
        Execute full multi-perspective inspection on a single captured frame.
        """
        meta = metadata.copy() if metadata else {}
        ins_id = meta.get("inspection_id", f"ins_{uuid.uuid4().hex[:12]}")

        img_rec = image_record or ImageRecord(
            image_id=meta.get("image_id", f"img_{uuid.uuid4().hex[:8]}"),
            file_path=meta.get("file_path", "memory://capture.png"),
            timestamp=datetime.now(timezone.utc),
            camera_id=meta.get("camera_id", meta.get("view_id", "cam_top")),
            part_id=meta.get("part_id", "part_unknown"),
        )

        # -------------------------------------------------------------
        # Step 1: Quality Gate Check (Stage 0)
        # -------------------------------------------------------------
        if self.mock_mode:
            quality = ImageQualityAssessment(
                is_acceptable=meta.get("is_acceptable", True),
                recommended_action=QualityAction.PROCEED_TO_INFERENCE if meta.get("is_acceptable", True) else QualityAction.FLAG_FOR_REVIEW,
                blur_score=100.0,
                is_blurry=False,
                saturation_score=0.0,
                is_overexposed=False,
                is_underexposed=False,
                coverage_score=0.95,
                has_missing_region=False,
            )
        else:
            quality = evaluate_image_quality(image)

        # Check early exit at Stage 0
        if not bypass_cascade:
            exit_0 = self.cascade_engine.evaluate_stage0_quality(quality)
            if exit_0 is not None:
                fallback_pred = Prediction(label="Unusable", confidence=0.99)
                det_out = DetectionOutput(primary_prediction=fallback_pred)
                return self.fusion_engine.fuse(
                    image_record=img_rec,
                    detection_output=det_out,
                    evidences=[],
                    quality_assessment=quality,
                    inspection_id=ins_id,
                )

        # -------------------------------------------------------------
        # Step 2: Part Localization & Backbone Screening (Stage 1)
        # -------------------------------------------------------------
        det_output = self.detector.predict(image, metadata=meta)

        # Step 3: Fast Appearance Screening
        ev_app = self.appearance_perspective.extract(image, metadata=meta)

        # Step 4: Texture Perspective (High-Res Tiles + LBP)
        ev_tex = self.texture_perspective.extract(image, metadata=meta)

        # Step 5: Anomaly & Novelty Detection
        anomaly_res = self.novelty_detector.detect(image, metadata=meta)
        ev_nov = Evidence(
            perspective="novelty",
            status=EvidenceStatus.AVAILABLE,
            confidence=anomaly_res.confidence,
            score=anomaly_res.anomaly_score,
            region=anomaly_res.anomaly_region,
            reference="novelty_patch_distribution",
            metadata={
                "is_anomalous": anomaly_res.is_anomalous,
                "novelty_type": anomaly_res.novelty_type.value,
            },
        )

        # Check early exit at Stage 1 (Screening Pass on clean normal parts)
        if not bypass_cascade:
            exit_1 = self.cascade_engine.evaluate_stage1_screening(
                detection_output=det_output,
                initial_evidences=[ev_app, ev_tex],
                novelty_score=anomaly_res.anomaly_score,
            )
            if exit_1 is not None and quality.is_acceptable:
                meta_unc = {"evidences": [ev_app, ev_tex]}
                ev_unc = self.uncertainty_perspective.extract(image, metadata=meta_unc)
                return self.fusion_engine.fuse(
                    image_record=img_rec,
                    detection_output=det_output,
                    evidences=[ev_app, ev_tex, ev_unc],
                    quality_assessment=quality,
                    anomaly_score=anomaly_res.anomaly_score,
                    is_novel=anomaly_res.is_anomalous,
                    inspection_id=ins_id,
                )

        # -------------------------------------------------------------
        # Step 6: 3D Geometry Surface Expert
        # -------------------------------------------------------------
        geo_meta = meta.copy()
        if depth_map is not None:
            geo_meta["depth_map"] = depth_map
        if ev_app.score is not None:
            geo_meta["appearance_score"] = ev_app.score

        ev_geo = self.geometry_perspective.extract(image, metadata=geo_meta)

        # -------------------------------------------------------------
        # Step 7: Operational & Environmental Context
        # -------------------------------------------------------------
        ctx_meta = meta.copy()
        if det_output.detections:
            ctx_meta["candidate_region"] = det_output.detections[0].bbox
        elif ev_app.region:
            ctx_meta["candidate_region"] = ev_app.region

        ev_ctx = self.context_perspective.extract(image, metadata=ctx_meta)

        # -------------------------------------------------------------
        # Step 8: Multi-Perspective Uncertainty Estimation
        # -------------------------------------------------------------
        gathered_evidences = [ev_app, ev_tex, ev_geo, ev_ctx, ev_nov]
        unc_meta = meta.copy()
        unc_meta["evidences"] = gathered_evidences
        if det_output.embedding is not None:
            unc_meta["embedding"] = det_output.embedding
        if getattr(self.novelty_detector, "nominal_mean_features", None) is not None:
            unc_meta["nominal_embedding"] = self.novelty_detector.nominal_mean_features

        ev_unc = self.uncertainty_perspective.extract(image, metadata=unc_meta)
        all_evidences = gathered_evidences + [ev_unc]

        # -------------------------------------------------------------
        # Step 9: Contradiction Resolution
        # -------------------------------------------------------------
        resolved_evidences = self.contradiction_detector.resolve_and_mark_contradictions(all_evidences)

        # -------------------------------------------------------------
        # Step 10: Severity Estimation
        # -------------------------------------------------------------
        sev_est = self.severity_estimator.estimate(
            detection_output=det_output,
            evidences=resolved_evidences,
            depth_map=depth_map,
        )
        det_output.severity_level = sev_est.level.value
        det_output.severity_score = sev_est.score

        # -------------------------------------------------------------
        # Step 11: Calibrated Late Fusion & Evidence Graph Generation
        # -------------------------------------------------------------
        result = self.fusion_engine.fuse(
            image_record=img_rec,
            detection_output=det_output,
            evidences=resolved_evidences,
            quality_assessment=quality,
            anomaly_score=anomaly_res.anomaly_score,
            is_novel=anomaly_res.is_anomalous,
            model_metadata=self.detector.get_model_metadata(),
            inspection_id=ins_id,
        )
        result.__dict__["severity_estimate"] = sev_est
        return result

    def inspect_batch_views(
        self,
        images: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        image_records: Optional[List[ImageRecord]] = None,
        depth_maps: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Tuple[List[InspectionResult], CrossViewReport]:
        """
        Execute multi-view inspection across all cameras for a single mudguard part.
        """
        n_views = len(images)
        meta_list = metadata_list or [{} for _ in range(n_views)]
        records = image_records or [None for _ in range(n_views)]
        d_maps = depth_maps or [None for _ in range(n_views)]

        results: List[InspectionResult] = []
        for i in range(n_views):
            res = self.inspect_image(
                image=images[i],
                metadata=meta_list[i],
                image_record=records[i],
                depth_map=d_maps[i],
            )
            results.append(res)

        report = self.cross_view_checker.evaluate_part_views(results)
        return results, report

    def inspect_part_views(
        self,
        part_id: str,
        view_images: Dict[str, np.ndarray],
        view_depths: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CrossViewReport:
        """
        Inspect all camera views for a single part and synthesize a cross-view consensus report.
        """
        depth_dict = view_depths or {}
        view_results: List[InspectionResult] = []

        for view_name, img_arr in view_images.items():
            view_meta = {**(metadata or {}), "view_id": view_name, "part_id": part_id}
            depth_arr = depth_dict.get(view_name)
            res = self.inspect_image(image=img_arr, depth_map=depth_arr, metadata=view_meta)
            view_results.append(res)

        return self.cross_view_checker.evaluate_part_views(view_results)

    def get_last_evidence_graph(self) -> Optional[EvidenceGraph]:
        """Return the most recent EvidenceGraph generated during inspection."""
        return self.fusion_engine.get_last_evidence_graph()
