import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus
from src.geometry.calibration import (
    DepthCalibrationMetadata,
    DepthQualityAssessment,
    assess_depth_quality,
)


@dataclass
class DentMetrics:
    """Quantitative physical measurements of detected surface denting."""
    max_dent_depth_mm: float
    mean_dent_depth_mm: float
    dent_area_mm2: float
    dent_volume_mm3: float
    geometric_error_rmse_mm: float
    surface_coverage: float
    has_dent: bool
    dent_region: Optional[List[float]] = None  # [x1, y1, x2, y2]


class DentGeometryExpert:
    """
    Expert engine for physical 3D surface dent quantification and optical shadow disambiguation.
    Fits robust nominal polynomial surfaces, calculates millimeter-scale residuals,
    and handles missing sensor inputs with explicit graceful degradation.
    """

    def __init__(
        self,
        dent_depth_threshold_mm: float = 2.0,
        pixel_size_mm: float = 1.0,  # mm per pixel at target inspection distance
        calibration: Optional[DepthCalibrationMetadata] = None,
    ):
        self.dent_depth_threshold_mm = dent_depth_threshold_mm
        self.pixel_size_mm = pixel_size_mm
        self.calibration = calibration or DepthCalibrationMetadata()

    def fit_nominal_surface(
        self,
        depth_mm: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fits a smooth nominal 2D quadratic reference surface Z(x, y) = c0 + c1*x + c2*y + c3*x^2 + c4*y^2
        using least-squares regression over valid body points.
        """
        h, w = depth_mm.shape
        y_coords, x_coords = np.nonzero(valid_mask)
        z_vals = depth_mm[valid_mask]

        if len(z_vals) < 20:
            # Fallback to median plane if sparse
            return np.full_like(depth_mm, np.median(depth_mm) if len(z_vals) > 0 else 500.0)

        # Build polynomial design matrix A
        x_norm = x_coords / float(w)
        y_norm = y_coords / float(h)
        A = np.column_stack([
            np.ones_like(x_norm),
            x_norm,
            y_norm,
            x_norm ** 2,
            y_norm ** 2,
        ])

        # Solve least squares c = (A^T A)^(-1) A^T z
        coeffs, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)

        # Evaluate fitted nominal surface over full grid
        all_y, all_x = np.mgrid[:h, :w]
        ax_norm = all_x / float(w)
        ay_norm = all_y / float(h)
        all_A = np.stack([
            np.ones_like(ax_norm),
            ax_norm,
            ay_norm,
            ax_norm ** 2,
            ay_norm ** 2,
        ], axis=-1)

        nominal_surface = np.dot(all_A, coeffs)
        return nominal_surface.astype(np.float32)

    def analyze_surface(
        self,
        depth_map: Optional[np.ndarray],
        calibration: Optional[DepthCalibrationMetadata] = None,
    ) -> Tuple[DepthQualityAssessment, Optional[DentMetrics], Optional[np.ndarray]]:
        """
        Analyze depth surface against nominal reference.
        Returns: (quality_assessment, dent_metrics, residual_map)
        """
        calib = calibration or self.calibration
        quality = assess_depth_quality(depth_map, calibration=calib)

        if not quality.is_usable or depth_map is None:
            return quality, None, None

        depth_mm = depth_map.astype(np.float32) * calib.scale_z_mm
        min_d, max_d = calib.valid_depth_range_mm
        valid_mask = np.isfinite(depth_mm) & (depth_mm >= min_d) & (depth_mm <= max_d)

        # Fit nominal reference surface
        nominal_surface = self.fit_nominal_surface(depth_mm, valid_mask)

        # Calculate residuals (positive = indentation / dent)
        residual_map = np.zeros_like(depth_mm)
        residual_map[valid_mask] = depth_mm[valid_mask] - nominal_surface[valid_mask]

        valid_residuals = residual_map[valid_mask]
        # Estimate max dent depth using raw depth statistics to avoid fitting bias.
        # Use the difference between the maximum valid depth and the median depth.
        if len(valid_residuals) > 0:
            max_depth = float(np.max(depth_mm[valid_mask]))
            median_depth = float(np.median(depth_mm[valid_mask]))
            max_dent_depth = max_depth - median_depth
        else:
            max_dent_depth = 0.0
        rmse = float(np.sqrt(np.mean(valid_residuals ** 2))) if len(valid_residuals) > 0 else 0.0

        # Dent segmentation mask
        dent_mask = valid_mask & (residual_map >= (self.dent_depth_threshold_mm * 0.5))
        dent_pixel_count = int(np.count_nonzero(dent_mask))

        pixel_area_mm2 = self.pixel_size_mm ** 2
        dent_area_mm2 = float(dent_pixel_count * pixel_area_mm2)
        dent_volume_mm3 = float(np.sum(residual_map[dent_mask]) * pixel_area_mm2) if dent_pixel_count > 0 else 0.0
        mean_dent_depth = float(np.mean(residual_map[dent_mask])) if dent_pixel_count > 0 else 0.0

        has_dent = max_dent_depth >= self.dent_depth_threshold_mm

        dent_region: Optional[List[float]] = None
        if dent_pixel_count > 10:
            y_idx, x_idx = np.nonzero(dent_mask)
            dent_region = [
                float(np.min(x_idx)),
                float(np.min(y_idx)),
                float(np.max(x_idx) + 1),
                float(np.max(y_idx) + 1),
            ]

        metrics = DentMetrics(
            max_dent_depth_mm=max_dent_depth,
            mean_dent_depth_mm=mean_dent_depth,
            dent_area_mm2=dent_area_mm2,
            dent_volume_mm3=dent_volume_mm3,
            geometric_error_rmse_mm=rmse,
            surface_coverage=quality.surface_coverage,
            has_dent=has_dent,
            dent_region=dent_region,
        )

        return quality, metrics, residual_map

    def evaluate_to_evidence(
        self,
        depth_map: Optional[np.ndarray],
        calibration: Optional[DepthCalibrationMetadata] = None,
        rgb_appearance_score: Optional[float] = None,
    ) -> Evidence:
        """
        Produce standardized Evidence contract from 3D surface analysis.
        """
        start_time = time.perf_counter()
        quality, metrics, _ = self.analyze_surface(depth_map, calibration=calibration)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not quality.is_usable or metrics is None:
            return Evidence(
                perspective="geometry",
                status=EvidenceStatus.UNAVAILABLE,
                confidence=0.0,
                score=0.0,
                region=None,
                reference=None,
                runtime_ms=elapsed_ms,
                metadata={
                    "reasons": quality.reasons,
                    "coverage": quality.surface_coverage,
                    "usable": False,
                },
            )

        # Dent score based on depression depth relative to threshold
        raw_score = metrics.max_dent_depth_mm / self.dent_depth_threshold_mm
        score = float(np.clip(raw_score, 0.0, 1.0))
        confidence = float(np.clip(quality.surface_coverage * 0.95, 0.70, 0.98))

        # Check optical shadow disagreement
        is_optical_shadow_dispute = False
        # Optical shadow dispute: high RGB appearance confidence but low geometric dent score.
        # Adjusted threshold to detect disputes when the dent score is modest (<= 0.5).
        if rgb_appearance_score is not None and rgb_appearance_score >= 0.70 and score <= 0.5:
            is_optical_shadow_dispute = True

        return Evidence(
            perspective="geometry",
            status=EvidenceStatus.AVAILABLE,
            confidence=confidence,
            score=score,
            region=metrics.dent_region,
            reference="3d_depth_residual_surface_map",
            runtime_ms=elapsed_ms,
            metadata={
                "max_dent_depth_mm": metrics.max_dent_depth_mm,
                "mean_dent_depth_mm": metrics.mean_dent_depth_mm,
                "dent_volume_mm3": metrics.dent_volume_mm3,
                "dent_area_mm2": metrics.dent_area_mm2,
                "geometric_error_rmse_mm": metrics.geometric_error_rmse_mm,
                "surface_coverage": metrics.surface_coverage,
                "optical_shadow_dispute": is_optical_shadow_dispute,
                "usable": True,
            },
        )
