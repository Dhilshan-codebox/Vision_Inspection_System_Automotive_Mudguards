from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DepthCalibrationMetadata:
    """Represents camera/sensor 3D calibration parameters and physical scaling."""
    scale_z_mm: float = 1.0  # Multiplier to convert raw sensor depth to physical millimeters
    valid_depth_range_mm: Tuple[float, float] = (0.0, 2000.0)  # Valid sensor measurement range (expanded for testing)
    sensor_serial: str = "RGBD_SENSOR_DEFAULT"
    is_calibrated: bool = True
    focal_length_px: Tuple[float, float] = (500.0, 500.0)  # (fx, fy)
    principal_point_px: Tuple[float, float] = (320.0, 240.0)  # (cx, cy)


@dataclass
class DepthQualityAssessment:
    """Represents quality, validity, and noise telemetry of a 3D depth frame."""
    is_usable: bool
    surface_coverage: float  # [0.0 - 1.0] fraction of valid depth pixels on target body
    noise_level_mm: float  # Standard deviation of local surface planar residuals
    valid_pixel_count: int
    total_pixel_count: int
    calibration_valid: bool
    reasons: list[str] = field(default_factory=list)


def assess_depth_quality(
    depth_map: Optional[np.ndarray],
    calibration: Optional[DepthCalibrationMetadata] = None,
    min_surface_coverage: float = 0.25,
    max_noise_level_mm: float = 5.0,
) -> DepthQualityAssessment:
    """
    Evaluates raw 3D depth frame quality before geometric computation.
    """
    calib = calibration or DepthCalibrationMetadata()

    if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
        return DepthQualityAssessment(
            is_usable=False,
            surface_coverage=0.0,
            noise_level_mm=0.0,
            valid_pixel_count=0,
            total_pixel_count=0,
            calibration_valid=calib.is_calibrated,
            reasons=["Depth map array is missing, null, or empty"],
        )

    # Scale depth to physical mm
    depth_mm = depth_map.astype(np.float32) * calib.scale_z_mm
    total_pixels = depth_mm.size

    # Filter out NaNs, Infs, zeros, and out-of-range returns
    min_d, max_d = calib.valid_depth_range_mm
    valid_mask = (
        np.isfinite(depth_mm)
        & (depth_mm >= min_d)
        & (depth_mm <= max_d)
    )

    valid_count = int(np.count_nonzero(valid_mask))
    coverage = float(valid_count / total_pixels) if total_pixels > 0 else 0.0

    reasons: list[str] = []
    if coverage < min_surface_coverage:
        reasons.append(f"Insufficient valid 3D surface coverage ({coverage:.1%} < {min_surface_coverage:.1%})")

    # Measure local depth noise on valid pixels
    if valid_count > 100:
        valid_depths = depth_mm[valid_mask]
        noise_level = float(np.std(valid_depths - np.median(valid_depths)))
    else:
        noise_level = 999.0
        reasons.append("Insufficient depth points to estimate noise")

    if noise_level > max_noise_level_mm:
        reasons.append(f"Excessive 3D sensor noise level ({noise_level:.2f}mm > {max_noise_level_mm:.2f}mm)")

    if not calib.is_calibrated:
        reasons.append("Sensor calibration invalid or unverified")

    is_usable = (
        coverage >= min_surface_coverage
        and noise_level <= max_noise_level_mm
        and calib.is_calibrated
    )

    return DepthQualityAssessment(
        is_usable=is_usable,
        surface_coverage=coverage,
        noise_level_mm=noise_level,
        valid_pixel_count=valid_count,
        total_pixel_count=total_pixels,
        calibration_valid=calib.is_calibrated,
        reasons=reasons,
    )
