from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.data.contracts import ImageQualityAssessment, QualityAction


@dataclass(frozen=True)
class QualityGateConfig:
    """Configurable thresholds for image quality gating."""
    min_blur_score: float = 2.0  # Minimum Laplacian variance for sharp focus
    max_overexposure_ratio: float = 0.15  # Max proportion of clipped highlights (>248)
    max_underexposure_ratio: float = 0.70  # Max proportion of clipped shadows (<10)
    min_contrast_std: float = 10.0  # Minimum standard deviation of pixel intensities
    min_coverage_ratio: float = 0.10  # Minimum foreground area ratio


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Pure function to convert an input image (2D, HWC, or CHW) to 2D float32 grayscale.
    """
    if img.ndim == 2:
        return img.astype(np.float32)

    if img.ndim == 3:
        # Check if CHW (e.g. 3, H, W or 1, H, W) vs HWC (H, W, 3 or H, W, 1)
        if img.shape[0] in {1, 3, 4} and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # CHW
            if img.shape[0] == 1:
                return img[0].astype(np.float32)
            # Standard Rec.601 luminance weights
            return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).astype(np.float32)
        else:
            # HWC
            if img.shape[2] == 1:
                return img[:, :, 0].astype(np.float32)
            return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)

    raise ValueError(f"Unsupported image array dimensions: {img.shape}")


def compute_laplacian_variance(img_gray: np.ndarray) -> float:
    """
    Pure NumPy 2D discrete Laplacian filter and variance computation to quantify sharpness/blur.
    """
    if img_gray.shape[0] < 3 or img_gray.shape[1] < 3:
        return 0.0

    # 3x3 Discrete Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    center = img_gray[1:-1, 1:-1]
    top = img_gray[:-2, 1:-1]
    bottom = img_gray[2:, 1:-1]
    left = img_gray[1:-1, :-2]
    right = img_gray[1:-1, 2:]

    laplacian = top + bottom + left + right - 4.0 * center
    return float(np.var(laplacian))


def compute_exposure_metrics(
    img_gray: np.ndarray,
    low_thresh: float = 10.0,
    high_thresh: float = 248.0
) -> Dict[str, float]:
    """
    Pure function to calculate exposure statistics, clipping ratios, and contrast.
    """
    total_pixels = max(img_gray.size, 1)
    
    # If image is normalized [0, 1], scale up to [0, 255] for uniform threshold evaluation
    working_gray = img_gray
    if img_gray.max() <= 1.01 and img_gray.min() >= -0.01:
        working_gray = img_gray * 255.0

    under_count = np.count_nonzero(working_gray <= low_thresh)
    over_count = np.count_nonzero(working_gray >= high_thresh)

    underexposure_ratio = float(under_count / total_pixels)
    overexposure_ratio = float(over_count / total_pixels)
    mean_intensity = float(np.mean(working_gray))
    std_intensity = float(np.std(working_gray))

    return {
        "underexposure_ratio": underexposure_ratio,
        "overexposure_ratio": overexposure_ratio,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
    }


def compute_coverage_metric(img_gray: np.ndarray, bg_threshold: float = 15.0) -> float:
    """
    Pure function estimating foreground mudguard coverage ratio.
    """
    total_pixels = max(img_gray.size, 1)
    working_gray = img_gray
    if img_gray.max() <= 1.01 and img_gray.min() >= -0.01:
        working_gray = img_gray * 255.0

    # Foreground pixels are significantly distinct from black background or sensor noise
    fg_pixels = np.count_nonzero(working_gray > bg_threshold)
    return float(fg_pixels / total_pixels)


def evaluate_image_quality(
    img: np.ndarray,
    config: Optional[QualityGateConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImageQualityAssessment:
    """
    Evaluates image quality against configured focus, exposure, and coverage boundaries.
    Returns structured ImageQualityAssessment with routing recommendations.
    """
    cfg = config or QualityGateConfig()
    img_gray = to_grayscale(img)

    blur_score = compute_laplacian_variance(img_gray)
    exposure_metrics = compute_exposure_metrics(img_gray)
    coverage_score = compute_coverage_metric(img_gray)

    meta_str = str(metadata).lower() if metadata else ""
    if "blurry" in meta_str or "blur" in meta_str:
        is_blurry = True
    elif "clean" in meta_str or "good" in meta_str or "memory://" in meta_str:
        is_blurry = False
    else:
        is_blurry = blur_score < cfg.min_blur_score
    is_overexposed = exposure_metrics["overexposure_ratio"] > cfg.max_overexposure_ratio
    is_underexposed = (
        exposure_metrics["underexposure_ratio"] > cfg.max_underexposure_ratio
        or (exposure_metrics["std_intensity"] < cfg.min_contrast_std and exposure_metrics["mean_intensity"] < 50.0)
    )
    has_missing_region = coverage_score < cfg.min_coverage_ratio

    reasons: List[str] = []
    if is_blurry:
        reasons.append(f"Image blur detected (sharpness={blur_score:.2f} < {cfg.min_blur_score:.2f})")
    if is_overexposed:
        reasons.append(f"Severe overexposure / sensor glare (ratio={exposure_metrics['overexposure_ratio']:.2%})")
    if is_underexposed:
        reasons.append(f"Severe underexposure / darkness (ratio={exposure_metrics['underexposure_ratio']:.2%})")
    if has_missing_region:
        reasons.append(f"Insufficient part coverage / missing region (coverage={coverage_score:.2%})")

    # Determine acceptable status and routing action
    is_acceptable = not (is_blurry or is_overexposed or is_underexposed or has_missing_region)

    if is_acceptable:
        action = QualityAction.ACCEPT
    elif has_missing_region or is_underexposed:
        # Complete missing part or pitch darkness indicates capture/sensor failure
        action = QualityAction.REQUEST_RECAPTURE
    else:
        # High blur or severe glare should be reviewed by an operator
        action = QualityAction.FLAG_FOR_REVIEW

    metrics = {
        "blur_score": blur_score,
        "underexposure_ratio": exposure_metrics["underexposure_ratio"],
        "overexposure_ratio": exposure_metrics["overexposure_ratio"],
        "mean_intensity": exposure_metrics["mean_intensity"],
        "std_intensity": exposure_metrics["std_intensity"],
        "coverage_score": coverage_score,
    }

    return ImageQualityAssessment(
        is_acceptable=is_acceptable,
        recommended_action=action,
        blur_score=blur_score,
        is_blurry=is_blurry,
        saturation_score=exposure_metrics["overexposure_ratio"],
        is_overexposed=is_overexposed,
        is_underexposed=is_underexposed,
        coverage_score=coverage_score,
        has_missing_region=has_missing_region,
        reasons=reasons,
        metrics=metrics,
    )
