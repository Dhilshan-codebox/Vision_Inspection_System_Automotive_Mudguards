import pytest
import numpy as np
from PIL import Image, ImageFilter

from src.preprocessing import (
    QualityGateConfig,
    to_grayscale,
    compute_laplacian_variance,
    compute_exposure_metrics,
    compute_coverage_metric,
    evaluate_image_quality,
    resize_image,
    letterbox_pad,
    normalize_image,
    extract_roi,
    create_tiles,
    preprocess_image,
)
from src.data.contracts import QualityAction


@pytest.fixture
def sharp_image() -> np.ndarray:
    """Creates a high-contrast synthetic image with sharp lines and edges."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Mudguard body
    img[50:250, 50:350, :] = [180, 180, 180]
    # Edge patterns / text
    img[100:120, 100:300, :] = [20, 20, 20]
    img[150:200, 120:140, :] = [240, 40, 40]
    return img


@pytest.fixture
def blurry_image(sharp_image: np.ndarray) -> np.ndarray:
    """Creates a heavily blurred version of the sharp image."""
    pil_img = Image.fromarray(sharp_image)
    # Apply strong Gaussian blur
    blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=10))
    return np.array(blurred_pil)


def test_deterministic_preprocessing(sharp_image: np.ndarray):
    """Verify that preprocessing is strictly deterministic."""
    out1 = preprocess_image(sharp_image, target_size=(320, 320), normalize=True)
    out2 = preprocess_image(sharp_image, target_size=(320, 320), normalize=True)

    assert np.array_equal(out1, out2)
    assert out1.shape == (3, 320, 320)
    assert out1.dtype == np.float32


def test_channel_order_and_range(sharp_image: np.ndarray):
    """Test channel transposition and normalization ranges."""
    # CHW normalized float32
    chw_norm = normalize_image(sharp_image, channel_first=True)
    assert chw_norm.shape == (3, 300, 400)
    assert chw_norm.dtype == np.float32

    # HWC normalized float32
    hwc_norm = normalize_image(sharp_image, channel_first=False)
    assert hwc_norm.shape == (300, 400, 3)

    # Values should be centered around ImageNet statistics
    assert -3.0 < hwc_norm.min() < 0.0
    assert 0.0 < hwc_norm.max() < 4.0


def test_letterbox_pad_aspect_ratio():
    """Verify letterbox padding maintains aspect ratio without distortion."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]

    padded, meta = letterbox_pad(img, target_size=(300, 300), pad_value=114)
    assert padded.shape == (300, 300, 3)
    assert meta["resized_shape"] == (150, 300)
    assert meta["pad_top"] == 75
    assert meta["pad_left"] == 0
    assert np.all(padded[0:70, :, :] == 114)


def test_roi_and_tile_extraction(sharp_image: np.ndarray):
    """Test region-of-interest cropping and overlapping tile generation."""
    roi = extract_roi(sharp_image, bbox=[50, 50, 150, 150])
    assert roi.shape == (100, 100, 3)

    tiles = create_tiles(sharp_image, tile_size=(128, 128), overlap=16)
    assert len(tiles) > 1
    for tile_arr, coords in tiles:
        assert tile_arr.shape == (128, 128, 3)
        assert coords[2] - coords[0] == 128
        assert coords[3] - coords[1] == 128


def test_quality_gate_sharp_image_passes(sharp_image: np.ndarray):
    """Test that a clear, sharp, well-exposed image passes the quality gate."""
    assessment = evaluate_image_quality(sharp_image)
    assert assessment.is_acceptable
    assert assessment.recommended_action == QualityAction.PROCEED_TO_INFERENCE
    assert not assessment.is_blurry
    assert not assessment.is_overexposed
    assert not assessment.is_underexposed
    assert not assessment.has_missing_region
    assert len(assessment.reasons) == 0


def test_quality_gate_blurry_image_fails(sharp_image: np.ndarray, blurry_image: np.ndarray):
    """Test that severe blur is caught and routed to review."""
    sharp_eval = evaluate_image_quality(sharp_image)
    blur_eval = evaluate_image_quality(blurry_image)

    assert sharp_eval.blur_score > blur_eval.blur_score
    assert blur_eval.is_blurry
    assert not blur_eval.is_acceptable
    assert blur_eval.recommended_action == QualityAction.FLAG_FOR_REVIEW
    assert any("blur" in r.lower() for r in blur_eval.reasons)


def test_quality_gate_overexposed():
    """Test that sensor glare / overexposure is flagged."""
    overexposed = np.full((200, 200, 3), 255, dtype=np.uint8)
    assessment = evaluate_image_quality(overexposed)

    assert not assessment.is_acceptable
    assert assessment.is_overexposed
    assert assessment.recommended_action == QualityAction.FLAG_FOR_REVIEW


def test_quality_gate_underexposed_and_missing_region():
    """Test that pitch black or empty frames request recapture."""
    black_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assessment = evaluate_image_quality(black_frame)

    assert not assessment.is_acceptable
    assert assessment.is_underexposed
    assert assessment.has_missing_region
    assert assessment.recommended_action == QualityAction.REQUEST_RECAPTURE


def test_grayscale_and_single_channel():
    """Test handling of 2D grayscale, single channel, and 4-channel inputs."""
    gray_2d = np.full((100, 100), 128, dtype=np.uint8)
    padded, meta = letterbox_pad(gray_2d, target_size=(200, 200))
    assert padded.shape == (200, 200, 1)

    rgba = np.full((100, 100, 4), 128, dtype=np.uint8)
    gray = to_grayscale(rgba)
    assert gray.shape == (100, 100)


def test_empty_image_handling():
    """Test graceful handling of zero-sized or tiny image cropping."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_roi = extract_roi(img, bbox=[10, 10, 10, 10])
    assert invalid_roi.shape[0] == 0 or invalid_roi.shape[1] == 0
