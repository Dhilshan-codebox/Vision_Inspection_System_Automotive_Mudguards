from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from PIL import Image


def to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Ensures input image is in (H, W, C) uint8 layout [0, 255].
    """
    if img.ndim == 2:
        # Expand 2D grayscale to HWC
        img = np.expand_dims(img, axis=-1)

    if img.ndim == 3 and img.shape[0] in {1, 3, 4} and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        # Convert CHW to HWC
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        if img.max() <= 1.01 and img.min() >= -0.01:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    return img


def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    resample: Image.Resampling = Image.Resampling.BILINEAR
) -> np.ndarray:
    """
    Pure deterministic image resize to (target_height, target_width).
    """
    hwc = to_hwc_uint8(img)
    target_h, target_w = target_size

    pil_img = Image.fromarray(hwc if hwc.shape[2] > 1 else hwc[:, :, 0])
    resized_pil = pil_img.resize((target_w, target_h), resample=resample)
    resized_arr = np.array(resized_pil)

    if resized_arr.ndim == 2 and hwc.shape[2] == 1:
        resized_arr = np.expand_dims(resized_arr, axis=-1)

    return resized_arr


def letterbox_pad(
    img: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_value: int = 114,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resizes image maintaining aspect ratio and pads symmetrically to target_size (height, width).
    Returns (padded_image, metadata).
    """
    hwc = to_hwc_uint8(img)
    orig_h, orig_w = hwc.shape[:2]
    target_h, target_w = target_size

    # Compute scaling ratio
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))

    # Resize using PIL
    pil_img = Image.fromarray(hwc if hwc.shape[2] > 1 else hwc[:, :, 0])
    resized_pil = pil_img.resize((new_w, new_h), resample=resample)
    resized_arr = np.array(resized_pil)
    if resized_arr.ndim == 2 and hwc.shape[2] == 1:
        resized_arr = np.expand_dims(resized_arr, axis=-1)

    # Compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Create padded container
    channels = hwc.shape[2]
    if channels == 1:
        padded = np.full((target_h, target_w, 1), pad_value, dtype=np.uint8)
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized_arr
    else:
        padded = np.full((target_h, target_w, channels), pad_value, dtype=np.uint8)
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized_arr

    meta = {
        "ratio": ratio,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "original_shape": (orig_h, orig_w),
        "resized_shape": (new_h, new_w),
        "target_shape": (target_h, target_w),
    }

    return padded, meta


def normalize_image(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    channel_first: bool = True,
) -> np.ndarray:
    """
    Converts uint8 [0, 255] or float to normalized float32 tensor with specified mean and std.
    """
    hwc = to_hwc_uint8(img)
    arr = hwc.astype(np.float32) / 255.0

    channels = arr.shape[2]
    mean_arr = np.array(mean[:channels], dtype=np.float32)
    std_arr = np.array(std[:channels], dtype=np.float32)

    normalized = (arr - mean_arr) / std_arr

    if channel_first:
        normalized = np.transpose(normalized, (2, 0, 1))

    return normalized.astype(np.float32)


def extract_roi(
    img: np.ndarray,
    bbox: List[int | float],
    is_normalized: bool = False,
) -> np.ndarray:
    """
    Extracts region of interest bounded by [x1, y1, x2, y2].
    """
    hwc = to_hwc_uint8(img)
    h, w = hwc.shape[:2]

    x1, y1, x2, y2 = bbox
    if is_normalized:
        x1 = x1 * w
        x2 = x2 * w
        y1 = y1 * h
        y2 = y2 * h

    x1_idx = max(0, min(w - 1, int(round(x1))))
    x2_idx = max(0, min(w, int(round(x2))))
    y1_idx = max(0, min(h - 1, int(round(y1))))
    y2_idx = max(0, min(h, int(round(y2))))

    if x2_idx <= x1_idx or y2_idx <= y1_idx:
        return np.zeros((0, 0, hwc.shape[2]), dtype=hwc.dtype)

    return hwc[y1_idx:y2_idx, x1_idx:x2_idx, :]


def create_tiles(
    img: np.ndarray,
    tile_size: Tuple[int, int] = (256, 256),
    overlap: int = 32,
) -> List[Tuple[np.ndarray, List[int]]]:
    """
    Splits image into overlapping grid tiles for high-detail local defect analysis.
    Returns list of (tile_array, [x1, y1, x2, y2]).
    """
    hwc = to_hwc_uint8(img)
    h, w = hwc.shape[:2]
    tile_h, tile_w = tile_size

    step_y = max(1, tile_h - overlap)
    step_x = max(1, tile_w - overlap)

    tiles: List[Tuple[np.ndarray, List[int]]] = []

    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            # Clamp right and bottom edges to image boundary
            x1 = x
            y1 = y
            x2 = min(w, x1 + tile_w)
            y2 = min(h, y1 + tile_h)

            # If tile touches edge and is smaller than tile_size, shift back
            if x2 - x1 < tile_w and w >= tile_w:
                x1 = w - tile_w
                x2 = w
            if y2 - y1 < tile_h and h >= tile_h:
                y1 = h - tile_h
                y2 = h

            tile_crop = hwc[y1:y2, x1:x2, :]
            tiles.append((tile_crop, [x1, y1, x2, y2]))

    return tiles


def preprocess_image(
    img: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    channel_first: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Unified deterministic preprocessing pipeline function.
    Resizes/letterboxes, normalizes, and reorders dimensions for neural network input.
    """
    padded, _ = letterbox_pad(img, target_size=target_size)
    if normalize:
        return normalize_image(padded, mean=mean, std=std, channel_first=channel_first)
    if channel_first:
        return np.transpose(padded, (2, 0, 1))
    return padded
