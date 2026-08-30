import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from src.anomaly.anomaly_protocol import (
    AnomalyDetectorProtocol,
    AnomalyResult,
    NoveltyType,
)
from src.preprocessing.quality_gate import to_grayscale


class NoveltyAnomalyDetector:
    """
    Novelty and unknown-defect anomaly detector implementing AnomalyDetectorProtocol.
    Learns normal-part feature representations from training data and identifies
    unseen/out-of-distribution defects without forcing them into a known class.
    """

    def __init__(
        self,
        patch_size: int = 16,
        anomaly_threshold: float = 0.50,
        mock_mode: bool = False,
    ):
        self.patch_size = patch_size
        self.anomaly_threshold = anomaly_threshold
        self.mock_mode = mock_mode

        # Learned nominal representations
        self.nominal_mean_features: Optional[np.ndarray] = None
        self.nominal_std_features: Optional[np.ndarray] = None
        self.nominal_global_mean: float = 128.0
        self.is_fitted: bool = False

    def _extract_patch_features(self, img_gray: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Extracts statistical and gradient features across sliding patches.
        Returns: (features_array [N, D], patch_coordinates [[x1, y1, x2, y2], ...])
        """
        h, w = img_gray.shape[:2]
        p = self.patch_size
        features_list: List[np.ndarray] = []
        coords_list: List[List[int]] = []

        img_mean = float(np.mean(img_gray))
        for y in range(0, h - p + 1, p // 2):
            for x in range(0, w - p + 1, p // 2):
                patch = img_gray[y : y + p, x : x + p]
                # Feature vector: [relative_mean, std, grad_x_mean, grad_y_mean, range]
                p_mean = float(np.mean(patch))
                p_std = float(np.std(patch))
                grad_x = float(np.mean(np.abs(patch[:, 1:] - patch[:, :-1]))) if p > 1 else 0.0
                grad_y = float(np.mean(np.abs(patch[1:, :] - patch[:-1, :]))) if p > 1 else 0.0
                p_range = float(np.max(patch) - np.min(patch))

                feat = np.array([p_mean - img_mean, p_std, grad_x, grad_y, p_range], dtype=np.float32)
                features_list.append(feat)
                coords_list.append([x, y, x + p, y + p])

        if not features_list:
            # Fallback for very small images
            return np.zeros((1, 5), dtype=np.float32), [[0, 0, w, h]]

        return np.stack(features_list), coords_list

    def fit(self, normal_images: List[np.ndarray]) -> None:
        """
        Fit nominal representation strictly on clean, normal images.
        """
        if not normal_images:
            return

        all_features: List[np.ndarray] = []
        global_means: List[float] = []

        for img in normal_images:
            img_gray = to_grayscale(img)
            feats, _ = self._extract_patch_features(img_gray)
            all_features.append(feats)
            global_means.append(float(np.mean(img_gray)))

        stacked = np.concatenate(all_features, axis=0)
        self.nominal_mean_features = np.mean(stacked, axis=0)
        # Ensure standard deviation has safe minimum floor to avoid exploding z-scores
        self.nominal_std_features = np.maximum(np.std(stacked, axis=0), 2.0)
        self.nominal_global_mean = float(np.mean(global_means))

        # Compute baseline nominal distance on training set
        train_z = np.abs((stacked - self.nominal_mean_features) / self.nominal_std_features)
        train_distances = np.mean(train_z, axis=1)
        self.nominal_baseline_dist = float(np.percentile(train_distances, 95.0))
        self.is_fitted = True

    def calibrate_thresholds(
        self,
        val_normal_scores: List[float],
        val_anomaly_scores: Optional[List[float]] = None,
        target_fpr: float = 0.05,
    ) -> float:
        """
        Calibrate detection threshold on validation data to achieve target False Positive Rate.
        """
        if not val_normal_scores:
            return self.anomaly_threshold

        # Pick threshold corresponding to (1 - target_fpr) percentile of normal scores
        calibrated_thresh = float(np.percentile(val_normal_scores, 100.0 * (1.0 - target_fpr)))
        self.anomaly_threshold = float(np.clip(calibrated_thresh, 0.10, 0.90))
        return self.anomaly_threshold

    def detect(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnomalyResult:
        """
        Detect unknown anomalies, foreign defects, or lighting shifts.
        """
        start_time = time.perf_counter()

        if self.mock_mode:
            mock_score = metadata.get("mock_anomaly_score", 0.05) if metadata else 0.05
            mock_is_anom = mock_score >= self.anomaly_threshold
            mock_type = NoveltyType(metadata.get("mock_novelty_type", "NOMINAL")) if metadata else NoveltyType.NOMINAL
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AnomalyResult(
                anomaly_score=mock_score,
                is_anomalous=mock_is_anom,
                novelty_type=mock_type,
                confidence=0.90,
                metrics={"mock": 1.0},
                runtime_ms=elapsed_ms,
            )

        img_gray = to_grayscale(image)
        h, w = img_gray.shape[:2]

        # 1. Check for blank or invalid input
        img_std = float(np.std(img_gray))
        img_mean = float(np.mean(img_gray))
        if img_mean < 5.0 or img_mean > 250.0 or (img_std < 1.0 and img_mean < 30.0):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AnomalyResult(
                anomaly_score=0.95,
                is_anomalous=True,
                novelty_type=NoveltyType.BLANK_OR_INVALID,
                confidence=0.99,
                metrics={"img_std": img_std, "img_mean": img_mean},
                runtime_ms=elapsed_ms,
            )

        # 2. Extract patch features
        patch_feats, coords = self._extract_patch_features(img_gray)

        # Compute z-score distance against nominal representation if fitted
        if self.is_fitted and self.nominal_mean_features is not None:
            z_scores = np.abs((patch_feats - self.nominal_mean_features) / self.nominal_std_features)
            patch_distances = np.mean(z_scores, axis=1)
            ref_scale = max(getattr(self, "nominal_baseline_dist", 2.0) * 3.5, 4.0)
        else:
            # Fallback baseline feature deviation
            patch_distances = patch_feats[:, 1] / 30.0  # Normalized patch std
            ref_scale = 3.0

        # Top-k patch anomaly score
        top_k = min(5, len(patch_distances))
        top_distances = np.partition(patch_distances, -top_k)[-top_k:]
        raw_anomaly_score = float(np.mean(top_distances) / ref_scale)
        anomaly_score = float(np.clip(raw_anomaly_score, 0.0, 1.0))

        # 3. Check for Global Lighting Shift
        global_mean_diff = abs(img_mean - self.nominal_global_mean)
        is_lighting_shift = global_mean_diff > 45.0 and anomaly_score < 0.65

        # 4. Determine Novelty Type and Anomaly Region
        is_anomalous = anomaly_score >= self.anomaly_threshold or is_lighting_shift
        anomaly_region: Optional[List[float]] = None

        if is_lighting_shift:
            novelty_type = NoveltyType.LIGHTING_SHIFT
        elif is_anomalous:
            novelty_type = NoveltyType.UNKNOWN_DEFECT
            # Bounding box of most anomalous patch
            max_idx = int(np.argmax(patch_distances))
            c = coords[max_idx]
            anomaly_region = [float(c[0]), float(c[1]), float(c[2]), float(c[3])]
        else:
            novelty_type = NoveltyType.NOMINAL

        confidence = float(np.clip(0.70 + abs(anomaly_score - self.anomaly_threshold) * 0.5, 0.5, 0.98))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            novelty_type=novelty_type,
            confidence=confidence,
            anomaly_region=anomaly_region,
            metrics={
                "max_patch_distance": float(np.max(patch_distances)),
                "mean_patch_distance": float(np.mean(patch_distances)),
                "global_mean_diff": global_mean_diff,
            },
            runtime_ms=elapsed_ms,
        )
