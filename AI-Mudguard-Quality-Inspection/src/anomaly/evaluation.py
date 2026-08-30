from typing import List, Dict, Any, Tuple
import numpy as np

from src.anomaly.anomaly_protocol import AnomalyDetectorProtocol, AnomalyResult, NoveltyType


class AnomalyEvaluationProtocol:
    """
    Evaluation protocol to assess anomaly detector performance across:
    1. Clean normal parts (False Positive Rate).
    2. Known defects.
    3. Unknown/novel defects (Novelty Detection Rate).
    4. Environmental lighting shifts.
    5. Blank/invalid inputs.
    """

    @staticmethod
    def evaluate(
        detector: AnomalyDetectorProtocol,
        normal_images: List[np.ndarray],
        known_defect_images: List[np.ndarray],
        unknown_defect_images: List[np.ndarray],
        lighting_shift_images: List[np.ndarray],
        blank_invalid_images: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation across all 5 test categories.
        """
        results: Dict[str, List[AnomalyResult]] = {
            "normal": [detector.detect(img) for img in normal_images],
            "known_defects": [detector.detect(img) for img in known_defect_images],
            "unknown_defects": [detector.detect(img) for img in unknown_defect_images],
            "lighting_shifts": [detector.detect(img) for img in lighting_shift_images],
            "blank_invalid": [detector.detect(img) for img in blank_invalid_images],
        }

        # 1. Normal parts FPR
        normal_anom_count = sum(1 for r in results["normal"] if r.is_anomalous)
        fpr = float(normal_anom_count / len(normal_images)) if normal_images else 0.0

        # 2. Unknown defects detection rate
        unknown_detected = sum(1 for r in results["unknown_defects"] if r.is_anomalous)
        unknown_detection_rate = float(unknown_detected / len(unknown_defect_images)) if unknown_defect_images else 0.0

        # 3. Lighting shift detection rate
        shift_detected = sum(1 for r in results["lighting_shifts"] if r.novelty_type == NoveltyType.LIGHTING_SHIFT)
        shift_detection_rate = float(shift_detected / len(lighting_shift_images)) if lighting_shift_images else 0.0

        # 4. Blank/invalid detection rate
        blank_detected = sum(1 for r in results["blank_invalid"] if r.novelty_type == NoveltyType.BLANK_OR_INVALID)
        blank_detection_rate = float(blank_detected / len(blank_invalid_images)) if blank_invalid_images else 0.0

        # 5. Known defects detection rate
        known_detected = sum(1 for r in results["known_defects"] if r.is_anomalous)
        known_detection_rate = float(known_detected / len(known_defect_images)) if known_defect_images else 0.0

        # Compute simple AUROC (binary discrimination between normal vs all defects/anomalies)
        y_true = [0] * len(normal_images) + [1] * (len(known_defect_images) + len(unknown_defect_images) + len(blank_invalid_images))
        y_scores = (
            [r.anomaly_score for r in results["normal"]]
            + [r.anomaly_score for r in results["known_defects"]]
            + [r.anomaly_score for r in results["unknown_defects"]]
            + [r.anomaly_score for r in results["blank_invalid"]]
        )

        auroc = AnomalyEvaluationProtocol.calculate_auroc(y_true, y_scores)

        return {
            "false_positive_rate_on_normal": fpr,
            "unknown_defects_detection_rate": unknown_detection_rate,
            "known_defects_anomaly_rate": known_detection_rate,
            "lighting_shift_detection_rate": shift_detection_rate,
            "blank_invalid_detection_rate": blank_detection_rate,
            "auroc": auroc,
            "sample_counts": {k: len(v) for k, v in results.items()},
        }

    @staticmethod
    def calculate_auroc(y_true: List[int], y_scores: List[float]) -> float:
        """Calculate Area Under the ROC Curve without external dependencies."""
        if not y_true or not y_scores or len(set(y_true)) < 2:
            return 1.0

        # Rank-based Mann-Whitney U calculation of AUROC
        pos_scores = [s for t, s in zip(y_true, y_scores) if t == 1]
        neg_scores = [s for t, s in zip(y_true, y_scores) if t == 0]

        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return 1.0

        u_stat = sum(
            1.0 if p > n else 0.5 if p == n else 0.0
            for p in pos_scores
            for n in neg_scores
        )
        return float(u_stat / (n_pos * n_neg))
