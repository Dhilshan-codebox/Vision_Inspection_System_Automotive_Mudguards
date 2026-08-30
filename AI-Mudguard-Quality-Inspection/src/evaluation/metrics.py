from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np


class EvaluationEngine:
    """
    Comprehensive evaluation metrics calculator for vision inspection systems.
    Computes per-class precision/recall, confusion matrix, localization IoU,
    Expected Calibration Error (ECE), and p50/p95 latency percentiles.
    """

    @staticmethod
    def compute_confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Computes 2D confusion matrix mapping actual to predicted classes.
        """
        all_classes = sorted(list(set(y_true + y_pred + (classes or []))))
        matrix = {c_true: {c_pred: 0 for c_pred in all_classes} for c_true in all_classes}

        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1

        return {
            "classes": all_classes,
            "matrix": matrix,
        }

    @staticmethod
    def compute_classification_metrics(
        y_true: List[str],
        y_pred: List[str],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Computes per-class Precision, Recall, F1-Score, and overall Macro/Weighted averages.
        """
        all_classes = sorted(list(set(y_true + y_pred + (classes or []))))
        per_class: Dict[str, Dict[str, float]] = {}

        for cls in all_classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            support = sum(1 for t in y_true if t == cls)

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }

        # Calculate macro averages
        valid_classes = [c for c in all_classes if per_class[c]["support"] > 0]
        if valid_classes:
            macro_precision = float(np.mean([per_class[c]["precision"] for c in valid_classes]))
            macro_recall = float(np.mean([per_class[c]["recall"] for c in valid_classes]))
            macro_f1 = float(np.mean([per_class[c]["f1_score"] for c in valid_classes]))
        else:
            macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0

        return {
            "per_class": per_class,
            "macro_avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1_score": macro_f1,
            },
            "total_samples": len(y_true),
        }

    @staticmethod
    def compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        union_area = boxA_area + boxB_area - inter_area
        return float(inter_area / union_area) if union_area > 0 else 0.0

    @staticmethod
    def compute_localization_quality(
        true_boxes: List[Optional[List[float]]],
        pred_boxes: List[Optional[List[float]]],
        iou_threshold: float = 0.50
    ) -> Dict[str, float]:
        """
        Evaluates bounding box localization accuracy and mean IoU.
        """
        ious: List[float] = []
        matches = 0
        total_eval = 0

        for t_box, p_box in zip(true_boxes, pred_boxes):
            if t_box is not None and p_box is not None:
                iou = EvaluationEngine.compute_iou(t_box, p_box)
                ious.append(iou)
                if iou >= iou_threshold:
                    matches += 1
                total_eval += 1

        mean_iou = float(np.mean(ious)) if ious else 0.0
        loc_accuracy = float(matches / total_eval) if total_eval > 0 else 0.0

        return {
            "mean_iou": mean_iou,
            "localization_accuracy": loc_accuracy,
            "iou_threshold": iou_threshold,
            "evaluated_pairs": total_eval,
        }

    @staticmethod
    def compute_confidence_calibration(
        y_true_binary: List[int],
        y_confidence: List[float],
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculates Expected Calibration Error (ECE) and reliability diagram bin statistics.
        y_true_binary: 1 if prediction was correct, 0 if incorrect.
        y_confidence: predicted confidence [0.0, 1.0].
        """
        if not y_true_binary or not y_confidence:
            return {"ece": 0.0, "bins": []}

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        total_samples = len(y_confidence)
        ece = 0.0
        bins_data: List[Dict[str, Any]] = []

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = [
                (t, c) for t, c in zip(y_true_binary, y_confidence)
                if bin_lower <= c < bin_upper or (i == num_bins - 1 and c == bin_upper)
            ]

            bin_size = len(in_bin)
            if bin_size > 0:
                bin_acc = float(np.mean([t for t, _ in in_bin]))
                bin_conf = float(np.mean([c for _, c in in_bin]))
                bin_error = abs(bin_acc - bin_conf)
                ece += (bin_size / total_samples) * bin_error

                bins_data.append({
                    "bin_range": (round(bin_lower, 2), round(bin_upper, 2)),
                    "count": bin_size,
                    "accuracy": bin_acc,
                    "confidence": bin_conf,
                    "error": bin_error,
                })

        return {
            "ece": float(ece),
            "num_bins": num_bins,
            "bins": bins_data,
        }

    @staticmethod
    def compute_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
        """
        Calculates p50 (median), p90, p95, p99, and mean latency percentiles in ms.
        """
        if not latencies_ms:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

        arr = np.array(latencies_ms, dtype=np.float64)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sample_count": len(latencies_ms),
        }
