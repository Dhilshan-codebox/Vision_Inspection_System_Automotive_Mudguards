import time
from typing import Optional, Dict, Any, List
import numpy as np

from src.data.contracts import Prediction, ModelMetadata
from src.detection.detector_protocol import (
    DetectorProtocol,
    DetectionOutput,
    DetectionItem,
)
from src.detection.backbone import SharedBackbone, BackboneOutput
from src.preprocessing.quality_gate import to_grayscale


DEFAULT_CLASSES: List[str] = [
    "Scratch",
    "Dent",
    "Paint Defect",
    "Paint Misalignment",
    "Good",
]


class RGBBaselineDetector:
    """
    Baseline 4-class RGB detector and localization engine implementing DetectorProtocol.
    Capable of operating with neural backbone checkpoints or deterministic feature extractors.
    """

    def __init__(
        self,
        version: str = "1.0.0",
        name: str = "RGB_Baseline_Detector",
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.50,
        checkpoint_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.version = version
        self.name = name
        self.classes = classes or DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = checkpoint_path
        self.mock_mode = mock_mode

        self.thresholds: Dict[str, float] = {
            "confidence_threshold": self.confidence_threshold,
            "scratch": 0.50,
            "dent": 0.55,
            "paint_defect": 0.50,
            "paint_misalignment": 0.60,
        }

    def get_model_metadata(self) -> ModelMetadata:
        """Return standardized model metadata contract."""
        return ModelMetadata(
            version=self.version,
            name=self.name,
            thresholds=self.thresholds,
        )

    def predict(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionOutput:
        """
        Execute forward pass inference on image array.
        """
        start_time = time.perf_counter()

        if self.mock_mode:
            mock_label = metadata.get("mock_label", "Good") if metadata else "Good"
            mock_conf = metadata.get("mock_confidence", 0.94) if metadata else 0.94
            mock_bbox = metadata.get("mock_bbox", [100.0, 100.0, 200.0, 200.0]) if mock_label != "Good" else None

            probs = {c: 0.05 for c in self.classes}
            probs[mock_label] = mock_conf

            detections: List[DetectionItem] = []
            if mock_bbox and mock_label != "Good":
                detections.append(DetectionItem(class_name=mock_label, confidence=mock_conf, bbox=mock_bbox))

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return DetectionOutput(
                primary_prediction=Prediction(label=mock_label, confidence=mock_conf),
                detections=detections,
                class_probabilities=probs,
                latency_ms=elapsed_ms,
            )

        # Realistic feature-based image analysis
        img_gray = to_grayscale(image)
        h, w = img_gray.shape[:2]

        # Extract multi-channel features
        grad_x = np.abs(img_gray[:, 1:] - img_gray[:, :-1])
        grad_y = np.abs(img_gray[1:, :] - img_gray[:-1, :])
        mean_grad = float(np.mean(grad_x) + np.mean(grad_y))

        # 1. Scratch feature: high narrow edge gradients
        scratch_score = float(np.clip(mean_grad / 60.0, 0.02, 0.98))

        # 2. Dent feature: smooth parabolic intensity gradient across surface
        center_patch = img_gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        dent_score = float(np.clip(np.std(center_patch) / 80.0, 0.02, 0.98))

        # 3. Paint Defect feature: local color/brightness spots
        spots_count = np.count_nonzero(np.abs(img_gray - np.mean(img_gray)) > 50.0)
        paint_defect_score = float(np.clip(spots_count / max(img_gray.size * 0.02, 1), 0.02, 0.98))

        # 4. Paint Misalignment feature: boundary edge variance
        edge_pixels = np.concatenate([img_gray[0, :], img_gray[-1, :], img_gray[:, 0], img_gray[:, -1]])
        misalignment_score = float(np.clip(np.std(edge_pixels) / 70.0, 0.02, 0.98))

        # 5. Good score
        max_defect = max(scratch_score, dent_score, paint_defect_score, misalignment_score)
        good_score = float(np.clip(1.0 - max_defect, 0.02, 0.98))

        raw_scores = {
            "Scratch": scratch_score,
            "Dent": dent_score,
            "Paint Defect": paint_defect_score,
            "Paint Misalignment": misalignment_score,
            "Good": good_score,
        }

        # Softmax normalization over class probabilities
        exp_scores = {k: np.exp(v * 2.0) for k, v in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        class_probs = {k: float(v / sum_exp) for k, v in exp_scores.items()}

        # Top prediction
        top_label = max(class_probs, key=class_probs.get)
        top_confidence = class_probs[top_label]

        # Precise localized bounding box calculation
        detections: List[DetectionItem] = []
        if top_label != "Good" and top_confidence >= self.confidence_threshold:
            bbox = None
            try:
                from scipy.ndimage import gaussian_filter, label
                blur_bg = gaussian_filter(img_gray.astype(np.float32), sigma=15)
                diff = np.abs(img_gray.astype(np.float32) - blur_bg)

                grad = np.zeros((h, w), dtype=np.float32)
                grad[:, :-1] += grad_x
                grad[:-1, :] += grad_y

                defect_mask = (diff > 25.0) & (grad < 40.0) & (img_gray > 20) & (img_gray < 250)
                labeled, num_features = label(defect_mask)

                best_bbox = None
                max_defect_size = 0

                for i in range(1, num_features + 1):
                    mask_i = (labeled == i)
                    sz = np.sum(mask_i)
                    if 15 < sz < (h * w * 0.3):
                        if sz > max_defect_size:
                            ys, xs = np.nonzero(mask_i)
                            max_defect_size = sz
                            best_bbox = [
                                float(np.min(xs)),
                                float(np.min(ys)),
                                float(np.max(xs) + 1),
                                float(np.max(ys) + 1),
                            ]

                if best_bbox:
                    bbox = best_bbox
                else:
                    high_mask = grad_x > np.percentile(grad_x, 92)
                    if np.count_nonzero(high_mask) > 0:
                        y_idx, x_idx = np.nonzero(high_mask)
                        bbox = [
                            float(np.min(x_idx)),
                            float(np.min(y_idx)),
                            float(np.max(x_idx) + 1),
                            float(np.max(y_idx) + 1),
                        ]
                    else:
                        bbox = [float(w * 0.3), float(h * 0.3), float(w * 0.7), float(h * 0.7)]
            except Exception:
                high_mask = grad_x > np.percentile(grad_x, 90)
                if np.count_nonzero(high_mask) > 0:
                    y_idx, x_idx = np.nonzero(high_mask)
                    bbox = [
                        float(np.min(x_idx)),
                        float(np.min(y_idx)),
                        float(np.max(x_idx) + 1),
                        float(np.max(y_idx) + 1),
                    ]
                else:
                    bbox = [float(w * 0.3), float(h * 0.3), float(w * 0.7), float(h * 0.7)]

            detections.append(
                DetectionItem(
                    class_name=top_label,
                    confidence=top_confidence,
                    bbox=bbox,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionOutput(
            primary_prediction=Prediction(label=top_label, confidence=top_confidence),
            detections=detections,
            class_probabilities=class_probs,
            features=np.array(list(raw_scores.values()), dtype=np.float32),
            latency_ms=elapsed_ms,
        )


class BackboneDetector:
    """
    Detector implementation powered by the multi-head SharedBackbone.
    Conforms to DetectorProtocol while enriching output with severity,
    embedding, image quality, and geometry residual predictions.
    """

    def __init__(
        self,
        version: str = "2.0.0",
        name: str = "Backbone_Detector",
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.50,
        checkpoint_path: Optional[str] = None,
        embedding_dim: int = 128,
        device: str = "cpu",
        mock_mode: bool = False,
    ):
        self.version = version
        self.name = name
        self.classes = classes or DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = checkpoint_path
        self.mock_mode = mock_mode

        self.backbone = SharedBackbone(
            classes=self.classes,
            embedding_dim=embedding_dim,
            device=device,
            checkpoint_path=checkpoint_path,
            mock_mode=mock_mode,
        )

        self.thresholds: Dict[str, float] = {
            "confidence_threshold": self.confidence_threshold,
            "scratch": 0.50,
            "dent": 0.55,
            "paint_defect": 0.50,
            "paint_misalignment": 0.60,
        }

    def get_model_metadata(self) -> ModelMetadata:
        """Return standardized model metadata contract."""
        return ModelMetadata(
            version=self.version,
            name=self.name,
            thresholds=self.thresholds,
        )

    def predict(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionOutput:
        """
        Execute forward prediction using SharedBackbone.
        """
        if self.mock_mode:
            mock_label = metadata.get("mock_label", "Good") if metadata else "Good"
            mock_conf = metadata.get("mock_confidence", 0.94) if metadata else 0.94
            mock_bbox = metadata.get("mock_bbox", [100.0, 100.0, 200.0, 200.0]) if mock_label != "Good" else None

            probs = {c: 0.05 for c in self.classes}
            probs[mock_label] = mock_conf

            detections: List[DetectionItem] = []
            if mock_bbox and mock_label != "Good":
                detections.append(DetectionItem(class_name=mock_label, confidence=mock_conf, bbox=mock_bbox))

            return DetectionOutput(
                primary_prediction=Prediction(label=mock_label, confidence=mock_conf),
                detections=detections,
                class_probabilities=probs,
                severity_level="none" if mock_label == "Good" else "medium",
                severity_score=0.0 if mock_label == "Good" else 0.5,
                embedding=np.zeros(128, dtype=np.float32),
                quality_scores={"is_acceptable_prob": 0.99, "blur_score": 100.0, "exposure_score": 0.5},
                geometry_residual=0.0,
                latency_ms=1.0,
            )

        out: BackboneOutput = self.backbone.forward(image)

        detections: List[DetectionItem] = []
        if out.bbox_hint and out.top_class != "Good" and out.top_confidence >= self.confidence_threshold:
            detections.append(
                DetectionItem(
                    class_name=out.top_class,
                    confidence=out.top_confidence,
                    bbox=out.bbox_hint,
                )
            )

        return DetectionOutput(
            primary_prediction=Prediction(
                label=out.top_class,
                confidence=out.top_confidence,
            ),
            detections=detections,
            class_probabilities=out.defect_class_probs,
            features=out.features,
            severity_level=out.severity_level,
            severity_score=out.severity_score,
            embedding=out.embedding,
            quality_scores=out.quality_scores,
            geometry_residual=out.geometry_residual,
            latency_ms=out.latency_ms,
        )


def create_detector(
    detector_type: str = "backbone",
    version: str = "2.0.0",
    confidence_threshold: float = 0.50,
    checkpoint_path: Optional[str] = None,
    mock_mode: bool = False,
) -> DetectorProtocol:
    """Factory function to instantiate detector instances."""
    if detector_type.lower() == "baseline":
        return RGBBaselineDetector(
            version=version,
            confidence_threshold=confidence_threshold,
            checkpoint_path=checkpoint_path,
            mock_mode=mock_mode,
        )
    return BackboneDetector(
        version=version,
        confidence_threshold=confidence_threshold,
        checkpoint_path=checkpoint_path,
        mock_mode=mock_mode,
    )
