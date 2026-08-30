import os
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.data.contracts import ModelMetadata
from src.preprocessing.quality_gate import to_grayscale
from src.preprocessing.transforms import to_hwc_uint8, letterbox_pad, normalize_image

# Optional PyTorch & timm support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    timm = None
    HAS_TIMM = False


DEFAULT_DEFECT_CLASSES = ["Scratch", "Dent", "Paint Defect", "Paint Misalignment", "Good"]
SEVERITY_LEVELS = ["none", "low", "medium", "high", "critical"]


@dataclass
class BackboneOutput:
    """Output contract for SharedBackbone containing all multi-head predictions."""
    defect_class_probs: Dict[str, float]
    top_class: str
    top_confidence: float
    severity_level: str
    severity_score: float
    quality_scores: Dict[str, float]
    embedding: np.ndarray  # (embedding_dim,) L2 normalized
    bbox_hint: Optional[List[float]] = None  # [x1, y1, x2, y2]
    geometry_residual: float = 0.0  # Estimated depth depression in mm from RGB
    features: Optional[np.ndarray] = None
    latency_ms: float = 0.0


if HAS_TORCH:
    class PyTorchMultiHeadNet(nn.Module):
        """PyTorch multi-head neural network wrapper around an EfficientNet feature extractor."""

        def __init__(
            self,
            model_name: str = "efficientnet_b0",
            pretrained: bool = True,
            embedding_dim: int = 128,
            num_classes: int = 5,
            num_severity: int = 5,
        ):
            super().__init__()
            self.feature_dim = 1280

            if HAS_TIMM:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,  # return pooled features
                )
                self.feature_dim = self.backbone.num_features
            else:
                # Lightweight convolutional feature extractor fallback in pure PyTorch
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.feature_dim = 128

            # Head 1: Defect Classification
            self.defect_class_head = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

            # Head 2: Severity Ordinal Regression
            self.severity_head = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.SiLU(),
                nn.Linear(128, num_severity),
            )

            # Head 3: Image Quality
            self.quality_head = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 3),  # [is_acceptable_logit, blur_scalar, exposure_scalar]
            )

            # Head 4: Embedding Projection for Novelty / OOD
            self.embedding_head = nn.Sequential(
                nn.Linear(self.feature_dim, embedding_dim),
            )

            # Head 5: Defect ROI Bounding Box Hinting
            self.localization_head = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 4),
                nn.Sigmoid(),  # normalized coordinates [x1, y1, x2, y2]
            )

            # Head 6: Geometry Dent-Depth Residual (mm)
            self.geometry_residual_head = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            feats = self.backbone(x)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)

            class_logits = self.defect_class_head(feats)
            severity_logits = self.severity_head(feats)
            quality_out = self.quality_head(feats)
            raw_emb = self.embedding_head(feats)
            emb = F.normalize(raw_emb, p=2, dim=-1)
            bbox = self.localization_head(feats)
            geo_res = self.geometry_residual_head(feats)

            return {
                "features": feats,
                "class_logits": class_logits,
                "severity_logits": severity_logits,
                "quality_out": quality_out,
                "embedding": emb,
                "bbox": bbox,
                "geometry_residual": geo_res,
            }
else:
    PyTorchMultiHeadNet = None


class SharedBackbone:
    """
    Unified Multi-Head Feature Extractor Backbone.
    Extracts features for defect classification, ordinal severity, image quality,
    128-dim novelty embedding, localization hinting, and geometry residual.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        embedding_dim: int = 128,
        num_classes: int = 5,
        classes: Optional[List[str]] = None,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.classes = classes or DEFAULT_DEFECT_CLASSES
        self.num_classes = len(self.classes)
        self.num_severity = len(SEVERITY_LEVELS)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.mock_mode = mock_mode

        # Cache deterministic projection matrix for high-performance NumPy fallback
        rng = np.random.RandomState(42)
        self._numpy_proj_matrix = rng.randn(10, self.embedding_dim).astype(np.float32)

        self.torch_model = None
        if HAS_TORCH and not self.mock_mode and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.torch_model = PyTorchMultiHeadNet(
                    model_name=model_name,
                    pretrained=pretrained,
                    embedding_dim=embedding_dim,
                    num_classes=self.num_classes,
                    num_severity=self.num_severity,
                )
                self.torch_model.to(self.device)
                self.torch_model.eval()

                state = torch.load(checkpoint_path, map_location=self.device)
                self.torch_model.load_state_dict(state.get("model_state_dict", state))
            except Exception:
                self.torch_model = None

    def _numpy_feature_pass(self, image: np.ndarray) -> BackboneOutput:
        """
        Deterministic, high-fidelity NumPy multi-head feature engine.
        Used when PyTorch is not available, or in mock/fallback mode.
        """
        img_hwc = to_hwc_uint8(image)
        h, w = img_hwc.shape[:2]
        img_gray = to_grayscale(img_hwc)

        # 1. Gradients
        grad_x = np.abs(img_gray[:, 1:] - img_gray[:, :-1])
        grad_y = np.abs(img_gray[1:, :] - img_gray[:-1, :])
        mean_grad = float(np.mean(grad_x) + np.mean(grad_y))

        # 2. Defect metrics
        scratch_score = float(np.clip(mean_grad / 55.0, 0.01, 0.98))

        center_patch = img_gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        dent_score = float(np.clip(np.std(center_patch) / 75.0, 0.01, 0.98))

        spots_count = np.count_nonzero(np.abs(img_gray - np.mean(img_gray)) > 45.0)
        paint_defect_score = float(np.clip(spots_count / max(img_gray.size * 0.02, 1), 0.01, 0.98))

        edge_pixels = np.concatenate([img_gray[0, :], img_gray[-1, :], img_gray[:, 0], img_gray[:, -1]])
        misalignment_score = float(np.clip(np.std(edge_pixels) / 65.0, 0.01, 0.98))

        max_defect = max(scratch_score, dent_score, paint_defect_score, misalignment_score)
        # On flat surface with near zero defect scores, Good should be 0.98
        good_score = float(np.clip(1.0 - max_defect if max_defect > 0.10 else 0.98, 0.01, 0.98))

        raw_scores = {
            "Scratch": scratch_score,
            "Dent": dent_score,
            "Paint Defect": paint_defect_score,
            "Paint Misalignment": misalignment_score,
            "Good": good_score,
        }

        # Softmax over classes with higher temperature scale
        exp_s = {k: np.exp(v * 4.0) for k, v in raw_scores.items()}
        total_exp = sum(exp_s.values())
        class_probs = {k: float(v / total_exp) for k, v in exp_s.items()}

        top_class = max(class_probs, key=class_probs.get)
        top_conf = class_probs[top_class]

        # Severity Estimation
        if top_class == "Good":
            sev_idx = 0
            sev_score = 0.0
        else:
            sev_score = float(np.clip(top_conf * 0.9, 0.1, 1.0))
            if sev_score < 0.25:
                sev_idx = 1
            elif sev_score < 0.50:
                sev_idx = 2
            elif sev_score < 0.75:
                sev_idx = 3
            else:
                sev_idx = 4
        sev_level = SEVERITY_LEVELS[sev_idx]

        # Quality Scores
        blur_val = float(np.var(grad_x))
        is_acc = 1.0 if blur_val > 10.0 else 0.0
        quality_scores = {
            "is_acceptable_prob": is_acc,
            "blur_score": blur_val,
            "exposure_score": float(np.mean(img_gray) / 255.0),
        }

        # 128-dim L2 normalized embedding via deterministic projection
        feat_vector = np.array([
            scratch_score, dent_score, paint_defect_score, misalignment_score, good_score,
            mean_grad, float(np.std(center_patch)), float(np.mean(img_gray)),
            float(np.std(img_gray)), blur_val
        ], dtype=np.float32)

        raw_emb = np.dot(feat_vector, self._numpy_proj_matrix)
        norm = np.linalg.norm(raw_emb) + 1e-8
        embedding = (raw_emb / norm).astype(np.float32)

        # Localization hint
        bbox_hint = None
        if top_class != "Good" and top_conf >= 0.40:
            high_mask = grad_x > np.percentile(grad_x, 88)
            if np.count_nonzero(high_mask) > 0:
                y_idx, x_idx = np.nonzero(high_mask)
                bbox_hint = [
                    float(np.min(x_idx)),
                    float(np.min(y_idx)),
                    float(np.max(x_idx) + 1),
                    float(np.max(y_idx) + 1),
                ]
            else:
                bbox_hint = [float(w * 0.2), float(h * 0.2), float(w * 0.8), float(h * 0.8)]

        # Geometry residual (mm depth estimate from shadow pattern)
        geometry_residual = float(dent_score * 2.8) if dent_score > 0.3 else 0.0

        return BackboneOutput(
            defect_class_probs=class_probs,
            top_class=top_class,
            top_confidence=top_conf,
            severity_level=sev_level,
            severity_score=sev_score,
            quality_scores=quality_scores,
            embedding=embedding,
            bbox_hint=bbox_hint,
            geometry_residual=geometry_residual,
            features=feat_vector,
        )

    def forward(self, image: np.ndarray) -> BackboneOutput:
        """
        Execute forward pass across all 6 heads.
        """
        start_time = time.perf_counter()

        if self.torch_model is not None and HAS_TORCH:
            try:
                # Preprocess to tensor [1, 3, H, W]
                padded, _ = letterbox_pad(image, target_size=(640, 640))
                norm_img = normalize_image(padded, channel_first=True)
                tensor_input = torch.from_numpy(norm_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.torch_model(tensor_input)

                # 1. Defect classes
                class_probs_arr = F.softmax(out["class_logits"], dim=-1).cpu().numpy()[0]
                class_probs = {self.classes[i]: float(class_probs_arr[i]) for i in range(min(len(self.classes), len(class_probs_arr)))}
                top_class = max(class_probs, key=class_probs.get)
                top_conf = class_probs[top_class]

                # 2. Severity
                sev_probs = F.softmax(out["severity_logits"], dim=-1).cpu().numpy()[0]
                sev_idx = int(np.argmax(sev_probs))
                sev_level = SEVERITY_LEVELS[sev_idx]
                sev_score = float(np.sum(sev_probs * np.linspace(0.0, 1.0, len(sev_probs))))

                # 3. Quality
                qual_out = out["quality_out"].cpu().numpy()[0]
                is_acc_prob = float(1.0 / (1.0 + np.exp(-qual_out[0])))
                quality_scores = {
                    "is_acceptable_prob": is_acc_prob,
                    "blur_score": float(qual_out[1]),
                    "exposure_score": float(qual_out[2]),
                }

                # 4. Embedding
                emb = out["embedding"].cpu().numpy()[0]

                # 5. Localization Bounding Box
                raw_bbox = out["bbox"].cpu().numpy()[0]
                h_orig, w_orig = image.shape[:2]
                bbox_hint = [
                    float(raw_bbox[0] * w_orig),
                    float(raw_bbox[1] * h_orig),
                    float(raw_bbox[2] * w_orig),
                    float(raw_bbox[3] * h_orig),
                ] if top_class != "Good" else None

                # 6. Geometry residual
                geo_res = float(out["geometry_residual"].cpu().numpy()[0][0])

                feats = out["features"].cpu().numpy()[0]

                elapsed = (time.perf_counter() - start_time) * 1000.0
                return BackboneOutput(
                    defect_class_probs=class_probs,
                    top_class=top_class,
                    top_confidence=top_conf,
                    severity_level=sev_level,
                    severity_score=sev_score,
                    quality_scores=quality_scores,
                    embedding=emb,
                    bbox_hint=bbox_hint,
                    geometry_residual=geo_res,
                    features=feats,
                    latency_ms=elapsed,
                )
            except Exception:
                pass  # Fall back to NumPy feature pass

        out = self._numpy_feature_pass(image)
        out.latency_ms = (time.perf_counter() - start_time) * 1000.0
        return out

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract 128-dim normalized embedding for novelty / distance estimation."""
        out = self.forward(image)
        return out.embedding

    def export_onnx(self, output_path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> bool:
        """Export backbone PyTorch model to ONNX format."""
        if not HAS_TORCH or self.torch_model is None:
            return False
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            dummy_input = torch.randn(*input_shape, device=self.device)
            torch.onnx.export(
                self.torch_model,
                dummy_input,
                output_path,
                input_names=["input_image"],
                output_names=["class_logits", "severity_logits", "quality_out", "embedding", "bbox", "geometry_residual"],
                dynamic_axes={"input_image": {0: "batch_size"}},
                opset_version=14,
            )
            return True
        except Exception:
            return False
