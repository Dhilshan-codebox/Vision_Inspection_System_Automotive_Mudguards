from typing import Optional, Dict, Any, List
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus
from src.perspectives.base import PerspectiveInterface


class UncertaintyPerspective(PerspectiveInterface):
    """
    Evaluates epistemic (model knowledge & inter-perspective disagreement),
    aleatoric (sensor noise & exposure variance), and out-of-distribution (OOD) uncertainty.
    """

    def __init__(
        self,
        name: str = "uncertainty",
        epistemic_weight: float = 0.60,
        aleatoric_weight: float = 0.40,
        mock_mode: bool = False,
    ):
        super().__init__(name=name)
        self.epistemic_weight = epistemic_weight
        self.aleatoric_weight = aleatoric_weight
        self.mock_mode = mock_mode

    def _process(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        if self.mock_mode:
            mock_score = metadata.get("mock_uncertainty_score", 0.05) if metadata else 0.05
            mock_conf = metadata.get("mock_uncertainty_conf", 0.90) if metadata else 0.90
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.AVAILABLE,
                confidence=mock_conf,
                score=mock_score,
                reference="mock_uncertainty_bounds",
                metadata={
                    "mock": True,
                    "uncertainty_type": "nominal",
                    "perspective_disagreement_range": 0.05,
                    "perspective_variance": 0.01,
                },
            )

        meta = metadata or {}
        other_evidences: List[Evidence] = meta.get("evidences", [])
        backbone_embedding: Optional[np.ndarray] = meta.get("embedding")
        nominal_embedding: Optional[np.ndarray] = meta.get("nominal_embedding")
        class_probs: Optional[Dict[str, float]] = meta.get("class_probabilities")

        # 1. Epistemic Uncertainty (Perspective Disagreement + Model Entropy)
        valid_scores = [
            e.score for e in other_evidences
            if e.score is not None and e.status == EvidenceStatus.AVAILABLE and e.perspective != self.name
        ]

        if len(valid_scores) >= 2:
            score_variance = float(np.var(valid_scores))
            score_range = float(np.max(valid_scores) - np.min(valid_scores))
        else:
            score_variance = 0.05
            score_range = 0.10

        # Model entropy from predicted class distribution
        model_entropy = 0.0
        if class_probs and len(class_probs) > 1:
            p_vals = np.array(list(class_probs.values()), dtype=np.float32)
            p_vals = p_vals[p_vals > 1e-6]
            model_entropy = float(-np.sum(p_vals * np.log(p_vals)) / np.log(len(class_probs)))

        epistemic_score = float(np.clip(score_variance * 2.0 + score_range * 0.4 + model_entropy * 0.3, 0.0, 1.0))

        # 2. Aleatoric Uncertainty (Sensor Noise + Illumination Extremes)
        img_float = image.astype(np.float32)
        # High-frequency noise estimation via Laplacian / diff
        noise_level = float(np.std(img_float) / 255.0)
        mean_lum = float(np.mean(img_float))
        exposure_extremity = 0.0
        if mean_lum < 35.0 or mean_lum > 220.0:
            exposure_extremity = float(abs(mean_lum - 128.0) / 128.0)

        aleatoric_score = float(np.clip(noise_level * 0.5 + exposure_extremity * 0.5, 0.0, 1.0))

        # 3. Out-Of-Distribution (OOD) Distance from Embedding
        ood_score = 0.0
        if backbone_embedding is not None and nominal_embedding is not None:
            # Cosine distance
            cos_sim = float(np.dot(backbone_embedding, nominal_embedding) / (
                (np.linalg.norm(backbone_embedding) * np.linalg.norm(nominal_embedding)) + 1e-8
            ))
            ood_score = float(np.clip(1.0 - max(0.0, cos_sim), 0.0, 1.0))

        # Combined Total Uncertainty
        combined_score = float(np.clip(
            self.epistemic_weight * epistemic_score + self.aleatoric_weight * aleatoric_score + 0.20 * ood_score,
            0.0,
            1.0
        ))

        # Determine dominant uncertainty type
        if ood_score > 0.45:
            uncertainty_type = "ood"
        elif score_range > 0.50:
            uncertainty_type = "sensor_conflict"
        elif epistemic_score > aleatoric_score and epistemic_score > 0.30:
            uncertainty_type = "epistemic"
        elif aleatoric_score > 0.30:
            uncertainty_type = "aleatoric"
        else:
            uncertainty_type = "nominal"

        confidence = float(np.clip(1.0 - (combined_score * 0.5), 0.5, 1.0))

        return Evidence(
            perspective=self.name,
            status=EvidenceStatus.AVAILABLE,
            confidence=confidence,
            score=combined_score,
            reference="epistemic_aleatoric_ood_estimator",
            metadata={
                "perspective_disagreement_range": score_range,
                "perspective_variance": score_variance,
                "model_entropy": model_entropy,
                "sensor_noise_level": noise_level,
                "epistemic_score": epistemic_score,
                "aleatoric_score": aleatoric_score,
                "ood_score": ood_score,
                "uncertainty_type": uncertainty_type,
                "evaluated_perspectives_count": len(valid_scores),
            },
        )
