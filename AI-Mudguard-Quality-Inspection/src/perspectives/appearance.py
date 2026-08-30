from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus
from src.perspectives.base import PerspectiveInterface
from src.preprocessing.quality_gate import to_grayscale
from src.preprocessing.transforms import to_hwc_uint8


class AppearancePerspective(PerspectiveInterface):
    """
    Evaluates 2D visual appearance features, looking for scratch lines,
    color discontinuities, multi-scale edge deviations, and paint-boundary misalignment.
    """

    def __init__(
        self,
        name: str = "appearance",
        scratch_sensitivity: float = 0.5,
        multiscale_sobel: bool = True,
        mock_mode: bool = False,
    ):
        super().__init__(name=name)
        self.scratch_sensitivity = scratch_sensitivity
        self.multiscale_sobel = multiscale_sobel
        self.mock_mode = mock_mode

    def _compute_multiscale_gradients(self, img_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute fine (3x3 equivalent) and coarse (7x7 equivalent) gradients."""
        # 3x3 simple difference
        grad_3x = np.abs(img_gray[:, 1:] - img_gray[:, :-1])
        grad_3y = np.abs(img_gray[1:, :] - img_gray[:-1, :])

        # 7x7 coarse difference (step 3)
        if img_gray.shape[0] > 7 and img_gray.shape[1] > 7:
            grad_7x = np.abs(img_gray[:, 3:] - img_gray[:, :-3])
            grad_7y = np.abs(img_gray[3:, :] - img_gray[:-3, :])
        else:
            grad_7x, grad_7y = grad_3x, grad_3y

        return {
            "fine_x": grad_3x,
            "fine_y": grad_3y,
            "coarse_x": grad_7x,
            "coarse_y": grad_7y,
        }

    def _compute_color_discontinuity(self, image: np.ndarray) -> float:
        """Compute channel-wise gradient disparity across RGB channels."""
        hwc = to_hwc_uint8(image)
        if hwc.shape[2] < 3:
            return 0.0

        r_grad = np.abs(hwc[:, 1:, 0].astype(float) - hwc[:, :-1, 0].astype(float))
        g_grad = np.abs(hwc[:, 1:, 1].astype(float) - hwc[:, :-1, 1].astype(float))
        b_grad = np.abs(hwc[:, 1:, 2].astype(float) - hwc[:, :-1, 2].astype(float))

        channel_diff = np.abs(r_grad - g_grad) + np.abs(g_grad - b_grad) + np.abs(b_grad - r_grad)
        return float(np.mean(channel_diff) / 40.0)

    def _compute_boundary_alignment(self, img_gray: np.ndarray) -> Tuple[float, float]:
        """Compute edge density and gradient variance along outer boundary zones."""
        h, w = img_gray.shape[:2]
        pad_y = max(1, h // 10)
        pad_x = max(1, w // 10)

        top_strip = img_gray[:pad_y, :]
        bottom_strip = img_gray[-pad_y:, :]
        left_strip = img_gray[:, :pad_x]
        right_strip = img_gray[:, -pad_x:]

        boundary_pixels = np.concatenate([
            top_strip.ravel(), bottom_strip.ravel(), left_strip.ravel(), right_strip.ravel()
        ])

        boundary_std = float(np.std(boundary_pixels))
        boundary_grad = float(np.mean(np.abs(np.diff(boundary_pixels))))
        return boundary_std, boundary_grad

    def _process(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        if self.mock_mode:
            mock_score = metadata.get("mock_appearance_score", 0.15) if metadata else 0.15
            mock_conf = metadata.get("mock_appearance_conf", 0.90) if metadata else 0.90
            mock_region = metadata.get("mock_region", [50.0, 50.0, 150.0, 150.0]) if metadata else None
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.AVAILABLE,
                confidence=mock_conf,
                score=mock_score,
                region=mock_region,
                reference="mock_appearance_features",
                metadata={"mock": True, "defect_type_hint": "nominal"},
            )

        img_hwc = to_hwc_uint8(image)
        img_gray = to_grayscale(img_hwc)
        h, w = img_gray.shape[:2]

        if h < 5 or w < 5:
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.UNAVAILABLE,
                confidence=0.0,
                score=0.0,
                metadata={"reason": "Image too small for appearance analysis"},
            )

        # 1. Multi-scale Gradients
        grads = self._compute_multiscale_gradients(img_gray)
        grad_x = grads["fine_x"]
        grad_y = grads["fine_y"]

        # 2. High gradient scratch / chip mask
        threshold = np.mean(grad_x) + 2.5 * np.std(grad_x)
        high_grad_mask = grad_x > max(threshold, 20.0)
        anomaly_count = int(np.count_nonzero(high_grad_mask))
        total_pixels = img_gray.size

        fine_scratch_score = min(1.0, (anomaly_count / 100.0) * self.scratch_sensitivity)

        # 3. Color Discontinuity
        color_disc = self._compute_color_discontinuity(img_hwc)

        # 4. Paint Boundary Alignment
        boundary_std, boundary_grad = self._compute_boundary_alignment(img_gray)
        boundary_score = float(np.clip(boundary_std / 60.0 + boundary_grad / 50.0, 0.0, 1.0))

        # Overall appearance defect score
        combined_score = float(np.clip(
            0.6 * fine_scratch_score + 0.25 * color_disc + 0.15 * boundary_score,
            0.0,
            1.0
        ))

        # Defect type hint
        if fine_scratch_score > 0.45 and fine_scratch_score >= boundary_score:
            defect_type_hint = "scratch_candidate"
        elif boundary_score > 0.45:
            defect_type_hint = "paint_misalignment_candidate"
        elif color_disc > 0.40:
            defect_type_hint = "paint_defect_candidate"
        else:
            defect_type_hint = "nominal"

        confidence = 0.88 if total_pixels > 1000 else 0.50

        # Discover bounding region of defect cluster
        region: Optional[List[float]] = None
        if anomaly_count > 10:
            y_indices, x_indices = np.nonzero(high_grad_mask)
            x1 = float(np.min(x_indices))
            x2 = float(np.max(x_indices) + 1)
            y1 = float(np.min(y_indices))
            y2 = float(np.max(y_indices) + 1)
            region = [x1, y1, x2, y2]

        return Evidence(
            perspective=self.name,
            status=EvidenceStatus.AVAILABLE,
            confidence=confidence,
            score=combined_score,
            region=region,
            reference="multiscale_gradient_appearance_map",
            metadata={
                "anomaly_pixel_count": anomaly_count,
                "mean_gradient": float(np.mean(grad_x)),
                "fine_scratch_score": float(fine_scratch_score),
                "color_discontinuity_score": float(color_disc),
                "boundary_score": float(boundary_score),
                "defect_type_hint": defect_type_hint,
            },
        )
