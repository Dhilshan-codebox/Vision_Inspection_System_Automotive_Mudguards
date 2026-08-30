from typing import Optional, Dict, Any, List
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus
from src.perspectives.base import PerspectiveInterface


class ContextPerspective(PerspectiveInterface):
    """
    Evaluates operational and environmental context including camera viewpoint alignment,
    part SKU consistency, part pose/aspect ratio plausibility, paint boundary zone validity,
    and factory ambient lighting stability.
    """

    def __init__(
        self,
        name: str = "context",
        expected_viewpoints: Optional[list] = None,
        expected_aspect_ratio: float = 1.0,
        aspect_ratio_tolerance: float = 0.5,
        mock_mode: bool = False,
    ):
        super().__init__(name=name)
        self.expected_viewpoints = expected_viewpoints or [
            "view_top", "view_side", "view_front", "cam_top", "cam_side", "angle_front", "angle_top"
        ]
        self.expected_aspect_ratio = expected_aspect_ratio
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.mock_mode = mock_mode

    def _process(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        if self.mock_mode:
            mock_score = metadata.get("mock_context_score", 0.02) if metadata else 0.02
            mock_conf = metadata.get("mock_context_conf", 0.95) if metadata else 0.95
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.AVAILABLE,
                confidence=mock_conf,
                score=mock_score,
                reference="mock_context_contract",
                metadata={"mock": True, "details": {"known_viewpoint": True}},
            )

        meta = metadata or {}
        part_id = meta.get("part_id")
        view_id = meta.get("view_id")
        defect_bbox = meta.get("defect_bbox") or meta.get("region")

        context_penalties = 0.0
        details: Dict[str, Any] = {}

        # 1. Viewpoint alignment check
        if view_id:
            is_known_view = any(v.lower() in str(view_id).lower() for v in self.expected_viewpoints)
            details["known_viewpoint"] = is_known_view
            if not is_known_view:
                context_penalties += 0.25
        else:
            details["known_viewpoint"] = False

        # 2. Part Pose & Aspect Ratio Plausibility
        h, w = image.shape[:2]
        observed_aspect_ratio = float(w / max(h, 1))
        details["observed_aspect_ratio"] = observed_aspect_ratio
        ar_disparity = abs(observed_aspect_ratio - self.expected_aspect_ratio)
        if ar_disparity > self.aspect_ratio_tolerance:
            pose_penalty = min(0.20, (ar_disparity - self.aspect_ratio_tolerance) * 0.2)
            context_penalties += pose_penalty
            details["pose_plausible"] = False
        else:
            details["pose_plausible"] = True

        # 3. Ambient Lighting Stability
        mean_intensity = float(np.mean(image))
        details["mean_ambient_intensity"] = mean_intensity
        if mean_intensity < 30.0 or mean_intensity > 220.0:
            context_penalties += 0.30

        # 4. Paint Boundary Zone Validation for Candidate Defect Region
        if defect_bbox and len(defect_bbox) == 4:
            x1, y1, x2, y2 = defect_bbox
            # If defect box is completely outside [0, w] x [0, h], mark impossible
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h or x2 <= x1 or y2 <= y1:
                context_penalties += 0.35
                details["zone_valid"] = False
            else:
                details["zone_valid"] = True
        else:
            details["zone_valid"] = True

        score = float(np.clip(context_penalties, 0.0, 1.0))
        confidence = 0.90 if part_id and view_id else 0.70

        return Evidence(
            perspective=self.name,
            status=EvidenceStatus.AVAILABLE,
            confidence=confidence,
            score=score,
            reference="line_environment_context",
            metadata={
                "part_id": part_id,
                "view_id": view_id,
                "details": details,
            },
        )
