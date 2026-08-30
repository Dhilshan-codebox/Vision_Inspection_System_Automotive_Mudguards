from typing import Optional, Dict, Any, List
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus
from src.perspectives.base import PerspectiveInterface
from src.geometry.surface_expert import DentGeometryExpert


class GeometryPerspective(PerspectiveInterface):
    """Evaluates 3D surface geometry, depth variations, and physical dent depressions.
    Gracefully handles missing 3D sensor data by returning an explicit UNAVAILABLE status.
    """

    def __init__(self, name: str = "geometry", dent_depth_threshold_mm: float = 2.0, mock_mode: bool = False):
        super().__init__(name=name)
        self.dent_depth_threshold_mm = dent_depth_threshold_mm
        self.mock_mode = mock_mode
        # Instantiate expert (calibration can be set later via metadata)
        self._expert = DentGeometryExpert(dent_depth_threshold_mm=self.dent_depth_threshold_mm)

    def _process(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Evidence:
        """Process an image with optional depth map and return standardized Evidence.

        * `metadata["depth_map"]` – required depth map as a NumPy array (raw sensor units).
        * `metadata["appearance_score"]` – optional appearance perspective confidence (0‑1).
        * `metadata["has_depth_sensor"]` – optional flag used in mock mode.
        """
        if self.mock_mode:
            has_depth = metadata.get("has_depth_sensor", False) if metadata else False
            if not has_depth:
                return Evidence(
                    perspective=self.name,
                    status=EvidenceStatus.UNAVAILABLE,
                    confidence=0.0,
                    score=0.0,
                    reference=None,
                    metadata={"reason": "3D depth sensor offline or unavailable in mock mode"},
                )
            # Simple mock response preserving caller‑provided fields
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.AVAILABLE,
                confidence=metadata.get("mock_geometry_conf", 0.92),
                score=metadata.get("mock_geometry_score", 0.05),
                region=metadata.get("mock_region"),
                reference="mock_3d_profile",
                metadata={"mock": True, "sensor": "RGB-D"},
            )

        # Retrieve depth map from metadata
        depth_map = metadata.get("depth_map") if metadata else None
        if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.UNAVAILABLE,
                confidence=0.0,
                score=0.0,
                region=None,
                reference=None,
                metadata={"reason": "No 3D/depth sensor data provided for geometry analysis"},
            )

        # Optional appearance confidence from other perspective (used for optical‑shadow dispute)
        rgb_appearance_score = metadata.get("appearance_score") if metadata else None
        evidence = self._expert.evaluate_to_evidence(
            depth_map=depth_map,
            calibration=None,
            rgb_appearance_score=rgb_appearance_score,
        )
        # Ensure perspective label matches this class name
        evidence.perspective = self.name
        return evidence
