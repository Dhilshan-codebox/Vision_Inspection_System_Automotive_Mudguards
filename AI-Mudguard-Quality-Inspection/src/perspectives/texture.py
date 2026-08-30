from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus
from src.perspectives.base import PerspectiveInterface
from src.preprocessing.quality_gate import to_grayscale
from src.preprocessing.transforms import create_tiles, extract_roi, to_hwc_uint8


def compute_lbp_approximation(img_gray: np.ndarray) -> np.ndarray:
    """
    Computes an 8-neighbor Local Binary Pattern (LBP) representation using pure NumPy vectorization.
    """
    h, w = img_gray.shape[:2]
    if h < 3 or w < 3:
        return np.zeros_like(img_gray, dtype=np.uint8)

    center = img_gray[1:-1, 1:-1].astype(np.int16)
    lbp = np.zeros(center.shape, dtype=np.uint8)

    # 8 neighbor offsets: (dy, dx, bit_shift)
    neighbors = [
        (-1, -1, 0),
        (-1,  0, 1),
        (-1,  1, 2),
        ( 0,  1, 3),
        ( 1,  1, 4),
        ( 1,  0, 5),
        ( 1, -1, 6),
        ( 0, -1, 7),
    ]

    for dy, dx, bit in neighbors:
        y1, y2 = 1 + dy, h - 1 + dy
        x1, x2 = 1 + dx, w - 1 + dx
        nb = img_gray[y1:y2, x1:x2].astype(np.int16)
        lbp |= ((nb >= center).astype(np.uint8) << bit)

    return lbp


class TexturePerspective(PerspectiveInterface):
    """
    Evaluates surface micro-texture, roughness, and paint uniformity
    (detecting orange-peel, pin-holes, scoring, and coating graininess)
    using high-resolution overlapping tile analysis and Local Binary Patterns (LBP).
    """

    def __init__(
        self,
        name: str = "texture",
        roughness_threshold: float = 25.0,
        tile_size: int = 224,
        tile_stride: int = 160,
        use_lbp: bool = True,
        mock_mode: bool = False,
    ):
        super().__init__(name=name)
        self.roughness_threshold = roughness_threshold
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.use_lbp = use_lbp
        self.mock_mode = mock_mode

    def _evaluate_tile_roughness(self, tile_gray: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean local std, std of stds, and LBP entropy for a tile."""
        th, tw = tile_gray.shape[:2]
        if th < 8 or tw < 8:
            return 0.0, 0.0, 0.0

        patch_size = 8
        stds = []
        for y in range(0, th - patch_size + 1, patch_size):
            for x in range(0, tw - patch_size + 1, patch_size):
                p = tile_gray[y : y + patch_size, x : x + patch_size]
                stds.append(float(np.std(p)))

        if not stds:
            return 0.0, 0.0, 0.0

        mean_std = float(np.mean(stds))
        std_of_stds = float(np.std(stds))

        # LBP texture irregularity
        lbp_score = 0.0
        if self.use_lbp and th >= 16 and tw >= 16:
            lbp_map = compute_lbp_approximation(tile_gray)
            lbp_hist, _ = np.histogram(lbp_map, bins=32, range=(0, 256), density=True)
            non_zero_hist = lbp_hist[lbp_hist > 0]
            entropy = -float(np.sum(non_zero_hist * np.log2(non_zero_hist)))
            lbp_score = float(np.clip((entropy - 3.0) / 2.0, 0.0, 1.0))

        return mean_std, std_of_stds, lbp_score

    def _process(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        if self.mock_mode:
            mock_score = metadata.get("mock_texture_score", 0.08) if metadata else 0.08
            mock_conf = metadata.get("mock_texture_conf", 0.88) if metadata else 0.88
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.AVAILABLE,
                confidence=mock_conf,
                score=mock_score,
                reference="mock_texture_profile",
                metadata={"mock": True, "mean_roughness": 12.0},
            )

        img_hwc = to_hwc_uint8(image)
        img_gray = to_grayscale(img_hwc)
        h, w = img_gray.shape[:2]

        if h < 8 or w < 8:
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.UNAVAILABLE,
                confidence=0.0,
                score=0.0,
                metadata={"reason": "Image too small for micro-texture analysis"},
            )

        # 1. Global / Patch Roughness Baseline
        patch_size = 8
        global_stds = []
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = img_gray[y : y + patch_size, x : x + patch_size]
                global_stds.append(float(np.std(patch)))

        if not global_stds:
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.UNAVAILABLE,
                confidence=0.0,
                score=0.0,
                metadata={"reason": "Insufficient patches for texture analysis"},
            )

        mean_local_std = float(np.mean(global_stds))
        std_of_stds = float(np.std(global_stds))

        # 2. High-Resolution Tiled Analysis
        overlap = max(16, self.tile_size - self.tile_stride)
        tiles = create_tiles(img_hwc, tile_size=(min(self.tile_size, h), min(self.tile_size, w)), overlap=overlap)

        tile_scores: List[float] = []
        worst_tile_bbox: Optional[List[int]] = None
        max_tile_defect = 0.0

        for tile_arr, tile_bbox in tiles:
            tile_g = to_grayscale(tile_arr)
            t_mean_std, t_std_std, t_lbp = self._evaluate_tile_roughness(tile_g)

            t_rough_score = float(np.clip(t_mean_std / self.roughness_threshold, 0.0, 1.0))
            t_unif_score = float(np.clip(t_std_std / 15.0, 0.0, 1.0))
            t_score = 0.5 * t_rough_score + 0.3 * t_unif_score + 0.2 * t_lbp

            tile_scores.append(t_score)
            if t_score > max_tile_defect:
                max_tile_defect = t_score
                worst_tile_bbox = tile_bbox

        avg_tile_score = float(np.mean(tile_scores)) if tile_scores else 0.0

        # Global Defect Scores
        roughness_score = float(np.clip(mean_local_std / self.roughness_threshold, 0.0, 1.0))
        uniformity_defect_score = float(np.clip(std_of_stds / 15.0, 0.0, 1.0))

        # Combined final score incorporating worst tile to prevent dilution
        final_score = float(np.clip(
            0.35 * roughness_score + 0.25 * uniformity_defect_score + 0.40 * max_tile_defect,
            0.0,
            1.0
        ))

        region: Optional[List[float]] = None
        if max_tile_defect > 0.40 and worst_tile_bbox is not None:
            region = [float(worst_tile_bbox[0]), float(worst_tile_bbox[1]), float(worst_tile_bbox[2]), float(worst_tile_bbox[3])]

        return Evidence(
            perspective=self.name,
            status=EvidenceStatus.AVAILABLE,
            confidence=0.88,
            score=final_score,
            region=region,
            reference="tiled_lbp_texture_map",
            metadata={
                "mean_roughness": mean_local_std,
                "texture_non_uniformity": std_of_stds,
                "total_patches_evaluated": len(global_stds),
                "total_tiles_evaluated": len(tiles),
                "worst_tile_defect_score": float(max_tile_defect),
                "avg_tile_defect_score": float(avg_tile_score),
            },
        )
