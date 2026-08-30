from typing import Tuple, Optional
import numpy as np


class GeometryFixtureGenerator:
    """
    Generates synthetic, predefined RGB-D / 3D test fixtures for nominal mudguard surfaces,
    authentic physical dent depressions, and optical shadow edge cases.
    """

    @staticmethod
    def generate_nominal_mudguard_depth(
        shape: Tuple[int, int] = (200, 200),
        base_depth_mm: float = 500.0,
        curvature_coeff: float = 0.001,
    ) -> np.ndarray:
        """
        Creates a nominal cylindrical/curved mudguard surface depth map in mm.
        """
        h, w = shape
        y_grid, x_grid = np.ogrid[:h, :w]
        x_center = w / 2.0

        # Parabolic surface profile along width: Z(x) = Z0 + a * (x - x0)^2
        depth_map = base_depth_mm + curvature_coeff * ((x_grid - x_center) ** 2)
        # Subtle sensor noise (+/- 0.2mm)
        noise = np.random.uniform(-0.15, 0.15, shape).astype(np.float32)
        return (depth_map + noise).astype(np.float32)

    @staticmethod
    def inject_dent(
        depth_map: np.ndarray,
        center: Tuple[int, int] = (100, 100),
        radius_px: float = 20.0,
        max_depression_mm: float = 3.5,
    ) -> np.ndarray:
        """
        Injects a smooth physical Gaussian dent indentation (increasing depth/depression) into the surface.
        """
        modified = depth_map.copy()
        h, w = modified.shape
        cy, cx = center

        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2

        # Gaussian depression profile
        dent_depression = max_depression_mm * np.exp(-dist_sq / (2.0 * (radius_px / 2.0) ** 2))
        modified += dent_depression.astype(np.float32)
        return modified

    @staticmethod
    def generate_shadow_vs_dent_pair(
        shape: Tuple[int, int] = (200, 200),
        center: Tuple[int, int] = (100, 100),
        radius_px: float = 20.0,
        is_true_dent: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates an (RGB, Depth) pair:
        - If is_true_dent=False: 2D RGB has dark optical shadow spot, but 3D Depth is completely nominal!
        - If is_true_dent=True: 2D RGB has dark spot, and 3D Depth has physical 3.5mm depression!
        """
        h, w = shape
        cy, cx = center

        # Create nominal RGB background
        rgb = np.full((h, w, 3), 160, dtype=np.uint8)

        # Draw dark optical spot in RGB in both cases
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        shadow_mask = dist_sq <= (radius_px ** 2)
        rgb[shadow_mask] = [40, 40, 40]

        # Generate Depth
        nominal_depth = GeometryFixtureGenerator.generate_nominal_mudguard_depth(shape=shape)
        if is_true_dent:
            depth_map = GeometryFixtureGenerator.inject_dent(
                nominal_depth,
                center=center,
                radius_px=radius_px,
                max_depression_mm=3.5,
            )
        else:
            # Depth remains nominal flat/curved (optical shadow only)
            depth_map = nominal_depth

        return rgb, depth_map
