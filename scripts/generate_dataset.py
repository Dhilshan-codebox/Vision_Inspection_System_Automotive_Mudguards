#!/usr/bin/env python3
"""
Generate a predefined deterministic dataset for Automotive Mudguard Inspection.
Creates data/mudguard_dataset/ with train, val, and test splits containing:
  - normal
  - scratch
  - dent
  - paint_defect
  - paint_misalignment
Also populates data/raw and exports manifest to data/processed/manifest.json.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def create_mudguard_base(width=640, height=640, seed=0) -> np.ndarray:
    """Generates base mudguard surface (dark metallic grey with subtle curved contour)."""
    rng = np.random.RandomState(seed)
    # Metallic grey base
    img = np.full((height, width, 3), 140, dtype=np.uint8)

    # Add subtle texture noise
    noise = rng.normal(0, 3, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw curved mudguard boundary mask (smooth dark outer edges)
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2 + 100
    radius = int(width * 0.75)
    outer_mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
    img[outer_mask] = 30 # dark background outside mudguard
    return img


def add_scratch(img: np.ndarray, seed: int, severity: str) -> np.ndarray:
    """Adds a sharp linear scratch defect."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    rng = np.random.RandomState(seed)

    x1, y1 = rng.randint(150, 450), rng.randint(150, 450)
    angle = rng.uniform(0, 2 * np.pi)
    length = 80 if severity == "low" else (140 if severity == "medium" else 200)
    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))

    width = 2 if severity == "low" else (4 if severity == "medium" else 6)
    draw.line([(x1, y1), (x2, y2)], fill=(20, 20, 20), width=width)
    draw.line([(x1+1, y1+1), (x2+1, y2+1)], fill=(230, 230, 230), width=1) # scratch highlight
    return np.array(pil_img)


def add_dent(img: np.ndarray, seed: int, severity: str) -> np.ndarray:
    """Adds a smooth gradient shadow ellipse simulating a dent."""
    height, width, _ = img.shape
    rng = np.random.RandomState(seed)
    cx, cy = rng.randint(200, 440), rng.randint(200, 440)
    rx = 40 if severity == "low" else (70 if severity == "medium" else 100)
    ry = int(rx * rng.uniform(0.6, 0.9))

    y, x = np.ogrid[:height, :width]
    dist_sq = ((x - cx) / rx)**2 + ((y - cy) / ry)**2
    mask = dist_sq <= 1.0

    # Shadow factor
    factor = np.ones((height, width), dtype=np.float32)
    factor[mask] = 0.5 + 0.4 * dist_sq[mask] # darker in center

    res = img.astype(np.float32)
    for c in range(3):
        res[:, :, c] *= factor
    return np.clip(res, 0, 255).astype(np.uint8)


def add_paint_defect(img: np.ndarray, seed: int, severity: str) -> np.ndarray:
    """Adds paint contamination / runs / orange peel specks."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    rng = np.random.RandomState(seed)

    num_spots = 5 if severity == "low" else (15 if severity == "medium" else 30)
    cx, cy = rng.randint(200, 440), rng.randint(200, 440)

    for _ in range(num_spots):
        sx = cx + rng.randint(-50, 50)
        sy = cy + rng.randint(-50, 50)
        r = rng.randint(3, 10)
        color = (220, 180, 50) if rng.rand() > 0.5 else (40, 40, 40)
        draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=color)
    return np.array(pil_img)


def add_paint_misalignment(img: np.ndarray, seed: int, severity: str) -> np.ndarray:
    """Adds uneven paint border or color offset line."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    rng = np.random.RandomState(seed)

    offset = 15 if severity == "low" else (30 if severity == "medium" else 50)
    y_line = rng.randint(220, 380)

    # Draw uneven paint border polygon
    points = [(100, y_line), (540, y_line + offset), (540, y_line + offset + 40), (100, y_line + 40)]
    draw.polygon(points, fill=(180, 50, 50))
    return np.array(pil_img)


def generate_dataset(dataset_root="data/mudguard_dataset"):
    root = Path(dataset_root).resolve()
    raw_root = Path("data/raw").resolve()
    proc_root = Path("data/processed").resolve()

    for d in [root, raw_root, proc_root]:
        d.mkdir(parents=True, exist_ok=True)

    classes = ["normal", "scratch", "dent", "paint_defect", "paint_misalignment"]
    splits_spec = {
        "train": {"samples_per_class": 12, "part_offset": 100},
        "val": {"samples_per_class": 4, "part_offset": 200},
        "test": {"samples_per_class": 4, "part_offset": 300},
    }

    cameras = ["cam_top", "cam_side", "cam_front"]
    severities = ["low", "medium", "high"]

    manifest_records = []

    total_idx = 0
    for split, spec in splits_spec.items():
        n_per_class = spec["samples_per_class"]
        part_offset = spec["part_offset"]

        for cls_idx, cls_name in enumerate(classes):
            cls_dir = root / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_per_class):
                total_idx += 1
                seed = total_idx * 1000 + cls_idx * 10 + i
                part_id = f"part_{part_offset + i}"
                cam_id = cameras[i % len(cameras)]
                sev = "none" if cls_name == "normal" else severities[i % len(severities)]

                # Base image
                img = create_mudguard_base(width=640, height=640, seed=seed)

                # Add defect
                if cls_name == "scratch":
                    img = add_scratch(img, seed, sev)
                elif cls_name == "dent":
                    img = add_dent(img, seed, sev)
                elif cls_name == "paint_defect":
                    img = add_paint_defect(img, seed, sev)
                elif cls_name == "paint_misalignment":
                    img = add_paint_misalignment(img, seed, sev)

                filename = f"{part_id}_{cam_id}_{cls_name}_{sev}_{i+1:02d}.png"
                filepath = cls_dir / filename

                pil_img = Image.fromarray(img)
                pil_img.save(filepath)

                # Copy to raw
                raw_filepath = raw_root / filename
                pil_img.save(raw_filepath)

                rel_path = str(filepath.relative_to(root.parent.parent if root.parent.parent.name == "data" else root.parent)).replace("\\", "/")

                manifest_records.append({
                    "path": str(filepath.relative_to(root)).replace("\\", "/"),
                    "full_path": str(filepath),
                    "label": cls_name,
                    "split": split,
                    "part_id": part_id,
                    "camera_id": cam_id,
                    "severity": sev,
                    "width": 640,
                    "height": 640,
                    "channels": 3,
                })

    manifest_path = proc_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "mudguard_dataset",
            "version": "1.0.0",
            "total_images": len(manifest_records),
            "records": manifest_records,
        }, f, indent=2)

    print(f"Dataset generated successfully at {root}!")
    print(f"Total images: {len(manifest_records)}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    generate_dataset()
