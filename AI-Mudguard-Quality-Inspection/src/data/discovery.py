"""
Dataset Discovery Engine for Automotive Mudguard Quality Inspection.
Scans directories, parses structured filenames for part_id, view_id, and severity.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.data.contracts import ManifestRecord, SeverityLevel

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DatasetDiscovery:
    """
    Scans dataset directories for image files and parses metadata from paths and filenames.
    Supports both split-based (train/val/test) and class-folder direct layouts.
    """

    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root).resolve()

    def discover(self) -> List[ManifestRecord]:
        """Alias for discover_all."""
        return self.discover_all()

    def discover_all(self) -> List[ManifestRecord]:
        """Scans dataset root and returns list of ManifestRecords."""
        records: List[ManifestRecord] = []
        if not self.dataset_root.exists() or not self.dataset_root.is_dir():
            return records

        # Check if root contains split subdirectories (train/val/test)
        subdirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        split_names = {"train", "val", "test", "validation"}
        has_splits = any(d.name.lower() in split_names for d in subdirs)

        if has_splits:
            for split_dir in subdirs:
                if split_dir.name.lower() in split_names:
                    split_label = "val" if split_dir.name.lower() == "validation" else split_dir.name.lower()
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            self._scan_folder(class_dir, records, split=split_label, default_label=class_dir.name)
        else:
            for class_dir in subdirs:
                if class_dir.is_dir():
                    self._scan_folder(class_dir, records, split="train", default_label=class_dir.name)

        return records

    def _scan_folder(
        self,
        folder: Path,
        records: List[ManifestRecord],
        split: str,
        default_label: str,
    ):
        for root, _, files in os.walk(folder):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    full_path = Path(root) / file
                    try:
                        rel_path = str(full_path.relative_to(self.dataset_root)).replace("\\", "/")
                    except ValueError:
                        rel_path = str(full_path).replace("\\", "/")

                    part_id, view_id, severity = self.parse_filename_metadata(file)

                    records.append(
                        ManifestRecord(
                            path=rel_path,
                            label=default_label,
                            split=split,
                            part_id=part_id,
                            view_id=view_id,
                            severity=severity,
                            is_valid=True,
                        )
                    )

    @staticmethod
    def parse_filename_metadata(filename: str) -> Tuple[Optional[str], Optional[str], SeverityLevel]:
        """
        Extracts part_id, view_id, and severity level from structured filename.
        Example: part001_camTop_scratch_high_01.png -> part_001, view_Top, SeverityLevel.HIGH
        """
        name_no_ext = Path(filename).stem

        # Parse part_id
        part_match = re.search(r"part[-_]?(\d+|[a-zA-Z0-9]+)", name_no_ext, re.IGNORECASE)
        part_id = f"part_{part_match.group(1)}" if part_match else None

        # Parse view_id / camera_id
        cam_match = re.search(r"(cam|view)[-_]?([a-zA-Z0-9]+)", name_no_ext, re.IGNORECASE)
        view_id = f"view_{cam_match.group(2)}" if cam_match else None

        # Parse severity
        lower_name = name_no_ext.lower()
        if "critical" in lower_name:
            severity = SeverityLevel.CRITICAL
        elif "high" in lower_name:
            severity = SeverityLevel.HIGH
        elif "medium" in lower_name or "med" in lower_name:
            severity = SeverityLevel.MEDIUM
        elif "low" in lower_name:
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.NONE

        return part_id, view_id, severity
