"""
Manifest Builder & Manager for Dataset Serialization and Ingestion.
Supports JSON and CSV manifest save, load, and round-trip verification.
"""

import csv
import json
from pathlib import Path
from typing import List, Union

from src.data.contracts import DatasetManifest, ManifestRecord, SeverityLevel


class ManifestBuilder:
    """Utility class for building, saving, and loading dataset manifests."""

    @staticmethod
    def build_manifest(
        records: List[ManifestRecord],
        dataset_name: str = "mudguard_dataset",
        version: str = "1.0.0",
    ) -> DatasetManifest:
        """Constructs a DatasetManifest from a list of ManifestRecord instances."""
        classes = sorted(list({r.label for r in records}))
        return DatasetManifest(
            dataset_name=dataset_name,
            version=version,
            records=records,
            total_images=len(records),
            classes_found=classes,
        )

    @staticmethod
    def save_json(manifest: DatasetManifest, output_path: Union[str, Path]):
        """Serializes manifest to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

    @staticmethod
    def load_json(input_path: Union[str, Path]) -> DatasetManifest:
        """Deserializes manifest from JSON."""
        with open(Path(input_path), "r", encoding="utf-8") as f:
            data = f.read()
        return DatasetManifest.model_validate_json(data)

    @staticmethod
    def save_csv(manifest: DatasetManifest, output_path: Union[str, Path]):
        """Serializes manifest to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "path", "label", "split", "part_id", "view_id", "severity",
            "image_hash", "width", "height", "channels", "is_valid"
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in manifest.records:
                row = r.model_dump()
                row["severity"] = r.severity.value if isinstance(r.severity, SeverityLevel) else str(r.severity)
                writer.writerow(row)

    @staticmethod
    def load_csv(input_path: Union[str, Path]) -> DatasetManifest:
        """Deserializes manifest records from CSV."""
        records = []
        with open(Path(input_path), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    ManifestRecord(
                        path=row["path"],
                        label=row["label"],
                        split=row.get("split", "train"),
                        part_id=row.get("part_id") or None,
                        view_id=row.get("view_id") or None,
                        severity=SeverityLevel(row["severity"]) if row.get("severity") in SeverityLevel._value2member_map_ else SeverityLevel.NONE,
                        image_hash=row.get("image_hash") or None,
                        width=int(row["width"]) if row.get("width") else None,
                        height=int(row["height"]) if row.get("height") else None,
                        channels=int(row["channels"]) if row.get("channels") else 3,
                        is_valid=row.get("is_valid", "True").lower() == "true",
                    )
                )
        return ManifestBuilder.build_manifest(records)


ManifestManager = ManifestBuilder
