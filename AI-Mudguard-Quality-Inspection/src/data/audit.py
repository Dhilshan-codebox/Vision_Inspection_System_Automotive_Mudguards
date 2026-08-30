"""
Dataset Audit Engine for Automotive Mudguard Inspection.
Performs non-destructive image verification, SHA-256 duplicate detection,
cross-split data leakage detection, and dimension statistics.
"""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union
from PIL import Image

from src.data.contracts import (
    AuditFinding,
    DatasetAuditReport,
    DatasetManifest,
    ManifestRecord,
)
from src.data.discovery import DatasetDiscovery
from src.data.manifest import ManifestBuilder


class DatasetAuditor:
    """
    Audits a mudguard inspection dataset for file corruption, SHA-256 duplicates,
    split leakage, and label distribution anomalies without modifying any files.
    """

    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root).resolve()
        self.discovery = DatasetDiscovery(self.dataset_root)

    @staticmethod
    def compute_sha256(file_path: Union[str, Path]) -> str:
        """Computes SHA-256 hex digest of a file in chunks."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def audit(self) -> Tuple[DatasetAuditReport, DatasetManifest]:
        """Runs complete non-destructive dataset audit."""
        records = self.discovery.discover_all()
        findings: List[AuditFinding] = []

        valid_records: List[ManifestRecord] = []
        corrupted_records: List[ManifestRecord] = []

        hash_to_records: Dict[str, List[ManifestRecord]] = defaultdict(list)
        class_counts: Dict[str, int] = defaultdict(int)
        split_counts: Dict[str, int] = defaultdict(int)

        widths: List[int] = []
        heights: List[int] = []

        for record in records:
            full_path = self.dataset_root / record.path
            if not full_path.exists():
                full_path = Path(record.path)

            # Check image integrity
            try:
                with Image.open(full_path) as img:
                    img.verify()

                # Re-open after verify to fetch dimensions
                with Image.open(full_path) as img:
                    w, h = img.size
                    channels = len(img.getbands())
                    record.width = w
                    record.height = h
                    record.channels = channels
                    widths.append(w)
                    heights.append(h)

                file_hash = self.compute_sha256(full_path)
                record.image_hash = file_hash
                record.is_valid = True
                valid_records.append(record)

                hash_to_records[file_hash].append(record)
                class_counts[record.label] += 1
                split_counts[record.split] += 1

            except Exception as e:
                record.is_valid = False
                corrupted_records.append(record)

        # Audit finding: Corrupted Images
        if corrupted_records:
            corrupt_paths = [r.path for r in corrupted_records]
            findings.append(
                AuditFinding(
                    category="corruption",
                    severity="ERROR",
                    message=f"Found {len(corrupted_records)} corrupted or unreadable image files.",
                    affected_files=corrupt_paths,
                )
            )

        # Audit finding: Duplicates & Split Leakage
        duplicate_count = 0
        split_leakage_count = 0
        leaking_files: List[str] = []

        for img_hash, r_list in hash_to_records.items():
            if len(r_list) > 1:
                duplicate_count += len(r_list) - 1
                splits_involved = {r.split for r in r_list}
                if len(splits_involved) > 1:
                    split_leakage_count += 1
                    leaking_files.extend([r.path for r in r_list])

        if split_leakage_count > 0:
            findings.append(
                AuditFinding(
                    category="split_leakage",
                    severity="ERROR",
                    message=f"Found {split_leakage_count} exact duplicate images leaked across data splits.",
                    affected_files=leaking_files,
                )
            )

        dimension_stats = {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
        }

        passed_quality_gate = len(corrupted_records) == 0 and split_leakage_count == 0

        report = DatasetAuditReport(
            dataset_root=str(self.dataset_root),
            total_images=len(records),
            valid_images=len(valid_records),
            corrupted_images=len(corrupted_records),
            duplicate_images=duplicate_count,
            split_leakage_count=split_leakage_count,
            class_counts=dict(class_counts),
            split_counts=dict(split_counts),
            dimension_stats=dimension_stats,
            findings=findings,
            passed_quality_gate=passed_quality_gate,
        )

        manifest = ManifestBuilder.build_manifest(records, dataset_name=self.dataset_root.name)
        return report, manifest
