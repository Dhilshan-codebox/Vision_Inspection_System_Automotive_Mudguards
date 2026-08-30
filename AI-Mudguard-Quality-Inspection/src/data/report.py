"""
Report Generator for Dataset Audit Results.
Generates formatted Markdown and JSON reports from DatasetAuditReport instances.
"""

import json
from pathlib import Path
from typing import Union

from src.data.contracts import DatasetAuditReport


class ReportGenerator:
    """Generates human-readable Markdown and structured JSON audit reports."""

    @staticmethod
    def generate_markdown(report: DatasetAuditReport) -> str:
        """Generates Markdown representation of DatasetAuditReport."""
        status_str = "PASS" if report.passed_quality_gate else "FAIL"
        md = [
            "# Dataset Audit Report",
            f"**Dataset Root**: `{report.dataset_root}`",
            f"**Quality Gate Status**: **{status_str}**",
            "",
            "## Summary Metrics",
            f"- **Total Images**: {report.total_images}",
            f"- **Valid Images**: {report.valid_images}",
            f"- **Corrupted Images**: {report.corrupted_images}",
            f"- **Duplicate Images**: {report.duplicate_images}",
            f"- **Split Leakage Count**: {report.split_leakage_count}",
            "",
            "## Class Distribution",
        ]

        for cls_name, count in report.class_counts.items():
            md.append(f"- **{cls_name}**: {count}")

        md.extend([
            "",
            "## Image Dimensions",
            f"- **Width Range**: {report.dimension_stats.get('min_width', 0)}px - {report.dimension_stats.get('max_width', 0)}px",
            f"- **Height Range**: {report.dimension_stats.get('min_height', 0)}px - {report.dimension_stats.get('max_height', 0)}px",
            "",
            "## Audit Findings",
        ])

        if not report.findings:
            md.append("No critical audit findings detected.")
        else:
            for finding in report.findings:
                md.append(f"### [{finding.severity}] {finding.category.upper()}")
                md.append(f"{finding.message}")
                if finding.affected_files:
                    md.append("Affected files (first 5):")
                    for f in finding.affected_files[:5]:
                        md.append(f"  - `{f}`")

        return "\n".join(md)

    @staticmethod
    def save_markdown_report(report: DatasetAuditReport, output_path: Union[str, Path]):
        """Saves Markdown audit report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        md_text = ReportGenerator.generate_markdown(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_text)

    @staticmethod
    def save_json_report(report: DatasetAuditReport, output_path: Union[str, Path]):
        """Saves JSON audit report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
