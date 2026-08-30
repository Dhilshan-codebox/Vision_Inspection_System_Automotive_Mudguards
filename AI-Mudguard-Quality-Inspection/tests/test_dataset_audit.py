import pytest
from pathlib import Path
from PIL import Image
import os

from src.data import (
    DatasetDiscovery,
    ManifestBuilder,
    DatasetAuditor,
    ReportGenerator,
    SeverityLevel,
    DatasetManifest,
    DatasetAuditReport,
)


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory with synthetic test images."""
    dataset_dir = tmp_path / "mock_mudguard_dataset"
    dataset_dir.mkdir()

    # Create train split
    train_dir = dataset_dir / "train"
    train_scratch = train_dir / "Scratch"
    train_good = train_dir / "Good"
    train_scratch.mkdir(parents=True)
    train_good.mkdir(parents=True)

    # Create val split
    val_dir = dataset_dir / "val"
    val_dent = val_dir / "Dent"
    val_good = val_dir / "Good"
    val_dent.mkdir(parents=True)
    val_good.mkdir(parents=True)

    # Create images
    img_red = Image.new("RGB", (128, 128), color=(255, 0, 0))
    img_green = Image.new("RGB", (128, 128), color=(0, 255, 0))
    img_blue = Image.new("RGB", (256, 256), color=(0, 0, 255))

    img_red.save(train_scratch / "part001_camTop_scratch_high_01.png")
    img_green.save(train_good / "part002_camSide_good_01.png")
    img_blue.save(val_dent / "part003_camFront_dent_medium_01.png")
    img_green.save(val_good / "part004_camSide_good_02.png")

    return dataset_dir


def test_dataset_discovery_split_layout(temp_dataset_dir: Path):
    discovery = DatasetDiscovery(temp_dataset_dir)
    records = discovery.discover()

    assert len(records) == 4
    splits = {r.split for r in records}
    assert "train" in splits
    assert "val" in splits

    labels = {r.label for r in records}
    assert "Scratch" in labels
    assert "Dent" in labels
    assert "Good" in labels


def test_filename_metadata_parsing(temp_dataset_dir: Path):
    discovery = DatasetDiscovery(temp_dataset_dir)
    records = discovery.discover()

    scratch_record = next(r for r in records if "scratch_high" in r.path)
    assert scratch_record.part_id == "part_001"
    assert scratch_record.view_id == "view_Top"
    assert scratch_record.severity == SeverityLevel.HIGH


def test_dataset_auditor_clean(temp_dataset_dir: Path):
    auditor = DatasetAuditor(temp_dataset_dir)
    report, manifest = auditor.audit()

    assert report.total_images == 4
    assert report.valid_images == 4
    assert report.corrupted_images == 0
    assert report.class_counts["Scratch"] == 1
    assert report.class_counts["Dent"] == 1
    assert report.class_counts["Good"] == 2
    assert report.dimension_stats["min_width"] == 128
    assert report.dimension_stats["max_width"] == 256


def test_corrupted_image_detection(tmp_path: Path):
    dataset_dir = tmp_path / "corrupt_dataset"
    class_dir = dataset_dir / "Scratch"
    class_dir.mkdir(parents=True)

    # Valid image
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(class_dir / "valid_01.jpg")

    # Corrupted image file (invalid binary content)
    corrupt_file = class_dir / "corrupted_02.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE")

    auditor = DatasetAuditor(dataset_dir)
    report, manifest = auditor.audit()

    assert report.total_images == 2
    assert report.valid_images == 1
    assert report.corrupted_images == 1
    assert not report.passed_quality_gate
    assert any(f.category == "corruption" for f in report.findings)


def test_split_leakage_detection(tmp_path: Path):
    dataset_dir = tmp_path / "leakage_dataset"
    train_dir = dataset_dir / "train" / "Scratch"
    val_dir = dataset_dir / "val" / "Scratch"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Save exact same pixel image to both train and val
    shared_img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    shared_img.save(train_dir / "leak_img_train.png")
    shared_img.save(val_dir / "leak_img_val.png")

    auditor = DatasetAuditor(dataset_dir)
    report, manifest = auditor.audit()

    assert report.split_leakage_count >= 1
    assert not report.passed_quality_gate
    assert any(f.category == "split_leakage" for f in report.findings)


def test_manifest_builder_json_csv_roundtrip(temp_dataset_dir: Path, tmp_path: Path):
    auditor = DatasetAuditor(temp_dataset_dir)
    report, manifest = auditor.audit()

    # JSON export and import
    json_path = tmp_path / "manifest.json"
    ManifestBuilder.save_json(manifest, json_path)
    loaded_json = ManifestBuilder.load_json(json_path)

    assert loaded_json.total_images == manifest.total_images
    assert len(loaded_json.records) == len(manifest.records)
    assert loaded_json.records[0].path == manifest.records[0].path

    # CSV export and import
    csv_path = tmp_path / "manifest.csv"
    ManifestBuilder.save_csv(manifest, csv_path)
    loaded_csv = ManifestBuilder.load_csv(csv_path)

    assert loaded_csv.total_images == manifest.total_images
    assert loaded_csv.records[0].path == manifest.records[0].path


def test_non_destructive_audit(temp_dataset_dir: Path):
    """Verify that auditing never modifies source image timestamps or contents."""
    files_before = {}
    for p in temp_dataset_dir.rglob("*.png"):
        files_before[p] = (p.stat().st_mtime, p.stat().st_size, DatasetAuditor.compute_sha256(p))

    auditor = DatasetAuditor(temp_dataset_dir)
    report, manifest = auditor.audit()

    for p, (mtime, size, file_hash) in files_before.items():
        assert p.exists(), f"File {p} was deleted!"
        assert p.stat().st_size == size, f"File size changed for {p}!"
        assert DatasetAuditor.compute_sha256(p) == file_hash, f"File content altered for {p}!"


def test_report_generator(temp_dataset_dir: Path, tmp_path: Path):
    auditor = DatasetAuditor(temp_dataset_dir)
    report, _ = auditor.audit()

    md = ReportGenerator.generate_markdown(report)
    assert "# Dataset Audit Report" in md
    assert "Scratch" in md
    assert "128px" in md

    md_path = tmp_path / "AUDIT_REPORT.md"
    ReportGenerator.save_markdown_report(report, md_path)
    assert md_path.exists()
    assert len(md_path.read_text(encoding="utf-8")) > 100
