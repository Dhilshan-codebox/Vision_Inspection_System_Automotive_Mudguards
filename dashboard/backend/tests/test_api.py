import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_inspect_endpoint():
    file_data = ("test.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xafA\x0c\x00\x00\x00\x00IEND\xaeB`\x82", "image/png")
    response = client.post(
        "/api/v1/inspect",
        files={"file": file_data},
        data={"camera_id": "cam_top", "part_id": "part_001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "inspection_id" in data
    assert data["decision"] in {"PASS", "REVIEW", "FAIL", "REQUEST_RECAPTURE"}


def test_evaluate_endpoint():
    response = client.post("/api/v1/evaluate", params={"dataset_dir": "data/mudguard_dataset", "split": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "macro_f1" in data
    assert "sample_count" in data


def test_dataset_audit_endpoint():
    response = client.get("/api/v1/dataset/audit", params={"dataset_dir": "data/mudguard_dataset"})
    assert response.status_code == 200
    data = response.json()
    assert "total_images" in data
    assert "passed_quality_gate" in data


def test_models_endpoint():
    response = client.get("/api/v1/models/current")
    assert response.status_code == 200
    data = response.json()
    assert "active_backbone" in data


def test_feedback_endpoint():
    response = client.post(
        "/api/v1/feedback",
        json={
            "inspection_id": "ins_123",
            "corrected_decision": "FAIL",
            "corrected_label": "Scratch",
            "notes": "Verified scratch under review",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
