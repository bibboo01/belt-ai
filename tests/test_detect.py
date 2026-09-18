from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


# -----------------------------------------------------------------------------
# Test Client
# -----------------------------------------------------------------------------

client = TestClient(app)


# -----------------------------------------------------------------------------
# Test Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE = (
    PROJECT_ROOT
    / "images"
    / "original"
    / "153_recovery_audit.jpg"
)


# -----------------------------------------------------------------------------
# Test: Successful Detection
# -----------------------------------------------------------------------------

def test_detect_success():
    """
    Verify that /detect accepts a valid image
    and returns a successful detection response.
    """

    assert TEST_IMAGE.exists(), (
        f"Test image not found: {TEST_IMAGE}"
    )

    with TEST_IMAGE.open("rb") as image_file:
        response = client.post(
            "/detect?camera_id=1",
            files={
                "image": (
                    TEST_IMAGE.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"

    assert "image" in data
    assert "model" in data
    assert "result" in data
    assert "detections" in data

    assert data["image"]["image_id"] > 0

    assert data["model"]["model_id"] == 2
    assert data["model"]["name"] == "YOLO11"
    assert data["model"]["version"] == "ODBv54"

    assert data["result"]["overall"] in {
        "GOOD",
        "NG",
    }

    assert data["result"]["total_detections"] == len(
        data["detections"]
    )


# -----------------------------------------------------------------------------
# Test: Invalid Camera
# -----------------------------------------------------------------------------

def test_detect_invalid_camera():
    """
    Verify that /detect rejects an unknown camera.
    """

    with TEST_IMAGE.open("rb") as image_file:
        response = client.post(
            "/detect?camera_id=999999",
            files={
                "image": (
                    TEST_IMAGE.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code != 200


# -----------------------------------------------------------------------------
# Test: Invalid Content Type
# -----------------------------------------------------------------------------

def test_detect_invalid_content_type():
    """
    Verify that /detect rejects unsupported content types.
    """

    with TEST_IMAGE.open("rb") as image_file:
        response = client.post(
            "/detect?camera_id=1",
            files={
                "image": (
                    "test.txt",
                    image_file,
                    "text/plain",
                )
            },
        )

    assert response.status_code != 200

