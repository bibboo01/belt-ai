import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import main as main_module
from app.main import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(
    app,
    raise_server_exceptions=False,
)


# ============================================================
# TEST CONFIG
# ============================================================

CAMERA_ID = 1
TEST_IMAGE = "tests/sample.jpg"


# ============================================================
# HELPERS
# ============================================================

def post_detect(
    camera_id=CAMERA_ID,
    image_path=TEST_IMAGE,
):
    """
    Send a multipart request to /api/v1/detect.
    """

    with open(
        image_path,
        "rb",
    ) as f:

        return client.post(
            "/api/v1/detect",
            params={
                "camera_id": camera_id,
            },
            files={
                "image": (
                    "sample.jpg",
                    f,
                    "image/jpeg",
                )
            },
        )


def assert_error_contract(
    response,
    expected_status,
    expected_code,
):
    """
    Validate standard Belt AI error response.
    """

    assert (
        response.status_code
        == expected_status
    ), response.text

    data = response.json()

    assert (
        data["status"]
        == "error"
    )

    assert (
        data["code"]
        == expected_code
    ), data

    assert (
        "message"
        in data
    )

    assert (
        "request_id"
        in data
    )


# ============================================================
# API-16
# MODEL NOT READY
# ============================================================

def test_api_16_model_not_ready(
    monkeypatch,
):
    """
    API-16

    Simulate model manager reporting
    that the model is not ready.
    """

    # --------------------------------------------------------
    # Model unavailable
    # --------------------------------------------------------

    monkeypatch.setattr(
        main_module.model_manager,
        "get_model",
        lambda: None,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "is_ready",
        lambda: False,
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert_error_contract(
        response=response,
        expected_status=503,
        expected_code="MODEL_NOT_READY",
    )


# ============================================================
# API-17
# MODEL NOT REGISTERED
# ============================================================

def test_api_17_model_not_registered(
    monkeypatch,
):
    """
    API-17

    Simulate a valid loaded model but
    the model is missing from the database registry.
    """

    # ========================================================
    # Fake model
    # ========================================================

    class FakeModel:

        names = {
            0: "dogear-belt",
            1: "good-belt",
            2: "splice-belt",
        }

    fake_model = FakeModel()

    # --------------------------------------------------------
    # Model manager
    # --------------------------------------------------------

    monkeypatch.setattr(
        main_module.model_manager,
        "get_model",
        lambda: fake_model,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "is_ready",
        lambda: True,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "model_name",
        "YOLO11",
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "model_version",
        "FAULT-TEST",
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "confidence",
        0.5,
    )

    # --------------------------------------------------------
    # Model registry returns nothing
    # --------------------------------------------------------

    monkeypatch.setattr(
        main_module.repository,
        "get_model_version",
        lambda **kwargs: None,
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert_error_contract(
        response=response,
        expected_status=503,
        expected_code="MODEL_NOT_REGISTERED",
    )


# ============================================================
# API-18
# INFERENCE FAILED
# ============================================================

def test_api_18_inference_failed(
    monkeypatch,
):
    """
    API-18

    Simulate model inference failure.
    """

    # ========================================================
    # Fake model
    # ========================================================

    class FakeModel:

        names = {
            0: "dogear-belt",
            1: "good-belt",
            2: "splice-belt",
        }

        def predict(
            self,
            source,
            conf,
            verbose,
        ):

            raise RuntimeError(
                "SIMULATED_INFERENCE_FAILURE"
            )

    fake_model = FakeModel()

    # --------------------------------------------------------
    # Model manager
    # --------------------------------------------------------

    monkeypatch.setattr(
        main_module.model_manager,
        "get_model",
        lambda: fake_model,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "is_ready",
        lambda: True,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "model_name",
        "YOLO11",
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "model_version",
        "FAULT-TEST",
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "confidence",
        0.5,
    )

    # --------------------------------------------------------
    # Model registry
    # --------------------------------------------------------

    monkeypatch.setattr(
        main_module.repository,
        "get_model_version",
        lambda **kwargs: {
            "model_id": 2,
            "model_name": "YOLO11",
            "model_version": "FAULT-TEST",
        },
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert_error_contract(
        response=response,
        expected_status=500,
        expected_code="INFERENCE_FAILED",
    )


# ============================================================
# API-19
# DATABASE TRANSACTION FAILED
# ============================================================

def test_api_19_database_transaction_failed(
    monkeypatch,
):
    """
    API-19

    Simulate:

        Model inference
            ↓
        Detection parsing
            ↓
        Database transaction
            ↓
        SQLAlchemyError

    Expected:

        HTTP 500
        DATABASE_ERROR
        Image recovery attempt
    """

    # ========================================================
    # Fake YOLO Box
    # ========================================================
    #
    # IMPORTANT:
    #
    # main.py uses:
    #
    #   box.cls[0]
    #   box.conf[0]
    #   box.xyxy[0].tolist()
    #
    # Therefore all values must be NumPy arrays.
    # ========================================================

    class FakeBox:

        def __init__(self):

            self.cls = np.array(
                [2]
            )

            self.conf = np.array(
                [0.95]
            )

            self.xyxy = np.array(
                [
                    [
                        100,
                        100,
                        500,
                        500,
                    ]
                ]
            )

    # ========================================================
    # Fake YOLO Boxes
    # ========================================================

    class FakeBoxes:

        def __iter__(self):

            return iter(
                [
                    FakeBox()
                ]
            )

    # ========================================================
    # Fake YOLO Result
    # ========================================================

    class FakeResult:

        def __init__(self):

            self.boxes = FakeBoxes()

        def plot(self):

            return np.zeros(
                (
                    1080,
                    1920,
                    3,
                ),
                dtype=np.uint8,
            )

    # ========================================================
    # Fake YOLO Model
    # ========================================================

    class FakeModel:

        names = {
            0: "dogear-belt",
            1: "good-belt",
            2: "splice-belt",
        }

        def predict(
            self,
            source,
            conf,
            verbose,
        ):

            return [
                FakeResult()
            ]

    fake_model = FakeModel()

    # ========================================================
    # Model Manager
    # ========================================================

    monkeypatch.setattr(
        main_module.model_manager,
        "get_model",
        lambda: fake_model,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "is_ready",
        lambda: True,
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "model_name",
        "YOLO11",
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "model_version",
        "FAULT-TEST",
    )

    monkeypatch.setattr(
        main_module.model_manager,
        "confidence",
        0.5,
    )

    # ========================================================
    # Model Registry
    # ========================================================

    monkeypatch.setattr(
        main_module.repository,
        "get_model_version",
        lambda **kwargs: {
            "model_id": 2,
            "model_name": "YOLO11",
            "model_version": "FAULT-TEST",
        },
    )

    # ========================================================
    # Force Database Transaction Failure
    # ========================================================

    def fake_begin():

        raise SQLAlchemyError(
            "SIMULATED_DATABASE_FAILURE"
        )

    monkeypatch.setattr(
        main_module.repository,
        "begin",
        fake_begin,
    )

    # ========================================================
    # Capture ERROR Status Update
    # ========================================================

    status_updates = []

    original_update = (
        main_module.repository
        .update_image_processed
    )

    def tracking_update(
        image_id,
        status,
        inference_time_ms=None,
        result_path=None,
    ):

        status_updates.append(
            {
                "image_id": image_id,
                "status": status,
            }
        )

        return original_update(
            image_id=image_id,
            status=status,
            inference_time_ms=(
                inference_time_ms
            ),
            result_path=result_path,
        )

    monkeypatch.setattr(
        main_module.repository,
        "update_image_processed",
        tracking_update,
    )

    # ========================================================
    # Request
    # ========================================================

    response = post_detect()

    # ========================================================
    # Validate API Error Contract
    # ========================================================

    assert_error_contract(
        response=response,
        expected_status=500,
        expected_code="DATABASE_ERROR",
    )

    # ========================================================
    # Validate ERROR Recovery
    # ========================================================

    assert len(
        status_updates
    ) >= 1

    assert any(
        update["status"]
        == "ERROR"
        for update in status_updates
    )


# ============================================================
# SUMMARY
# ============================================================

def test_fault_injection_summary():
    """
    Summary marker.

    The actual validation is performed by
    API-16 through API-19.
    """

    assert True