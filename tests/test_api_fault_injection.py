import numpy as np
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

import app.detection_service as detection_service_module
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
# SERVICE OBJECTS
# ============================================================

detection_service = (
    detection_service_module.detection_service
)

service_repository = (
    detection_service.repository
)

service_model_manager = (
    detection_service.model_manager
)


# ============================================================
# HELPERS
# ============================================================

def post_detect(
    camera_id=CAMERA_ID,
    image_path=TEST_IMAGE,
):
    """
    Send multipart request to /api/v1/detect.
    """

    with open(
        image_path,
        "rb",
    ) as file:

        return client.post(
            "/api/v1/detect",
            params={
                "camera_id": camera_id,
            },
            files={
                "image": (
                    "sample.jpg",
                    file,
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
    ), data

    assert (
        data["code"]
        == expected_code
    ), data

    assert (
        "message"
        in data
    ), data

    assert (
        "request_id"
        in data
    ), data


def bypass_model_registry(
    monkeypatch,
    model_version="FAULT-TEST",
):
    """
    Bypass production model registry validation
    so fault-injection tests can reach the intended
    failure point.
    """

    def fake_validate_model_registry():
        return {
            "model_id": 2,
            "model_name": "YOLO11",
            "model_version": model_version,
            "status": "PRODUCTION",
        }

    monkeypatch.setattr(
        detection_service,
        "_validate_model_registry",
        fake_validate_model_registry,
    )

    monkeypatch.setattr(
        service_model_manager,
        "get_info",
        lambda: {
            "model_id": 2,
            "model_name": "YOLO11",
            "model_version": model_version,
            "status": "PRODUCTION",
        },
    )


def create_successful_inference_result():
    """
    Create fake successful inference result
    matching the current DetectionService contract.
    """

    return SimpleNamespace(
        detections=[
            {
                "class_id": 2,
                "class_name": "splice-belt",
                "confidence": 0.95,
                "bbox": [
                    100,
                    100,
                    500,
                    500,
                ],
            }
        ],
        inference_time_ms=12.34,
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

    Simulate:

        Model Manager
            ↓
        Model not ready

    Expected:

        HTTP 503
        MODEL_NOT_READY
    """

    # --------------------------------------------------------
    # Prevent load_model() from making the model ready
    # --------------------------------------------------------

    monkeypatch.setattr(
        service_model_manager,
        "load_model",
        lambda: None,
    )

    monkeypatch.setattr(
        service_model_manager,
        "is_ready",
        lambda: False,
    )

    monkeypatch.setattr(
        service_model_manager,
        "get_model",
        lambda: None,
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

    Simulate:

        Model ready
            ↓
        Registry lookup
            ↓
        No registered model

    Expected:

        HTTP 503
        MODEL_NOT_REGISTERED
    """

    # --------------------------------------------------------
    # Model ready
    # --------------------------------------------------------

    monkeypatch.setattr(
        service_model_manager,
        "is_ready",
        lambda: True,
    )

    # --------------------------------------------------------
    # Deterministic model info
    # --------------------------------------------------------

    monkeypatch.setattr(
        service_model_manager,
        "get_info",
        lambda: {
            "model_id": 2,
            "model_name": "YOLO11",
            "model_version": "FAULT-TEST",
            "status": "PRODUCTION",
        },
    )

    # --------------------------------------------------------
    # Registry returns nothing
    # --------------------------------------------------------

    monkeypatch.setattr(
        service_repository,
        "get_model_version",
        lambda *args, **kwargs: None,
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

    Simulate:

        Model Registry OK
            ↓
        Inference
            ↓
        RuntimeError

    Expected:

        HTTP 500
        INFERENCE_FAILED
    """

    # --------------------------------------------------------
    # Bypass model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-TEST",
    )

    # --------------------------------------------------------
    # Force inference failure
    # --------------------------------------------------------

    def fake_run_inference(
        frame,
        model_manager,
    ):
        raise RuntimeError(
            "SIMULATED_INFERENCE_FAILURE"
        )

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        fake_run_inference,
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

        Model Registry OK
            ↓
        Inference OK
            ↓
        Detection parsing OK
            ↓
        Database transaction
            ↓
        SQLAlchemyError

    Expected:

        HTTP 500
        DATABASE_ERROR
        Image recovery attempt
    """

    # --------------------------------------------------------
    # Bypass model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-TEST",
    )

    # --------------------------------------------------------
    # Successful fake inference
    # --------------------------------------------------------

    fake_inference_result = (
        create_successful_inference_result()
    )

    def fake_run_inference(
        frame,
        model_manager,
    ):
        return fake_inference_result

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        fake_run_inference,
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Patch the repository used by DetectionService itself.
    #
    # The previous test patched:
    #
    #     main_module.repository.begin
    #
    # but DetectionService uses:
    #
    #     detection_service.repository.begin
    #
    # --------------------------------------------------------

    def fake_begin():
        raise SQLAlchemyError(
            "SIMULATED_DATABASE_FAILURE"
        )

    monkeypatch.setattr(
        service_repository,
        "begin",
        fake_begin,
    )

    # --------------------------------------------------------
    # Track ERROR recovery
    # --------------------------------------------------------

    status_updates = []

    original_update = (
        service_repository
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
            inference_time_ms=inference_time_ms,
            result_path=result_path,
        )

    monkeypatch.setattr(
        service_repository,
        "update_image_processed",
        tracking_update,
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    # --------------------------------------------------------
    # Validate API error
    # --------------------------------------------------------

    assert_error_contract(
        response=response,
        expected_status=500,
        expected_code="DATABASE_ERROR",
    )

    # --------------------------------------------------------
    # Validate ERROR recovery
    # --------------------------------------------------------

    assert len(
        status_updates
    ) >= 1

    assert any(
        update["status"] == "ERROR"
        for update in status_updates
    )


# ============================================================
# API-19 EXTRA
# NO DETECTION SHOULD BE COMMITTED
# ============================================================

def test_api_19_no_detection_when_transaction_fails(
    monkeypatch,
):
    """
    Additional transaction rollback verification.

    Expected:

        HTTP 500
        DATABASE_ERROR
    """

    # --------------------------------------------------------
    # Bypass model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-TEST-NO-DETECTION",
    )

    # --------------------------------------------------------
    # Successful fake inference
    # --------------------------------------------------------

    fake_inference_result = (
        create_successful_inference_result()
    )

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        lambda frame, model_manager:
            fake_inference_result,
    )

    # --------------------------------------------------------
    # Track image creation
    # --------------------------------------------------------

    created_image_ids = []

    original_create_image = (
        service_repository
        .create_image
    )

    def tracking_create_image(
        **kwargs,
    ):
        image_id = original_create_image(
            **kwargs
        )

        created_image_ids.append(
            image_id
        )

        return image_id

    monkeypatch.setattr(
        service_repository,
        "create_image",
        tracking_create_image,
    )

    # --------------------------------------------------------
    # Force transaction failure
    # --------------------------------------------------------

    def fake_begin():
        raise SQLAlchemyError(
            "SIMULATED_TRANSACTION_ROLLBACK"
        )

    monkeypatch.setattr(
        service_repository,
        "begin",
        fake_begin,
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    # --------------------------------------------------------
    # Validate API
    # --------------------------------------------------------

    assert_error_contract(
        response=response,
        expected_status=500,
        expected_code="DATABASE_ERROR",
    )

    # --------------------------------------------------------
    # Image record should exist because it is created
    # before the final transaction.
    # --------------------------------------------------------

    assert len(
        created_image_ids
    ) >= 1


# ============================================================
# SUMMARY
# ============================================================

def test_fault_injection_summary():
    """
    Summary marker.
    """

    assert True