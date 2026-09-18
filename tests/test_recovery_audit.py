from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import app.detection_service as detection_service_module
from app.main import app
from app.config import settings


# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 1
TEST_IMAGE = Path("tests/sample.jpg")

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ORIGINAL_DIR = (
    PROJECT_DIR
    / "images"
    / "original"
)

RESULT_DIR = (
    PROJECT_DIR
    / "images"
    / "result"
)


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(
    app,
    raise_server_exceptions=False,
)


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(
    settings.database_url
)


# ============================================================
# DETECTION SERVICE OBJECTS
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

def post_detect():
    """
    Send detection request using the real API.
    """

    with open(
        TEST_IMAGE,
        "rb",
    ) as file:

        return client.post(
            "/api/v1/detect",
            params={
                "camera_id": CAMERA_ID,
            },
            files={
                "image": (
                    "recovery_audit.jpg",
                    file,
                    "image/jpeg",
                )
            },
        )


def bypass_model_registry(
    monkeypatch,
    model_version="FAULT-RECOVERY",
):
    """
    Bypass model registry validation.

    This makes the recovery tests fail exactly
    at the intended failure point.
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
    Fake successful inference result.

    This avoids using the real YOLO model during
    fault-injection / recovery tests.
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


def get_latest_image_id():
    """
    Get latest image_id from database.
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT image_id
                FROM belt.image
                ORDER BY image_id DESC
                LIMIT 1
                """
            )
        )

        row = result.fetchone()

    if row is None:
        return None

    return row[0]


def get_image_record(
    image_id,
):
    """
    Read image record directly from PostgreSQL.
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT
                    image_id,
                    filename,
                    image_path,
                    result_path,
                    status,
                    processed_time,
                    inference_time_ms
                FROM belt.image
                WHERE image_id = :image_id
                """
            ),
            {
                "image_id": image_id,
            },
        )

        row = result.mappings().first()

    if row is None:
        return None

    return dict(row)


def count_detections(
    image_id,
):
    """
    Count detections for image.
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM belt.detection
                WHERE image_id = :image_id
                """
            ),
            {
                "image_id": image_id,
            },
        )

        return int(
            result.scalar()
        )


def count_inspection_results(
    image_id,
):
    """
    Count inspection results for image.
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM belt.inspection_result
                WHERE image_id = :image_id
                """
            ),
            {
                "image_id": image_id,
            },
        )

        return int(
            result.scalar()
        )


def find_files_for_image(
    image_id,
):
    """
    Find original/result files generated for image.
    """

    original_files = list(
        ORIGINAL_DIR.glob(
            f"{image_id}_*"
        )
    )

    result_files = list(
        RESULT_DIR.glob(
            f"{image_id}_*"
        )
    )

    return (
        original_files,
        result_files,
    )


def cleanup_image_files(
    image_id,
):
    """
    Remove audit files generated by the test.
    """

    original_files, result_files = (
        find_files_for_image(
            image_id
        )
    )

    for file_path in (
        original_files
        + result_files
    ):

        try:

            file_path.unlink(
                missing_ok=True
            )

        except Exception:

            pass


def track_created_image(
    monkeypatch,
):
    """
    Track image_id created by DetectionService.
    """

    created_image_ids = []

    original_create = (
        service_repository
        .create_image
    )

    def tracking_create_image(
        **kwargs,
    ):
        image_id = original_create(
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

    return created_image_ids


def force_database_failure(
    monkeypatch,
    message="RECOVERY_DATABASE_FAILURE",
):
    """
    Force the final DB transaction to fail.

    IMPORTANT:
    Patch DetectionService's repository,
    not main_module.repository.
    """

    def fake_begin():
        raise SQLAlchemyError(
            message
        )

    monkeypatch.setattr(
        service_repository,
        "begin",
        fake_begin,
    )


# ============================================================
# REC-01
# DATABASE FAILURE
# ============================================================

def test_rec_01_database_failure(
    monkeypatch,
):
    """
    REC-01

    Simulate:

        Model Registry
            ↓
        Inference SUCCESS
            ↓
        Database transaction
            ↓
        SQLAlchemyError

    Expected:

        HTTP 500
        DATABASE_ERROR
    """

    # --------------------------------------------------------
    # Model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-RECOVERY-01",
    )

    # --------------------------------------------------------
    # Successful inference
    # --------------------------------------------------------

    fake_result = (
        create_successful_inference_result()
    )

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        lambda frame, model_manager:
            fake_result,
    )

    # --------------------------------------------------------
    # Force DB failure
    # --------------------------------------------------------

    force_database_failure(
        monkeypatch,
        message="RECOVERY_AUDIT_DATABASE_FAILURE",
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert (
        response.status_code
        == 500
    ), response.text

    data = response.json()

    assert (
        data["status"]
        == "error"
    ), data

    assert (
        data["code"]
        == "DATABASE_ERROR"
    ), data

    assert (
        "request_id"
        in data
    ), data


# ============================================================
# REC-02
# IMAGE STATUS MUST NOT REMAIN PROCESSING
# ============================================================

def test_rec_02_image_status_after_database_failure(
    monkeypatch,
):
    """
    REC-02

    Verify that an image created before
    transaction failure does not remain PROCESSING.

    Expected final state:

        ERROR
    """

    # --------------------------------------------------------
    # Track created image
    # --------------------------------------------------------

    created_image_ids = (
        track_created_image(
            monkeypatch
        )
    )

    # --------------------------------------------------------
    # Model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-RECOVERY-02",
    )

    # --------------------------------------------------------
    # Successful inference
    # --------------------------------------------------------

    fake_result = (
        create_successful_inference_result()
    )

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        lambda frame, model_manager:
            fake_result,
    )

    # --------------------------------------------------------
    # Force DB failure
    # --------------------------------------------------------

    force_database_failure(
        monkeypatch,
        message="RECOVERY_AUDIT_FAILURE_02",
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    assert (
        response.status_code
        == 500
    ), response.text

    data = response.json()

    assert (
        data["code"]
        == "DATABASE_ERROR"
    ), data

    # --------------------------------------------------------
    # Image must have been created
    # --------------------------------------------------------

    assert (
        len(created_image_ids)
        == 1
    )

    image_id = (
        created_image_ids[0]
    )

    # --------------------------------------------------------
    # Read database record
    # --------------------------------------------------------

    image_record = (
        get_image_record(
            image_id
        )
    )

    assert (
        image_record
        is not None
    )

    # --------------------------------------------------------
    # Must not remain PROCESSING
    # --------------------------------------------------------

    assert (
        image_record["status"]
        == "ERROR"
    ), (
        f"Image {image_id} "
        f"remained in status "
        f"{image_record['status']}"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cleanup_image_files(
        image_id
    )


# ============================================================
# REC-03
# NO DETECTION AFTER ROLLBACK
# ============================================================

def test_rec_03_no_detection_after_rollback(
    monkeypatch,
):
    """
    REC-03

    Verify that a failed final transaction
    does not leave detections or inspection results.
    """

    # --------------------------------------------------------
    # Track created image
    # --------------------------------------------------------

    created_image_ids = (
        track_created_image(
            monkeypatch
        )
    )

    # --------------------------------------------------------
    # Model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-RECOVERY-03",
    )

    # --------------------------------------------------------
    # Successful inference
    # --------------------------------------------------------

    fake_result = (
        create_successful_inference_result()
    )

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        lambda frame, model_manager:
            fake_result,
    )

    # --------------------------------------------------------
    # Force DB failure
    # --------------------------------------------------------

    force_database_failure(
        monkeypatch,
        message="ROLLBACK_TEST_FAILURE",
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    assert (
        response.status_code
        == 500
    ), response.text

    data = response.json()

    assert (
        data["code"]
        == "DATABASE_ERROR"
    ), data

    # --------------------------------------------------------
    # Image should exist
    # --------------------------------------------------------

    assert (
        len(created_image_ids)
        == 1
    )

    image_id = (
        created_image_ids[0]
    )

    # --------------------------------------------------------
    # Detection must not exist
    # --------------------------------------------------------

    detection_count = (
        count_detections(
            image_id
        )
    )

    assert (
        detection_count
        == 0
    ), (
        f"Found {detection_count} "
        f"detection(s) after rollback"
    )

    # --------------------------------------------------------
    # Inspection result must not exist
    # --------------------------------------------------------

    inspection_count = (
        count_inspection_results(
            image_id
        )
    )

    assert (
        inspection_count
        == 0
    ), (
        "Inspection result exists "
        "after failed transaction"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cleanup_image_files(
        image_id
    )


# ============================================================
# REC-04
# ORPHAN FILE AUDIT
# ============================================================

def test_rec_04_orphan_file_audit(
    monkeypatch,
):
    """
    REC-04

    Audit files created before database failure.

    Current intended behavior:

        Original image
            -> retained for audit

        Result image
            -> removed during error recovery

    This test reports the final file state.
    """

    # --------------------------------------------------------
    # Track created image
    # --------------------------------------------------------

    created_image_ids = (
        track_created_image(
            monkeypatch
        )
    )

    # --------------------------------------------------------
    # Model registry
    # --------------------------------------------------------

    bypass_model_registry(
        monkeypatch,
        model_version="FAULT-RECOVERY-04",
    )

    # --------------------------------------------------------
    # Successful inference
    # --------------------------------------------------------

    fake_result = (
        create_successful_inference_result()
    )

    monkeypatch.setattr(
        detection_service_module,
        "run_inference",
        lambda frame, model_manager:
            fake_result,
    )

    # --------------------------------------------------------
    # Force DB failure
    # --------------------------------------------------------

    force_database_failure(
        monkeypatch,
        message="ORPHAN_FILE_TEST_FAILURE",
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = post_detect()

    assert (
        response.status_code
        == 500
    ), response.text

    data = response.json()

    assert (
        data["code"]
        == "DATABASE_ERROR"
    ), data

    # --------------------------------------------------------
    # Image ID
    # --------------------------------------------------------

    assert (
        len(created_image_ids)
        == 1
    )

    image_id = (
        created_image_ids[0]
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    original_files, result_files = (
        find_files_for_image(
            image_id
        )
    )

    # --------------------------------------------------------
    # Audit output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "REC-04 ORPHAN FILE AUDIT"
    )

    print(
        "=" * 60
    )

    print(
        f"Image ID      : {image_id}"
    )

    print(
        f"Original count: {len(original_files)}"
    )

    print(
        f"Result count  : {len(result_files)}"
    )

    if original_files:

        print(
            "Original files:"
        )

        for file_path in original_files:

            print(
                f"  - {file_path}"
            )

    if result_files:

        print(
            "Result files:"
        )

        for file_path in result_files:

            print(
                f"  - {file_path}"
            )

    # --------------------------------------------------------
    # Current expected recovery behavior
    #
    # Original is preserved for audit.
    # Result should be removed after failure.
    # --------------------------------------------------------

    assert (
        len(result_files)
        == 0
    ), (
        "Incomplete result image "
        "was not removed"
    )

    # --------------------------------------------------------
    # Original may remain intentionally.
    # --------------------------------------------------------

    print(
        "Result file cleanup: PASS"
    )

    print(
        "Original retention : "
        f"{len(original_files)} file(s)"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Cleanup audit files
    # --------------------------------------------------------

    cleanup_image_files(
        image_id
    )


# ============================================================
# REC-05
# RECOVERY SUMMARY
# ============================================================

def test_recovery_audit_summary():
    """
    REC-05

    Recovery audit summary marker.
    """

    assert True