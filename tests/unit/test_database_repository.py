from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.database.repository import DatabaseRepository


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_engine():
    """
    Create a mocked SQLAlchemy engine.
    """
    return MagicMock()


@pytest.fixture
def repository(mock_engine):
    """
    Create DatabaseRepository with a mocked engine.
    """
    return DatabaseRepository(
        engine=mock_engine,
    )


@pytest.fixture
def mock_connection(mock_engine):
    """
    Create a mocked database connection and wire it to:

        engine.connect()
        engine.begin()
    """

    connection = MagicMock()

    # engine.connect()
    mock_engine.connect.return_value.__enter__.return_value = (
        connection
    )

    # engine.begin()
    mock_engine.begin.return_value.__enter__.return_value = (
        connection
    )

    return connection


# ============================================================================
# CONNECTION
# ============================================================================


def test_test_connection_success(
    repository,
    mock_connection,
):
    """
    test_connection() should return True when SELECT 1 succeeds.
    """

    result = repository.test_connection()

    assert result is True

    mock_connection.execute.assert_called_once()


def test_test_connection_failure(
    repository,
    mock_connection,
):
    """
    test_connection() should return False when database access fails.
    """

    mock_connection.execute.side_effect = SQLAlchemyError(
        "Database error"
    )

    result = repository.test_connection()

    assert result is False


def test_begin_returns_engine_transaction(
    repository,
):
    """
    begin() should return engine.begin().
    """

    transaction = MagicMock()

    with patch.object(
        repository.engine,
        "begin",
        return_value=transaction,
    ) as mock_begin:
        result = repository.begin()

    assert result is transaction

    mock_begin.assert_called_once()


# ============================================================================
# MODEL REGISTRY
# ============================================================================


def test_get_model_version_success(
    repository,
    mock_connection,
):
    """
    get_model_version() should return the matching model.
    """

    mock_result = MagicMock()

    mock_result.mappings.return_value.first.return_value = {
        "model_id": 2,
        "model_name": "YOLO11",
        "model_version": "ODBv54",
        "model_type": "detect",
        "model_path": "models/production/best.pt",
        "confidence_threshold": 0.5,
        "status": "PRODUCTION",
    }

    mock_connection.execute.return_value = mock_result

    result = repository.get_model_version(
        model_name="YOLO11",
        model_version="ODBv54",
    )

    assert result is not None
    assert result["model_id"] == 2
    assert result["model_name"] == "YOLO11"
    assert result["model_version"] == "ODBv54"
    assert result["status"] == "PRODUCTION"

    mock_connection.execute.assert_called_once()


def test_get_model_version_not_found(
    repository,
    mock_connection,
):
    """
    get_model_version() should return None when the model is not found.
    """

    mock_result = MagicMock()

    mock_result.mappings.return_value.first.return_value = None

    mock_connection.execute.return_value = mock_result

    result = repository.get_model_version(
        model_name="YOLO11",
        model_version="UNKNOWN",
    )

    assert result is None


def test_get_model_version_database_error(
    repository,
    mock_connection,
):
    """
    get_model_version() should re-raise SQLAlchemy errors.
    """

    mock_connection.execute.side_effect = SQLAlchemyError(
        "Database error"
    )

    with pytest.raises(SQLAlchemyError):
        repository.get_model_version(
            model_name="YOLO11",
            model_version="ODBv54",
        )


# ============================================================================
# CAMERA
# ============================================================================


def test_get_camera_success(
    repository,
    mock_connection,
):
    """
    get_camera() should return the matching camera.
    """

    mock_result = MagicMock()

    mock_result.mappings.return_value.first.return_value = {
        "camera_id": 1,
        "machine_id": 1,
        "camera_name": "Camera 1",
        "serial_number": "OAK-001",
        "ip_address": "192.168.1.10",
    }

    mock_connection.execute.return_value = mock_result

    result = repository.get_camera(
        camera_id=1,
    )

    assert result is not None
    assert result["camera_id"] == 1
    assert result["machine_id"] == 1
    assert result["camera_name"] == "Camera 1"

    mock_connection.execute.assert_called_once()


def test_get_camera_not_found(
    repository,
    mock_connection,
):
    """
    get_camera() should return None when the camera is not found.
    """

    mock_result = MagicMock()

    mock_result.mappings.return_value.first.return_value = None

    mock_connection.execute.return_value = mock_result

    result = repository.get_camera(
        camera_id=999999,
    )

    assert result is None


def test_get_camera_database_error(
    repository,
    mock_connection,
):
    """
    get_camera() should re-raise SQLAlchemy errors.
    """

    mock_connection.execute.side_effect = SQLAlchemyError(
        "Database error"
    )

    with pytest.raises(SQLAlchemyError):
        repository.get_camera(
            camera_id=1,
        )


# ============================================================================
# IMAGE - NORMAL
# ============================================================================


def test_create_image(
    repository,
    mock_connection,
):
    """
    create_image() should execute an INSERT and return image_id.
    """

    mock_result = MagicMock()

    mock_result.scalar_one.return_value = 123

    mock_connection.execute.return_value = mock_result

    result = repository.create_image(
        camera_id=1,
        filename="test.jpg",
        image_path="images/original/test.jpg",
        capture_time="2026-09-17 09:00:00",
    )

    assert result == 123

    mock_connection.execute.assert_called_once()


# ============================================================================
# DETECTION - NORMAL
# ============================================================================


def test_create_detection(
    repository,
    mock_connection,
):
    """
    create_detection() should execute an INSERT
    and return detection_id.
    """

    mock_result = MagicMock()

    mock_result.scalar_one.return_value = 456

    mock_connection.execute.return_value = mock_result

    result = repository.create_detection(
        image_id=123,
        model_id=2,
        class_id=1,
        class_name="good-belt",
        confidence=0.95,
        x1=100,
        y1=200,
        x2=300,
        y2=400,
    )

    assert result == 456

    mock_connection.execute.assert_called_once()


# ============================================================================
# INSPECTION RESULT - NORMAL
# ============================================================================


def test_create_inspection_result(
    repository,
    mock_connection,
):
    """
    create_inspection_result() should execute an INSERT.
    """

    mock_result = MagicMock()

    mock_result.scalar_one.return_value = 789

    mock_connection.execute.return_value = mock_result

    result = repository.create_inspection_result(
        image_id=123,
        overall_result="GOOD",
        good_count=1,
        splice_count=0,
        dogear_count=0,
    )

    assert result == 789

    mock_connection.execute.assert_called_once()


# ============================================================================
# IMAGE STATUS - NORMAL
# ============================================================================


def test_update_image_processed(
    repository,
    mock_connection,
):
    """
    update_image_processed() should execute an UPDATE.
    """

    mock_result = MagicMock()

    mock_result.rowcount = 1

    mock_connection.execute.return_value = mock_result

    result = repository.update_image_processed(
        image_id=123,
        status="DONE",
        inference_time_ms=3850.24,
        result_path=(
            "images/result/"
            "123_test_result.jpg"
        ),
    )

    assert result is None

    mock_connection.execute.assert_called_once()


# ============================================================================
# IMAGE - TRANSACTION
# ============================================================================


def test_create_image_tx(
    repository,
    mock_connection,
):
    """
    create_image_tx() should execute an INSERT
    inside an existing transaction.
    """

    mock_result = MagicMock()

    mock_result.scalar_one.return_value = 123

    mock_connection.execute.return_value = mock_result

    result = repository.create_image_tx(
        connection=mock_connection,
        camera_id=1,
        filename="test.jpg",
        image_path="images/original/test.jpg",
        capture_time="2026-09-17 09:00:00",
    )

    assert result == 123

    mock_connection.execute.assert_called_once()


# ============================================================================
# DETECTION - TRANSACTION
# ============================================================================


def test_create_detection_tx(
    repository,
    mock_connection,
):
    """
    create_detection_tx() should execute an INSERT
    inside an existing transaction.
    """

    mock_result = MagicMock()

    mock_result.scalar_one.return_value = 456

    mock_connection.execute.return_value = mock_result

    result = repository.create_detection_tx(
        connection=mock_connection,
        image_id=123,
        model_id=2,
        class_id=1,
        class_name="good-belt",
        confidence=0.95,
        x1=100,
        y1=200,
        x2=300,
        y2=400,
    )

    assert result == 456

    mock_connection.execute.assert_called_once()


# ============================================================================
# INSPECTION RESULT - TRANSACTION
# ============================================================================


def test_create_inspection_result_tx(
    repository,
    mock_connection,
):
    """
    create_inspection_result_tx() should execute an INSERT
    inside an existing transaction.
    """

    mock_result = MagicMock()

    mock_result.scalar_one.return_value = 789

    mock_connection.execute.return_value = mock_result

    result = repository.create_inspection_result_tx(
        connection=mock_connection,
        image_id=123,
        overall_result="GOOD",
        good_count=1,
        splice_count=0,
        dogear_count=0,
    )

    assert result == 789

    mock_connection.execute.assert_called_once()


# ============================================================================
# REPOSITORY CONSTRUCTION
# ============================================================================


def test_repository_uses_provided_engine(
    mock_engine,
):
    """
    DatabaseRepository should use the engine provided by the caller.
    """

    repository = DatabaseRepository(
        engine=mock_engine,
    )

    assert repository.engine is mock_engine
