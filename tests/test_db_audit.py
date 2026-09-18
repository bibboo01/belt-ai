"""
Belt AI - Database Audit
========================
Production-oriented database integrity and transaction audit.

Run:
    .\.venv\Scripts\python.exe -m pytest .\tests\test_db_audit.py -v -s
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.database.repository import DatabaseRepository


@pytest.fixture(scope="module")
def repo():
    return DatabaseRepository()


def _scalar(repo, sql, params=None):
    with repo.engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


def _camera_id(repo):
    return _scalar(
        repo,
        """
        SELECT camera_id
        FROM belt.camera
        ORDER BY camera_id
        LIMIT 1
        """,
    )


def _model_id(repo):
    return _scalar(
        repo,
        """
        SELECT model_id
        FROM belt.model_version
        WHERE model_name = 'YOLO11'
          AND model_version = 'ODBv54'
        LIMIT 1
        """,
    )


def test_db_01_connection(repo):
    value = _scalar(repo, "SELECT 1")
    assert value == 1
    print("DB-01 PASS connection")


def test_db_02_select_1(repo):
    value = _scalar(repo, "SELECT 1 AS result")
    assert value == 1
    print(f"DB-02 PASS SELECT 1 result={value}")


def test_db_03_camera_exists(repo):
    camera_id = _camera_id(repo)
    assert camera_id is not None
    print(f"DB-03 PASS Camera {camera_id} exists")


def test_db_04_invalid_camera_not_found(repo):
    invalid_camera_id = 999999
    result = _scalar(
        repo,
        """
        SELECT camera_id
        FROM belt.camera
        WHERE camera_id = :camera_id
        """,
        {"camera_id": invalid_camera_id},
    )
    assert result is None
    print(f"DB-04 PASS Camera {invalid_camera_id} not found")


def test_db_05_model_registry(repo):
    model_id = _model_id(repo)
    assert model_id is not None
    print(f"DB-05 PASS YOLO11/ODBv54 registered")


def test_db_06_fake_model_not_found(repo):
    result = _scalar(
        repo,
        """
        SELECT model_id
        FROM belt.model_version
        WHERE model_name = :name
          AND model_version = :version
        """,
        {
            "name": "FAULT-TEST-NOT-REAL",
            "version": f"NOPE-{uuid.uuid4().hex[:8]}",
        },
    )
    assert result is None
    print("DB-06 PASS fake model not found")


def test_db_07_transaction_rollback(repo):
    marker = f"DB-AUDIT-ROLLBACK-{uuid.uuid4().hex}"
    camera_id = _camera_id(repo)
    assert camera_id is not None

    try:
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO belt.image
                    (
                        camera_id,
                        filename,
                        image_path,
                        capture_time,
                        status
                    )
                    VALUES
                    (
                        :camera_id,
                        :filename,
                        :image_path,
                        CURRENT_TIMESTAMP,
                        'PROCESSING'
                    )
                    """
                ),
                {
                    "camera_id": camera_id,
                    "filename": marker,
                    "image_path": marker,
                },
            )
            raise RuntimeError("forced rollback")
    except RuntimeError:
        pass

    count = _scalar(
        repo,
        "SELECT COUNT(*) FROM belt.image WHERE filename = :filename",
        {"filename": marker},
    )
    assert count == 0
    print("DB-07 PASS transaction rollback")


def test_db_08_fk_violation_camera(repo):
    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO belt.image
                    (
                        camera_id,
                        filename,
                        image_path,
                        capture_time,
                        status
                    )
                    VALUES
                    (
                        999999999,
                        :filename,
                        :image_path,
                        CURRENT_TIMESTAMP,
                        'PROCESSING'
                    )
                    """
                ),
                {
                    "filename": f"FK-AUDIT-{uuid.uuid4().hex}",
                    "image_path": "db-audit",
                },
            )
    print("DB-08 PASS FK violation IntegrityError")


def test_db_09_nonexistent_image(repo):
    result = _scalar(
        repo,
        """
        SELECT image_id
        FROM belt.image
        WHERE image_id = 999999999
        """,
    )
    assert result is None
    print("DB-09 PASS nonexistent image Image 999999999 not found")


def test_db_10_not_null_violation(repo):
    camera_id = _camera_id(repo)
    assert camera_id is not None

    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO belt.image
                    (
                        camera_id,
                        filename,
                        image_path,
                        capture_time,
                        status
                    )
                    VALUES
                    (
                        :camera_id,
                        NULL,
                        'db-audit',
                        CURRENT_TIMESTAMP,
                        'PROCESSING'
                    )
                    """
                ),
                {"camera_id": camera_id},
            )
    print("DB-10 PASS NOT NULL IntegrityError")


def test_db_11_unique_violation_inspection(repo):
    image_id = _scalar(
        repo,
        """
        SELECT image_id
        FROM belt.inspection_result
        ORDER BY inspection_id
        LIMIT 1
        """,
    )

    if image_id is None:
        pytest.skip("No existing inspection_result row available")

    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO belt.inspection_result
                    (
                        image_id,
                        good_count,
                        splice_count,
                        dogear_count,
                        overall_result
                    )
                    VALUES
                    (
                        :image_id,
                        0,
                        0,
                        0,
                        'GOOD'
                    )
                    """
                ),
                {"image_id": image_id},
            )
    print("DB-11 PASS UNIQUE IntegrityError")


def test_db_12_detection_image_fk(repo):
    model_id = _model_id(repo)

    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO belt.detection
                    (
                        image_id,
                        class_id,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        model_id
                    )
                    VALUES
                    (
                        999999999,
                        2,
                        'splice-belt',
                        0.5,
                        0,
                        0,
                        10,
                        10,
                        :model_id
                    )
                    """
                ),
                {"model_id": model_id},
            )
    print("DB-12 PASS detection image FK IntegrityError")


def test_db_13_inspection_image_fk(repo):
    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO belt.inspection_result
                    (
                        image_id,
                        good_count,
                        splice_count,
                        dogear_count,
                        overall_result
                    )
                    VALUES
                    (
                        999999999,
                        0,
                        0,
                        0,
                        'GOOD'
                    )
                    """
                )
            )
    print("DB-13 PASS inspection image FK IntegrityError")


def test_db_14_full_transaction_rollback(repo):
    camera_id = _camera_id(repo)
    model_id = _model_id(repo)
    assert camera_id is not None
    assert model_id is not None

    marker = f"DB-AUDIT-FULL-{uuid.uuid4().hex}"
    image_id = None

    try:
        with repo.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO belt.image
                    (
                        camera_id,
                        filename,
                        image_path,
                        capture_time,
                        status
                    )
                    VALUES
                    (
                        :camera_id,
                        :filename,
                        :image_path,
                        CURRENT_TIMESTAMP,
                        'PROCESSING'
                    )
                    RETURNING image_id
                    """
                ),
                {
                    "camera_id": camera_id,
                    "filename": marker,
                    "image_path": "db-audit",
                },
            )
            image_id = result.scalar_one()

            conn.execute(
                text(
                    """
                    INSERT INTO belt.detection
                    (
                        image_id,
                        class_id,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        model_id
                    )
                    VALUES
                    (
                        :image_id,
                        2,
                        'splice-belt',
                        0.5,
                        0,
                        0,
                        10,
                        10,
                        :model_id
                    )
                    """
                ),
                {"image_id": image_id, "model_id": model_id},
            )

            raise RuntimeError("forced full transaction rollback")
    except RuntimeError:
        pass

    assert _scalar(
        repo,
        "SELECT COUNT(*) FROM belt.image WHERE image_id = :image_id",
        {"image_id": image_id},
    ) == 0

    print(f"DB-14 PASS full transaction rollback image_id={image_id} rolled back")


def test_db_15_partial_failure_rollback(repo):
    camera_id = _camera_id(repo)
    model_id = _model_id(repo)
    assert camera_id is not None
    assert model_id is not None

    marker = f"DB-AUDIT-PARTIAL-{uuid.uuid4().hex}"
    image_id = None

    try:
        with repo.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO belt.image
                    (
                        camera_id,
                        filename,
                        image_path,
                        capture_time,
                        status
                    )
                    VALUES
                    (
                        :camera_id,
                        :filename,
                        :image_path,
                        CURRENT_TIMESTAMP,
                        'PROCESSING'
                    )
                    RETURNING image_id
                    """
                ),
                {
                    "camera_id": camera_id,
                    "filename": marker,
                    "image_path": "db-audit",
                },
            )
            image_id = result.scalar_one()

            conn.execute(
                text(
                    """
                    INSERT INTO belt.detection
                    (
                        image_id,
                        class_id,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        model_id
                    )
                    VALUES
                    (
                        :image_id,
                        2,
                        'splice-belt',
                        0.5,
                        0,
                        0,
                        10,
                        10,
                        :model_id
                    )
                    """
                ),
                {"image_id": image_id, "model_id": model_id},
            )

            raise RuntimeError("forced partial transaction failure")
    except RuntimeError:
        pass

    image_count = _scalar(
        repo,
        "SELECT COUNT(*) FROM belt.image WHERE image_id = :image_id",
        {"image_id": image_id},
    )
    detection_count = _scalar(
        repo,
        "SELECT COUNT(*) FROM belt.detection WHERE image_id = :image_id",
        {"image_id": image_id},
    )

    assert image_count == 0
    assert detection_count == 0

    print(
        f"DB-15 PASS partial failure rollback image_id={image_id}; "
        "image and detection rolled back"
    )


def test_db_16_database_recovery(repo):
    value = _scalar(repo, "SELECT 1")
    assert value == 1
    print("DB-16 PASS database recovery")


def test_database_audit_summary():
    print("\n" + "=" * 60)
    print("DATABASE AUDIT SUMMARY")
    print("=" * 60)
    print("16 database integrity / transaction tests")
    print("Run this file with pytest -v -s")
    print("=" * 60)
