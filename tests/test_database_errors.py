from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.config import settings
from app.database.repository import DatabaseRepository


# ==========================================================
# CONFIG
# ==========================================================

repo = DatabaseRepository()

passed = 0
failed = 0


# ==========================================================
# HELPERS
# ==========================================================

def print_section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def pass_test(message: str, detail: str = ""):
    global passed

    passed += 1
    print(f"[PASS] {message}")

    if detail:
        print(f"       {detail}")


def fail_test(message: str, detail: str = ""):
    global failed

    failed += 1
    print(f"[FAIL] {message}")

    if detail:
        print(f"       {detail}")

def _pass(
    message: str,
    detail: str = "",
):
    print(
        f"[PASS] {message}"
        + (f" | {detail}" if detail else "")
    )

def _fail(
    message: str,
    detail: str = "",
):
    print(
        f"[FAIL] {message}"
        + (f" | {detail}" if detail else "")
    )


# ==========================================================
# ENGINE
# ==========================================================

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("BELT AI - DATABASE ERROR AUDIT")
print("=" * 70)


# ==========================================================
# DB-01 : DATABASE CONNECTION
# ==========================================================

print_section("DB-01 : DATABASE CONNECTION")

try:
    if repo.test_connection():
        _pass("Database connection")
    else:
        _fail("Database connection")

except Exception as exc:
    _fail(
        "Database connection",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-02 : SIMPLE QUERY
# ==========================================================

print_section("DB-02 : SIMPLE QUERY")

try:

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        value = result.scalar_one()

    if value == 1:
        _pass(
            "SELECT 1",
            f"result={value}",
        )
    else:
        _fail(
            "SELECT 1",
            f"unexpected result={value}",
        )

except Exception as exc:
    _fail(
        "SELECT 1",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-03 : CAMERA LOOKUP
# ==========================================================

print_section("DB-03 : CAMERA LOOKUP")

try:

    camera = repo.get_camera(1)

    if camera is not None:
        _pass(
            "Camera 1 exists",
            str(camera),
        )
    else:
        _fail(
            "Camera 1 exists",
            "Camera returned None",
        )

except Exception as exc:
    _fail(
        "Camera 1 exists",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-04 : CAMERA NOT FOUND
# ==========================================================

print_section("DB-04 : CAMERA NOT FOUND")

try:

    camera = repo.get_camera(999999)

    if camera is None:
        _pass(
            "Camera 999999 not found",
            "None",
        )
    else:
        _fail(
            "Camera 999999 not found",
            f"Unexpected result: {camera}",
        )

except Exception as exc:
    _fail(
        "Camera 999999 not found",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-05 : MODEL REGISTRY
# ==========================================================

print_section("DB-05 : MODEL REGISTRY")

try:

    model = repo.get_model_version(
        model_name="YOLO11",
        model_version="ODBv54",
    )

    if model is not None:
        _pass(
            "YOLO11 / ODBv54 registered",
            str(model),
        )
    else:
        _fail(
            "YOLO11 / ODBv54 registered",
            "Model returned None",
        )

except Exception as exc:
    _fail(
        "YOLO11 / ODBv54 registered",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-06 : MODEL NOT FOUND
# ==========================================================

print_section("DB-06 : MODEL NOT FOUND")

try:

    model = repo.get_model_version(
        model_name="FAKE_MODEL",
        model_version="999999",
    )

    if model is None:
        _pass(
            "Non-existing model",
            "None",
        )
    else:
        _fail(
            "Non-existing model",
            f"Unexpected result: {model}",
        )

except Exception as exc:
    _fail(
        "Non-existing model",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-07 : TRANSACTION ROLLBACK
# ==========================================================

print_section("DB-07 : TRANSACTION ROLLBACK")

try:

    with repo.begin() as connection:

        connection.execute(
            text(
                """
                CREATE TEMP TABLE belt_ai_tx_test (
                    id INTEGER
                )
                """
            )
        )

        connection.execute(
            text(
                """
                INSERT INTO belt_ai_tx_test (id)
                VALUES (1)
                """
            )
        )

        raise RuntimeError(
            "Intentional exception triggered rollback"
        )

except RuntimeError as exc:

    _pass(
        "Transaction rollback",
        str(exc),
    )

except Exception as exc:

    _fail(
        "Transaction rollback",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-08 : FOREIGN KEY VIOLATION
# ==========================================================

print_section("DB-08 : FOREIGN KEY VIOLATION")

try:

    with repo.begin() as connection:

        repo.create_image_tx(
            connection=connection,
            camera_id=999999999,
            filename="fk_test.jpg",
            image_path="test/fk_test.jpg",
            capture_time=datetime.now(),
            status="PROCESSING",
            image_width=1920,
            image_height=1080,
        )

    _fail(
        "Foreign key violation",
        "INSERT unexpectedly succeeded",
    )

except IntegrityError:

    _pass(
        "Foreign key violation",
        "IntegrityError",
    )

except SQLAlchemyError as exc:

    _fail(
        "Foreign key violation",
        f"SQLAlchemyError: {exc}",
    )

except Exception as exc:

    _fail(
        "Foreign key violation",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-09 : UPDATE NON-EXISTENT IMAGE
# ==========================================================

print_section("DB-09 : UPDATE NON-EXISTENT IMAGE")

try:

    repo.update_image_path(
        image_id=999999999,
        image_path="test/nonexistent.jpg",
    )

    _fail(
        "Update non-existent image",
        "Update unexpectedly succeeded",
    )

except RuntimeError as exc:

    if "not found" in str(exc).lower():

        _pass(
            "Update non-existent image",
            str(exc),
        )

    else:

        _fail(
            "Update non-existent image",
            f"Unexpected RuntimeError: {exc}",
        )

except Exception as exc:

    _fail(
        "Update non-existent image",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-10 : NOT NULL VIOLATION
# ==========================================================

print_section("DB-10 : NOT NULL VIOLATION")

try:

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                INSERT INTO belt.image (
                    camera_id,
                    filename,
                    image_path,
                    capture_time,
                    status
                )
                VALUES (
                    :camera_id,
                    NULL,
                    :image_path,
                    CURRENT_TIMESTAMP,
                    'PROCESSING'
                )
                """
            ),
            {
                "camera_id": 1,
                "image_path": "test/not_null.jpg",
            },
        )

    _fail(
        "NOT NULL violation",
        "INSERT unexpectedly succeeded",
    )

except IntegrityError:

    _pass(
        "NOT NULL violation",
        "IntegrityError",
    )

except Exception as exc:

    _fail(
        "NOT NULL violation",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-11 : UNIQUE VIOLATION
# ==========================================================

print_section("DB-11 : UNIQUE VIOLATION")

try:

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                INSERT INTO belt.model_version (
                    model_name,
                    model_version,
                    model_type,
                    status
                )
                VALUES (
                    'YOLO11',
                    'ODBv54',
                    'detect',
                    'STAGING'
                )
                """
            )
        )

    _fail(
        "UNIQUE violation",
        "Duplicate model unexpectedly inserted",
    )

except IntegrityError:

    _pass(
        "UNIQUE violation",
        "IntegrityError",
    )

except Exception as exc:

    _fail(
        "UNIQUE violation",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-12 : DETECTION -> IMAGE FOREIGN KEY
# ==========================================================

print_section("DB-12 : DETECTION IMAGE FOREIGN KEY")

try:

    with repo.begin() as connection:

        repo.create_detection_tx(
            connection=connection,
            image_id=999999999,
            model_id=2,
            class_id=2,
            class_name="splice-belt",
            confidence=0.75,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        )

    _fail(
        "Detection image FK",
        "INSERT unexpectedly succeeded",
    )

except IntegrityError:

    _pass(
        "Detection image FK",
        "IntegrityError",
    )

except Exception as exc:

    _fail(
        "Detection image FK",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-13 : INSPECTION -> IMAGE FOREIGN KEY
# ==========================================================

print_section("DB-13 : INSPECTION IMAGE FOREIGN KEY")

try:

    with repo.begin() as connection:

        repo.create_inspection_result_tx(
            connection=connection,
            image_id=999999999,
            good_count=0,
            splice_count=1,
            dogear_count=0,
            overall_result="NG",
        )

    _fail(
        "Inspection image FK",
        "INSERT unexpectedly succeeded",
    )

except IntegrityError:

    _pass(
        "Inspection image FK",
        "IntegrityError",
    )

except Exception as exc:

    _fail(
        "Inspection image FK",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-14 : FULL TRANSACTION ROLLBACK
# ==========================================================

print_section("DB-14 : FULL TRANSACTION ROLLBACK")

test_image_id = None

try:

    with repo.begin() as connection:

        test_image_id = repo.create_image_tx(
            connection=connection,
            camera_id=1,
            filename="transaction_test.jpg",
            image_path="test/transaction_test.jpg",
            capture_time=datetime.now(),
            status="PROCESSING",
            image_width=1920,
            image_height=1080,
        )

        repo.create_detection_tx(
            connection=connection,
            image_id=test_image_id,
            model_id=2,
            class_id=2,
            class_name="splice-belt",
            confidence=0.75,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        )

        # Force rollback
        raise RuntimeError(
            "Intentional full transaction failure"
        )

except RuntimeError:
    pass

# Verify image was rolled back
try:

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM belt.image
                WHERE image_id = :image_id
                """
            ),
            {
                "image_id": test_image_id,
            },
        )

        count = result.scalar_one()

    if count == 0:

        _pass(
            "Full transaction rollback",
            f"image_id={test_image_id} rolled back",
        )

    else:

        _fail(
            "Full transaction rollback",
            f"image_id={test_image_id} still exists",
        )

except Exception as exc:

    _fail(
        "Full transaction rollback",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-15 : PARTIAL FAILURE ROLLBACK
# ==========================================================

print_section("DB-15 : PARTIAL FAILURE ROLLBACK")

partial_image_id = None

try:

    with repo.begin() as connection:

        partial_image_id = repo.create_image_tx(
            connection=connection,
            camera_id=1,
            filename="partial_failure.jpg",
            image_path="test/partial_failure.jpg",
            capture_time=datetime.now(),
            status="PROCESSING",
            image_width=1920,
            image_height=1080,
        )

        # First operation succeeds
        repo.create_detection_tx(
            connection=connection,
            image_id=partial_image_id,
            model_id=2,
            class_id=2,
            class_name="splice-belt",
            confidence=0.75,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        )

        # Second operation intentionally fails
        repo.create_detection_tx(
            connection=connection,
            image_id=999999999,
            model_id=2,
            class_id=2,
            class_name="splice-belt",
            confidence=0.75,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        )

except IntegrityError:

    pass

except Exception:

    pass


# Verify first successful operation was also rolled back
try:

    with engine.connect() as connection:

        image_result = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM belt.image
                WHERE image_id = :image_id
                """
            ),
            {
                "image_id": partial_image_id,
            },
        )

        image_count = image_result.scalar_one()

        detection_result = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM belt.detection
                WHERE image_id = :image_id
                """
            ),
            {
                "image_id": partial_image_id,
            },
        )

        detection_count = detection_result.scalar_one()

    if image_count == 0 and detection_count == 0:

        _pass(
            "Partial failure rollback",
            (
                f"image_id={partial_image_id}; "
                "image and detection rolled back"
            ),
        )

    else:

        _fail(
            "Partial failure rollback",
            (
                f"image_count={image_count}, "
                f"detection_count={detection_count}"
            ),
        )

except Exception as exc:

    _fail(
        "Partial failure rollback",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# DB-16 : DATABASE RECOVERY / CONNECTION
# ==========================================================

print_section("DB-16 : DATABASE RECOVERY")

try:

    # New connection after previous transaction failures
    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        value = result.scalar_one()

    if value == 1:

        _pass(
            "Database recovery",
            "Connection works after transaction failures",
        )

    else:

        _fail(
            "Database recovery",
            f"Unexpected result={value}",
        )

except Exception as exc:

    _fail(
        "Database recovery",
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# SUMMARY
# ==========================================================

print()
print()
print("=" * 70)
print("DATABASE ERROR AUDIT SUMMARY")
print("=" * 70)

print(f"Passed : {passed}/16")
print(f"Failed : {failed}/16")

print("=" * 70)

if failed == 0:

    print("DATABASE AUDIT : PASSED")

else:

    print("DATABASE AUDIT : NEED REVIEW")

print("=" * 70)