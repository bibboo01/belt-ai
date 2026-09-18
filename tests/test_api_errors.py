import os
import time
import requests


# ==========================================================
# CONFIG
# ==========================================================

BASE_URL = os.getenv(
    "BELT_AI_URL",
    "http://127.0.0.1:8000",
)

API_URL = f"{BASE_URL}/api/v1"

SAMPLE_IMAGE = "tests/sample.jpg"


# ==========================================================
# COUNTER
# ==========================================================

passed = 0
failed = 0


# ==========================================================
# HELPERS
# ==========================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check(
    name,
    condition,
    detail="",
):
    global passed
    global failed

    if condition:
        passed += 1
        print(f"[PASS] {name}")

        if detail:
            print(f"       {detail}")

    else:
        failed += 1
        print(f"[FAIL] {name}")

        if detail:
            print(f"       {detail}")


def request_id_exists(response):
    try:
        data = response.json()

        return (
            isinstance(data, dict)
            and (
                data.get("request_id")
                or response.headers.get("x-request-id")
            )
        )

    except Exception:
        return False


def error_contract_valid(
    response,
    expected_code=None,
):
    try:
        data = response.json()

    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    if data.get("status") != "error":
        return False

    if not data.get("code"):
        return False

    if not data.get("message"):
        return False

    if expected_code is not None:
        if data.get("code") != expected_code:
            return False

    return True


# ==========================================================
# START
# ==========================================================

print("=" * 70)
print("BELT AI - API ERROR AUDIT")
print("=" * 70)

print()
print(f"Base URL : {BASE_URL}")
print(f"API URL  : {API_URL}")


# ==========================================================
# API-01 : HEALTH
# ==========================================================

section("API-01 : HEALTH")

try:

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    check(
        "Health endpoint",
        response.status_code == 200,
        f"HTTP {response.status_code}",
    )

except Exception as exc:

    check(
        "Health endpoint",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-02 : LIVE
# ==========================================================

section("API-02 : LIVE")

try:

    response = requests.get(
        f"{API_URL}/health/live",
        timeout=10,
    )

    check(
        "Liveness",
        response.status_code == 200,
        f"HTTP {response.status_code}",
    )

except Exception as exc:

    check(
        "Liveness",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-03 : READY
# ==========================================================

section("API-03 : READY")

try:

    response = requests.get(
        f"{API_URL}/health/ready",
        timeout=10,
    )

    check(
        "Readiness",
        response.status_code in (200, 503),
        f"HTTP {response.status_code}",
    )

except Exception as exc:

    check(
        "Readiness",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-04 : CAMERA NOT FOUND
# ==========================================================

section("API-04 : CAMERA NOT FOUND")

try:

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 999999,
        },
        files={
            "image": (
                "sample.jpg",
                open(SAMPLE_IMAGE, "rb"),
                "image/jpeg",
            ),
        },
        timeout=30,
    )

    data = response.json()

    check(
        "Camera not found HTTP status",
        response.status_code == 404,
        f"HTTP {response.status_code}",
    )

    check(
        "Camera not found error contract",
        error_contract_valid(
            response,
            "CAMERA_NOT_FOUND",
        ),
        str(data),
    )

    check(
        "Camera not found request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Camera not found",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-05 : INVALID FILE TYPE
# ==========================================================

section("API-05 : INVALID FILE TYPE")

try:

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 1,
        },
        files={
            "image": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            ),
        },
        timeout=30,
    )

    data = response.json()

    check(
        "Invalid file type HTTP status",
        response.status_code == 400,
        f"HTTP {response.status_code}",
    )

    check(
        "Invalid file type contract",
        error_contract_valid(
            response,
            "INVALID_FILE_TYPE",
        ),
        str(data),
    )

    check(
        "Invalid file type request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Invalid file type",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-06 : CORRUPT IMAGE
# ==========================================================

section("API-06 : CORRUPT IMAGE")

try:

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 1,
        },
        files={
            "image": (
                "corrupt.jpg",
                b"THIS_IS_NOT_A_REAL_JPEG",
                "image/jpeg",
            ),
        },
        timeout=30,
    )

    data = response.json()

    check(
        "Corrupt image HTTP status",
        response.status_code == 400,
        f"HTTP {response.status_code}",
    )

    check(
        "Corrupt image contract",
        error_contract_valid(
            response,
            "IMAGE_DECODE_FAILED",
        ),
        str(data),
    )

    check(
        "Corrupt image request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Corrupt image",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-07 : MISSING IMAGE
# ==========================================================

section("API-07 : MISSING IMAGE")

try:

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 1,
        },
        timeout=30,
    )

    data = response.json()

    check(
        "Missing image HTTP status",
        response.status_code == 422,
        f"HTTP {response.status_code}",
    )

    check(
        "Missing image error contract",
        error_contract_valid(
            response,
            "VALIDATION_ERROR",
        ),
        str(data),
    )

    check(
        "Missing image request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Missing image",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-08 : MISSING CAMERA ID
# ==========================================================

section("API-08 : MISSING CAMERA ID")

try:

    with open(
        SAMPLE_IMAGE,
        "rb",
    ) as image_file:

        response = requests.post(
            f"{API_URL}/detect",
            files={
                "image": (
                    "sample.jpg",
                    image_file,
                    "image/jpeg",
                ),
            },
            timeout=30,
        )

    data = response.json()

    check(
        "Missing camera_id HTTP status",
        response.status_code == 422,
        f"HTTP {response.status_code}",
    )

    check(
        "Missing camera_id contract",
        error_contract_valid(
            response,
            "VALIDATION_ERROR",
        ),
        str(data),
    )

    check(
        "Missing camera_id request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Missing camera_id",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-09 : EMPTY FILE
# ==========================================================

section("API-09 : EMPTY FILE")

try:

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 1,
        },
        files={
            "image": (
                "empty.jpg",
                b"",
                "image/jpeg",
            ),
        },
        timeout=30,
    )

    data = response.json()

    check(
        "Empty file HTTP status",
        response.status_code == 400,
        f"HTTP {response.status_code}",
    )

    check(
        "Empty file contract",
        error_contract_valid(
            response,
            "EMPTY_IMAGE",
        ),
        str(data),
    )

    check(
        "Empty file request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Empty file",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-10 : INVALID CAMERA ID TYPE
# ==========================================================

section("API-10 : INVALID CAMERA ID TYPE")

try:

    with open(
        SAMPLE_IMAGE,
        "rb",
    ) as image_file:

        response = requests.post(
            f"{API_URL}/detect",
            params={
                "camera_id": "abc",
            },
            files={
                "image": (
                    "sample.jpg",
                    image_file,
                    "image/jpeg",
                ),
            },
            timeout=30,
        )

    data = response.json()

    check(
        "Invalid camera_id HTTP status",
        response.status_code == 422,
        f"HTTP {response.status_code}",
    )

    check(
        "Invalid camera_id contract",
        error_contract_valid(
            response,
            "VALIDATION_ERROR",
        ),
        str(data),
    )

    check(
        "Invalid camera_id request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Invalid camera_id",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-11 : OVERSIZED FILE
# ==========================================================

section("API-11 : OVERSIZED FILE")

try:

    oversized_data = b"0" * (
        10 * 1024 * 1024 + 1
    )

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 1,
        },
        files={
            "image": (
                "large.jpg",
                oversized_data,
                "image/jpeg",
            ),
        },
        timeout=30,
    )

    data = response.json()

    check(
        "Oversized file HTTP status",
        response.status_code == 413,
        f"HTTP {response.status_code}",
    )

    check(
        "Oversized file contract",
        error_contract_valid(
            response,
            "FILE_TOO_LARGE",
        ),
        str(data),
    )

    check(
        "Oversized file request_id",
        request_id_exists(response),
        str(data.get("request_id")),
    )

except Exception as exc:

    check(
        "Oversized file",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-12 : REQUEST ID
# ==========================================================

section("API-12 : REQUEST ID")

try:

    response = requests.get(
        f"{API_URL}/health/live",
        timeout=10,
    )

    request_id = response.headers.get(
        "x-request-id"
    )

    check(
        "X-Request-ID header",
        bool(request_id),
        str(request_id),
    )

except Exception as exc:

    check(
        "X-Request-ID header",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-13 : ERROR DOES NOT EXPOSE TRACEBACK
# ==========================================================

section("API-13 : ERROR INFORMATION DISCLOSURE")

try:

    response = requests.post(
        f"{API_URL}/detect",
        params={
            "camera_id": 999999,
        },
        files={
            "image": (
                "sample.jpg",
                open(SAMPLE_IMAGE, "rb"),
                "image/jpeg",
            ),
        },
        timeout=30,
    )

    body = response.text.lower()

    forbidden = [
        "traceback",
        "sqlalchemy",
        "psycopg",
        "exception",
        "file \"",
        "line ",
    ]

    leaked = [
        item
        for item in forbidden
        if item in body
    ]

    check(
        "No internal error details exposed",
        len(leaked) == 0,
        f"leaked={leaked}",
    )

except Exception as exc:

    check(
        "No internal error details exposed",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-14 : HEALTH RESPONSE
# ==========================================================

section("API-14 : HEALTH RESPONSE CONTRACT")

try:

    response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )

    data = response.json()

    check(
        "Health response is JSON",
        isinstance(data, dict),
        str(data),
    )

    check(
        "Health has status",
        "status" in data,
        str(data),
    )

    check(
        "Health has checks",
        "checks" in data,
        str(data),
    )

except Exception as exc:

    check(
        "Health response contract",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# API-15 : READY RESPONSE
# ==========================================================

section("API-15 : READINESS RESPONSE CONTRACT")

try:

    response = requests.get(
        f"{API_URL}/health/ready",
        timeout=10,
    )

    data = response.json()

    check(
        "Ready response JSON",
        isinstance(data, dict),
        str(data),
    )

    check(
        "Ready has status",
        data.get("status")
        in (
            "ready",
            "not_ready",
        ),
        str(data),
    )

    check(
        "Ready has checks",
        "checks" in data,
        str(data),
    )

except Exception as exc:

    check(
        "Readiness response contract",
        False,
        f"{type(exc).__name__}: {exc}",
    )


# ==========================================================
# SUMMARY
# ==========================================================

print()
print()
print("=" * 70)
print("API ERROR AUDIT SUMMARY")
print("=" * 70)

print(f"Passed : {passed}/15")
print(f"Failed : {failed}/15")

print("=" * 70)

if failed == 0:

    print("API ERROR AUDIT : PASSED")

else:

    print("API ERROR AUDIT : NEED REVIEW")

print("=" * 70)