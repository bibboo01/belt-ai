from pathlib import Path
import time
from datetime import datetime
import uuid
import logging

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Query,
    APIRouter,
)
from app.detection_service import detection_service

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.model_manager import model_manager
from app.database.repository import DatabaseRepository
from app.logging_config import setup_logging
from app.errors import BeltAIError
from app.schemas import DetectResponse

from app.exception_handlers import (
    belt_ai_exception_handler,
    unexpected_exception_handler,
)


# ==========================================================
# Logging
# ==========================================================

setup_logging()

logger = logging.getLogger("belt_ai")


# ==========================================================
# Database
# ==========================================================

repository = DatabaseRepository()


# ==========================================================
# Project Paths
# ==========================================================

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


ORIGINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# File Cleanup
# ==========================================================

def cleanup_detection_files(
    image_path: Path | None = None,
    result_path: Path | None = None,
):
    """
    Remove files created during a detection request
    when processing fails.

    This prevents orphan files when database,
    inference, model-output, or result-processing
    operations fail after files have already been created.
    """

    for file_path in (
        image_path,
        result_path,
    ):

        if file_path is None:
            continue

        try:

            if file_path.exists():

                file_path.unlink()

                logger.warning(
                    "Detection file cleanup | "
                    "path=%s",
                    file_path,
                )

        except Exception:

            logger.exception(
                "Detection file cleanup failed | "
                "path=%s",
                file_path,
            )


# ==========================================================
# Upload Validation
# ==========================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

MAX_FILE_SIZE = (
    10 * 1024 * 1024
)


# ==========================================================
# Expected Model Classes
# ==========================================================

EXPECTED_CLASSES = {
    0: "dogear-belt",
    1: "good-belt",
    2: "splice-belt",
}


# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="Belt AI",
    description="AI-powered belt inspection system",
    version="1.0.0",
)


# ==========================================================
# Exception Handlers
# ==========================================================

app.add_exception_handler(
    BeltAIError,
    belt_ai_exception_handler,
)

app.add_exception_handler(
    Exception,
    unexpected_exception_handler,
)


# ==========================================================
# Request Validation Exception Handler
# ==========================================================

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    logger.warning(
        "Request validation failed | "
        "request_id=%s | "
        "method=%s | "
        "path=%s | "
        "errors=%s",
        request_id,
        request.method,
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "request_id": request_id,
        },
    )


# ==========================================================
# API v1 Router
# ==========================================================

api_v1 = APIRouter(
    prefix="/api/v1",
    tags=["v1"],
)


# ==========================================================
# Request ID Middleware
# ==========================================================

class RequestIDMiddleware(
    BaseHTTPMiddleware
):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):

        request_id = str(
            uuid.uuid4()
        )

        request.state.request_id = (
            request_id
        )

        start_time = (
            time.perf_counter()
        )

        response = None

        try:

            response = await call_next(
                request
            )

            return response

        finally:

            request_time_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            if response is not None:

                response.headers[
                    "X-Request-ID"
                ] = request_id

            status_code = (
                response.status_code
                if response is not None
                else 500
            )

            logger.info(
                "Request completed | "
                "request_id=%s | "
                "method=%s | "
                "path=%s | "
                "status=%s | "
                "request_time_ms=%.2f",

                request_id,
                request.method,
                request.url.path,
                status_code,
                request_time_ms,
            )


# ==========================================================
# Register Middleware
# ==========================================================

app.add_middleware(
    RequestIDMiddleware
)


# ==========================================================
# Root
# ==========================================================

@app.get("/")
def root():

    return {
        "system": "Belt AI",
        "status": "running",
        "version": "1.0.0",
    }


# ==========================================================
# Health
# ==========================================================

@app.get("/health")
def health():

    # ------------------------------------------------------
    # Model Check
    # ------------------------------------------------------

    try:

        model_manager.get_model()

        model_info = (
            model_manager.get_info()
        )

        model_ready = (
            model_manager.is_ready()
        )

    except Exception:

        logger.exception(
            "Health model check failed"
        )

        model_info = {
            "status": "ERROR"
        }

        model_ready = False


    # ------------------------------------------------------
    # Database Check
    # ------------------------------------------------------

    try:

        database_ready = (
            repository.test_connection()
        )

    except Exception:

        logger.exception(
            "Health database check failed"
        )

        database_ready = False


    overall_status = (
        "healthy"
        if (
            model_ready
            and database_ready
        )
        else "unhealthy"
    )


    return {

        "status": overall_status,

        "checks": {

            "database": (
                "healthy"
                if database_ready
                else "unhealthy"
            ),

            "model": (
                "healthy"
                if model_ready
                else "unhealthy"
            ),

        },

        "model": model_info,

    }


# ==========================================================
# API v1 Health
# ==========================================================

@api_v1.get("/health")
def health_v1():

    try:

        model_manager.get_model()

        model_info = (
            model_manager.get_info()
        )

        model_ready = (
            model_manager.is_ready()
        )

    except Exception:

        logger.exception(
            "API v1 model health check failed"
        )

        model_info = {
            "status": "ERROR"
        }

        model_ready = False


    try:

        database_ready = (
            repository.test_connection()
        )

    except Exception:

        logger.exception(
            "API v1 database check failed"
        )

        database_ready = False


    overall_status = (
        "healthy"
        if (
            model_ready
            and database_ready
        )
        else "unhealthy"
    )


    return {

        "status": overall_status,

        "checks": {

            "database": (
                "healthy"
                if database_ready
                else "unhealthy"
            ),

            "model": (
                "healthy"
                if model_ready
                else "unhealthy"
            ),

        },

        "model": model_info,

    }


# ==========================================================
# Liveness
# ==========================================================

@app.get("/health/live")
def health_live():

    return {
        "status": "alive"
    }


@api_v1.get("/health/live")
def health_live_v1():

    return {
        "status": "alive"
    }


# ==========================================================
# Readiness
# ==========================================================

def _check_readiness():

    try:

        model_manager.get_model()

        model_ready = (
            model_manager.is_ready()
        )

    except Exception:

        logger.exception(
            "Readiness model check failed"
        )

        model_ready = False


    try:

        database_ready = (
            repository.test_connection()
        )

    except Exception:

        logger.exception(
            "Readiness database check failed"
        )

        database_ready = False


    ready = (
        model_ready
        and database_ready
    )


    return (
        ready,
        model_ready,
        database_ready,
    )


@app.get("/health/ready")
def health_ready():

    (
        ready,
        model_ready,
        database_ready,
    ) = _check_readiness()


    payload = {

        "status": (
            "ready"
            if ready
            else "not_ready"
        ),

        "checks": {

            "database": (
                "healthy"
                if database_ready
                else "unhealthy"
            ),

            "model": (
                "healthy"
                if model_ready
                else "unhealthy"
            ),

        },

    }


    if not ready:

        return JSONResponse(
            status_code=503,
            content=payload,
        )


    return payload


@api_v1.get("/health/ready")
def health_ready_v1():

    (
        ready,
        model_ready,
        database_ready,
    ) = _check_readiness()


    payload = {

        "status": (
            "ready"
            if ready
            else "not_ready"
        ),

        "checks": {

            "database": (
                "healthy"
                if database_ready
                else "unhealthy"
            ),

            "model": (
                "healthy"
                if model_ready
                else "unhealthy"
            ),

        },

    }


    if not ready:

        return JSONResponse(
            status_code=503,
            content=payload,
        )


    return payload


# ==========================================================
# Detect
# ==========================================================

@app.post(
    "/detect",
    response_model=DetectResponse,
)
async def detect(
    request: Request,
    camera_id: int = Query(...),
    image: UploadFile = File(...),
):
    """
    Production detection endpoint.

    Responsibilities of this endpoint:
        - Receive HTTP request
        - Read uploaded image
        - Call DetectionService
        - Return API response
        - Handle unexpected HTTP-level errors

    All detection / database / model logic is handled by
    DetectionService.
    """

    request_start_time = time.perf_counter()

    request_id = request.state.request_id

    logger.info(
        "Detection started | "
        "request_id=%s | "
        "camera_id=%s | "
        "filename=%s",
        request_id,
        camera_id,
        image.filename,
    )

    try:

        # ==================================================
        # 1. Read Uploaded File
        # ==================================================

        image_bytes = await image.read()

        # ==================================================
        # 2. Detection Service
        # ==================================================

        result = detection_service.process(
            camera_id=camera_id,
            filename=image.filename or "unknown.jpg",
            content_type=image.content_type,
            image_bytes=image_bytes,
        )

        # ==================================================
        # 3. Request Time
        #
        # DetectionService already calculates its own
        # request_time_ms.
        #
        # This measurement is for server-side logging.
        # ==================================================

        total_request_time_ms = (
            time.perf_counter()
            - request_start_time
        ) * 1000.0

        # ==================================================
        # 4. Completion Log
        # ==================================================

        logger.info(
            "Detection completed | "
            "request_id=%s | "
            "image_id=%s | "
            "camera_id=%s | "
            "model=%s:%s | "
            "result=%s | "
            "detections=%s | "
            "inference_ms=%.2f | "
            "request_ms=%.2f",
            request_id,
            result["image"]["image_id"],
            camera_id,
            result["model"]["name"],
            result["model"]["version"],
            result["result"]["overall"],
            result["result"]["total_detections"],
            result["inference_time_ms"],
            total_request_time_ms,
        )

        return result

    # ======================================================
    # Belt AI Error
    # ======================================================

    except BeltAIError:

        logger.exception(
            "Belt AI detection error | "
            "request_id=%s | "
            "camera_id=%s",
            request_id,
            camera_id,
        )

        raise

    # ======================================================
    # Unexpected Error
    # ======================================================

    except Exception as exc:

        logger.exception(
            "Unexpected detection error | "
            "request_id=%s | "
            "camera_id=%s",
            request_id,
            camera_id,
        )

        raise BeltAIError(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
        ) from exc


# ==========================================================
# API v1 Detect
# ==========================================================

@api_v1.post(
    "/detect",
    response_model=DetectResponse,
)
async def detect_v1(

    request: Request,

    camera_id: int = Query(...),

    image: UploadFile = File(...),

):

    return await detect(

        request=request,

        camera_id=camera_id,

        image=image,

    )


# ==========================================================
# Include API v1
# ==========================================================

app.include_router(
    api_v1
)
