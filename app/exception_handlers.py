import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.errors import BeltAIError


logger = logging.getLogger("belt_ai")


async def belt_ai_exception_handler(
    request: Request,
    exc: BeltAIError,
):
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    logger.error(
        "Belt AI exception | "
        "request_id=%s | "
        "method=%s | "
        "path=%s | "
        "status=%s | "
        "code=%s | "
        "message=%s",
        request_id,
        request.method,
        request.url.path,
        exc.status_code,
        exc.code,
        exc.message,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
        },
    )


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    logger.exception(
        "Unhandled exception | "
        "request_id=%s | "
        "method=%s | "
        "path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Internal server error",
            "request_id": request_id,
        },
    )