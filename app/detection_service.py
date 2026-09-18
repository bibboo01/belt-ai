from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import BASE_DIR
from app.database.repository import repository
from app.errors import BeltAIError
from app.inference import run_inference
from app.model_manager import model_manager


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = BASE_DIR / "images" / "original"
RESULT_DIR = BASE_DIR / "images" / "result"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


EXPECTED_CLASSES = {
    0: "dogear-belt",
    1: "good-belt",
    2: "splice-belt",
}


# ============================================================
# Detection Service
# ============================================================

class DetectionService:
    """
    Production detection service.

    Pipeline:

        Request
          ↓
        Validate Camera
          ↓
        Validate Image
          ↓
        Validate Production Model
          ↓
        Create PROCESSING record
          ↓
        Save Original
          ↓
        Run YOLO Inference
          ↓
        Validate Detection Classes
          ↓
        Save Result Image
          ↓
        DB Transaction
          ├── Detections
          ├── Inspection Result
          └── Image DONE
          ↓
        API Response
    """

    def __init__(self):
        self.repository = repository
        self.model_manager = model_manager

        IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        RESULT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _error(
        status_code: int,
        code: str,
        message: str,
    ) -> BeltAIError:
        """
        Create a consistent BeltAIError.
        """
        return BeltAIError(
            status_code=status_code,
            code=code,
            message=message,
        )

    @staticmethod
    def _safe_filename(
        filename: str,
    ) -> str:
        """
        Prevent path traversal and unsafe filenames.
        """

        filename = Path(
            filename
        ).name.strip()

        if not filename:
            filename = "upload.jpg"

        safe_chars = []

        for char in filename:

            if (
                char.isalnum()
                or char in {"-", "_", "."}
            ):
                safe_chars.append(char)
            else:
                safe_chars.append("_")

        filename = "".join(
            safe_chars
        )

        if not Path(filename).suffix:
            filename += ".jpg"

        return filename

    @staticmethod
    def _relative_path(
        path: Path,
    ) -> str:
        """
        Convert absolute path to project-relative path.

        Example:
            images/original/123_test.jpg
        """

        return (
            path.resolve()
            .relative_to(
                BASE_DIR.resolve()
            )
            .as_posix()
        )

    # ========================================================
    # Image Validation
    # ========================================================

    @staticmethod
    def _validate_content_type(
        content_type: str | None,
        filename: str,
    ) -> None:

        if content_type not in ALLOWED_CONTENT_TYPES:

            raise BeltAIError(
                status_code=415,
                code="UNSUPPORTED_MEDIA_TYPE",
                message=(
                    "Unsupported image content type: "
                    f"{content_type}"
                ),
            )

        extension = Path(
            filename
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:

            raise BeltAIError(
                status_code=400,
                code="INVALID_IMAGE_EXTENSION",
                message=(
                    "Unsupported image extension: "
                    f"{extension}"
                ),
            )

    @staticmethod
    def _decode_image(
        image_bytes: bytes,
    ) -> np.ndarray:
        """
        Decode uploaded image safely.
        """

        if not image_bytes:

            raise BeltAIError(
                status_code=400,
                code="EMPTY_IMAGE",
                message="Empty image",
            )

        if len(image_bytes) > MAX_IMAGE_SIZE:

            raise BeltAIError(
                status_code=413,
                code="IMAGE_TOO_LARGE",
                message=(
                    "Image size exceeds "
                    f"{MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB"
                ),
            )

        buffer = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if frame is None:

            raise BeltAIError(
                status_code=400,
                code="INVALID_IMAGE",
                message="Unable to decode image",
            )

        if frame.size == 0:

            raise BeltAIError(
                status_code=400,
                code="EMPTY_DECODED_IMAGE",
                message="Decoded image is empty",
            )

        return frame

    # ========================================================
    # Model Registry / Lineage
    # ========================================================

    def _validate_model_registry(
        self,
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # 1. Ensure production model is loaded
        # ----------------------------------------------------

        if not self.model_manager.is_ready():
            try:
                self.model_manager.load_model()

            except Exception as exc:
                raise BeltAIError(
                    status_code=503,
                    code="MODEL_LOAD_FAILED",
                    message="Unable to load production model",
                ) from exc

        # ----------------------------------------------------
        # 2. Verify model is actually ready
        # ----------------------------------------------------

        if not self.model_manager.is_ready():
            raise BeltAIError(
                status_code=503,
                code="MODEL_NOT_READY",
                message="Production model is not ready",
            )

        # ----------------------------------------------------
        # 3. Get model information
        # ----------------------------------------------------

        model_info = self.model_manager.get_info()

        model_id = model_info.get("model_id")
        model_name = model_info.get("model_name")
        model_version = model_info.get("model_version")
        model_status = str(
            model_info.get("status", "")
        ).upper()

        # ----------------------------------------------------
        # 4. Lookup database registry
        # ----------------------------------------------------

        registry = (
            self.repository.get_model_version(
                model_name,
                model_version,
            )
        )

        if registry is None:

            raise BeltAIError(
                status_code=503,
                code="MODEL_NOT_REGISTERED",
                message=(
                    "Model is not registered in database: "
                    f"{model_name}:{model_version}"
                ),
            )

        # ----------------------------------------------------
        # 5. Validate model_id lineage
        # ----------------------------------------------------

        db_model_id = registry.get(
            "model_id"
        )

        if db_model_id is None:

            raise BeltAIError(
                status_code=503,
                code="MODEL_REGISTRY_INVALID",
                message=(
                    "Database model registry does not "
                    "have model_id"
                ),
            )

        if int(db_model_id) != int(model_id):

            raise BeltAIError(
                status_code=503,
                code="MODEL_LINEAGE_MISMATCH",
                message=(
                    "Model lineage mismatch: "
                    f"production model_id={model_id}, "
                    f"database model_id={db_model_id}"
                ),
            )

        # ----------------------------------------------------
        # 6. Validate database status
        # ----------------------------------------------------

        db_status = str(
            registry.get(
                "status",
                "",
            )
        ).upper()

        if db_status != "PRODUCTION":

            raise BeltAIError(
                status_code=503,
                code="MODEL_NOT_PRODUCTION",
                message=(
                    "Model is not PRODUCTION in database: "
                    f"{model_name}:{model_version} "
                    f"(status={db_status})"
                ),
            )

        return registry

    # ========================================================
    # Result Image
    # ========================================================

    def _build_result_path(
        self,
        image_id: int,
        filename: str,
    ) -> Path:

        original_name = Path(
            filename
        )

        stem = original_name.stem
        suffix = original_name.suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            suffix = ".jpg"

        result_filename = (
            f"{image_id}_{stem}_result{suffix}"
        )

        return (
            RESULT_DIR
            / result_filename
        )

    def _save_result_image(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        result_path: Path,
    ) -> None:
        """
        Draw detection results using OpenCV.
        """

        output = frame.copy()

        for detection in detections:

            class_name = str(
                detection["class_name"]
            )

            confidence = float(
                detection["confidence"]
            )

            bbox = detection["bbox"]

            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            # ----------------------------------------------
            # Clamp bounding box
            # ----------------------------------------------

            x1 = max(
                0,
                x1,
            )

            y1 = max(
                0,
                y1,
            )

            x2 = min(
                output.shape[1] - 1,
                x2,
            )

            y2 = min(
                output.shape[0] - 1,
                y2,
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            # ----------------------------------------------
            # Label
            # ----------------------------------------------

            text_y = max(
                25,
                y1 - 8,
            )

            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        success = cv2.imwrite(
            str(result_path),
            output,
        )

        if not success:

            raise BeltAIError(
                status_code=500,
                code="RESULT_SAVE_FAILED",
                message="Unable to save result image",
            )

    # ========================================================
    # Main Process
    # ========================================================

    def process(
        self,
        camera_id: int,
        filename: str,
        content_type: str | None,
        image_bytes: bytes,
    ) -> dict[str, Any]:

        image_id: int | None = None
        original_path: Path | None = None
        result_path: Path | None = None

        started_at = time.perf_counter()

        try:

            # =================================================
            # 1. Validate Camera
            # =================================================

            camera = (
                self.repository.get_camera(
                    camera_id
                )
            )

            if camera is None:

                raise BeltAIError(
                    status_code=404,
                    code="CAMERA_NOT_FOUND",
                    message=(
                        f"Camera not found: {camera_id}"
                    ),
                )

            # =================================================
            # 2. Validate Filename / Content Type
            # =================================================

            safe_filename = (
                self._safe_filename(
                    filename
                )
            )

            self._validate_content_type(
                content_type,
                safe_filename,
            )

            # =================================================
            # 3. Decode Image
            # =================================================

            frame = (
                self._decode_image(
                    image_bytes
                )
            )

            height, width = (
                frame.shape[:2]
            )

            # =================================================
            # 4. Validate Production Model
            # =================================================

            registry = (
                self._validate_model_registry()
            )

            model_info = (
                self.model_manager.get_info()
            )

            # =================================================
            # 5. Create Image Record
            # =================================================

            capture_time = datetime.now()

            image_id = (
                self.repository.create_image(
                    camera_id=camera_id,
                    filename=safe_filename,
                    image_path="",
                    result_path=None,
                    image_width=width,
                    image_height=height,
                    capture_time=capture_time,
                    status="PROCESSING",
                )
            )

            # =================================================
            # 6. Save Original Image
            # =================================================

            original_filename = (
                f"{image_id}_{safe_filename}"
            )

            original_path = (
                IMAGE_DIR
                / original_filename
            )

            # -------------------------------------------------
            # Preserve original uploaded bytes.
            # -------------------------------------------------

            with open(
                original_path,
                "wb",
            ) as file:

                file.write(
                    image_bytes
                )

            original_path_db = (
                self._relative_path(
                    original_path
                )
            )

            # -------------------------------------------------
            # Save original path.
            #
            # update_image_path() only accepts image_path.
            # -------------------------------------------------

            self.repository.update_image_path(
                image_id=image_id,
                image_path=original_path_db,
            )

            # =================================================
            # 7. Run AI Inference
            # =================================================

            inference_result = run_inference(
                frame=frame,
                model_manager=self.model_manager,
            )

            detections = (
                inference_result.detections
            )

            # =================================================
            # 8. Validate Detection Classes
            # =================================================

            for detection in detections:

                class_id = int(
                    detection["class_id"]
                )

                class_name = str(
                    detection["class_name"]
                )

                expected_name = (
                    EXPECTED_CLASSES.get(
                        class_id
                    )
                )

                if expected_name is None:

                    raise BeltAIError(
                        status_code=500,
                        code="UNEXPECTED_CLASS_ID",
                        message=(
                            f"Unexpected class_id: "
                            f"{class_id}"
                        ),
                    )

                if class_name != expected_name:

                    raise BeltAIError(
                        status_code=500,
                        code="CLASS_MAPPING_MISMATCH",
                        message=(
                            "Class mapping mismatch: "
                            f"class_id={class_id}, "
                            f"class_name={class_name}, "
                            f"expected={expected_name}"
                        ),
                    )

            # =================================================
            # 9. Save Result Image
            # =================================================

            result_path = (
                self._build_result_path(
                    image_id=image_id,
                    filename=safe_filename,
                )
            )

            self._save_result_image(
                frame=frame,
                detections=detections,
                result_path=result_path,
            )

            result_path_db = (
                self._relative_path(
                    result_path
                )
            )

            # =================================================
            # 10. Calculate Counts
            # =================================================

            good_count = sum(
                1
                for detection in detections
                if int(
                    detection["class_id"]
                ) == 1
            )

            splice_count = sum(
                1
                for detection in detections
                if int(
                    detection["class_id"]
                ) == 2
            )

            dogear_count = sum(
                1
                for detection in detections
                if int(
                    detection["class_id"]
                ) == 0
            )

            total_detections = len(
                detections
            )

            ng_count = (
                splice_count
                + dogear_count
            )

            overall_result = (
                "NG"
                if ng_count > 0
                else "GOOD"
            )

            # =================================================
            # 11. Inference Time
            # =================================================

            inference_time_ms = float(
                inference_result.inference_time_ms
            )

            # =================================================
            # 12. Final DB Transaction
            #
            # Atomic:
            #
            #   detections
            #   inspection result
            #   result path
            #   DONE
            #
            # If anything fails, transaction rolls back.
            # =================================================

            with self.repository.begin() as connection:

                # ---------------------------------------------
                # Insert detections
                # ---------------------------------------------

                for detection in detections:

                    bbox = detection["bbox"]

                    self.repository.create_detection_tx(
                        connection=connection,
                        image_id=image_id,
                        model_id=int(
                            registry["model_id"]
                        ),
                        class_id=int(
                            detection["class_id"]
                        ),
                        class_name=str(
                            detection["class_name"]
                        ),
                        confidence=float(
                            detection["confidence"]
                        ),
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3]),
                    )

                # ---------------------------------------------
                # Insert inspection result
                # ---------------------------------------------

                self.repository.create_inspection_result_tx(
                    connection=connection,
                    image_id=image_id,
                    good_count=good_count,
                    splice_count=splice_count,
                    dogear_count=dogear_count,
                    overall_result=overall_result,
                )

                # ---------------------------------------------
                # Mark DONE + result path
                # ---------------------------------------------

                self.repository.update_image_processed_tx(
                    connection=connection,
                    image_id=image_id,
                    status="DONE",
                    inference_time_ms=inference_time_ms,
                    result_path=result_path_db,
                )

            # =================================================
            # 13. Request Time
            # =================================================

            request_time_ms = (
                time.perf_counter()
                - started_at
            ) * 1000.0

            # =================================================
            # 14. API Response
            # =================================================

            response = {

                "status": "success",

                "image": {
                    "image_id": int(
                        image_id
                    ),
                    "filename": safe_filename,
                    "width": int(
                        width
                    ),
                    "height": int(
                        height
                    ),
                    "original_path": (
                        original_path_db
                    ),
                    "result_path": (
                        result_path_db
                    ),
                },

                "model": {
                    "model_id": int(
                        registry["model_id"]
                    ),
                    "name": str(
                        model_info["model_name"]
                    ),
                    "version": str(
                        model_info["model_version"]
                    ),
                },

                "result": {
                    "overall": overall_result,
                    "good_count": good_count,
                    "splice_count": splice_count,
                    "dogear_count": dogear_count,
                    "ng_count": ng_count,
                    "total_detections": (
                        total_detections
                    ),
                },

                "inference_time_ms": round(
                    inference_time_ms,
                    2,
                ),

                "request_time_ms": round(
                    request_time_ms,
                    2,
                ),

                "detections": [
                    {
                        "class_id": int(
                            detection["class_id"]
                        ),
                        "class_name": str(
                            detection["class_name"]
                        ),
                        "confidence": float(
                            detection["confidence"]
                        ),
                        "bbox": [
                            int(value)
                            for value
                            in detection["bbox"]
                        ],
                    }
                    for detection
                    in detections
                ],
            }

            return response

        # =====================================================
        # BeltAIError
        # =====================================================

        except BeltAIError:

            # -------------------------------------------------
            # Mark image ERROR
            # -------------------------------------------------

            if image_id is not None:

                try:

                    self.repository.update_image_processed(
                        image_id=image_id,
                        status="ERROR",
                        inference_time_ms=None,
                    )

                except Exception:
                    pass

            # -------------------------------------------------
            # Keep original for audit.
            #
            # Remove only incomplete result.
            # -------------------------------------------------

            if result_path is not None:

                try:

                    if result_path.exists():
                        result_path.unlink()

                except Exception:
                    pass

            raise

        # =====================================================
        # Unexpected Error
        # =====================================================

        except Exception as exc:

            # -------------------------------------------------
            # Log original error
            # -------------------------------------------------

            try:

                from app.logging_config import get_logger

                logger = get_logger()

                logger.exception(
                    "Unexpected detection service error",
                    extra={
                        "camera_id": camera_id,
                        "image_id": image_id,
                        "filename": safe_filename
                        if "safe_filename" in locals()
                        else filename,
                    },
                )

            except Exception:
                pass

            # -------------------------------------------------
            # Mark image ERROR
            # -------------------------------------------------

            if image_id is not None:

                try:

                    self.repository.update_image_processed(
                        image_id=image_id,
                        status="ERROR",
                        inference_time_ms=None,
                    )

                except Exception:
                    pass

            # -------------------------------------------------
            # Remove incomplete result
            # -------------------------------------------------

            if result_path is not None:

                try:

                    if result_path.exists():
                        result_path.unlink()

                except Exception:
                    pass

            raise BeltAIError(
                status_code=500,
                code="DETECTION_FAILED",
                message="Detection failed",
            ) from exc


# ============================================================
# Singleton
# ============================================================

detection_service = DetectionService()
