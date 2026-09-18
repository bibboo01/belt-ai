from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

class DatabaseRepository:
    """
    Database repository for Belt AI.

    Responsibilities:
    - Database connection
    - Model registry lookup
    - Camera lookup
    - Image CRUD
    - Detection CRUD
    - Inspection result CRUD
    - Transaction-safe operations
    """

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or create_engine(
            settings.database_url,
            pool_pre_ping=True,
        )

    # ==========================================================
    # TRANSACTION
    # ==========================================================

    def begin(self):
        """
        Start a database transaction.

        Usage:
            with repo.begin() as connection:
                ...
        """
        return self.engine.begin()

    # ==========================================================
    # CONNECTION TEST
    # ==========================================================

    def test_connection(self) -> bool:
        """
        Test database connection.
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            return True

        except SQLAlchemyError as exc:
            logger.error(
                "Database connection failed: %s",
                exc,
                exc_info=True,
            )
            return False

    # ==========================================================
    # MODEL REGISTRY
    # ==========================================================

    def get_model_version(
        self,
        model_name: str,
        model_version: str,
    ) -> Optional[dict]:
        """
        Get registered model information.
        """

        query = text(
            """
            SELECT
                model_id,
                model_name,
                model_version,
                model_type,
                model_path,
                confidence_threshold,
                status
            FROM belt.model_version
            WHERE model_name = :model_name
              AND model_version = :model_version
            LIMIT 1
            """
        )

        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    query,
                    {
                        "model_name": model_name,
                        "model_version": model_version,
                    },
                )

                row = result.mappings().first()

                if row is None:
                    return None

                return dict(row)

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to get model version "
                "model_name=%s model_version=%s: %s",
                model_name,
                model_version,
                exc,
                exc_info=True,
            )
            raise

    # ==========================================================
    # CAMERA
    # ==========================================================

    def get_camera(
        self,
        camera_id: int,
    ) -> Optional[dict]:
        """
        Get camera information.
        """

        query = text(
            """
            SELECT
                camera_id,
                machine_id,
                camera_name,
                serial_number,
                ip_address
            FROM belt.camera
            WHERE camera_id = :camera_id
            LIMIT 1
            """
        )

        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    query,
                    {
                        "camera_id": camera_id,
                    },
                )

                row = result.mappings().first()

                if row is None:
                    return None

                return dict(row)

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to get camera camera_id=%s: %s",
                camera_id,
                exc,
                exc_info=True,
            )
            raise

    # ==========================================================
    # IMAGE - TRANSACTION
    # ==========================================================

    def create_image_tx(
        self,
        connection,
        camera_id: int,
        filename: str,
        image_path: str,
        capture_time: datetime,
        status: str = "PROCESSING",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> int:
        """
        Create image inside an existing transaction.
        """

        query = text(
            """
            INSERT INTO belt.image (
                camera_id,
                filename,
                image_path,
                image_width,
                image_height,
                capture_time,
                status
            )
            VALUES (
                :camera_id,
                :filename,
                :image_path,
                :image_width,
                :image_height,
                :capture_time,
                :status
            )
            RETURNING image_id
            """
        )

        result = connection.execute(
            query,
            {
                "camera_id": camera_id,
                "filename": filename,
                "image_path": image_path,
                "image_width": image_width,
                "image_height": image_height,
                "capture_time": capture_time,
                "status": status,
            },
        )

        image_id = result.scalar_one()

        return int(image_id)

    # ==========================================================
    # IMAGE - NORMAL
    # ==========================================================

    def create_image(
        self,
        camera_id: int,
        filename: str,
        image_path: str,
        capture_time: datetime,
        status: str = "PROCESSING",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        result_path: Optional[str] = None,
    ) -> int:
        """
        Create image using its own transaction.
        """

        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    INSERT INTO belt.image (
                        camera_id,
                        filename,
                        image_path,
                        result_path,
                        image_width,
                        image_height,
                        capture_time,
                        status
                    )
                    VALUES (
                        :camera_id,
                        :filename,
                        :image_path,
                        :result_path,
                        :image_width,
                        :image_height,
                        :capture_time,
                        :status
                    )
                    RETURNING image_id
                    """
                ),
                {
                    "camera_id": camera_id,
                    "filename": filename,
                    "image_path": image_path,
                    "result_path": result_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "capture_time": capture_time,
                    "status": status,
                },
            )

            return int(result.scalar_one())

    # ==========================================================
    # IMAGE PATH - TRANSACTION
    # ==========================================================

    def update_image_path_tx(
        self,
        connection,
        image_id: int,
        image_path: str,
    ) -> None:
        """
        Update image path inside an existing transaction.

        IMPORTANT:
        rowcount must be exactly 1.
        """

        query = text(
            """
            UPDATE belt.image
            SET image_path = :image_path
            WHERE image_id = :image_id
            """
        )

        result = connection.execute(
            query,
            {
                "image_id": image_id,
                "image_path": image_path,
            },
        )

        if result.rowcount != 1:
            raise RuntimeError(
                f"Image {image_id} not found"
            )

    # ==========================================================
    # IMAGE PATH - NORMAL
    # ==========================================================

    def update_image_path(
        self,
        image_id: int,
        image_path: str,
        result_path: Optional[str] = None,
    ) -> None:
        """
        Update image path using its own transaction.
        """

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE belt.image
                    SET
                        image_path = :image_path,
                        result_path = :result_path
                    WHERE image_id = :image_id
                    """
                ),
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "result_path": result_path,
                },
            )

    # ==========================================================
    # DETECTION - TRANSACTION
    # ==========================================================

    def create_detection_tx(
        self,
        connection,
        image_id: int,
        model_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> int:
        """
        Create detection inside an existing transaction.
        """

        query = text(
            """
            INSERT INTO belt.detection (
                image_id,
                model_id,
                class_id,
                class_name,
                confidence,
                x1,
                y1,
                x2,
                y2
            )
            VALUES (
                :image_id,
                :model_id,
                :class_id,
                :class_name,
                :confidence,
                :x1,
                :y1,
                :x2,
                :y2
            )
            RETURNING detection_id
            """
        )

        result = connection.execute(
            query,
            {
                "image_id": image_id,
                "model_id": model_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        )

        detection_id = result.scalar_one()

        return int(detection_id)

    # ==========================================================
    # DETECTION - NORMAL
    # ==========================================================

    def create_detection(
        self,
        image_id: int,
        model_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> int:
        """
        Create detection using its own transaction.
        """

        try:
            with self.engine.begin() as connection:
                return self.create_detection_tx(
                    connection=connection,
                    image_id=image_id,
                    model_id=model_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to create detection image_id=%s: %s",
                image_id,
                exc,
                exc_info=True,
            )
            raise

    # ==========================================================
    # INSPECTION RESULT - TRANSACTION
    # ==========================================================

    def create_inspection_result_tx(
        self,
        connection,
        image_id: int,
        good_count: int,
        splice_count: int,
        dogear_count: int,
        overall_result: str,
    ) -> int:
        """
        Create inspection result inside an existing transaction.
        """

        query = text(
            """
            INSERT INTO belt.inspection_result (
                image_id,
                good_count,
                splice_count,
                dogear_count,
                overall_result
            )
            VALUES (
                :image_id,
                :good_count,
                :splice_count,
                :dogear_count,
                :overall_result
            )
            RETURNING inspection_id
            """
        )

        result = connection.execute(
            query,
            {
                "image_id": image_id,
                "good_count": good_count,
                "splice_count": splice_count,
                "dogear_count": dogear_count,
                "overall_result": overall_result,
            },
        )

        inspection_id = result.scalar_one()

        return int(inspection_id)

    # ==========================================================
    # INSPECTION RESULT - NORMAL
    # ==========================================================

    def create_inspection_result(
        self,
        image_id: int,
        good_count: int,
        splice_count: int,
        dogear_count: int,
        overall_result: str,
    ) -> int:
        """
        Create inspection result using its own transaction.
        """

        try:
            with self.engine.begin() as connection:
                return self.create_inspection_result_tx(
                    connection=connection,
                    image_id=image_id,
                    good_count=good_count,
                    splice_count=splice_count,
                    dogear_count=dogear_count,
                    overall_result=overall_result,
                )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to create inspection result "
                "image_id=%s: %s",
                image_id,
                exc,
                exc_info=True,
            )
            raise

    # ==========================================================
    # IMAGE STATUS - TRANSACTION
    # ==========================================================

    def update_image_processed_tx(
        self,
        connection,
        image_id: int,
        status: str,
        inference_time_ms: Optional[float] = None,
        result_path: Optional[str] = None,
    ) -> None:
        """
        Update image processing status inside transaction.

        IMPORTANT:
        rowcount must be exactly 1.
        """

        query = text(
            """
            UPDATE belt.image
            SET
                status = :status,
                processed_time = CURRENT_TIMESTAMP,
                inference_time_ms = :inference_time_ms,
                result_path = :result_path
            WHERE image_id = :image_id
            """
        )

        result = connection.execute(
            query,
            {
                "image_id": image_id,
                "status": status,
                "inference_time_ms": inference_time_ms,
                "result_path": result_path,
            },
        )

        if result.rowcount != 1:
            raise RuntimeError(
                f"Image {image_id} not found"
            )

    # ==========================================================
    # IMAGE STATUS - NORMAL
    # ==========================================================

    def update_image_processed(
        self,
        image_id: int,
        status: str,
        inference_time_ms: Optional[float] = None,
        result_path: Optional[str] = None,
    ) -> None:
        """
        Update image processing status.
        """

        try:
            with self.engine.begin() as connection:
                self.update_image_processed_tx(
                    connection=connection,
                    image_id=image_id,
                    status=status,
                    inference_time_ms=inference_time_ms,
                    result_path=result_path,
                )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to update image status "
                "image_id=%s status=%s: %s",
                image_id,
                status,
                exc,
                exc_info=True,
            )
            raise


    # ============================================================

    # Repository Singleton

    # ============================================================

repository = DatabaseRepository()
