from sqlalchemy import text

from app.database.repository import DatabaseRepository


def main():

    repository = DatabaseRepository()

    print("=" * 60)
    print("BELT AI - TRANSACTION ROLLBACK TEST")
    print("=" * 60)

    # ------------------------------------------------------
    # Test data
    # ------------------------------------------------------

    camera_id = 1
    test_filename = "__ROLLBACK_TEST__.jpg"

    image_id = None

    try:

        # --------------------------------------------------
        # Start Transaction
        # --------------------------------------------------

        print("\n[1] BEGIN TRANSACTION")

        with repository.begin() as connection:

            # ----------------------------------------------
            # Create Image
            # ----------------------------------------------

            image_id = repository.create_image_tx(
                connection=connection,
                camera_id=camera_id,
                filename=test_filename,
                image_path="__rollback_test__",
                image_width=100,
                image_height=100,
                status="PROCESSING",
            )

            print(f"[2] Image created: {image_id}")

            # ----------------------------------------------
            # Create Detection
            # ----------------------------------------------

            detection_id = repository.create_detection_tx(
                connection=connection,
                image_id=image_id,
                model_id=2,
                class_id=2,
                class_name="splice-belt",
                confidence=0.9999,
                x1=1,
                y1=1,
                x2=10,
                y2=10,
            )

            print(f"[3] Detection created: {detection_id}")

            # ----------------------------------------------
            # FORCE ERROR
            # ----------------------------------------------

            print("[4] FORCE ERROR")

            raise RuntimeError(
                "Intentional rollback test error"
            )

    except RuntimeError as exc:

        print(f"[5] Exception caught: {exc}")
        print("[6] Transaction should be ROLLED BACK")


    # ------------------------------------------------------
    # Verify Database
    # ------------------------------------------------------

    print("\n[7] VERIFY DATABASE")

    with repository.engine.connect() as connection:

        image_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM belt.image
                WHERE filename = :filename
                """
            ),
            {
                "filename": test_filename,
            },
        ).scalar_one()

        detection_count = 0

        if image_id is not None:

            detection_count = connection.execute(
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
            ).scalar_one()

        inspection_count = 0

        if image_id is not None:

            inspection_count = connection.execute(
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
            ).scalar_one()

    print(f"Image records       : {image_count}")
    print(f"Detection records   : {detection_count}")
    print(f"Inspection records  : {inspection_count}")

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    if (
        image_count == 0
        and detection_count == 0
        and inspection_count == 0
    ):

        print("\nTRANSACTION ROLLBACK TEST : PASSED")

    else:

        print("\nTRANSACTION ROLLBACK TEST : FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()