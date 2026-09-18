from datetime import datetime

from app.database.repository import DatabaseRepository


def main():
    print("=" * 70)
    print("BELT AI - DATABASE WRITE TEST")
    print("=" * 70)

    repository = DatabaseRepository()

    # 1. Connection
    if not repository.test_connection():
        print("Database connection : FAILED")
        return

    print("Database connection : OK")

    # 2. Get model
    model = repository.get_model_version(
        model_name="YOLO11",
        model_version="ODBv54",
    )

    if model is None:
        print("Model registry      : FAILED")
        return

    print(f"Model registry      : OK")
    print(f"Model ID            : {model['model_id']}")

    # 3. Create test image
    image_id = repository.create_image(
        camera_id=1,
        filename="db_test.jpg",
        image_path="tests/db_test.jpg",
        capture_time=datetime.now(),
        status="WAITING",
    )

    print(f"Image created       : {image_id}")

    # 4. Create test detection
    detection_id = repository.create_detection(
        image_id=image_id,
        model_id=model["model_id"],
        class_id=2,
        class_name="splice-belt",
        confidence=0.95,
        x1=100,
        y1=100,
        x2=500,
        y2=500,
    )

    print(f"Detection created   : {detection_id}")

    # 5. Create inspection result
    inspection_id = repository.create_inspection_result(
        image_id=image_id,
        good_count=0,
        splice_count=1,
        dogear_count=0,
        overall_result="NG",
    )

    print(f"Inspection created  : {inspection_id}")

    # 6. Update image
    repository.update_image_processed(
        image_id=image_id,
        status="DONE",
        inference_time_ms=123.45,
        result_path="tests/db_test_result.jpg",
    )

    print("Image updated       : DONE")

    print("=" * 70)
    print("DATABASE WRITE TEST : PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()