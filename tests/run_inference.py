from pathlib import Path
import time

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "staging"
    / "best (yolo11_ODBv54).pt"
)

IMAGE_PATH = BASE_DIR / "tests" / "sample.jpg"


def main():

    print("=" * 70)
    print("BELT AI - REAL INFERENCE TEST")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        return

    print(f"Model : {MODEL_PATH.name}")
    print(f"Image : {IMAGE_PATH.name}")

    model = YOLO(str(MODEL_PATH))

    start = time.perf_counter()

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.5,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            bbox = [
                round(float(x), 2)
                for x in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                }
            )

    print()
    print("-" * 70)
    print(f"Inference time : {elapsed_ms:.2f} ms")
    print(f"Detections     : {len(detections)}")
    print("-" * 70)

    for i, detection in enumerate(detections, start=1):

        print(
            f"{i}. "
            f"{detection['class_name']} | "
            f"confidence={detection['confidence']:.4f} | "
            f"bbox={detection['bbox']}"
        )

    print("-" * 70)

    ng_classes = {
        "splice-belt",
        "dogear-belt",
    }

    ng_count = sum(
        1
        for detection in detections
        if detection["class_name"] in ng_classes
    )

    overall = "NG" if ng_count > 0 else "GOOD"

    print(f"Overall result : {overall}")
    print(f"NG count       : {ng_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()