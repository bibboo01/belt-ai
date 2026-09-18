from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "staging"
    / "best (yolo11_ODBv54).pt"
)


def main():

    print("=" * 70)
    print("BELT AI - STAGING INFERENCE TEST")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print("ERROR: Model not found")
        return

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully")

    print(f"Task  : {model.task}")
    print(f"Names : {model.names}")

    print("=" * 70)
    print("STAGING MODEL READY")
    print("=" * 70)


if __name__ == "__main__":
    main()