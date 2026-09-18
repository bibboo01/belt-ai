from pathlib import Path
import shutil

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

STAGING_DIR = BASE_DIR / "models" / "staging"
PRODUCTION_DIR = BASE_DIR / "models" / "production"
ARCHIVE_DIR = BASE_DIR / "models" / "archive"


def inspect_model(model_file: Path):
    """Read basic metadata from YOLO model."""

    model = YOLO(str(model_file))

    return {
        "name": model_file.name,
        "path": str(model_file),
        "size_mb": round(
            model_file.stat().st_size / (1024 * 1024),
            2
        ),
        "task": model.task,
        "classes": model.names,
        "parameters": sum(
            p.numel()
            for p in model.model.parameters()
        ),
    }


def list_models():
    """List and inspect models in staging."""

    models = []

    if not STAGING_DIR.exists():
        return models

    for model_file in STAGING_DIR.glob("*.pt"):
        try:
            info = inspect_model(model_file)
            models.append(info)

        except Exception as e:
            models.append({
                "name": model_file.name,
                "error": str(e),
            })

    return models


def promote_to_production(model_name: str):
    """Copy a staging model to production."""

    source = STAGING_DIR / model_name
    destination = PRODUCTION_DIR / model_name

    if not source.exists():
        raise FileNotFoundError(
            f"Staging model not found: {source}"
        )

    PRODUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(source, destination)

    return {
        "status": "PROMOTED",
        "model": model_name,
        "production_path": str(destination),
    }


if __name__ == "__main__":

    print("=" * 70)
    print("BELT AI - MODEL REGISTRY")
    print("=" * 70)

    print("\nSTAGING MODELS:")

    models = list_models()

    if not models:
        print("No models found.")

    for model in models:

        print("\n" + "-" * 70)

        if "error" in model:
            print(f"Model : {model['name']}")
            print(f"ERROR : {model['error']}")
            continue

        print(f"Model      : {model['name']}")
        print(f"Size       : {model['size_mb']} MB")
        print(f"Task       : {model['task']}")
        print(f"Parameters : {model['parameters']:,}")
        print(f"Classes    : {model['classes']}")

    print("\n" + "=" * 70)