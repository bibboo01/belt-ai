from pathlib import Path
import json
import shutil
import sys


BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCTION_DIR = BASE_DIR / "models" / "production"
ARCHIVE_DIR = BASE_DIR / "models" / "archive"

PRODUCTION_POINTER = PRODUCTION_DIR / "model.json"


def rollback_model(model_name: str):

    print("=" * 70)
    print("BELT AI - MODEL ROLLBACK")
    print("=" * 70)

    archive_model = ARCHIVE_DIR / model_name

    if not archive_model.exists():
        raise FileNotFoundError(
            f"Archived model not found: {archive_model}"
        )

    print(f"Rollback model : {model_name}")

    # ------------------------------------------------------
    # 1. Read current production model
    # ------------------------------------------------------

    current_model = None

    if PRODUCTION_POINTER.exists():

        with open(
            PRODUCTION_POINTER,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

            current_model = data.get("model_file")

    # ------------------------------------------------------
    # 2. Archive current production model
    # ------------------------------------------------------

    if current_model:

        current_path = (
            PRODUCTION_DIR / current_model
        )

        if current_path.exists():

            archive_current = (
                ARCHIVE_DIR / current_model
            )

            print(
                f"Archiving current model: "
                f"{current_model}"
            )

            shutil.move(
                str(current_path),
                str(archive_current),
            )

    # ------------------------------------------------------
    # 3. Restore archived model
    # ------------------------------------------------------

    production_model = (
        PRODUCTION_DIR / model_name
    )

    shutil.copy2(
        str(archive_model),
        str(production_model),
    )

    # ------------------------------------------------------
    # 4. Update production pointer
    # ------------------------------------------------------

    model_version = model_name

    if "ODBv3" in model_name:
        model_version = "ODBv3"

    elif "ODBv54" in model_name:
        model_version = "ODBv54"

    pointer = {
        "system": "Belt AI",
        "model_name": "YOLO11",
        "model_version": model_version,
        "model_file": model_name,
        "status": "PRODUCTION",
    }

    with open(
        PRODUCTION_POINTER,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            pointer,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("ROLLBACK SUCCESS")
    print("=" * 70)

    print(
        f"Production model : "
        f"{production_model}"
    )

    print(
        f"Version          : "
        f"{model_version}"
    )

    print("=" * 70)


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python rollback_model.py "
            "<model_name>"
        )

        sys.exit(1)

    model_name = sys.argv[1]

    try:

        rollback_model(
            model_name
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ROLLBACK FAILED")
        print("=" * 70)

        print(
            f"Reason : {e}"
        )

        print("=" * 70)

        sys.exit(1)


if __name__ == "__main__":
    main()