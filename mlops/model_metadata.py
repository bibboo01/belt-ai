from pathlib import Path
import json
from datetime import datetime

from model_registry import list_models


BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "mlops" / "metadata"


def create_metadata():
    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    models = list_models()

    metadata = {
        "system": "Belt AI",
        "created_at": datetime.now().isoformat(),
        "models": models,
    }

    output_file = METADATA_DIR / "models.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 70)
    print("BELT AI - MODEL METADATA")
    print("=" * 70)

    print(f"Metadata file : {output_file}")
    print(f"Models found  : {len(models)}")

    print("=" * 70)


if __name__ == "__main__":
    create_metadata()