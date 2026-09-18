"""
Belt AI - Production Safe Model Promotion V2

Usage:
    .\.venv\Scripts\python.exe .\mlops\promote_model.py YOLO11 ODBv54

Promotion flow:

    STAGING
       ↓
    Validate model
       ↓
    Find REAL evaluation
       ↓
    PASSED?
       ↓
    Backup production
       ↓
    Deploy candidate atomically
       ↓
    Validate deployed file
       ↓
    Update production pointer atomically
       ↓
    Archive previous production
       ↓
    SUCCESS

If deployment fails:
    → rollback previous production model + pointer
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

STAGING_DIR = BASE_DIR / "models" / "staging"
PRODUCTION_DIR = BASE_DIR / "models" / "production"
ARCHIVE_DIR = BASE_DIR / "models" / "archive"

EVALUATION_DIR = BASE_DIR / "evaluation" / "results"

PRODUCTION_MODEL = PRODUCTION_DIR / "best.pt"
PRODUCTION_POINTER = PRODUCTION_DIR / "model.json"


# ============================================================
# SAFETY SETTINGS
# ============================================================

REQUIRED_EVALUATION_TYPE = "REAL"
REQUIRED_STATUS = "PASSED"


# ============================================================
# UTILS
# ============================================================

def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def atomic_write_json(path: Path, data: dict) -> None:
    """
    Write JSON atomically.

    The original file is untouched until the temporary file
    has been completely written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


# ============================================================
# EVALUATION
# ============================================================

def load_latest_evaluation(
    model_name: str,
    model_version: str,
):
    """
    Find the newest evaluation matching EXACT model name/version.

    Important:
    TEST_ONLY / SIMULATED files may exist but are not accepted
    by validate_evaluation().
    """

    if not EVALUATION_DIR.exists():
        raise RuntimeError(
            f"Evaluation directory not found: {EVALUATION_DIR}"
        )

    candidates = []

    for file_path in EVALUATION_DIR.glob("*.json"):

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8-sig",
            ) as f:
                data = json.load(f)

        except Exception:
            continue

        if data.get("model_name") != model_name:
            continue

        if data.get("model_version") != model_version:
            continue

        candidates.append(
            (
                file_path.stat().st_mtime,
                file_path,
                data,
            )
        )

    if not candidates:
        raise RuntimeError(
            f"No evaluation found for "
            f"{model_name} {model_version}"
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, path, data = candidates[0]

    return path, data


def validate_evaluation(
    evaluation_path: Path,
    evaluation: dict,
) -> None:
    """
    Production evaluation gate.

    Only:

        evaluation_type = REAL
        status = PASSED

    can pass.
    """

    evaluation_type = evaluation.get("evaluation_type")

    if evaluation_type != REQUIRED_EVALUATION_TYPE:
        raise RuntimeError(
            f"Evaluation is not REAL: "
            f"{evaluation_type!r} "
            f"({evaluation_path.name})"
        )

    status = evaluation.get("status")

    if status != REQUIRED_STATUS:
        raise RuntimeError(
            f"Evaluation status is not PASSED: "
            f"{status!r} "
            f"({evaluation_path.name})"
        )


# ============================================================
# MODEL DISCOVERY
# ============================================================

def find_staging_model(
    model_name: str,
    model_version: str,
) -> Path:
    """
    Find candidate model in staging.
    """

    candidates = [
        STAGING_DIR / f"best (yolo11_{model_version}).pt",
        STAGING_DIR / f"best ({model_name}_{model_version}).pt",
        STAGING_DIR / f"{model_name}_{model_version}.pt",
        STAGING_DIR / f"best_{model_name}_{model_version}.pt",
        STAGING_DIR / model_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Staging model not found.\n"
        f"Model     : {model_name}\n"
        f"Version   : {model_version}\n"
        f"Staging   : {STAGING_DIR}"
    )


def validate_model_file(path: Path) -> dict:
    """
    Basic model integrity validation.

    Does not load PyTorch model yet.
    """

    if not path.exists():
        raise RuntimeError(
            f"Model file does not exist: {path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Model path is not a file: {path}"
        )

    size = path.stat().st_size

    if size <= 0:
        raise RuntimeError(
            f"Model file is empty: {path}"
        )

    return {
        "file_name": path.name,
        "file_size_bytes": size,
        "sha256": sha256_file(path),
    }


# ============================================================
# BACKUP
# ============================================================

def create_rollback_backup(
    timestamp: str,
):
    """
    Create rollback copies BEFORE touching production.

    Returns:
        backup_model
        backup_pointer
    """

    rollback_dir = (
        ARCHIVE_DIR /
        f"rollback_{timestamp}"
    )

    rollback_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    backup_model = rollback_dir / "best.pt"
    backup_pointer = rollback_dir / "model.json"

    if PRODUCTION_MODEL.exists():
        shutil.copy2(
            PRODUCTION_MODEL,
            backup_model,
        )

    if PRODUCTION_POINTER.exists():
        shutil.copy2(
            PRODUCTION_POINTER,
            backup_pointer,
        )

    return backup_model, backup_pointer, rollback_dir


# ============================================================
# ROLLBACK
# ============================================================

def rollback_production(
    backup_model: Path,
    backup_pointer: Path,
) -> None:
    """
    Restore previous production state.
    """

    print("")
    print("=" * 70)
    print("ROLLBACK")
    print("=" * 70)

    try:

        if backup_model.exists():
            shutil.copy2(
                backup_model,
                PRODUCTION_MODEL,
            )
            print("Restored model   : OK")

        elif PRODUCTION_MODEL.exists():
            PRODUCTION_MODEL.unlink()
            print("Removed candidate model")

        if backup_pointer.exists():
            shutil.copy2(
                backup_pointer,
                PRODUCTION_POINTER,
            )
            print("Restored pointer : OK")

        elif PRODUCTION_POINTER.exists():
            PRODUCTION_POINTER.unlink()
            print("Removed candidate pointer")

        print("ROLLBACK         : SUCCESS")

    except Exception as exc:

        print(
            f"ROLLBACK        : FAILED - {exc}"
        )

        raise


# ============================================================
# PROMOTION POINTER
# ============================================================

def build_production_pointer(
    model_name: str,
    model_version: str,
    model_info: dict,
    evaluation_path: Path,
    evaluation: dict,
) -> dict:

    return {
        "system": "Belt AI",
        "model_id": evaluation.get("model_id"),
        "model_name": model_name,
        "model_version": model_version,
        "model_file": "best.pt",
        "status": "PRODUCTION",

        "evaluation_type": evaluation.get(
            "evaluation_type"
        ),

        "evaluation_status": evaluation.get(
            "status"
        ),

        "evaluation_file": evaluation_path.name,

        "sha256": model_info["sha256"],

        "file_size_bytes": model_info[
            "file_size_bytes"
        ],

        "promoted_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }


# ============================================================
# PROMOTION
# ============================================================

def promote_model(
    model_name: str,
    model_version: str,
) -> Path:

    print("=" * 70)
    print("BELT AI - PRODUCTION MODEL PROMOTION V2")
    print("=" * 70)

    print(f"Model       : {model_name}")
    print(f"Version     : {model_version}")
    print(f"Staging     : {STAGING_DIR}")
    print(f"Production  : {PRODUCTION_DIR}")
    print("")

    # --------------------------------------------------------
    # STEP 1 - FIND MODEL
    # --------------------------------------------------------

    print("[1/8] Finding staging model...")

    staging_model = find_staging_model(
        model_name,
        model_version,
    )

    print(
        f"      Candidate : {staging_model.name}"
    )

    # --------------------------------------------------------
    # STEP 2 - VALIDATE MODEL
    # --------------------------------------------------------

    print("[2/8] Validating model file...")

    model_info = validate_model_file(
        staging_model
    )

    print(
        f"      Size      : "
        f"{model_info['file_size_bytes']:,} bytes"
    )

    print(
        f"      SHA256    : "
        f"{model_info['sha256'][:16]}..."
    )

    # --------------------------------------------------------
    # STEP 3 - LOAD EVALUATION
    # --------------------------------------------------------

    print("[3/8] Loading evaluation...")

    evaluation_path, evaluation = (
        load_latest_evaluation(
            model_name,
            model_version,
        )
    )

    print(
        f"      Evaluation: "
        f"{evaluation_path.name}"
    )

    # --------------------------------------------------------
    # STEP 4 - SAFETY GATE
    # --------------------------------------------------------

    print("[4/8] Running production safety gate...")

    validate_evaluation(
        evaluation_path,
        evaluation,
    )

    print("      Evaluation type : REAL")
    print("      Status          : PASSED")
    print("      SAFETY GATE     : PASS")

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    PRODUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_model, backup_pointer, rollback_dir = (
        create_rollback_backup(timestamp)
    )

    try:

        # ----------------------------------------------------
        # STEP 5 - DEPLOY MODEL TEMPORARILY
        # ----------------------------------------------------

        print("[5/8] Deploying candidate...")

        temp_model = (
            PRODUCTION_DIR /
            "best.pt.tmp"
        )

        shutil.copy2(
            staging_model,
            temp_model,
        )

        # Verify copied file
        temp_info = validate_model_file(
            temp_model
        )

        if (
            temp_info["sha256"]
            != model_info["sha256"]
        ):
            raise RuntimeError(
                "Candidate SHA256 mismatch "
                "after copy"
            )

        # Atomic replace
        os.replace(
            temp_model,
            PRODUCTION_MODEL,
        )

        print("      Model deploy : OK")

        # ----------------------------------------------------
        # STEP 6 - VERIFY PRODUCTION MODEL
        # ----------------------------------------------------

        print("[6/8] Verifying production model...")

        production_info = validate_model_file(
            PRODUCTION_MODEL
        )

        if (
            production_info["sha256"]
            != model_info["sha256"]
        ):
            raise RuntimeError(
                "Production model SHA256 mismatch"
            )

        if (
            production_info["file_size_bytes"]
            != model_info["file_size_bytes"]
        ):
            raise RuntimeError(
                "Production model size mismatch"
            )

        print("      SHA256 : MATCH")
        print("      Size   : MATCH")

        # ----------------------------------------------------
        # STEP 7 - UPDATE POINTER
        # ----------------------------------------------------

        print("[7/8] Updating production pointer...")

        pointer = build_production_pointer(
            model_name,
            model_version,
            production_info,
            evaluation_path,
            evaluation,
        )

        atomic_write_json(
            PRODUCTION_POINTER,
            pointer,
        )

        # Verify pointer
        with open(
            PRODUCTION_POINTER,
            "r",
            encoding="utf-8-sig",
        ) as f:
            verified_pointer = json.load(f)

        if verified_pointer.get(
            "model_version"
        ) != model_version:

            raise RuntimeError(
                "Production pointer verification failed"
            )

        if verified_pointer.get(
            "status"
        ) != "PRODUCTION":

            raise RuntimeError(
                "Production pointer status "
                "verification failed"
            )

        print("      Pointer : OK")

        # ----------------------------------------------------
        # STEP 8 - ARCHIVE
        # ----------------------------------------------------

        print("[8/8] Finalizing archive...")

        archive_metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "promoted_at": pointer[
                "promoted_at"
            ],
            "sha256": production_info[
                "sha256"
            ],
            "evaluation_file": evaluation_path.name,
        }

        atomic_write_json(
            rollback_dir / "metadata.json",
            archive_metadata,
        )

        print(
            f"      Rollback archive : "
            f"{rollback_dir.name}"
        )

        print("")
        print("=" * 70)
        print("PROMOTION SUCCESS")
        print("=" * 70)

        print(
            f"Production model : "
            f"{PRODUCTION_MODEL}"
        )

        print(
            f"Model version    : "
            f"{model_version}"
        )

        print(
            f"Evaluation       : "
            f"{evaluation_path.name}"
        )

        print(
            f"SHA256           : "
            f"{production_info['sha256']}"
        )

        print("=" * 70)

        return PRODUCTION_MODEL

    except Exception as exc:

        print("")
        print(
            f"PROMOTION ERROR : {exc}"
        )

        rollback_production(
            backup_model,
            backup_pointer,
        )

        raise


# ============================================================
# CLI
# ============================================================

def main() -> int:

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "  python mlops/promote_model.py "
            "<model_name> <model_version>"
        )

        return 1

    model_name = sys.argv[1]
    model_version = sys.argv[2]

    try:

        promote_model(
            model_name,
            model_version,
        )

        return 0

    except Exception as exc:

        print("")
        print("=" * 70)
        print("PROMOTION BLOCKED")
        print("=" * 70)
        print(str(exc))
        print("=" * 70)

        return 1


if __name__ == "__main__":
    sys.exit(main())