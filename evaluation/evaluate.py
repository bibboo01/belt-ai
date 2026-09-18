from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import yaml
from ultralytics import YOLO

from evaluation.evaluation_result import EvaluationResult
from evaluation.evaluation_gate import evaluate_gate, save_result


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# Utilities
# ==========================================================

def sha256_file(path: Path) -> str:

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_dataset_config(data_path: Path) -> dict:

    with open(
        data_path,
        "r",
        encoding="utf-8-sig",
    ) as f:

        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Invalid YOLO dataset config: {data_path}"
        )

    return data


def validate_test_dataset(data_path: Path) -> dict:

    data = load_dataset_config(data_path)

    if "test" not in data:
        raise RuntimeError(
            "Dataset does not define a 'test' split."
        )

    if not data["test"]:
        raise RuntimeError(
            "Dataset test split is empty."
        )

    print("Dataset validation:")
    print(f"  Test split : {data['test']}")
    print()

    return data


def extract_model_identity(
    model_path: Path,
) -> tuple[str, str]:

    filename = model_path.stem

    # best (yolo11_ODBv54).pt
    if "yolo11_" in filename:

        version = (
            filename
            .split("yolo11_", 1)[1]
            .rstrip(")")
        )

        return "YOLO11", version

    return "YOLO11", filename


# ==========================================================
# Evaluation
# ==========================================================

def evaluate_model(
    model_path: str,
    data_path: str,
    imgsz: int = 1280,
    conf: float = 0.001,
):

    model_file = Path(model_path).resolve()
    data_file = Path(data_path).resolve()

    # ------------------------------------------------------
    # Validate files
    # ------------------------------------------------------

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    # ------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------

    dataset_config = validate_test_dataset(
        data_file
    )

    # ------------------------------------------------------
    # Model identity
    # ------------------------------------------------------

    model_name, model_version = (
        extract_model_identity(model_file)
    )

    model_sha256 = sha256_file(model_file)

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    print("=" * 70)
    print("BELT AI - REAL MODEL EVALUATION")
    print("=" * 70)

    print(f"Model         : {model_file}")
    print(f"Model name    : {model_name}")
    print(f"Model version : {model_version}")
    print(f"Dataset       : {data_file}")
    print(f"Image size    : {imgsz}")
    print(f"Confidence    : {conf}")
    print(f"SHA256        : {model_sha256}")
    print()

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    print("Loading model...")

    model = YOLO(str(model_file))

    print("Model loaded")
    print(f"Classes : {model.names}")
    print()

    # ------------------------------------------------------
    # REAL TEST
    # ------------------------------------------------------

    print("=" * 70)
    print("RUNNING REAL TEST EVALUATION")
    print("=" * 70)
    print()

    start_time = time.perf_counter()

    results = model.val(
        data=str(data_file),
        imgsz=imgsz,
        conf=conf,
        split="test",
        plots=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - start_time

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)

    # ------------------------------------------------------
    # EvaluationResult
    # ------------------------------------------------------

    result = EvaluationResult(
        model_name=model_name,
        model_version=model_version,
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        evaluation_type="REAL",
    )

    # ------------------------------------------------------
    # Evaluation Gate
    # ------------------------------------------------------

    result = evaluate_gate(result)

    # ------------------------------------------------------
    # Print result
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL EVALUATION RESULT")
    print("=" * 70)

    print(f"Model       : {model_name}")
    print(f"Version     : {model_version}")
    print("Evaluation  : REAL")
    print()
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP50       : {map50:.4f}")
    print(f"mAP50-95    : {map50_95:.4f}")
    print()
    print(f"Gate        : {result.status}")
    print(f"Notes       : {result.notes}")
    print()
    print(f"Evaluation time : {elapsed:.2f} sec")

    if hasattr(results, "speed"):

        print()
        print("Speed:")

        for key, value in results.speed.items():

            print(
                f"  {key}: {value:.2f} ms"
            )

    print("=" * 70)

    # ------------------------------------------------------
    # Save result
    # ------------------------------------------------------

    output_file = save_result(result)

    print()
    print("Evaluation result saved:")
    print(f"  {output_file}")
    print()

    # ------------------------------------------------------
    # Add metadata to output
    # ------------------------------------------------------

    # EvaluationResult.save_result() already stores
    # the core metrics and gate status.
    #
    # Model SHA256 / dataset / runtime information
    # can be added in a later MLOps metadata step.

    return result, output_file


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Belt AI REAL YOLO Model Evaluation"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO .pt model",
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to YOLO data.yaml",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold",
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()