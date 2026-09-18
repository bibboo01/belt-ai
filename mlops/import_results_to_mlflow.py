"""
Import training results from a CSV file into MLflow.

Project: belt-ai
Purpose:
    Convert an existing CSV of YOLO training/evaluation results into
    MLflow runs so historical experiments can be compared in MLflow UI.

Expected usage from the belt-ai project root:

    .\.venv\Scripts\python.exe .\mlops\import_results_to_mlflow.py

Optional:

    .\.venv\Scripts\python.exe .\mlops\import_results_to_mlflow.py `
        --csv ".\result-train - dataset-result-3-class.csv"

    .\.venv\Scripts\python.exe .\mlops\import_results_to_mlflow.py `
        --csv ".\result-train - dataset-result-3-class.csv" `
        --experiment "belt-training-results"
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


# ============================================================================
# PROJECT CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV = BASE_DIR / "result-train - Copy of dataset-result-3-class.csv"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "belt-training-results"

# These are common column-name variants that may occur in exported CSV files.
COLUMN_ALIASES = {
    "date": [
        "date",
        "Date",
        "วันที่",
    ],
    "dataset": [
        "name-dataset",
        "dataset",
        "Dataset",
        "dataset_name",
        "name_dataset",
    ],
    "image": [
        "image",
        "images",
        "Image",
        "Image Size",
        "image_size",
    ],
    "precision": [
        "Precision",
        "precision",
        "P",
        "p",
    ],
    "recall": [
        "Recall",
        "recall",
        "R",
        "r",
    ],
    "map50": [
        "mAP@50",
        "mAP50",
        "map50",
        "mAP 50",
        "mAP_50",
    ],
    "map5095": [
        "mAP@50-95",
        "mAP50-95",
        "map50-95",
        "mAP 50-95",
        "mAP_50_95",
    ],
}


# ============================================================================
# HELPERS
# ============================================================================

def clean_column_name(value: Any) -> str:
    """Normalize a CSV column name for comparison."""
    return re.sub(r"[\s_\-]+", "", str(value).strip().lower())


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Find the actual DataFrame column matching one of the aliases."""
    normalized = {
        clean_column_name(column): column
        for column in df.columns
    }

    for alias in aliases:
        key = clean_column_name(alias)
        if key in normalized:
            return normalized[key]

    return None


def build_column_map(df: pd.DataFrame) -> dict[str, str | None]:
    """Map logical fields to actual CSV column names."""
    return {
        field: find_column(df, aliases)
        for field, aliases in COLUMN_ALIASES.items()
    }


def parse_number(value: Any) -> float | None:
    """
    Convert a CSV value to float.

    Handles:
        0.9518
        "0.9518"
        "95.18%"
        "0.9518 "
        empty/NaN
    """
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None

    is_percent = "%" in text
    text = text.replace("%", "").replace(",", "").strip()

    # Keep numeric characters, sign and decimal notation.
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)

    if not match:
        return None

    number = float(match.group())

    # If CSV contains 95.18%, convert to 0.9518.
    if is_percent:
        number /= 100.0

    return number


def safe_text(value: Any, default: str = "unknown") -> str:
    """Convert a CSV value into a safe MLflow tag string."""
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    return text if text else default


def make_run_name(
    row: pd.Series,
    row_number: int,
    column_map: dict[str, str | None],
) -> str:
    """Create a readable MLflow run name."""
    dataset_col = column_map.get("dataset")
    date_col = column_map.get("date")

    dataset = safe_text(row.get(dataset_col)) if dataset_col else "dataset"
    date = safe_text(row.get(date_col), "no-date") if date_col else "no-date"

    dataset = re.sub(r"[^\w.\-]+", "_", dataset)
    date = re.sub(r"[^\w.\-]+", "_", date)

    return f"{dataset}_{date}_row-{row_number}"


def log_extra_columns(
    row: pd.Series,
    column_map: dict[str, str | None],
) -> None:
    """
    Log remaining useful numeric columns as MLflow metrics and
    remaining textual columns as tags.

    This makes the importer tolerant of future CSV columns.
    """
    known_columns = {
        column
        for column in column_map.values()
        if column is not None
    }

    for column in row.index:
        if column in known_columns:
            continue

        value = row[column]

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        numeric = parse_number(value)

        # Only treat a value as a metric when the original value is
        # clearly numeric. Avoid converting arbitrary text containing digits.
        if isinstance(value, (int, float)) and numeric is not None:
            metric_name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(column)).strip("_")
            if metric_name:
                try:
                    mlflow.log_metric(metric_name[:250], numeric)
                except Exception:
                    pass
        else:
            tag_name = re.sub(r"[^a-zA-Z0-9_.\-]+", "_", str(column)).strip("_")
            if tag_name:
                try:
                    mlflow.set_tag(tag_name[:250], safe_text(value)[:500])
                except Exception:
                    pass


# ============================================================================
# IMPORT
# ============================================================================

def import_csv(
    csv_path: Path,
    experiment_name: str,
    tracking_uri: str,
) -> None:
    """Import every CSV row as one MLflow run."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}"
        )

    print("=" * 78)
    print("belt-ai | CSV -> MLflow importer")
    print("=" * 78)
    print(f"CSV       : {csv_path}")
    print(f"Tracking  : {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print()

    # ------------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------------

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        # Useful for CSV files exported with Windows/Thai encoding.
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if df.empty:
        print("CSV contains no rows.")
        return

    print(f"Rows      : {len(df):,}")
    print(f"Columns   : {len(df.columns)}")
    print()

    print("CSV columns:")
    for index, column in enumerate(df.columns, start=1):
        print(f"  {index:02d}. {column}")
    print()

    # ------------------------------------------------------------------------
    # Detect columns
    # ------------------------------------------------------------------------

    column_map = build_column_map(df)

    print("Detected columns:")
    for logical_name, actual_column in column_map.items():
        print(f"  {logical_name:10s}: {actual_column}")
    print()

    # At least one useful field must exist.
    detected_metrics = [
        column_map["precision"],
        column_map["recall"],
        column_map["map50"],
        column_map["map5095"],
    ]

    if not any(detected_metrics):
        raise RuntimeError(
            "Could not find Precision / Recall / mAP columns in the CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------------
    # Connect to MLflow
    # ------------------------------------------------------------------------

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment was not created/found: {experiment_name}"
        )

    print(f"Experiment ID: {experiment.experiment_id}")
    print()

    # ------------------------------------------------------------------------
    # Import rows
    # ------------------------------------------------------------------------

    success_count = 0
    skipped_count = 0
    error_count = 0

    metric_columns = {
        "precision": column_map["precision"],
        "recall": column_map["recall"],
        "mAP50": column_map["map50"],
        "mAP50_95": column_map["map5095"],
    }

    for index, row in df.iterrows():
        row_number = index + 2  # +1 for zero-index, +1 for CSV header

        run_name = make_run_name(
            row=row,
            row_number=row_number,
            column_map=column_map,
        )

        try:
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name,
            ):

                # ------------------------------------------------------------
                # General tags
                # ------------------------------------------------------------

                mlflow.set_tag("project", "belt-ai")
                mlflow.set_tag("source", "csv-import")
                mlflow.set_tag("source_file", csv_path.name)
                mlflow.set_tag("csv_row", str(row_number))
                mlflow.set_tag("model_family", "YOLO")

                # ------------------------------------------------------------
                # Dataset/date/image information
                # ------------------------------------------------------------

                dataset_col = column_map["dataset"]
                date_col = column_map["date"]
                image_col = column_map["image"]

                if dataset_col:
                    mlflow.set_tag(
                        "dataset",
                        safe_text(row[dataset_col])[:500],
                    )

                if date_col:
                    mlflow.set_tag(
                        "training_date",
                        safe_text(row[date_col])[:500],
                    )

                if image_col:
                    image_value = safe_text(row[image_col])
                    mlflow.set_tag("image", image_value[:500])

                    # If image field is numeric, also store as parameter.
                    image_number = parse_number(row[image_col])
                    if image_number is not None:
                        mlflow.log_param("image_count", image_number)

                # ------------------------------------------------------------
                # Metrics
                # ------------------------------------------------------------

                for metric_name, actual_column in metric_columns.items():
                    if not actual_column:
                        continue

                    value = parse_number(row[actual_column])

                    if value is not None and math.isfinite(value):
                        mlflow.log_metric(metric_name, value)

                # ------------------------------------------------------------
                # Extra columns
                # ------------------------------------------------------------

                log_extra_columns(row, column_map)

            success_count += 1

            if success_count <= 10 or success_count % 25 == 0:
                print(
                    f"[OK] row={row_number:03d} "
                    f"run={run_name}"
                )

        except Exception as exc:
            error_count += 1
            print(
                f"[ERROR] row={row_number:03d} "
                f"run={run_name}\n"
                f"        {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("IMPORT COMPLETE")
    print("=" * 78)
    print(f"Total rows : {len(df):,}")
    print(f"Imported   : {success_count:,}")
    print(f"Skipped    : {skipped_count:,}")
    print(f"Errors     : {error_count:,}")
    print()
    print("Open MLflow UI:")
    print("  http://127.0.0.1:5000")
    print()
    print(f"Experiment:")
    print(f"  {experiment_name}")
    print("=" * 78)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import YOLO training results CSV into MLflow."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=(
            "Path to CSV file. "
            f"Default: {DEFAULT_CSV}"
        ),
    )

    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=(
            "MLflow experiment name. "
            f"Default: {DEFAULT_EXPERIMENT}"
        ),
    )

    parser.add_argument(
        "--tracking-uri",
        default=MLFLOW_TRACKING_URI,
        help=(
            "MLflow Tracking Server URI. "
            f"Default: {MLFLOW_TRACKING_URI}"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import_csv(
            csv_path=args.csv.resolve(),
            experiment_name=args.experiment,
            tracking_uri=args.tracking_uri,
        )
        return 0

    except FileNotFoundError as exc:
        print(f"\n[FILE ERROR] {exc}")
        return 1

    except Exception as exc:
        print(
            f"\n[FATAL ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
