from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mlflow
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV = BASE_DIR / "result-train - Copy of dataset-result-3-class.csv"
TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "result-train"

# ============================================================================
# EXPECTED CSV SCHEMA
# ============================================================================

EXPECTED_COLUMNS = [
    "date",
    "name-dataset",
    "image",
    "Precision",
    "Recall",
    "mAP@50",
    "mAP@50-95",
]

METRIC_MAPPING = {
    "precision": "Precision",
    "recall": "Recall",
    "mAP50": "mAP@50",
    "mAP50_95": "mAP@50-95",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_text(value) -> str:
    """Convert value to clean string. Return empty string if NaN/None."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "null") else text


def parse_metric(value, column_name: str, csv_row: int) -> float:
    """Parse percentage or float metric and normalize between 0.0 and 1.0."""
    raw = clean_text(value)
    if not raw:
        raise ValueError(f"CSV row {csv_row}: {column_name} is empty")

    is_percent = raw.endswith("%")
    number_text = raw[:-1].strip() if is_percent else raw

    try:
        number = float(number_text.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"CSV row {csv_row}: invalid {column_name}={raw!r}") from exc

    if is_percent:
        number /= 100.0

    if not math.isfinite(number):
        raise ValueError(f"CSV row {csv_row}: {column_name} is not finite")

    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"CSV row {csv_row}: {column_name}={raw!r} must be between 0 and 1"
        )

    return number


def parse_image_count(value, csv_row: int) -> int:
    """Parse image count to integer."""
    raw = clean_text(value)
    if not raw:
        raise ValueError(f"CSV row {csv_row}: image is empty")

    try:
        number = float(raw.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"CSV row {csv_row}: invalid image={raw!r}") from exc

    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"CSV row {csv_row}: invalid image count={raw!r}")

    return int(number)

# ============================================================================
# PROCESS DATA
# ============================================================================

def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read CSV, strip whitespace from columns, drop empty rows, and ffill dates."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df.columns = df.columns.str.strip()

    # Forward fill missing dates for grouped rows
    if "date" in df.columns:
        df["date"] = df["date"].replace(r"^\s*$", None, regex=True).ffill()

    # Drop completely empty rows
    empty_mask = df.apply(
        lambda row: all(clean_text(v) == "" for v in row), axis=1
    )
    cleaned_df = df.loc[~empty_mask].copy()
    cleaned_df.reset_index(drop=False, inplace=True)  # Keep original row numbers
    return cleaned_df


def validate_all_rows(df: pd.DataFrame) -> list[dict]:
    """Validate and normalize rows containing valid training runs."""
    validated_rows = []

    for _, row in df.iterrows():
        csv_row = int(row["index"]) + 2  # Convert index back to original CSV line number

        date = clean_text(row.get("date"))
        dataset = clean_text(row.get("name-dataset"))
        image_raw = clean_text(row.get("image"))

        # Skip non-run / partial header rows safely
        if not dataset or not image_raw or clean_text(row.get("mAP@50")) == "":
            continue

        try:
            image_count = parse_image_count(image_raw, csv_row)
            metrics = {
                mlflow_key: parse_metric(row[csv_col], csv_col, csv_row)
                for mlflow_key, csv_col in METRIC_MAPPING.items()
            }
        except ValueError as e:
            print(f"[SKIP ROW {csv_row}] {e}")
            continue

        validated_rows.append({
            "csv_row": csv_row,
            "date": date,
            "dataset": dataset,
            "image_count": image_count,
            "metrics": metrics,
        })

    return validated_rows

# ============================================================================
# MLFLOW IMPORT
# ============================================================================

def import_to_mlflow(csv_path: Path, experiment_name: str, tracking_uri: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print("=" * 80)
    print("belt-ai | Importing YOLO Results to MLflow")
    print("=" * 80)

    df = prepare_dataframe(csv_path)
    rows = validate_all_rows(df)

    if not rows:
        print("No valid data rows found to import.")
        return

    print(f"Validated {len(rows)} real run records for import.\n")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    imported = 0
    for row in rows:
        run_name = f"{row['dataset']} | {row['date']} | CSV row {row['csv_row']}"

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
            mlflow.log_param("dataset", row["dataset"])
            mlflow.log_param("image_count", row["image_count"])
            mlflow.log_metrics(row["metrics"])
            mlflow.set_tags({
                "project": "belt-ai",
                "source": "historical-csv",
                "source_file": csv_path.name,
                "csv_row": str(row["csv_row"]),
                "training_date": row["date"],
                "dataset_name": row["dataset"],
                "model_family": "YOLO",
            })

        imported += 1
        m = row["metrics"]
        print(
            f"[OK] Row {row['csv_row']:03d} | {row['dataset'][:30]:<30} | "
            f"Images: {row['image_count']:<5} | mAP50: {m['mAP50']:.3f}"
        )

    print("\n" + "=" * 80)
    print(f"IMPORT COMPLETE: {imported} MLflow Runs created.")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import YOLO results into MLflow.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV file path")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=TRACKING_URI, help="MLflow Tracking URI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import_to_mlflow(args.csv.resolve(), args.experiment, args.tracking_uri)
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())