from pathlib import Path
import json
from datetime import datetime

from evaluation.evaluation_result import EvaluationResult


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"


MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_MAP50 = 0.90
MIN_MAP50_95 = 0.70


def evaluate_gate(result: EvaluationResult) -> EvaluationResult:

    if not result.is_ready():
        result.status = "PENDING"
        result.notes = "Evaluation metrics are incomplete."
        return result

    failed = []

    if result.precision < MIN_PRECISION:
        failed.append(
            f"Precision {result.precision:.4f} < {MIN_PRECISION:.2f}"
        )

    if result.recall < MIN_RECALL:
        failed.append(
            f"Recall {result.recall:.4f} < {MIN_RECALL:.2f}"
        )

    if result.map50 < MIN_MAP50:
        failed.append(
            f"mAP50 {result.map50:.4f} < {MIN_MAP50:.2f}"
        )

    if result.map50_95 < MIN_MAP50_95:
        failed.append(
            f"mAP50-95 {result.map50_95:.4f} < {MIN_MAP50_95:.2f}"
        )

    if failed:
        result.status = "REJECTED"
        result.notes = "; ".join(failed)
    else:
        result.status = "PASSED"
        result.notes = "All evaluation gates passed."

    return result


def save_result(result: EvaluationResult):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now()

    data = result.to_dict()

    data["evaluated_at"] = timestamp.isoformat()

    filename = (
        f"{result.model_version}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    )

    output_file = RESULTS_DIR / filename

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    return output_file


if __name__ == "__main__":

    result = EvaluationResult(
        model_name="YOLO11",
        model_version="ODBv54",

        # TEST ONLY
        precision=0.95,
        recall=0.93,
        map50=0.96,
        map50_95=0.78,
    )

    result = evaluate_gate(result)

    output_file = save_result(result)

    print("=" * 70)
    print("BELT AI - EVALUATION GATE")
    print("=" * 70)

    print(f"Model   : {result.model_name}")
    print(f"Version : {result.model_version}")
    print(f"Status  : {result.status}")
    print(f"Notes   : {result.notes}")
    print()
    print(f"Result saved : {output_file}")

    print("=" * 70)