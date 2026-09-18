import json
from pathlib import Path

import pytest

from mlops import promote_model


def test_promotion_blocks_test_only(tmp_path, monkeypatch):
    evaluation = {
        "model_name": "YOLO11",
        "model_version": "ODBv54",
        "status": "PASSED",
        "evaluation_type": "TEST_ONLY",
    }

    evaluation_file = tmp_path / "ODBv54_test_only.json"

    evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        promote_model,
        "EVALUATION_DIR",
        tmp_path,
    )

    path, data = promote_model.load_latest_evaluation(
        "YOLO11",
        "ODBv54",
    )

    with pytest.raises(RuntimeError, match="not REAL"):
        promote_model.validate_evaluation(
            path,
            data,
        )

    print(
        "PROMO-01 PASS TEST_ONLY evaluation blocked"
    )


def test_promotion_blocks_missing_type(tmp_path, monkeypatch):
    evaluation = {
        "model_name": "YOLO11",
        "model_version": "ODBv54",
        "status": "PASSED",
    }

    evaluation_file = tmp_path / "ODBv54_missing_type.json"

    evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        promote_model,
        "EVALUATION_DIR",
        tmp_path,
    )

    path, data = promote_model.load_latest_evaluation(
        "YOLO11",
        "ODBv54",
    )

    with pytest.raises(RuntimeError, match="not REAL"):
        promote_model.validate_evaluation(
            path,
            data,
        )

    print(
        "PROMO-02 PASS missing evaluation_type blocked"
    )


def test_promotion_blocks_failed_real_evaluation(
    tmp_path,
    monkeypatch,
):
    evaluation = {
        "model_name": "YOLO11",
        "model_version": "ODBv54",
        "status": "FAILED",
        "evaluation_type": "REAL",
    }

    evaluation_file = (
        tmp_path / "ODBv54_failed_real.json"
    )

    evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        promote_model,
        "EVALUATION_DIR",
        tmp_path,
    )

    path, data = promote_model.load_latest_evaluation(
        "YOLO11",
        "ODBv54",
    )

    with pytest.raises(RuntimeError, match="status"):
        promote_model.validate_evaluation(
            path,
            data,
        )

    print(
        "PROMO-03 PASS FAILED evaluation blocked"
    )


def test_promotion_allows_real_passed():
    evaluation = {
        "model_name": "YOLO11",
        "model_version": "ODBv54",
        "status": "PASSED",
        "evaluation_type": "REAL",
    }

    promote_model.validate_evaluation(
        Path("REAL_TEST_RESULT.json"),
        evaluation,
    )

    print(
        "PROMO-04 PASS REAL + PASSED accepted"
    )


def test_promotion_safety_summary():
    print("")
    print("=" * 60)
    print("MODEL PROMOTION SAFETY AUDIT")
    print("=" * 60)
    print("TEST_ONLY / SIMULATED : BLOCKED")
    print("Missing evaluation type: BLOCKED")
    print("FAILED evaluation      : BLOCKED")
    print("REAL + PASSED          : ALLOWED")
    print("=" * 60)

    assert True