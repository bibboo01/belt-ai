from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.model_manager as model_manager_module
from app.model_manager import EXPECTED_CLASSES, ModelManager


# ============================================================================
# Fake YOLO Model
# ============================================================================


def create_fake_model(
    names=None,
    task="detect",
):
    """
    Create a minimal YOLO-like object.
    """

    if names is None:
        names = {
            0: "dogear-belt",
            1: "good-belt",
            2: "splice-belt",
        }

    return SimpleNamespace(
        names=names,
        task=task,
    )


class FakeYOLO:
    """
    Fake replacement for ultralytics.YOLO.
    """

    def __init__(
        self,
        model_path,
        task=None,
        verbose=False,
    ):
        self.model_path = model_path
        self.task = task
        self.verbose = verbose

        self.names = {
            0: "dogear-belt",
            1: "good-belt",
            2: "splice-belt",
        }


# ============================================================================
# Helpers
# ============================================================================


def create_manager(
    monkeypatch,
    tmp_path,
):
    """
    Create an isolated ModelManager using a temporary production directory.
    """

    production_dir = (
        tmp_path
        / "models"
        / "production"
    )

    production_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monkeypatch.setattr(
        model_manager_module,
        "PRODUCTION_DIR",
        production_dir,
    )

    monkeypatch.setattr(
        model_manager_module,
        "MODEL_POINTER",
        production_dir / "model.json",
    )

    manager = ModelManager()

    return manager, production_dir


def write_model_registry(
    production_dir: Path,
    model_file: str = "best.pt",
    model_id: int = 2,
    model_name: str = "YOLO11",
    model_version: str = "ODBv54",
):
    """
    Create a test model.json.
    """

    model_json = {
        "model_id": model_id,
        "model_name": model_name,
        "model_version": model_version,
        "model_file": model_file,
    }

    (
        production_dir / "model.json"
    ).write_text(
        json.dumps(model_json),
        encoding="utf-8",
    )


def create_fake_model_file(
    production_dir: Path,
    model_file: str = "best.pt",
):
    """
    Create a placeholder model file.

    The contents do not matter because YOLO is mocked.
    """

    model_path = production_dir / model_file

    model_path.write_bytes(
        b"fake-model",
    )

    return model_path


# ============================================================================
# Constants
# ============================================================================


def test_expected_classes():
    """
    Verify the expected production class mapping.
    """

    assert EXPECTED_CLASSES == {
        0: "dogear-belt",
        1: "good-belt",
        2: "splice-belt",
    }


# ============================================================================
# Initial State
# ============================================================================


def test_model_manager_initial_state(
    monkeypatch,
    tmp_path,
):
    """
    ModelManager should start in a non-ready state.
    """

    manager, _ = create_manager(
        monkeypatch,
        tmp_path,
    )

    assert manager.is_ready() is False


# ============================================================================
# Successful Model Loading
# ============================================================================


def test_load_model_success(
    monkeypatch,
    tmp_path,
):
    """
    Verify successful production model loading.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    create_fake_model_file(
        production_dir,
    )

    write_model_registry(
        production_dir,
    )

    fake_model = create_fake_model()

    def fake_yolo(
        model_path,
        task=None,
        verbose=False,
    ):
        return fake_model

    monkeypatch.setattr(
        model_manager_module,
        "YOLO",
        fake_yolo,
    )

    result = manager.load_model()

    assert result is True
    assert manager.is_ready() is True

    info = manager.get_info()

    assert info["model_id"] == 2
    assert info["model_name"] == "YOLO11"
    assert info["model_version"] == "ODBv54"


# ============================================================================
# model.json Validation
# ============================================================================


def test_load_model_fails_when_model_json_missing(
    monkeypatch,
    tmp_path,
):
    """
    Missing model.json must fail loading.
    """

    manager, _ = create_manager(
        monkeypatch,
        tmp_path,
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


def test_load_model_fails_when_model_json_is_invalid(
    monkeypatch,
    tmp_path,
):
    """
    Invalid JSON must fail loading.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    (
        production_dir / "model.json"
    ).write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


# ============================================================================
# Model File Validation
# ============================================================================


def test_load_model_fails_when_model_file_missing(
    monkeypatch,
    tmp_path,
):
    """
    Missing model file must fail loading.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    write_model_registry(
        production_dir,
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


def test_load_model_fails_when_model_file_is_invalid(
    monkeypatch,
    tmp_path,
):
    """
    YOLO loading failure must fail the ModelManager load.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    create_fake_model_file(
        production_dir,
    )

    write_model_registry(
        production_dir,
    )

    def failing_yolo(
        model_path,
        task=None,
        verbose=False,
    ):
        raise RuntimeError(
            "Invalid YOLO model"
        )

    monkeypatch.setattr(
        model_manager_module,
        "YOLO",
        failing_yolo,
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


# ============================================================================
# Class Mapping Validation
# ============================================================================


def test_load_model_fails_when_class_mapping_is_wrong(
    monkeypatch,
    tmp_path,
):
    """
    Model class mapping must match EXPECTED_CLASSES.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    create_fake_model_file(
        production_dir,
    )

    write_model_registry(
        production_dir,
    )

    wrong_model = create_fake_model(
        names={
            0: "dogear-belt",
            1: "wrong-class",
            2: "splice-belt",
        }
    )

    monkeypatch.setattr(
        model_manager_module,
        "YOLO",
        lambda *args, **kwargs: wrong_model,
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


def test_load_model_fails_when_class_is_missing(
    monkeypatch,
    tmp_path,
):
    """
    Missing class mapping must fail loading.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    create_fake_model_file(
        production_dir,
    )

    write_model_registry(
        production_dir,
    )

    incomplete_model = create_fake_model(
        names={
            0: "dogear-belt",
            1: "good-belt",
        }
    )

    monkeypatch.setattr(
        model_manager_module,
        "YOLO",
        lambda *args, **kwargs: incomplete_model,
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


# ============================================================================
# Task Validation
# ============================================================================


def test_load_model_fails_when_task_is_not_detect(
    monkeypatch,
    tmp_path,
):
    """
    Production model must be a detection model.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    create_fake_model_file(
        production_dir,
    )

    write_model_registry(
        production_dir,
    )

    wrong_task_model = create_fake_model(
        task="classify"
    )

    monkeypatch.setattr(
        model_manager_module,
        "YOLO",
        lambda *args, **kwargs: wrong_task_model,
    )

    result = manager.load_model()

    assert result is False
    assert manager.is_ready() is False


# ============================================================================
# get_model()
# ============================================================================


def test_get_model_returns_loaded_model(
    monkeypatch,
    tmp_path,
):
    """
    get_model() should return the loaded production model.
    """

    manager, production_dir = create_manager(
        monkeypatch,
        tmp_path,
    )

    create_fake_model_file(
        production_dir,
    )

    write_model_registry(
        production_dir,
    )

    fake_model = create_fake_model()

    monkeypatch.setattr(
        model_manager_module,
        "YOLO",
        lambda *args, **kwargs: fake_model,
    )

    model = manager.get_model()

    assert model is fake_model
    assert manager.is_ready() is True


# ============================================================================
# get_info()
# ============================================================================


def test_get_info_before_model_is_loaded(
    monkeypatch,
    tmp_path,
):
    """
    get_info() should return a safe representation before loading.
    """

    manager, _ = create_manager(
        monkeypatch,
        tmp_path,
    )

    info = manager.get_info()

    assert info is not None