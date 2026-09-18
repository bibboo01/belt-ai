from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.inference import InferenceResult, run_inference


# ============================================================================
# Fake YOLO Components
# ============================================================================


class FakeBoxes:
    """
    Minimal YOLO Boxes-like object used for unit testing.
    """

    def __init__(
        self,
        class_ids: list[int],
        confidences: list[float],
        xyxy: list[list[float]],
    ):
        self.cls = torch.tensor(
            class_ids,
            dtype=torch.float32,
        )

        self.conf = torch.tensor(
            confidences,
            dtype=torch.float32,
        )

        self.xyxy = torch.tensor(
            xyxy,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.cls)


class FakeResult:
    """
    Minimal YOLO Result-like object.
    """

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """
    Minimal YOLO model-like object.
    """

    def __init__(self, result):
        self.result = result
        self.last_source = None
        self.last_conf = None
        self.last_verbose = None

    def predict(
        self,
        source,
        conf,
        verbose,
    ):
        self.last_source = source
        self.last_conf = conf
        self.last_verbose = verbose

        return [self.result]


class FakeModelManager:
    """
    Minimal ModelManager-like object.
    """

    def __init__(
        self,
        model,
        confidence: float = 0.5,
    ):
        self.model = model
        self.confidence = confidence

    def get_model(self):
        return self.model


# ============================================================================
# Helper
# ============================================================================


def create_frame() -> np.ndarray:
    """
    Create a small valid BGR image for testing.
    """

    return np.zeros(
        (1080, 1920, 3),
        dtype=np.uint8,
    )


def create_manager(
    class_ids: list[int],
    confidences: list[float],
    xyxy: list[list[float]],
    confidence_threshold: float = 0.5,
):
    boxes = FakeBoxes(
        class_ids=class_ids,
        confidences=confidences,
        xyxy=xyxy,
    )

    result = FakeResult(
        boxes=boxes,
    )

    model = FakeModel(
        result=result,
    )

    manager = FakeModelManager(
        model=model,
        confidence=confidence_threshold,
    )

    return manager, model


# ============================================================================
# Tests
# ============================================================================


def test_run_inference_good_belt():
    """
    A good-belt detection must produce GOOD.
    """

    frame = create_frame()

    manager, model = create_manager(
        class_ids=[1],
        confidences=[0.92],
        xyxy=[
            [100.4, 200.6, 500.8, 800.2],
        ],
    )

    result = run_inference(
        frame=frame,
        model_manager=manager,
    )

    assert isinstance(result, InferenceResult)

    assert result.overall == "GOOD"

    assert result.good_count == 1
    assert result.splice_count == 0
    assert result.dogear_count == 0

    assert result.ng_count == 0
    assert result.total_detections == 1

    assert result.detections[0]["class_id"] == 1
    assert result.detections[0]["class_name"] == "good-belt"

    assert result.detections[0]["confidence"] == pytest.approx(
        0.92,
        abs=1e-5,
    )

    assert result.detections[0]["bbox"] == [
        100,
        201,
        501,
        800,
    ]

    assert result.inference_time_ms >= 0

    assert model.last_source is frame
    assert model.last_conf == pytest.approx(0.5)
    assert model.last_verbose is False


def test_run_inference_splice_is_ng():
    """
    A splice-belt detection must produce NG.
    """

    frame = create_frame()

    manager, _ = create_manager(
        class_ids=[2],
        confidences=[0.7443],
        xyxy=[
            [804.0, 472.0, 1003.0, 801.0],
        ],
    )

    result = run_inference(
        frame=frame,
        model_manager=manager,
    )

    assert result.overall == "NG"

    assert result.good_count == 0
    assert result.splice_count == 1
    assert result.dogear_count == 0

    assert result.ng_count == 1
    assert result.total_detections == 1

    assert result.detections[0]["class_name"] == "splice-belt"


def test_run_inference_dogear_is_ng():
    """
    A dogear-belt detection must produce NG.
    """

    frame = create_frame()

    manager, _ = create_manager(
        class_ids=[0],
        confidences=[0.88],
        xyxy=[
            [50.0, 60.0, 300.0, 400.0],
        ],
    )

    result = run_inference(
        frame=frame,
        model_manager=manager,
    )

    assert result.overall == "NG"

    assert result.good_count == 0
    assert result.splice_count == 0
    assert result.dogear_count == 1

    assert result.ng_count == 1
    assert result.total_detections == 1

    assert result.detections[0]["class_name"] == "dogear-belt"


def test_run_inference_multiple_classes():
    """
    Verify class counting when multiple detections exist.
    """

    frame = create_frame()

    manager, _ = create_manager(
        class_ids=[1, 1, 2, 0],
        confidences=[
            0.95,
            0.91,
            0.84,
            0.79,
        ],
        xyxy=[
            [100, 100, 300, 300],
            [400, 100, 700, 300],
            [800, 200, 1000, 700],
            [1100, 300, 1400, 800],
        ],
    )

    result = run_inference(
        frame=frame,
        model_manager=manager,
    )

    assert result.good_count == 2
    assert result.splice_count == 1
    assert result.dogear_count == 1

    assert result.ng_count == 2
    assert result.total_detections == 4

    assert result.overall == "NG"


def test_run_inference_no_detections():
    """
    No detections must produce GOOD according to current logic.
    """

    frame = create_frame()

    boxes = FakeBoxes(
        class_ids=[],
        confidences=[],
        xyxy=[],
    )

    result_object = FakeResult(
        boxes=boxes,
    )

    model = FakeModel(
        result=result_object,
    )

    manager = FakeModelManager(
        model=model,
        confidence=0.5,
    )

    result = run_inference(
        frame=frame,
        model_manager=manager,
    )

    assert result.overall == "GOOD"

    assert result.good_count == 0
    assert result.splice_count == 0
    assert result.dogear_count == 0

    assert result.ng_count == 0
    assert result.total_detections == 0

    assert result.detections == []


def test_run_inference_invalid_frame_type():
    """
    A non-numpy input must raise TypeError.
    """

    manager, _ = create_manager(
        class_ids=[],
        confidences=[],
        xyxy=[],
    )

    with pytest.raises(
        TypeError,
        match="Input frame must be a numpy.ndarray",
    ):
        run_inference(
            frame="invalid-frame",
            model_manager=manager,
        )


def test_run_inference_empty_frame():
    """
    An empty numpy array must raise ValueError.
    """

    manager, _ = create_manager(
        class_ids=[],
        confidences=[],
        xyxy=[],
    )

    empty_frame = np.array([])

    with pytest.raises(
        ValueError,
        match="Input frame is empty",
    ):
        run_inference(
            frame=empty_frame,
            model_manager=manager,
        )


def test_run_inference_model_unavailable():
    """
    A missing production model must raise RuntimeError.
    """

    manager = FakeModelManager(
        model=None,
        confidence=0.5,
    )

    frame = create_frame()

    with pytest.raises(
        RuntimeError,
        match="Production model is not available",
    ):
        run_inference(
            frame=frame,
            model_manager=manager,
        )


def test_run_inference_no_yolo_result():
    """
    YOLO returning no result must raise RuntimeError.
    """

    class NoResultModel:
        def predict(
            self,
            source,
            conf,
            verbose,
        ):
            return []

    manager = FakeModelManager(
        model=NoResultModel(),
        confidence=0.5,
    )

    frame = create_frame()

    with pytest.raises(
        RuntimeError,
        match="YOLO returned no result",
    ):
        run_inference(
            frame=frame,
            model_manager=manager,
        )


def test_run_inference_unexpected_class_id():
    """
    An unexpected class ID must raise RuntimeError.
    """

    frame = create_frame()

    manager, _ = create_manager(
        class_ids=[99],
        confidences=[0.90],
        xyxy=[
            [100, 100, 300, 300],
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected class_id from model: 99",
    ):
        run_inference(
            frame=frame,
            model_manager=manager,
        )