from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from app.model_manager import model_manager


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EXPECTED_CLASSES: Dict[int, str] = {
    0: "dogear-belt",
    1: "good-belt",
    2: "splice-belt",
}


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class InferenceResult:
    detections: List[Dict[str, Any]]
    overall: str
    good_count: int
    splice_count: int
    dogear_count: int
    inference_time_ms: float

    @property
    def ng_count(self) -> int:
        return self.splice_count + self.dogear_count

    @property
    def total_detections(self) -> int:
        return len(self.detections)


# -----------------------------------------------------------------------------
# Core Inference Function
# -----------------------------------------------------------------------------

def run_inference(
    frame: np.ndarray,
    model_manager=model_manager,
) -> InferenceResult:

    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. Validate input frame
    # -------------------------------------------------------------------------

    if not isinstance(frame, np.ndarray):
        raise TypeError(
            "Input frame must be a numpy.ndarray"
        )

    if frame.size == 0:
        raise ValueError(
            "Input frame is empty"
        )

    # -------------------------------------------------------------------------
    # 2. Get production model
    # -------------------------------------------------------------------------

    model = model_manager.get_model()

    if model is None:
        raise RuntimeError(
            "Production model is not available"
        )

    # -------------------------------------------------------------------------
    # 3. Execute YOLO inference
    # -------------------------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=model_manager.confidence,
        verbose=False,
    )

    if not results:
        raise RuntimeError(
            "YOLO returned no result"
        )

    result = results[0]

    detections: List[Dict[str, Any]] = []
    class_counter: Counter[int] = Counter()

    # -------------------------------------------------------------------------
    # 4. Extract detections
    # -------------------------------------------------------------------------

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        # Batch convert tensors for better performance.
        class_ids = (
            boxes.cls
            .int()
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )

        xyxy_coords = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        for (
            class_id,
            confidence,
            (x1, y1, x2, y2),
        ) in zip(
            class_ids,
            confidences,
            xyxy_coords,
        ):
            class_id = int(class_id)

            if class_id not in EXPECTED_CLASSES:
                raise RuntimeError(
                    f"Unexpected class_id from model: {class_id}"
                )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": EXPECTED_CLASSES[class_id],
                    "confidence": float(confidence),
                    "bbox": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ],
                }
            )

            class_counter[class_id] += 1

    # -------------------------------------------------------------------------
    # 5. Summarize class counts
    # -------------------------------------------------------------------------

    dogear_count = class_counter[0]
    good_count = class_counter[1]
    splice_count = class_counter[2]

    # -------------------------------------------------------------------------
    # 6. Determine overall inspection result
    #
    # Any dogear or splice defect = NG
    # No defect = GOOD
    # -------------------------------------------------------------------------

    overall = (
        "NG"
        if dogear_count > 0 or splice_count > 0
        else "GOOD"
    )

    # -------------------------------------------------------------------------
    # 7. Calculate inference time
    # -------------------------------------------------------------------------

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    # -------------------------------------------------------------------------
    # 8. Return result
    # -------------------------------------------------------------------------

    return InferenceResult(
        detections=detections,
        overall=overall,
        good_count=good_count,
        splice_count=splice_count,
        dogear_count=dogear_count,
        inference_time_ms=inference_time_ms,
    )
