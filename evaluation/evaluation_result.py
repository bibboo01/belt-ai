from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:

    model_name: str
    model_version: str

    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None

    status: str = "PENDING"

    # REAL / TEST_ONLY
    evaluation_type: str = "REAL"

    notes: str = ""


    def is_ready(self) -> bool:
        """Check whether evaluation has complete metrics."""

        return all([
            self.precision is not None,
            self.recall is not None,
            self.map50 is not None,
            self.map50_95 is not None,
        ])


    def to_dict(self):

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,

            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,

            "status": self.status,

            "evaluation_type": self.evaluation_type,

            "notes": self.notes,
        }


if __name__ == "__main__":

    result = EvaluationResult(
        model_name="YOLO11",
        model_version="ODBv54",
    )

    print("=" * 70)
    print("BELT AI - EVALUATION RESULT")
    print("=" * 70)

    print(result.to_dict())

    print()
    print(f"Evaluation ready : {result.is_ready()}")
    print(f"Evaluation type  : {result.evaluation_type}")

    print("=" * 70)