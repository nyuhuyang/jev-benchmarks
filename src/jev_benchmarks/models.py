from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Example:
    dataset: str
    task: str
    example_id: str
    text: str
    text_sha256: str
    labels: tuple[str, ...]
    target_index: int
    question_type: str = "choice"
    split: str = "test"
    permutation_id: str = "identity"
    locale: str | None = None
    letter_mode: str = "stable"
    option_order: tuple[int, ...] = ()
    instructions: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["labels"] = list(self.labels)
        row["option_order"] = list(self.option_order)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Example:
        return cls(
            **{
                **row,
                "labels": tuple(row["labels"]),
                "option_order": tuple(row.get("option_order", ())),
            }
        )


@dataclass(frozen=True)
class Prediction:
    experiment_id: str
    backend: str
    model_requested: str
    model_resolved: str
    dataset: str
    example_id: str
    target_index: int
    predicted_index: int
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    latency_seconds: float
    input_tokens: int | None = None
    probability_sum_raw: float | None = None
    error: str | None = None
    question_type: str = "choice"
    split: str = "test"
    condition: str = "A_raw"
    repeat_index: int = 0
    permutation_id: str = "identity"
    letter_mode: str = "stable"
    raw_probabilities: tuple[float, ...] = ()
    expected_score: float | None = None
    model_latency_seconds: float | None = None
    device: str | None = None
    cost_usd: float | None = None
    resolved_checkpoint: str | None = None
    attempt_id: int | None = None
    confidence_reported: float | None = None
    head_max_len_used: int | None = None
    dispatch_attempts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["labels"] = list(self.labels)
        row["probabilities"] = list(self.probabilities)
        row["raw_probabilities"] = list(self.raw_probabilities)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Prediction:
        return cls(
            **{
                **row,
                "labels": tuple(row["labels"]),
                "probabilities": tuple(row["probabilities"]),
                "raw_probabilities": tuple(row.get("raw_probabilities", ())),
            }
        )
