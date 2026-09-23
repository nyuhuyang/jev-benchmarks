from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models import Example, Prediction


class GLiNERV2Backend:
    name = "gliner"

    def __init__(
        self, model_id: str, revision: str, device: str = "cpu", *, upstream: Any = None
    ) -> None:
        if upstream is None:
            from .gliner import GLiNERBackend

            upstream = GLiNERBackend(model_id, revision, device=device)
        self.upstream = upstream

    def warmup(self, example: Example) -> None:
        self.upstream.warmup(example)

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        prediction = self.upstream.predict(experiment_id, example)
        expected = (
            sum((index + 1) * value for index, value in enumerate(prediction.probabilities))
            if example.question_type == "score"
            else None
        )
        return replace(
            prediction,
            question_type=example.question_type,
            split=example.split,
            permutation_id=example.permutation_id,
            letter_mode=example.letter_mode,
            raw_probabilities=prediction.probabilities,
            expected_score=expected,
            device="cpu",
        )

    def close(self) -> None:
        self.upstream.close()
