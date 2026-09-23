from __future__ import annotations

import contextlib
import json
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from jev_benchmarks import worker
from jev_benchmarks.adapters.qwen_logit import _HFScorer
from jev_benchmarks.models import Example, Prediction


class FakeTorch:
    def tensor(self, values, device):
        return np.asarray(values, dtype=int)

    def no_grad(self):
        return contextlib.nullcontext()

    def ones(self, shape, device):
        return np.ones(shape, dtype=int)

    def log_softmax(self, values, dim):
        peak = np.max(values, axis=dim, keepdims=True)
        return values - (peak + np.log(np.exp(values - peak).sum(axis=dim, keepdims=True)))


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        batch, length = input_ids.shape
        logits = np.zeros((batch, length, 128), dtype=float)
        for batch_index in range(batch):
            for position in range(length):
                prior = input_ids[batch_index, position]
                if prior == ord(" "):
                    logits[batch_index, position, ord("A")] = 2
                    logits[batch_index, position, ord("B")] = 1
                    logits[batch_index, position, ord("1")] = 3
                    logits[batch_index, position, ord("2")] = 2
                if prior == ord("1"):
                    logits[batch_index, position, ord("2")] = 4
                if prior == ord("2"):
                    logits[batch_index, position, ord("1")] = 2
        return SimpleNamespace(
            logits=logits.view(FakeArray),
            past_key_values=SimpleNamespace(batch_repeat_interleave=lambda n: None),
        )


class FakeArray(np.ndarray):
    def float(self):
        return self


def scorer() -> _HFScorer:
    value = object.__new__(_HFScorer)
    cast(Any, value).torch = FakeTorch()
    value.tokenizer = FakeTokenizer()
    value.model = FakeModel()
    value.device = "cpu"
    return value


def test_qwen_exact_suffix_scoring_modes_with_fake_logits() -> None:
    backend = scorer()
    letters = backend("P ", ["A", "B"], "letter")
    assert letters[0] > letters[1]
    assert backend.batch("P ", ["A", "B"], "letter", 10) == pytest.approx(letters)
    joint = backend("P ", ["12", "21"], "two_digit_joint")
    assert joint[0] > joint[1]
    assert backend.model.calls[-2]["use_cache"] is True
    assert backend.model.calls[-1]["past_key_values"] is not None
    full = backend("P ", ["12", "21"], "full_sequence")
    assert full[0] - full[1] == pytest.approx(joint[0] - joint[1])
    with pytest.raises(ValueError, match="one suffix token"):
        backend("P ", ["12"], "letter")
    with pytest.raises(ValueError, match="two suffix tokens"):
        backend("P ", ["1"], "two_digit_joint")
    with pytest.raises(ValueError, match="letter suffix"):
        backend.batch("P ", ["12"], "letter", 2)
    assert backend.batch("P ", ["12", "21"], "full_sequence", 2) == pytest.approx(full)


def test_worker_json_protocol_with_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = Example("d", "t", "id", "synthetic", "h", ("a", "b"), 0)

    class Backend:
        closed = False

        def predict(self, experiment_id, example):
            if experiment_id == "bad":
                raise ValueError("synthetic error")
            return Prediction(
                experiment_id,
                "fake",
                "m",
                "s",
                example.dataset,
                example.example_id,
                example.target_index,
                0,
                example.labels,
                (0.8, 0.2),
                0.1,
            )

        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr(worker, "load_config", lambda path: object())
    monkeypatch.setattr(worker, "_make_v2_backend", lambda *args: backend)
    lines = [
        json.dumps({"experiment_id": name, "example": row.to_dict()}) for name in ("good", "bad")
    ]
    monkeypatch.setattr(sys, "stdin", iter(lines))
    monkeypatch.setattr(sys, "argv", ["worker", "config", "backend", "attempt"])
    worker.main()
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["prediction"]["backend"] == "fake"
    assert "synthetic error" in output[1]["error"]
    assert backend.closed
