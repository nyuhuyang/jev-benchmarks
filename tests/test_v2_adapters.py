from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from jev_benchmarks.adapters.jev_openrouter import Budget, JevOpenRouterBackend
from jev_benchmarks.adapters.laya import LayaBackend, laya_head_fits, laya_request_tokens
from jev_benchmarks.adapters.qwen_logit import QwenLogitBackend, softmax_scores, two_digit_joint
from jev_benchmarks.models import Example


def example(kind: str = "choice", labels: tuple[str, ...] = ("alpha", "beta")) -> Example:
    return Example(
        "dataset",
        "task",
        "id",
        "synthetic text",
        "hash",
        labels,
        1,
        question_type=kind,
        instructions="Dataset-specific question?",
    )


def test_openrouter_request_shapes_ids_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    calls = []
    logged = []

    def transport(body, key, timeout):
        assert key == "fake-key" and timeout == 30
        request = json.loads(body)
        calls.append(request)
        kind = request["questions"]["label"]["type"]
        if kind == "choice":
            answer = {
                "type": "choice",
                "choice": "label_001",
                "probabilities": {"label_000": 0.1, "label_001": 0.9},
                "confidence": 0.8,
            }
        elif kind == "noul":
            answer = {"type": "noul", "noul": 0.9}
        else:
            answer = {
                "type": "score",
                "score": 1,
                "probabilities": {"0": 0.1, "1": 0.9},
                "confidence": 0.8,
            }
        return (
            200,
            {},
            {
                "model": "snapshot-1",
                "usage": {"cost": 0.001, "input_tokens": 20},
                "answers": {"label": answer},
            },
        )

    backend = JevOpenRouterBackend(
        "typesafe/jev-1.13",
        {"choice": "Choose", "noul": "Bool", "score": "Rate"},
        transport=transport,
        log_attempt=logged.append,
    )
    original = example()
    assert backend.predict("run", original).probabilities == (0.1, 0.9)
    assert backend.predict("run", original).cost_usd == 0.001
    assert calls[-1]["questions"]["label"]["instructions"] == "Dataset-specific question?"
    permuted = replace(original, option_order=(1, 0))
    backend.predict("run", permuted)
    assert list(calls[-1]["questions"]["label"]["criteria"]) == ["label_001", "label_000"]
    assert calls[-1]["questions"]["label"]["criteria"]["label_001"] == "beta"
    assert backend.predict("run", example("noul")).probabilities == pytest.approx((0.1, 0.9))
    assert calls[-1]["questions"]["label"]["criteria"] == {"true": "beta", "false": "alpha"}
    assert backend.predict("run", example("score")).probabilities == (0.1, 0.9)
    assert calls[-1]["questions"]["label"]["criteria"] == ["alpha", "beta"]
    assert len(logged) == 5
    assert backend.close() is None


def test_openrouter_retry_retry_after_timeout_and_cost_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    statuses = [429, 200]
    sleeps = []
    logs = []

    def transport(body, key, timeout):
        status = statuses.pop(0)
        if status == 429:
            return status, {"retry-after": "2"}, {"error": {"message": "limited"}}
        return (
            status,
            {},
            {
                "model": "snapshot",
                "usage": {"cost": 0.001},
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": "label_001",
                        "probabilities": {"label_000": 0.2, "label_001": 0.8},
                    }
                },
            },
        )

    backend = JevOpenRouterBackend(
        "m", {"choice": "q"}, transport=transport, sleep=sleeps.append, log_attempt=logs.append
    )
    assert backend.predict("run", example()).model_resolved == "snapshot"
    assert sleeps == [2.0]
    assert [row["status"] for row in logs] == [429, 200]
    assert backend.budget.settled == pytest.approx(0.001)
    assert backend.budget.reserved > 0 and not backend.budget.paused

    def missing(body, key, timeout):
        return 200, {}, {"model": "s", "usage": {}, "answers": {}}

    halted = JevOpenRouterBackend("m", {"choice": "q"}, transport=missing)
    with pytest.raises(RuntimeError, match=r"missing a valid usage\.cost"):
        halted.predict("run", example())
    assert halted.budget.paused and halted.budget.reserved > 0
    with pytest.raises(RuntimeError, match="paused"):
        halted.predict("run", example())

    def timeout(body, key, seconds):
        raise TimeoutError("late")

    timed = JevOpenRouterBackend("m", {"choice": "q"}, transport=timeout)
    with pytest.raises(TimeoutError):
        timed.predict("run", example())
    assert timed.budget.paused

    budget = Budget(maximum=0.000001, minimum=0.00001)
    guarded = JevOpenRouterBackend(
        "m",
        {"choice": "q"},
        budget=budget,
        transport=lambda *args: pytest.fail("must not dispatch"),
    )
    with pytest.raises(RuntimeError, match="budget exceeded"):
        guarded.predict("run", example())


def test_openrouter_http_400_no_retry_and_latency_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    bodies = []

    def transport(body, key, timeout):
        bodies.append(json.loads(body))
        if len(bodies) == 1:
            return 400, {}, {"error": {"message": "bad request"}}
        return (
            200,
            {},
            {
                "model": "s",
                "usage": {"cost": 0.001},
                "answers": {
                    "label_0": {
                        "type": "choice",
                        "choice": "label_001",
                        "probabilities": {"label_000": 0.2, "label_001": 0.8},
                    }
                },
            },
        )

    backend = JevOpenRouterBackend(
        "m",
        {"choice": "q"},
        transport=transport,
        latency_paraphrases=[f"q{index}" for index in range(10)],
    )
    with pytest.raises(RuntimeError, match="400"):
        backend.predict("run", example())
    assert not backend.budget.paused
    assert backend.predict("run", replace(example(), permutation_id="latency-10")).probabilities
    assert len(bodies[-1]["questions"]) == 10


def test_openrouter_500_exhausts_without_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    calls = []
    backend = JevOpenRouterBackend(
        "m",
        {},
        attempts=3,
        transport=lambda *args: (500, {}, {"error": {"message": "temporary"}}),
        sleep=lambda seconds: None,
        log_attempt=calls.append,
    )
    with pytest.raises(RuntimeError, match="after retries"):
        backend.predict("run", example())
    assert len(calls) == 3 and backend.budget.reserved > 0
    assert not backend.budget.paused


def test_openrouter_literal_live_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    payload = json.loads(
        '{"model":"typesafe/jev-1.13-20260917","answers":{'
        '"team":{"type":"choice","choice":"technical","probabilities":'
        '{"sales":0,"billing":0.22,"technical":0.78},"confidence":0.67},'
        '"frustration":{"type":"score","score":1,"legend":'
        '{"0":"calm","1":"frustrated","2":"very angry"},'
        '"probabilities":{"0":0.01,"1":0.99,"2":0},"confidence":0.99},'
        '"is_urgent":{"type":"noul","noul":0.98}},"usage":'
        '{"input_tokens":422,"output_tokens":73,"cost":0.000017724},'
        '"id":"gen-dec-…","provider":"TypeSafe"}'
    )
    backend = JevOpenRouterBackend(
        "m",
        {},
        transport=lambda *args: (
            200,
            {},
            {**payload, "answers": {"label": payload["answers"]["frustration"]}},
        ),
    )
    score = backend.predict("run", example("score", ("calm", "frustrated", "very angry")))
    assert score.probabilities == (0.01, 0.99, 0.0)
    assert score.confidence_reported == 0.99
    backend.transport = lambda *args: (
        200,
        {},
        {**payload, "answers": {"label": payload["answers"]["is_urgent"]}},
    )
    assert backend.predict("run", example("noul")).probabilities == pytest.approx((0.02, 0.98))


class FakeTokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class FakeAgent:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.tok = FakeTokenizer()
        self.device = "cpu"
        self.cfg = {"max_len": 512}
        self._forward = lambda *args, **kwargs: None

    def _to_internal(self, question):
        return {
            "t": question["type"],
            "ins": question["instructions"],
            "crit": question["criteria"],
        }

    def system_one(self, state, questions):
        # Laya's DecisionModel.forward takes five tensors.
        self._forward(None, None, None, None, qtype=None)
        return {"answers": {key: self.result for key in questions}}


@pytest.fixture
def fake_laya_common(monkeypatch: pytest.MonkeyPatch):
    common = SimpleNamespace(
        render_options=lambda q: (
            list(q["crit"].values()) if isinstance(q["crit"], dict) else q["crit"]
        ),
        serialize_state=lambda state: json.dumps(state),
    )
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(common=common))
    monkeypatch.setitem(sys.modules, "laya.common", common)


def test_laya_shapes_truncation_and_timing(fake_laya_common) -> None:
    questions = {"choice": "Choose", "noul": "True?", "score": "Rate"}
    backend = LayaBackend(
        "laya",
        "rev",
        "local",
        "english",
        {"dataset": 192},
        questions,
        agent=FakeAgent({"probabilities": {"label_000": 0.1, "label_001": 0.9}}),
    )
    choice = backend.predict("run", example())
    assert choice.probabilities == (0.1, 0.9)
    assert choice.resolved_checkpoint == "english"
    assert choice.model_latency_seconds is not None
    assert backend.agent._forward is not None
    assert backend.close() is None
    backend.agent.result = {"noul": 0.8}
    assert backend.predict("run", example("noul")).probabilities == pytest.approx((0.2, 0.8))
    backend.agent.result = {"score": 1.5}
    score = backend.predict("run", example("score", ("one", "two", "three")))
    assert score.probabilities == () and score.expected_score == 2.5
    backend.agent.result = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}
    assert backend.predict(
        "run", example("score", ("one", "two", "three"))
    ).expected_score == pytest.approx(2.6)
    with pytest.raises(ValueError, match="48-token"):
        backend.predict("run", example(labels=("a " * 60, "b")))
    backend.agent.cfg["max_len"] = 3
    with pytest.raises(ValueError, match="truncate text"):
        backend.predict("run", example())
    backend.head_max_len["dataset"] = 191
    with pytest.raises(ValueError, match="below the shipped"):
        backend.predict("run", example())


def test_laya_head_fits_matches_build_sequence() -> None:
    common = pytest.importorskip("laya.common")
    agent = pytest.importorskip("laya").Agent

    class Tok:
        mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 1, 2, 3

        def __call__(self, text, **kwargs):
            return {"input_ids": [10 + ord(char) for char in text]}

    question = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"a": "short", "b": "a somewhat longer option"},
    }
    sizes, instruction, _, _ = laya_request_tokens(question, "x", Tok(), agent._to_internal)
    internal = agent._to_internal(question)
    full, _ = common.build_sequence(Tok(), {"text": "x"}, internal, 10**6, 10**6)
    for head in range(1, sum(sizes) + instruction + 40):
        built, _ = common.build_sequence(Tok(), {"text": "x"}, internal, 10**6, head)
        assert laya_head_fits(sizes, instruction, head) == (built == full), head


def test_laya_ten_questions(fake_laya_common) -> None:
    backend = LayaBackend(
        "laya",
        "rev",
        "local",
        "english",
        {"dataset": 192},
        {"choice": "Choose"},
        agent=FakeAgent({"probabilities": {"label_000": 0.2, "label_001": 0.8}}),
        latency_paraphrases=[f"p{index}" for index in range(10)],
    )
    assert backend.predict("run", replace(example(), permutation_id="latency-10")).probabilities
    backend.agent.cfg["head_max_len"] = 7
    default = backend.predict("run", replace(example(), permutation_id="head-default"))
    assert default.head_max_len_used == 192


def test_qwen_label_modes_and_two_digit_math() -> None:
    calls = []

    def scorer(prompt, ids, mode):
        calls.append((prompt, ids, mode))
        return (0.0, 2.0)

    backend = QwenLogitBackend(
        "qwen",
        "rev",
        "local",
        "system",
        "{text}\n{options}",
        {"dataset": "letter"},
        scorer=scorer,
        renderer=lambda system, user: f"{system}:{user}:",
    )
    row = replace(example(), option_order=(1, 0))
    stable = backend.predict("run", row)
    assert "B) beta\nA) alpha" in calls[-1][0]
    assert stable.predicted_index == 1
    positional = backend.predict("run", replace(row, letter_mode="positional"))
    assert "A) beta\nB) alpha" in calls[-1][0]
    assert positional.predicted_index == 0
    assert backend.close() is None
    assert softmax_scores([0, 0]) == (0.5, 0.5)
    joint = two_digit_joint(
        {1: -0.1, 2: -0.2}, {1: {1: -0.3, 2: -0.5}, 2: {1: -0.1}}, [(1, 1), (1, 2), (2, 1)]
    )
    assert joint == pytest.approx(softmax_scores([-0.4, -0.6, -0.3]))
    with pytest.raises(ValueError, match="batched scorer"):
        backend.predict("run", replace(example(), permutation_id="latency-10"))


def test_qwen_score_and_large_ids() -> None:
    labels = tuple(f"intent {index}" for index in range(72))
    observed = []

    def scorer(prompt, ids, mode):
        observed.append((ids, mode))
        return list(range(len(ids)))

    backend = QwenLogitBackend(
        "q",
        "r",
        "p",
        "s",
        "{text}\n{options}",
        {"dataset": "two_digit_joint"},
        scorer=scorer,
        renderer=lambda *_: "prefix ",
    )
    output = backend.predict("e", example("choice", labels))
    assert observed[-1][0][0] == "01" and observed[-1][0][-1] == "72"
    assert output.predicted_index == 71
    backend.score_modes["dataset"] = "full_sequence"
    score = backend.predict("e", example("score", ("1", "2")))
    assert score.expected_score is not None
