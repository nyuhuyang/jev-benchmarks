from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jev_benchmarks.adapters.jev_openrouter import (
    Budget,
    BudgetLedger,
    JevOpenRouterBackend,
    valid_cost,
)
from jev_benchmarks.models import Example

ROW = Example("d", "t", "id", "synthetic", "h", ("a", "b"), 1, instructions="Pick?")


def answer(cost: object) -> dict:
    return {
        "model": "snapshot",
        "usage": {"cost": cost},
        "answers": {"label": {"probabilities": {"label_000": 0.1, "label_001": 0.9}}},
    }


def backend(path: Path, transport, **kwargs) -> JevOpenRouterBackend:
    budget = Budget(maximum=2.0, ledger=BudgetLedger(path))
    return JevOpenRouterBackend(
        "m", {}, budget=budget, transport=transport, sleep=lambda _: None, **kwargs
    )


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")


def test_crash_between_reserve_and_response_counts_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "budget-ledger.jsonl"

    def crash(body, key, timeout):
        raise KeyboardInterrupt  # not an Exception: no closing record is written

    first = backend(path, crash)
    with pytest.raises(KeyboardInterrupt):
        first.predict("run", ROW)
    first.close()
    reserved = events(path)[0]
    assert [row["event"] for row in events(path)] == ["reserved"]
    restarted = Budget(maximum=2.0, ledger=BudgetLedger(path))
    assert restarted.settled == pytest.approx(reserved["amount"])


def test_retained_429_still_counts_and_settled_cost_replaces(tmp_path: Path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    responses = [(429, {}, {"error": "limited"}), (200, {}, answer(0.001))]
    first = backend(path, lambda *_: responses.pop(0))
    first.predict("run", ROW)
    first.close()
    rows = events(path)
    assert [row["event"] for row in rows] == ["reserved", "retained", "reserved", "settled"]
    assert len({row["txn"] for row in rows}) == 2
    restarted = Budget(maximum=2.0, ledger=BudgetLedger(path))
    assert restarted.settled == pytest.approx(rows[0]["amount"] + 0.001)


@pytest.mark.parametrize("cost", [None, "NaN", math.nan, -1, "0.1", True, math.inf])
def test_malformed_cost_pauses_without_settling(tmp_path: Path, cost: object) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    body = {"model": "s", "usage": None, "answers": {}} if cost is None else answer(cost)
    jev = backend(path, lambda *_: (200, {}, body))
    with pytest.raises(RuntimeError, match=r"valid usage\.cost"):
        jev.predict("run", ROW)
    assert jev.budget.paused and jev.budget.settled == 0.0
    assert [row["event"] for row in events(path)] == ["reserved", "retained"]
    with pytest.raises(RuntimeError, match="paused"):
        jev.predict("run", ROW)


def test_valid_cost_accepts_only_finite_non_negative_numbers() -> None:
    assert valid_cost({"usage": {"cost": 0}}) == 0.0
    assert valid_cost({"usage": {"cost": 0.25}}) == 0.25
    assert valid_cost({"usage": []}) is None
    assert valid_cost([]) is None


def test_second_dispatcher_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    holder = BudgetLedger(path)
    with pytest.raises(RuntimeError, match="budget lock"):
        BudgetLedger(path)
    holder.close()
    BudgetLedger(path).close()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"event": "reserved", "txn": "a", "amount": 1}] * 2,
            "duplicate",
        ),
        ([{"event": "settled", "txn": "a", "cost": 0}], "unknown transaction"),
        (
            [
                {"event": "reserved", "txn": "a", "amount": 1},
                {"event": "retained", "txn": "a"},
                {"event": "settled", "txn": "a", "cost": 0},
            ],
            "twice",
        ),
    ],
)
def test_inconsistent_ledger_refuses_to_start(tmp_path: Path, rows: list, message: str) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ledger = BudgetLedger(path)
    with pytest.raises(RuntimeError, match=message):
        Budget(ledger=ledger)
    ledger.close()


def test_ledger_rows_pass_the_key_scrub_writer(tmp_path: Path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    jev = JevOpenRouterBackend(
        "m",
        {},
        budget=Budget(ledger=BudgetLedger(path, secret="fake-key")),
        transport=lambda *_: (200, {}, answer(0.001)),
    )
    jev.predict("run", ROW)
    jev.close()
    assert "fake-key" not in path.read_text()
    assert events(path)[-1] == {"cost": 0.001, "event": "settled", "txn": events(path)[0]["txn"]}
