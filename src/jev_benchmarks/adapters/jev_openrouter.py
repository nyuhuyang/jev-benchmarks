from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from ..io import append_jsonl, read_jsonl
from ..models import Example, Prediction

URL = "https://openrouter.ai/api/alpha/decisions"
Transport = Callable[[bytes, str, float], tuple[int, dict[str, str], dict[str, Any]]]


class BudgetExceeded(RuntimeError):
    pass


class JevResponseError(RuntimeError):
    """A response that could not be scored, carrying the snapshot that served it."""

    def __init__(self, message: str, model_resolved: str | None) -> None:
        super().__init__(message)
        self.model_resolved = model_resolved


def _retry_delay(value: str | None, attempt: int) -> float:
    if value is None:
        return float(min(2**attempt, 8))
    try:
        return max(0.0, float(value))
    except ValueError:
        return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())


def _transport(body: bytes, key: str, timeout: float) -> tuple[int, dict[str, str], dict[str, Any]]:
    request = urllib.request.Request(
        URL,
        body,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), json.loads(error.read())


def _money(value: object) -> float | None:
    """A finite, non-negative JSON number (booleans excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def valid_cost(response: object) -> float | None:
    """Return ``usage.cost`` only when it is a finite, non-negative number."""
    usage = response.get("usage") if isinstance(response, dict) else None
    return _money(usage.get("cost") if isinstance(usage, dict) else None)


class BudgetLedger:
    """Durable, single-dispatcher record of every Jev reservation and its closure.

    A reservation is written (fsync) before dispatch. Only a ``settled`` record with a finite
    cost replaces it; ``retained`` and unclosed reservations count at their full amount, so a
    crash or an unbilled error can only over-count spend.
    """

    def __init__(self, path: Path, *, secret: str | None = None) -> None:
        self.path = path
        self.secret = secret
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = path.with_name("budget.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise RuntimeError("another Jev dispatcher holds the budget lock") from None

    def liability(self) -> float:
        reserved: dict[str, float] = {}
        closed: dict[str, float | None] = {}
        for row in read_jsonl(self.path):
            txn, event = str(row["txn"]), row["event"]
            if event == "reserved":
                if txn in reserved:
                    raise RuntimeError(f"budget ledger has duplicate transaction {txn}")
                amount = _money(row.get("amount"))
                if amount is None:
                    raise RuntimeError(f"budget ledger has an invalid amount for {txn}")
                reserved[txn] = amount
            elif event in {"settled", "retained"}:
                if txn not in reserved:
                    raise RuntimeError(f"budget ledger closes unknown transaction {txn}")
                if txn in closed:
                    raise RuntimeError(f"budget ledger closes transaction {txn} twice")
                cost = _money(row.get("cost")) if event == "settled" else None
                if event == "settled" and cost is None:
                    raise RuntimeError(f"budget ledger has an invalid cost for {txn}")
                closed[txn] = cost
            else:
                raise RuntimeError(f"budget ledger has unknown event {event!r}")
        return sum(
            paid if (paid := closed.get(txn)) is not None else amount
            for txn, amount in reserved.items()
        )

    def append(self, row: dict[str, Any]) -> None:
        append_jsonl(self.path, row, secret=self.secret)

    def close(self) -> None:
        self._lock_file.close()


class Budget:
    def __init__(
        self,
        maximum: float = 2.0,
        price_per_token: float = 4.2e-8,
        multiplier: float = 1.5,
        minimum: float = 0.00001,
        *,
        ledger: BudgetLedger | None = None,
    ) -> None:
        self.maximum = maximum
        self.price_per_token = price_per_token
        self.multiplier = multiplier
        self.minimum = minimum
        self.ledger = ledger
        self.settled = ledger.liability() if ledger else 0.0
        self.reserved = 0.0
        self.paused = False
        self._lock = threading.Lock()

    def reserve(self, body: bytes, txn: str | None = None) -> float:
        amount = max(self.minimum, len(body) * self.price_per_token * self.multiplier)
        with self._lock:
            if self.paused:
                raise RuntimeError("cost dispatch paused pending operator review")
            if self.settled + self.reserved + amount > self.maximum:
                raise BudgetExceeded("cost budget exceeded before dispatch")
            if self.ledger and txn:
                self.ledger.append({"event": "reserved", "txn": txn, "amount": amount})
            self.reserved += amount
        return amount

    def settle(
        self, reservation: float, cost: float | None, txn: str | None = None, *, pause: bool = True
    ) -> None:
        """Close a reservation. ``cost=None`` retains it in full; ``pause`` stops dispatch."""
        with self._lock:
            if cost is None:
                if self.ledger and txn:
                    self.ledger.append({"event": "retained", "txn": txn})
                self.paused = self.paused or pause
                return
            if self.ledger and txn:
                self.ledger.append({"event": "settled", "txn": txn, "cost": cost})
            self.reserved -= reservation
            self.settled += cost
            if self.settled + self.reserved > self.maximum:
                self.paused = True

    def close(self) -> None:
        if self.ledger:
            self.ledger.close()


class JevOpenRouterBackend:
    name = "jev_openrouter"

    def __init__(
        self,
        model_id: str,
        questions: dict[str, str],
        *,
        budget: Budget | None = None,
        transport: Transport = _transport,
        timeout: float = 30,
        attempts: int = 3,
        log_attempt: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        latency_paraphrases: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.questions = questions
        self.budget = budget or Budget()
        self.transport = transport
        self.timeout = timeout
        self.attempts = attempts
        self.log_attempt = log_attempt or (lambda row: None)
        self.sleep = sleep
        self.latency_paraphrases = latency_paraphrases or []
        self._slots = threading.BoundedSemaphore(4)

    def warmup(self, example: Example) -> None:
        return None

    @staticmethod
    def criteria(example: Example) -> object:
        order = example.option_order or tuple(range(len(example.labels)))
        if example.question_type == "noul":
            return {"true": example.labels[1], "false": example.labels[0]}
        if example.question_type == "score":
            return [example.labels[index] for index in order]
        return {f"label_{index:03d}": example.labels[index] for index in order}

    def request_body(self, example: Example) -> bytes:
        question = {
            "type": example.question_type,
            "instructions": example.instructions or self.questions[example.question_type],
            "criteria": self.criteria(example),
        }
        questions = {"label": question}
        if example.permutation_id == "latency-10":
            if len(self.latency_paraphrases) != 10:
                raise ValueError("latency-10 requires ten frozen paraphrases")
            questions = {
                f"label_{index}": {**question, "instructions": paraphrase}
                for index, paraphrase in enumerate(self.latency_paraphrases)
            }
        payload = {"model": self.model_id, "state": {"text": example.text}, "questions": questions}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        body = self.request_body(example)
        response: dict[str, Any] | None = None
        start = time.perf_counter()
        with self._slots:
            for attempt in range(self.attempts):
                txn = str(uuid.uuid4())
                reservation = self.budget.reserve(body, txn)
                try:
                    status, headers, response = self.transport(body, key, self.timeout)
                except TimeoutError:
                    self.budget.settle(reservation, None, txn)
                    self.log_attempt(
                        {
                            "example_id": example.example_id,
                            "attempt": attempt,
                            "status": "timeout",
                            "reservation_usd": reservation,
                        }
                    )
                    raise
                except Exception:
                    self.budget.settle(reservation, None, txn)
                    self.log_attempt(
                        {
                            "example_id": example.example_id,
                            "attempt": attempt,
                            "status": "transport_error",
                            "reservation_usd": reservation,
                        }
                    )
                    raise
                cost = valid_cost(response)
                # A 2xx without a valid cost pauses; an unbilled error keeps its reservation.
                self.budget.settle(reservation, cost, txn, pause=status < 400)
                self.log_attempt(
                    {
                        "example_id": example.example_id,
                        "attempt": attempt,
                        "status": status,
                        "reservation_usd": reservation,
                        "cost_usd": cost,
                        "txn": txn,
                    }
                )
                if cost is None and status < 400:
                    raise RuntimeError("response missing a valid usage.cost; dispatch paused")
                if status == 429 or 500 <= status <= 599:
                    if attempt + 1 == self.attempts:
                        raise RuntimeError(f"OpenRouter HTTP {status} after retries")
                    retry_after = headers.get("retry-after", headers.get("Retry-After"))
                    self.sleep(_retry_delay(retry_after, attempt))
                    continue
                if status >= 400:
                    raise RuntimeError(f"OpenRouter HTTP {status}")
                break
        assert response is not None
        served = response.get("model") if isinstance(response, dict) else None
        try:
            return self._parse(experiment_id, example, response, start)
        except Exception as exc:
            # Keep the serving snapshot on unscorable 2xx responses for provenance checks.
            raise JevResponseError(
                f"{type(exc).__name__}: {exc}", str(served) if served else None
            ) from exc

    def _parse(
        self, experiment_id: str, example: Example, response: dict[str, Any], start: float
    ) -> Prediction:
        answer = response["answers"][
            "label_0" if example.permutation_id == "latency-10" else "label"
        ]
        values = answer.get("probabilities", {})
        if example.question_type == "choice":
            probabilities = tuple(
                float(values[f"label_{index:03d}"]) for index in range(len(example.labels))
            )
        elif example.question_type == "noul":
            positive = float(answer["noul"])
            probabilities = (1.0 - positive, positive)
        else:
            ordered = (
                [float(values[str(index)]) for index in range(len(example.labels))]
                if isinstance(values, dict)
                else [float(value) for value in values]
            )
            order = example.option_order or tuple(range(len(example.labels)))
            mapped = [0.0] * len(ordered)
            for position, index in enumerate(order):
                mapped[index] = ordered[position]
            probabilities = tuple(mapped)
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        expected = (
            sum((index + 1) * value for index, value in enumerate(probabilities))
            if example.question_type == "score"
            else None
        )
        return Prediction(
            experiment_id,
            self.name,
            self.model_id,
            str(response["model"]),
            example.dataset,
            example.example_id,
            example.target_index,
            predicted,
            example.labels,
            probabilities,
            time.perf_counter() - start,
            input_tokens=int(response["usage"].get("input_tokens", 0)),
            question_type=example.question_type,
            split=example.split,
            permutation_id=example.permutation_id,
            letter_mode=example.letter_mode,
            raw_probabilities=probabilities,
            expected_score=expected,
            cost_usd=float(response["usage"]["cost"]),
            confidence_reported=answer.get("confidence"),
        )

    def close(self) -> None:
        self.budget.close()
