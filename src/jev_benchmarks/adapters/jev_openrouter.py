from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from ..models import Example, Prediction

URL = "https://openrouter.ai/api/alpha/decisions"
Transport = Callable[[bytes, str, float], tuple[int, dict[str, str], dict[str, Any]]]


class BudgetExceeded(RuntimeError):
    pass


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


class Budget:
    def __init__(
        self,
        maximum: float = 2.0,
        price_per_token: float = 4.2e-8,
        multiplier: float = 1.5,
        minimum: float = 0.00001,
    ) -> None:
        self.maximum = maximum
        self.price_per_token = price_per_token
        self.multiplier = multiplier
        self.minimum = minimum
        self.settled = 0.0
        self.reserved = 0.0
        self.paused = False
        self._lock = threading.Lock()

    def reserve(self, body: bytes) -> float:
        amount = max(self.minimum, len(body) * self.price_per_token * self.multiplier)
        with self._lock:
            if self.paused:
                raise RuntimeError("cost dispatch paused pending operator review")
            if self.settled + self.reserved + amount > self.maximum:
                raise BudgetExceeded("cost budget exceeded before dispatch")
            self.reserved += amount
        return amount

    def settle(self, reservation: float, cost: float | None) -> None:
        with self._lock:
            if cost is None:
                self.paused = True
                return
            self.reserved -= reservation
            self.settled += cost
            if self.settled + self.reserved > self.maximum:
                self.paused = True


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
                reservation = self.budget.reserve(body)
                try:
                    status, headers, response = self.transport(body, key, self.timeout)
                except TimeoutError:
                    self.budget.settle(reservation, None)
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
                    self.budget.settle(reservation, None)
                    self.log_attempt(
                        {
                            "example_id": example.example_id,
                            "attempt": attempt,
                            "status": "transport_error",
                            "reservation_usd": reservation,
                        }
                    )
                    raise
                usage = response.get("usage", {})
                cost = usage.get("cost")
                if cost is not None:
                    self.budget.settle(reservation, float(cost))
                elif status < 400:
                    self.budget.settle(reservation, None)
                self.log_attempt(
                    {
                        "example_id": example.example_id,
                        "attempt": attempt,
                        "status": status,
                        "reservation_usd": reservation,
                        "cost_usd": cost,
                    }
                )
                if cost is None and status < 400:
                    raise RuntimeError("response missing usage.cost; dispatch paused")
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
        return None
