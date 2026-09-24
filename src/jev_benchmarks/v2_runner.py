from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from .adapters.base import Backend
from .adapters.jev_openrouter import BudgetExceeded
from .config import BenchmarkConfig
from .io import append_jsonl, read_jsonl, scrubbed_json, sha256_file, write_json
from .models import Example, Prediction
from .runner import _validate_prediction


def _relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(Path(__file__).resolve().parents[2]))


def verify_frozen(config: BenchmarkConfig, manifest: Path) -> None:
    frozen = config.raw["frozen"]
    record_path = config.path.parent.parent / frozen["hashes_file"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["config_sha256"] != sha256_file(config.path):
        raise RuntimeError("config hash mismatch with preregistered record")
    if record["manifest_sha256"] != sha256_file(manifest):
        raise RuntimeError("manifest hash mismatch with preregistered record")
    protocol_path = config.path.parent.parent / frozen["protocol_file"]
    if record["protocol_sha256"] != sha256_file(protocol_path):
        raise RuntimeError("protocol hash mismatch with preregistered record")
    tag = str(frozen["tag"])
    if tag.startswith("TODO-PIN"):
        raise RuntimeError("preregistration tag needs TODO-PIN resolution")
    tagged = subprocess.run(
        ["git", "show", f"{tag}:{_relative_to_repo(record_path)}"],
        check=True,
        capture_output=True,
    ).stdout
    if tagged != record_path.read_bytes():
        raise RuntimeError("hash record differs from preregistered tag")


def prediction_key(row: Example | Prediction | dict[str, Any]) -> tuple[str, str, str, str, int]:
    if isinstance(row, dict):
        return (
            str(row.get("split", "test")),
            str(row["example_id"]),
            str(row.get("permutation_id", "identity")),
            str(row.get("letter_mode", "stable")),
            int(row.get("repeat_index", 0)),
        )
    return (
        row.split,
        row.example_id,
        row.permutation_id,
        row.letter_mode,
        row.repeat_index if isinstance(row, Prediction) else 0,
    )


def _make_v2_backend(config: BenchmarkConfig, name: str, attempts_dir: Path) -> Backend:
    model = config.raw["models"][name]
    questions = config.raw.get("questions", {})
    if name == "jev_openrouter":
        from .adapters.jev_openrouter import Budget, BudgetLedger, JevOpenRouterBackend

        secret = os.environ.get("OPENROUTER_API_KEY")
        # One experiment-level ledger across attempts; holds the single-dispatcher lock.
        ledger = BudgetLedger(attempts_dir.parent / "budget-ledger.jsonl", secret=secret)
        budget = Budget(
            maximum=float(model["max_cost_usd"]),
            price_per_token=float(model["prompt_price_per_token"]),
            multiplier=float(model["reservation_multiplier"]),
            minimum=float(model["minimum_reservation_usd"]),
            ledger=ledger,
        )
        return JevOpenRouterBackend(
            model["model_id"],
            questions,
            budget=budget,
            latency_paraphrases=config.raw["latency_paraphrases"],
            log_attempt=lambda row: append_jsonl(
                attempts_dir / "http-attempts.jsonl", row, secret=secret
            ),
        )
    if name.startswith("laya"):
        from .adapters.laya import LayaBackend

        return LayaBackend(
            model["model_id"],
            model["revision"],
            model["local_repo"],
            model["checkpoint"],
            model["head_max_len"],
            questions,
            latency_paraphrases=config.raw["latency_paraphrases"],
            backend_name=name,
        )
    if name == "qwen_logit":
        from .adapters.qwen_logit import QwenLogitBackend

        return QwenLogitBackend(
            model["model_id"],
            model["revision"],
            model["local_path"],
            model["system_prompt"],
            model["user_template"],
            model["score_modes"],
        )
    if name == "gliner":
        from .adapters.gliner_v2 import GLiNERV2Backend

        return GLiNERV2Backend(model["model_id"], model["revision"], model["device"])
    raise ValueError(f"unknown v2 backend: {name}")


def local_runtime(config: BenchmarkConfig) -> dict[str, str]:
    """``local_runtime`` paths resolved like ``output_dir`` (relative to the repo root)."""
    resolved = {}
    for key in ("scratch_home", "hf_home"):
        value = Path(config.raw["local_runtime"][key])
        resolved[key] = str(value if value.is_absolute() else config.path.parent.parent / value)
    if not Path(resolved["hf_home"]).is_dir():
        raise RuntimeError(f"pinned hf_home is missing: {resolved['hf_home']}")
    return resolved


class LocalProcessBackend:
    def __init__(
        self,
        config: BenchmarkConfig,
        name: str,
        attempt: Path,
        *,
        runtime: dict[str, str] | None = None,
    ) -> None:
        # The worker builds the model from ``config``; ``runtime`` may come from another config.
        local = runtime or local_runtime(config)
        environment = {
            "PATH": str(Path(sys.executable).parent) + os.pathsep + "/usr/bin:/bin",
            "HOME": str(local["scratch_home"]),
            "HF_HOME": str(local["hf_home"]),
            "HF_HUB_OFFLINE": "1",
        }
        self.name = name
        attempt.mkdir(parents=True, exist_ok=True)
        self.stderr_file = (attempt / f"worker-{name}.stderr.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "jev_benchmarks.worker", str(config.path), name, str(attempt)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            env=environment,
        )

    def warmup(self, example: Example) -> None:
        self.predict("warmup", example)

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(
            scrubbed_json({"experiment_id": experiment_id, "example": example.to_dict()}) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("local backend process ended before returning a prediction")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return Prediction.from_dict(result["prediction"])

    def features(self, example: Example, layer: int) -> list[float]:
        assert self.process.stdin is not None and self.process.stdout is not None
        request = {"op": "features", "layer": layer, "example": example.to_dict()}
        self.process.stdin.write(scrubbed_json(request) + "\n")
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline() or '{"error": "worker ended"}')
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return [float(value) for value in result["features"]]

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=10)
        self.stderr_file.close()


def _attempt_dir(root: Path) -> Path:
    existing = sorted((int(path.name.split("-")[-1]), path) for path in root.glob("attempt-*"))
    for _, path in existing:
        status = path / "status.json"
        if status.exists():
            state = json.loads(status.read_text(encoding="utf-8"))["state"]
            if state in {"cost_paused", "cost_exhausted"}:
                raise RuntimeError("cost dispatch stopped pending operator review")
    if existing:
        latest = existing[-1][1]
        status = latest / "status.json"
        if (
            not status.exists()
            or json.loads(status.read_text(encoding="utf-8"))["state"] == "active"
        ):
            return latest
    return root / f"attempt-{existing[-1][0] + 1 if existing else 1}"


def run_v2_backend(
    config: BenchmarkConfig,
    backend_name: str,
    *,
    split: str = "test",
    backend_factory: Any = None,
) -> Path:
    if backend_name not in config.raw["models"]:
        raise ValueError(f"unknown v2 backend: {backend_name}")
    if split not in {"pilot", "calibration", "test", "permutation", "latency"}:
        raise ValueError(f"unknown split: {split}")
    manifest = config.output_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError("v2 run requires an existing frozen manifest")
    verify_frozen(config, manifest)
    examples = [
        Example.from_dict(row)
        for row in read_jsonl(manifest)
        if row.get("split") == split
        and row.get("dataset") in config.raw["models"][backend_name]["datasets"]
        and (row.get("letter_mode", "stable") != "positional" or backend_name == "qwen_logit")
        and (row.get("permutation_id") != "head-default" or backend_name.startswith("laya"))
    ]
    if backend_name == "gliner":
        summary_path = config.output_dir / "manifest-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        excluded = {
            dataset
            for dataset, details in summary.items()
            if "gliner" in details.get("excluded_datasets", [])
        }
        examples = [row for row in examples if row.dataset not in excluded]
    if not examples:
        raise ValueError(f"manifest has no examples for split {split}")
    if (
        backend_name == "jev_openrouter"
        and backend_factory is None
        and not os.environ.get("OPENROUTER_API_KEY")
    ):
        raise RuntimeError("OPENROUTER_API_KEY is required before Jev dispatch")
    root = config.output_dir / backend_name
    # Attempt state and pending work are read under the lock, so overlapping runs cannot both
    # dispatch the same paid calls.
    with _dispatch_lock(root):
        return _dispatch(config, backend_name, split, backend_factory, examples, root)


@contextmanager
def _dispatch_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "dispatch.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another {root.name} run holds the dispatch lock") from None
        yield


def _dispatch(
    config: BenchmarkConfig,
    backend_name: str,
    split: str,
    backend_factory: Any,
    examples: list[Example],
    root: Path,
) -> Path:
    attempt = _attempt_dir(root)
    attempt.mkdir(parents=True, exist_ok=True)
    output = attempt / "predictions.jsonl"
    done = {prediction_key(row) for row in read_jsonl(output) if row.get("error") is None}
    repeats = int(config.raw["models"][backend_name]["repeats"])
    pending = [
        (row, repeat)
        for row in examples
        for repeat in range(repeats)
        if (row.split, row.example_id, row.permutation_id, row.letter_mode, repeat) not in done
    ]
    if not pending:
        return output
    backend = (
        backend_factory(config, backend_name, attempt)
        if backend_factory
        else LocalProcessBackend(config, backend_name, attempt)
        if backend_name.startswith("laya") or backend_name in {"qwen_logit", "gliner"}
        else _make_v2_backend(config, backend_name, attempt)
    )
    secret = os.environ.get("OPENROUTER_API_KEY")
    snapshot = next(
        (
            row["model_resolved"]
            for row in read_jsonl(output)
            if row.get("model_resolved") not in (None, "unknown")
        ),
        None,
    )
    try:
        if backend_name.startswith("laya") or backend_name in {"qwen_logit", "gliner"}:
            for _ in range(5 if split == "latency" else 1):
                backend.warmup(pending[0][0])
        for example, repeat in pending:
            prediction: Prediction | None = None
            served_hint: str | None = None
            try:
                raw = backend.predict(config.experiment_id, example)
                # Keep the serving snapshot even if validation below rejects the prediction.
                served_hint = raw.model_resolved if raw.model_resolved != "unknown" else None
                prediction = _validate_prediction(raw)
                if (
                    prediction.dataset != example.dataset
                    or prediction.example_id != example.example_id
                    or prediction.target_index != example.target_index
                    or prediction.labels != example.labels
                    or prediction.question_type != example.question_type
                    or prediction.split != example.split
                    or prediction.permutation_id != example.permutation_id
                    or prediction.letter_mode != example.letter_mode
                ):
                    raise ValueError("backend prediction differs from manifest contract")
                prediction = replace(
                    prediction, repeat_index=repeat, attempt_id=int(attempt.name.split("-")[-1])
                )
                if (
                    backend_name == "jev_openrouter"
                    and snapshot
                    and prediction.model_resolved != snapshot
                ):
                    write_json(
                        attempt / "status.json",
                        {
                            "state": "snapshot_changed",
                            "old": snapshot,
                            "new": prediction.model_resolved,
                        },
                        secret=secret,
                    )
                    raise RuntimeError("Jev resolved snapshot changed; start a new attempt")
                snapshot = prediction.model_resolved
            except Exception as exc:
                if "snapshot changed" in str(exc):
                    raise
                served = getattr(exc, "model_resolved", None) or served_hint
                if backend_name == "jev_openrouter" and served and snapshot and served != snapshot:
                    write_json(
                        attempt / "status.json",
                        {"state": "snapshot_changed", "old": snapshot, "new": served},
                        secret=secret,
                    )
                    raise RuntimeError(
                        "Jev resolved snapshot changed; start a new attempt"
                    ) from exc
                snapshot = snapshot or served
                if isinstance(exc, BudgetExceeded):
                    write_json(attempt / "status.json", {"state": "cost_exhausted"}, secret=secret)
                    raise
                message = f"{type(exc).__name__}: {exc}"
                prediction = Prediction(
                    config.experiment_id,
                    backend_name,
                    str(config.raw["models"][backend_name]["model_id"]),
                    served or "unknown",
                    example.dataset,
                    example.example_id,
                    example.target_index,
                    -1,
                    example.labels,
                    (),
                    0.0,
                    error=message,
                    question_type=example.question_type,
                    split=example.split,
                    permutation_id=example.permutation_id,
                    letter_mode=example.letter_mode,
                    repeat_index=repeat,
                    attempt_id=int(attempt.name.split("-")[-1]),
                    dispatch_attempts=getattr(backend, "last_dispatch_attempts", None) or None,
                )
            append_jsonl(output, prediction.to_dict(), secret=secret)
            if getattr(getattr(backend, "budget", None), "paused", False):
                write_json(attempt / "status.json", {"state": "cost_paused"}, secret=secret)
                raise RuntimeError("cost dispatch paused pending operator review")
    finally:
        backend.close()
    write_json(attempt / "status.json", {"state": "active", "snapshot": snapshot}, secret=secret)
    return output
