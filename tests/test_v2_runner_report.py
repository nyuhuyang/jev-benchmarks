from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jev_benchmarks.adapters.jev_openrouter import BudgetExceeded
from jev_benchmarks.config import BenchmarkConfig
from jev_benchmarks.io import read_jsonl, scrubbed_json, sha256_file, write_jsonl
from jev_benchmarks.models import Example, Prediction
from jev_benchmarks.v2_report import _gliner_exclusions, _select_attempt, build_v2_report
from jev_benchmarks.v2_runner import (
    LocalProcessBackend,
    _attempt_dir,
    prediction_key,
    run_v2_backend,
    verify_frozen,
)


def config(tmp_path: Path) -> BenchmarkConfig:
    path = tmp_path / "configs" / "v2.yaml"
    path.parent.mkdir()
    path.write_text("v2: fake\n", encoding="utf-8")
    return BenchmarkConfig(
        {
            "schema_version": 2,
            "experiment_id": "fake-v2",
            "seed": 1,
            "output_dir": str(tmp_path / "runs"),
            "report_dir": "reports",
            "frozen": {
                "hashes_file": "configs/freeze.json",
                "protocol_file": "docs/PROTOCOL-v2.md",
                "tag": "v2-preregistered",
            },
            "dataset": {"datasets": [{"name": "agnews"}]},
            "models": {
                "jev_openrouter": {
                    "model_id": "jev",
                    "datasets": ["agnews"],
                    "repeats": 3,
                    "max_cost_usd": 2,
                    "prompt_price_per_token": 1e-8,
                    "reservation_multiplier": 1.5,
                    "minimum_reservation_usd": 1e-5,
                },
                "qwen_logit": {"model_id": "qwen", "datasets": ["agnews"], "repeats": 1},
            },
            "questions": {"choice": "Choose"},
            "latency_paraphrases": ["p"] * 10,
            "metrics": {"ece_bins": 10, "error_budget": 0.05, "bootstrap_resamples": 10},
            "confirmatory_family": {
                "k": 3,
                "tests": [
                    {
                        "id": "C1-accuracy",
                        "left": "jev_openrouter",
                        "right": "qwen_logit",
                        "datasets": ["agnews"],
                        "metric": "accuracy",
                        "condition": "A_raw",
                    },
                    {
                        "id": "C1-brier-A",
                        "left": "jev_openrouter",
                        "right": "qwen_logit",
                        "datasets": ["agnews"],
                        "metric": "brier",
                        "condition": "A_raw",
                    },
                    {
                        "id": "C1-brier-B",
                        "left": "jev_openrouter",
                        "right": "qwen_logit",
                        "datasets": ["agnews"],
                        "metric": "brier",
                        "condition": "B_scaled",
                    },
                ],
            },
            "local_runtime": {
                "scratch_home": str(tmp_path / "sandbox"),
                "hf_home": str(tmp_path / "cache"),
            },
        },
        path,
    )


def examples() -> list[Example]:
    return [
        Example(
            "agnews",
            "topic",
            f"id:{index}",
            f"synthetic {index}",
            f"h{index}",
            ("a", "b"),
            index % 2,
            split=split,
        )
        for split in ("calibration", "test")
        for index in range(4)
    ]


def prediction(backend: str, row: Example, repeat: int = 0) -> Prediction:
    vector = (0.8, 0.2) if row.target_index == 0 else (0.2, 0.8)
    return Prediction(
        "fake-v2",
        backend,
        backend,
        "snapshot-1",
        row.dataset,
        row.example_id,
        row.target_index,
        row.target_index,
        row.labels,
        vector,
        0.1,
        question_type=row.question_type,
        split=row.split,
        repeat_index=repeat,
        permutation_id=row.permutation_id,
        letter_mode=row.letter_mode,
        model_latency_seconds=0.05 if backend == "qwen_logit" else None,
    )


def test_frozen_hash_contract_and_key_scrub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(tmp_path)
    manifest = cfg.output_dir / "manifest.jsonl"
    write_jsonl(manifest, [row.to_dict() for row in examples()])
    protocol = tmp_path / "docs" / "PROTOCOL-v2.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol", encoding="utf-8")
    freeze = tmp_path / "configs" / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "config_sha256": sha256_file(cfg.path),
                "manifest_sha256": sha256_file(manifest),
                "protocol_sha256": sha256_file(protocol),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "jev_benchmarks.v2_runner._relative_to_repo", lambda path: "configs/freeze.json"
    )
    monkeypatch.setattr(
        "jev_benchmarks.v2_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=freeze.read_bytes()),
    )
    verify_frozen(cfg, manifest)
    # The committed probe record is bound by hash when the freeze record names it.
    probe = tmp_path / "results" / "reports" / "probe-v2.json"
    probe.parent.mkdir(parents=True)
    probe.write_text("{}", encoding="utf-8")
    record = json.loads(freeze.read_text())
    freeze.write_text(json.dumps({**record, "probe_results_sha256": sha256_file(probe)}))
    verify_frozen(cfg, manifest)
    probe.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="probe record hash"):
        verify_frozen(cfg, manifest)
    freeze.write_text(
        json.dumps({"config_sha256": "bad", "manifest_sha256": "bad"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="config hash"):
        verify_frozen(cfg, manifest)
    with pytest.raises(ValueError, match="API key"):
        scrubbed_json({"error": "fake-secret appeared"}, "fake-secret")
    assert prediction_key(examples()[0]) == ("calibration", "id:0", "identity", "stable", 0)


def test_runner_resume_repeat_keys_and_snapshot_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    manifest = cfg.output_dir / "manifest.jsonl"
    write_jsonl(manifest, [row.to_dict() for row in examples()])
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)

    class FakeBackend:
        def __init__(self):
            self.calls = []
            self.closed = False
            self.snapshot = "snapshot-1"

        def warmup(self, row):
            return None

        def predict(self, experiment_id, row):
            self.calls.append(row.example_id)
            return replace(prediction("jev_openrouter", row), model_resolved=self.snapshot)

        def close(self):
            self.closed = True

    backend = FakeBackend()

    def factory(*_):
        return backend

    output = run_v2_backend(cfg, "jev_openrouter", split="calibration", backend_factory=factory)
    assert output.parent.name == "attempt-1"
    assert len(read_jsonl(output)) == 12
    assert backend.closed
    assert (
        run_v2_backend(cfg, "jev_openrouter", split="calibration", backend_factory=factory)
        == output
    )
    assert len(backend.calls) == 12
    backend.snapshot = "snapshot-2"
    with pytest.raises(RuntimeError, match="snapshot changed"):
        run_v2_backend(cfg, "jev_openrouter", split="test", backend_factory=factory)
    assert json.loads((output.parent / "status.json").read_text())["state"] == "snapshot_changed"
    backend.snapshot = "snapshot-3"
    next_output = run_v2_backend(cfg, "jev_openrouter", split="test", backend_factory=factory)
    assert next_output.parent.name == "attempt-2"
    assert len(read_jsonl(next_output)) == 12


def test_runner_refuses_missing_manifest_and_scrubs_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)
    with pytest.raises(FileNotFoundError, match="existing frozen manifest"):
        run_v2_backend(cfg, "jev_openrouter")
    with pytest.raises(ValueError, match="unknown v2 backend"):
        run_v2_backend(cfg, "missing")
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in examples()])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_v2_backend(cfg, "jev_openrouter", split="test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-secret")

    class Broken:
        def warmup(self, row):
            return None

        def predict(self, experiment_id, row):
            raise ValueError("fake-secret in provider error")

        def close(self):
            return None

    with pytest.raises(ValueError, match="API key"):
        run_v2_backend(cfg, "jev_openrouter", split="test", backend_factory=lambda *_: Broken())
    assert read_jsonl(cfg.output_dir / "jev_openrouter" / "attempt-1" / "predictions.jsonl") == []

    class Exhausted(Broken):
        def predict(self, experiment_id, row):
            raise BudgetExceeded("cost budget exceeded before dispatch")

    with pytest.raises(BudgetExceeded):
        run_v2_backend(cfg, "jev_openrouter", split="test", backend_factory=lambda *_: Exhausted())
    assert read_jsonl(cfg.output_dir / "jev_openrouter" / "attempt-1" / "predictions.jsonl") == []
    assert (
        json.loads((cfg.output_dir / "jev_openrouter" / "attempt-1" / "status.json").read_text())[
            "state"
        ]
        == "cost_exhausted"
    )
    with pytest.raises(ValueError, match="unknown split"):
        run_v2_backend(cfg, "jev_openrouter", split="bad")


def test_cost_pause_status_and_minimal_local_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    root = cfg.output_dir / "jev_openrouter"
    path = root / "attempt-1"
    path.mkdir(parents=True)
    (path / "status.json").write_text('{"state":"cost_paused"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="operator review"):
        _attempt_dir(root)
    (path / "status.json").write_text('{"state":"snapshot_changed"}', encoding="utf-8")
    assert _attempt_dir(root).name == "attempt-2"
    captured: dict[str, Any] = {}

    class FakeProcess:
        stdin = io.StringIO()
        stdout = io.StringIO(
            '{"prediction": '
            + json.dumps(prediction("qwen_logit", examples()[0]).to_dict())
            + "}\n"
        )

        def wait(self, timeout):
            return 0

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("jev_benchmarks.v2_runner.subprocess.Popen", popen)
    with pytest.raises(RuntimeError, match="hf_home is missing"):
        LocalProcessBackend(cfg, "qwen_logit", tmp_path / "attempt")
    (tmp_path / "cache").mkdir()
    local = LocalProcessBackend(cfg, "qwen_logit", tmp_path / "attempt")
    assert local.predict("e", examples()[0]).backend == "qwen_logit"
    assert set(captured["env"]) == {"PATH", "HOME", "HF_HOME", "HF_HUB_OFFLINE"}
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["HF_HOME"] == str(tmp_path / "cache")
    assert captured["stderr"].name.endswith("worker-qwen_logit.stderr.log")
    local.close()


def test_gliner_excluded_dataset_is_not_expected_from_attempt(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cfg.raw["models"]["gliner"] = {
        "model_id": "gliner",
        "datasets": ["agnews", "banking77"],
        "repeats": 1,
    }
    cfg.output_dir.mkdir(parents=True)
    (cfg.output_dir / "manifest-summary.json").write_text(
        json.dumps(
            {"agnews": {"excluded_datasets": []}, "banking77": {"excluded_datasets": ["gliner"]}}
        ),
        encoding="utf-8",
    )
    assert _gliner_exclusions(cfg) == ["banking77"]
    rows = examples()
    manifest = rows + [
        replace(row, dataset="banking77", example_id=f"banking77:{row.example_id}") for row in rows
    ]
    attempt = cfg.output_dir / "gliner/attempt-1"
    attempt.mkdir(parents=True)
    write_jsonl(
        attempt / "predictions.jsonl", [prediction("gliner", row).to_dict() for row in rows]
    )
    selected, predictions, _ = _select_attempt(cfg, "gliner", manifest)
    assert selected == attempt
    assert len(predictions) == len(rows)


def test_v2_report_selects_single_snapshot_and_writes_aggregate_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    rows = examples()
    protocol = tmp_path / "docs" / "PROTOCOL-v2.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol", encoding="utf-8")
    # Add the derived suites without adding identifying text to aggregate outputs.
    rows += [
        replace(row, split="latency", permutation_id=f"latency-{size}")
        for row in examples()[:2]
        for size in (1, 10)
    ]
    rows += [
        replace(
            row,
            split="permutation",
            permutation_id=f"order-{index}",
            option_order=(1, 0) if index else (0, 1),
        )
        for row in examples()[:2]
        for index in range(4)
    ]
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in rows])
    monkeypatch.setattr("jev_benchmarks.v2_report.verify_frozen", lambda *args: None)
    for backend, model in cfg.raw["models"].items():
        attempt = cfg.output_dir / backend / "attempt-1"
        predictions = [
            prediction(backend, row, repeat) for row in rows for repeat in range(model["repeats"])
        ]
        write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in predictions])
        (attempt / "status.json").write_text('{"state":"active"}', encoding="utf-8")
    json_path, md_path = build_v2_report(cfg)
    payload = json.loads(json_path.read_text())
    assert payload["confirmatory_holm_k"] == 3
    assert payload["attempts"] == {"jev_openrouter": "attempt-1", "qwen_logit": "attempt-1"}
    assert "Frozen Holm family" in md_path.read_text()
    reports = json_path.parent
    assert {path.name for path in reports.iterdir()} == {
        "v2.json",
        "v2.md",
        "metrics.csv",
        "pairwise_ci.csv",
        "reliability_bins.csv",
        "latency.csv",
        "flip.csv",
        "repeat.csv",
        "temperature.csv",
        "SHA256SUMS",
    }
    assert "synthetic 0" not in (reports / "metrics.csv").read_text()
    assert "example_id" not in (reports / "latency.csv").read_text()
    assert len((reports / "SHA256SUMS").read_text().splitlines()) == 9
    # An incomplete newer attempt must not displace a complete selected attempt.
    write_jsonl(cfg.output_dir / "jev_openrouter" / "attempt-2" / "predictions.jsonl", [])
    assert build_v2_report(cfg)[0] == json_path
    qwen_path = cfg.output_dir / "qwen_logit" / "attempt-1" / "predictions.jsonl"
    qwen_rows = read_jsonl(qwen_path)
    qwen_rows[0]["target_index"] = 1 - qwen_rows[0]["target_index"]
    write_jsonl(qwen_path, qwen_rows)
    with pytest.raises(ValueError, match="prediction contract mismatch"):
        build_v2_report(cfg)
