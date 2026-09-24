from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from jev_benchmarks.adapters.gliner_v2 import GLiNERV2Backend
from jev_benchmarks.anchor import compare_anchor, run_anchor
from jev_benchmarks.config import BenchmarkConfig, load_config
from jev_benchmarks.io import write_jsonl
from jev_benchmarks.models import Example, Prediction


def example(kind: str = "choice") -> Example:
    labels = (
        ("ordinary message", "spam offer") if kind == "noul" else ("1 star", "2 star", "3 star")
    )
    return Example(
        "agnews",
        "topic",
        "agnews:1",
        "fixture",
        "hash",
        labels,
        0,
        question_type=kind,
        instructions="Which single label best describes the input text?",
    )


def prediction(row: Example, probabilities: tuple[float, ...]) -> Prediction:
    return Prediction(
        "pilot",
        "gliner",
        "model",
        "model@revision",
        row.dataset,
        row.example_id,
        row.target_index,
        max(range(len(probabilities)), key=probabilities.__getitem__),
        row.labels,
        probabilities,
        0.01,
        question_type=row.question_type,
        split=row.split,
    )


class FakeUpstream:
    def __init__(self, probabilities: tuple[float, ...]) -> None:
        self.probabilities = probabilities
        self.received: list[Example] = []
        self.closed = False

    def warmup(self, row: Example) -> None:
        self.received.append(row)

    def predict(self, experiment_id: str, row: Example) -> Prediction:
        self.received.append(row)
        return prediction(row, self.probabilities)

    def close(self) -> None:
        self.closed = True


def test_gliner_v2_question_types_and_instruction() -> None:
    upstream = FakeUpstream((0.2, 0.3, 0.5))
    backend = GLiNERV2Backend("model", "revision", upstream=upstream)
    score = backend.predict("pilot", example("score"))
    assert score.expected_score == pytest.approx(2.3)
    assert score.question_type == "score"
    assert upstream.received[0].instructions == "Which single label best describes the input text?"
    upstream.probabilities = (0.25, 0.75)
    noul = backend.predict("pilot", example("noul"))
    assert noul.probabilities == (0.25, 0.75)
    assert noul.expected_score is None
    backend.close()
    assert upstream.closed


def test_anchor_comparison_uses_both_macro_f1_definitions() -> None:
    row = example()
    records = [prediction(row, (0.8, 0.1, 0.1))]
    published = {"agnews": {"n": 1, "accuracy": 1.0, "macro_f1": 1 / 3, "brier": 0.06}}
    values = compare_anchor(records, published)["agnews"]
    assert values["macro_f1_all_classes"] == pytest.approx(1 / 3)
    assert values["macro_f1_targets_predictions"] == 1.0
    with pytest.raises(ValueError, match="brier"):
        compare_anchor(records, {"agnews": {**published["agnews"], "brier": 0.07}})


def _anchor_config(tmp_path: Path) -> BenchmarkConfig:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "results/reports").mkdir(parents=True)
    config_path = root / "configs/pilot-v1.yaml"
    raw = {
        "experiment_id": "pilot",
        "seed": 1,
        "output_dir": "results/runs/pilot",
        "dataset": {"datasets": [{"name": "agnews"}], "samples_per_dataset": 1},
        "models": {"gliner": {"model_id": "model", "revision": "revision", "device": "cpu"}},
        "metrics": {"ece_bins": 10, "error_budget": 0.05, "bootstrap_resamples": 2},
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(config_path)
    row = replace(example(), instructions="")
    cfg.output_dir.mkdir(parents=True)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict()])
    published = {
        "results": {
            "gliner": {
                "agnews": {
                    "n": 1,
                    "accuracy": 1.0,
                    "macro_f1": 1 / 3,
                    "brier": 0.06,
                }
            }
        }
    }
    (root / "results/reports/btzsc-pilot-v1.json").write_text(
        json.dumps(published), encoding="utf-8"
    )
    return cfg


def test_anchor_run_uses_existing_manifest_and_fake_backend(tmp_path: Path) -> None:
    cfg = _anchor_config(tmp_path)
    upstream = FakeUpstream((0.8, 0.1, 0.1))
    output = run_anchor(
        cfg, backend_factory=lambda config: GLiNERV2Backend("model", "revision", upstream=upstream)
    )
    assert output.exists() and upstream.closed
    assert json.loads(output.read_text())["agnews"]["macro_f1_targets_predictions"] == 1.0


def test_anchor_runs_gliner_in_the_minimal_env_worker_with_the_v2_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _anchor_config(tmp_path)
    (tmp_path / "pinned-cache").mkdir()
    (tmp_path / "configs" / "v2.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "v2",
                "seed": 1,
                "output_dir": "runs",
                "dataset": {"datasets": [{"name": "agnews"}], "samples_per_dataset": 1},
                "models": {"qwen_logit": {}},
                "metrics": {"ece_bins": 10, "error_budget": 0.05, "bootstrap_resamples": 2},
                "local_runtime": {"scratch_home": "sandbox", "hf_home": "pinned-cache"},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class Worker:
        def __init__(self, config, name, attempt, *, runtime):
            captured.update(config=config.path, name=name, runtime=runtime)
            self.backend = GLiNERV2Backend(
                "model", "revision", upstream=FakeUpstream((0.8, 0.1, 0.1))
            )

        def predict(self, experiment_id, example):
            return self.backend.predict(experiment_id, example)

        def close(self):
            return None

    monkeypatch.setattr("jev_benchmarks.v2_runner.LocalProcessBackend", Worker)
    run_anchor(cfg)
    assert captured["config"] == cfg.path and captured["name"] == "gliner"
    assert captured["runtime"]["hf_home"] == str(tmp_path / "pinned-cache")


def test_worker_builds_gliner_from_the_pilot_v1_model_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import sys

    from jev_benchmarks import worker
    from jev_benchmarks.adapters import gliner_v2

    cfg = _anchor_config(tmp_path)
    built = {}

    class Recorder:
        def __init__(self, model_id, revision, device):
            built.update(model_id=model_id, revision=revision, device=device)

        def close(self):
            return None

    monkeypatch.setattr(gliner_v2, "GLiNERV2Backend", Recorder)
    monkeypatch.setattr(sys, "argv", ["worker", str(cfg.path), "gliner", str(tmp_path / "a")])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    worker.main()
    assert built == {"model_id": "model", "revision": "revision", "device": "cpu"}
