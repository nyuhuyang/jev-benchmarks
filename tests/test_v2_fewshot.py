from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from test_v2_runner_report import config as report_config
from test_v2_runner_report import examples as report_examples
from test_v2_runner_report import prediction as report_prediction

from jev_benchmarks import fewshot
from jev_benchmarks.config import BenchmarkConfig
from jev_benchmarks.fewshot import FOLDS, PriorModel, cross_fit, make_model, run_fewshot
from jev_benchmarks.io import read_jsonl, write_jsonl
from jev_benchmarks.models import Example
from jev_benchmarks.v2_report import _select_attempt, build_v2_report, vector_datasets


class Spy:
    """Records what each model instance is fitted on and asked to predict."""

    instances: ClassVar[list[Spy]] = []

    def __init__(self) -> None:
        self.fitted: set[str] = set()
        self.predicted: list[str] = []
        Spy.instances.append(self)

    def fit(self, x, y):
        self.fitted = set(x)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x):
        self.predicted.extend(x)
        return np.full((len(x), len(self.classes_)), 1 / len(self.classes_))


def test_out_of_fold_rows_are_predicted_once_by_a_model_that_never_saw_them() -> None:
    Spy.instances = []
    texts = [f"cal {index}" for index in range(20)]
    y = np.arange(20) % 3
    test = [f"test {index}" for index in range(5)]
    out_of_fold, test_probabilities, absent = cross_fit(texts, y, test, 3, Spy, seed=1)
    folds, final = Spy.instances[:FOLDS], Spy.instances[FOLDS]
    assert sorted(text for spy in folds for text in spy.predicted) == sorted(texts)
    assert all(not (set(spy.predicted) & spy.fitted) for spy in Spy.instances)
    assert final.fitted == set(texts) and final.predicted == test
    assert out_of_fold.shape == (20, 3) and test_probabilities.shape == (5, 3) and absent == 0


def test_absent_class_gets_zero_and_is_counted() -> None:
    y = np.array([0, 1] * 5 + [2])
    out_of_fold, _, absent = cross_fit(list(range(11)), y, [0], 3, PriorModel, seed=1)
    assert absent == 1
    assert out_of_fold[-1, 2] == 0.0 and out_of_fold[-1].sum() == pytest.approx(1.0)
    assert PriorModel().fit(None, np.array([0, 0, 1])).predict_proba([1, 2]) == pytest.approx(
        np.array([[2 / 3, 1 / 3]] * 2)
    )


@pytest.mark.parametrize("classes", [72, 60])  # Banking77-like and MASSIVE-like calibration
def test_rare_class_datasets_run_and_are_deterministic(classes: int) -> None:
    y = np.arange(200) % classes
    texts = [f"intent {label} example {index}" for index, label in enumerate(y)]
    first = cross_fit(texts, y, texts[:7], classes, lambda: make_model("tfidf_lr", 1), seed=1)
    second = cross_fit(texts, y, texts[:7], classes, lambda: make_model("tfidf_lr", 1), seed=1)
    assert first[0].shape == (200, classes) and first[1].shape == (7, classes)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert np.allclose(first[0].sum(axis=1), 1.0)


def test_tfidf_vocabulary_is_fitted_per_training_fold() -> None:
    models = []

    def new_model():
        models.append(make_model("tfidf_lr", 1))
        return models[-1]

    texts = [f"ordinary text {index % 2}" for index in range(19)] + ["zzqq only here"]
    cross_fit(texts, np.arange(20) % 2, ["x"], 2, new_model, seed=1)
    has_token = [" zz" in model[0].vocabulary_ for model in models]
    assert has_token[:FOLDS].count(False) == 1 and has_token[FOLDS]


def fewshot_config(tmp_path: Path, rows: list[Example]) -> BenchmarkConfig:
    path = tmp_path / "configs" / "v2.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("fake: true", encoding="utf-8")
    cfg = BenchmarkConfig(
        {
            "schema_version": 2,
            "experiment_id": "few",
            "seed": 1,
            "output_dir": str(tmp_path / "runs"),
            "models": {
                "qwen_logit": {"model_id": "Qwen/Qwen3-1.7B", "revision": "rev"},
                "qwen_probe": {"datasets": ["agnews"], "repeats": 1, "layer": 18},
                "tfidf_lr": {"datasets": ["agnews"], "repeats": 1},
                "prior": {"datasets": ["agnews"], "repeats": 1},
            },
        },
        path,
    )
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in rows])
    return cfg


def rows(test_target: int) -> list[Example]:
    base = Example("agnews", "topic", "", "", "h", ("a", "b", "c"), 0, question_type="choice")
    calibration = [
        replace(
            base,
            example_id=f"c{i}",
            text=f"word{i % 3} text",
            target_index=i % 3,
            split="calibration",
        )
        for i in range(15)
    ]
    test = [
        replace(
            base, example_id=f"t{i}", text=f"word{i % 3}", target_index=test_target, split="test"
        )
        for i in range(6)
    ]
    permutation = [replace(test[0], split="permutation", permutation_id="order-1")]
    return calibration + test + permutation


class FakeExtractor:
    closed = False

    def features(self, row: Example, layer: int) -> list[float]:
        assert layer == 18
        return [float(ord(row.text[4])), float(len(row.text))]

    def close(self) -> None:
        FakeExtractor.closed = True


@pytest.mark.parametrize("backend", ["qwen_probe", "tfidf_lr", "prior"])
def test_run_fewshot_never_fits_on_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)
    outputs = []
    for target in (0, 2):
        cfg = fewshot_config(tmp_path / str(target), rows(target))
        output = run_fewshot(cfg, backend, feature_extractor=FakeExtractor())
        outputs.append(read_jsonl(output))
    # Changing every test label leaves every probability unchanged.
    assert [row["probabilities"] for row in outputs[0]] == [
        row["probabilities"] for row in outputs[1]
    ]
    predictions = outputs[0]
    assert {row["split"] for row in predictions} == {"calibration", "test"}
    assert len(predictions) == 21 and len({row["model_resolved"] for row in predictions}) == 1
    attempt = Path(output).parent
    metadata = json.loads((attempt / "fewshot-metadata.json").read_text())
    assert metadata["folds"] == FOLDS
    if backend == "qwen_probe":
        assert set(metadata["features_sha256"]) == {"features-calibration.npz", "features-test.npz"}
        assert FakeExtractor.closed
    assert run_fewshot(cfg, backend, feature_extractor=FakeExtractor()) == output


def test_fewshot_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="few-label"):
        run_fewshot(fewshot_config(tmp_path, rows(0)), "gliner")
    with pytest.raises(ValueError, match="few-label"):
        make_model("gliner", 1)


def test_report_drops_latency_for_few_label_rows_and_adds_jev_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = report_config(tmp_path)
    cfg.raw["models"]["prior"] = {"model_id": "prior", "datasets": ["agnews"], "repeats": 1}
    protocol = tmp_path / "docs" / "PROTOCOL-v2.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol", encoding="utf-8")
    (tmp_path / "docs" / "public-results.csv").write_text(
        "source,url,contenders,task,metric,values,protocol_note\n"
        "elcronos,https://example.org,Jev; Laya,emotion,accuracy,Jev 0.587,raw ECE\n",
        encoding="utf-8",
    )
    manifest = report_examples()
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in manifest])
    monkeypatch.setattr("jev_benchmarks.v2_report.verify_frozen", lambda *args: None)
    for backend, model in cfg.raw["models"].items():
        attempt = cfg.output_dir / backend / "attempt-1"
        predictions = [
            report_prediction(backend, row, repeat)
            for row in manifest
            for repeat in range(model["repeats"])
        ]
        write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in predictions])
        (attempt / "status.json").write_text('{"state":"active"}', encoding="utf-8")
    (cfg.output_dir / "prior" / "attempt-1" / "fewshot-metadata.json").write_text(
        '{"absent_class_held_out_rows": {"agnews": 3}}', encoding="utf-8"
    )
    json_path, _ = build_v2_report(cfg)
    with (json_path.parent / "metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    prior_rows = [row for row in metrics if row["backend"] == "prior"]
    jev_rows = [row for row in metrics if row["backend"] == "jev_openrouter"]
    assert prior_rows and all(not row["latency_p50_seconds"] for row in prior_rows)
    assert all(row["latency_p50_seconds"] for row in jev_rows if row["condition"] == "A_raw")
    payload = json.loads(json_path.read_text())
    few = [
        row
        for row in payload["descriptive_intervals"]
        if row["family"] == "descriptive_few_label" and row["right"] == "prior"
    ]
    assert {(row["metric"], row["condition"]) for row in few} == {
        ("accuracy", "A_raw"),
        ("brier", "A_raw"),
        ("brier", "B_scaled"),
    }
    assert any("not zero-shot" in text for text in payload["limitations"])
    assert payload["public_results"][0]["source"] == "elcronos"
    assert payload["few_label_absent_class_held_out_rows"] == {"prior": {"agnews": 3}}
    assert "Relation to public results" in json_path.with_suffix(".md").read_text()
    assert fewshot.FEWSHOT_BACKENDS == {"qwen_probe", "tfidf_lr", "prior"}


def test_scalar_only_outputs_are_excluded_from_vector_comparisons() -> None:
    manifest = report_examples()
    vector = {"agnews": {"calibration": [report_prediction("prior", row) for row in manifest]}}
    scalar = {
        "agnews": {
            "calibration": [
                replace(
                    report_prediction("jev_openrouter", row),
                    question_type="score",
                    probabilities=(),
                    expected_score=1.5,
                )
                for row in manifest
            ]
        }
    }
    assert vector_datasets(vector, vector, ["agnews"]) == ["agnews"]
    assert vector_datasets(scalar, vector, ["agnews"]) == []


def test_attempt_with_an_unscorable_call_from_another_snapshot_is_rejected(tmp_path: Path) -> None:
    cfg = report_config(tmp_path)
    manifest = report_examples()
    attempt = cfg.output_dir / "qwen_logit" / "attempt-1"
    rows = [report_prediction("qwen_logit", row) for row in manifest]
    rows[0] = replace(rows[0], model_resolved="snapshot-2", error="JevResponseError: bad answers")
    write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in rows])
    with pytest.raises(ValueError, match="single-snapshot"):
        _select_attempt(cfg, "qwen_logit", manifest)
    rows[0] = replace(rows[0], model_resolved="unknown", error="TimeoutError")
    write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in rows])
    assert _select_attempt(cfg, "qwen_logit", manifest)[0] == attempt


def test_runner_stops_when_an_unscorable_call_reveals_a_new_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.adapters.jev_openrouter import JevResponseError
    from jev_benchmarks.v2_runner import run_v2_backend

    cfg = report_config(tmp_path)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in report_examples()])
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)

    class Jev:
        calls = 0

        def predict(self, experiment_id, row):
            Jev.calls += 1
            if Jev.calls == 1:
                return report_prediction("jev_openrouter", row)
            raise JevResponseError("KeyError: 'answers'", "snapshot-2")

        def close(self):
            return None

    with pytest.raises(RuntimeError, match="snapshot changed"):
        run_v2_backend(cfg, "jev_openrouter", split="calibration", backend_factory=lambda *_: Jev())
    status = cfg.output_dir / "jev_openrouter" / "attempt-1" / "status.json"
    assert json.loads(status.read_text())["new"] == "snapshot-2"


def test_invalid_prediction_from_a_new_snapshot_still_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.v2_runner import run_v2_backend

    cfg = report_config(tmp_path)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in report_examples()])
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)

    class Jev:
        calls = 0

        def predict(self, experiment_id, row):
            Jev.calls += 1
            good = report_prediction("jev_openrouter", row)
            if Jev.calls == 1:
                return good
            return replace(good, model_resolved="snapshot-2", probabilities=(0.6, 0.45))

        def close(self):
            return None

    with pytest.raises(RuntimeError, match="snapshot changed"):
        run_v2_backend(cfg, "jev_openrouter", split="calibration", backend_factory=lambda *_: Jev())


def test_retried_rows_from_another_snapshot_reject_the_attempt(tmp_path: Path) -> None:
    cfg = report_config(tmp_path)
    manifest = report_examples()
    attempt = cfg.output_dir / "qwen_logit" / "attempt-1"
    rows = [replace(report_prediction("qwen_logit", row), model_resolved="B") for row in manifest]
    earlier = replace(rows[0], model_resolved="A", error="JevResponseError: bad answers")
    write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in [earlier, *rows]])
    with pytest.raises(ValueError, match="single-snapshot"):
        _select_attempt(cfg, "qwen_logit", manifest)


def test_dispatch_lock_refuses_an_overlapping_run(tmp_path: Path) -> None:
    from jev_benchmarks.v2_runner import _dispatch_lock

    with _dispatch_lock(tmp_path / "jev_openrouter"):
        with pytest.raises(RuntimeError, match="dispatch lock"):
            with _dispatch_lock(tmp_path / "jev_openrouter"):
                pass
    with _dispatch_lock(tmp_path / "jev_openrouter"):
        pass


def test_response_without_model_blocks_confirmatory_selection(tmp_path: Path) -> None:
    from jev_benchmarks.adapters.jev_openrouter import MISSING_MODEL

    cfg = report_config(tmp_path)
    manifest = report_examples()
    attempt = cfg.output_dir / "qwen_logit" / "attempt-1"
    rows = [
        replace(report_prediction("qwen_logit", row), model_resolved=MISSING_MODEL, error="x")
        for row in manifest
    ]
    write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in rows])
    with pytest.raises(ValueError, match="single-snapshot"):
        _select_attempt(cfg, "qwen_logit", manifest)


def test_report_and_fewshot_refuse_while_a_run_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.v2_runner import _dispatch_lock

    cfg = report_config(tmp_path)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in report_examples()])
    monkeypatch.setattr("jev_benchmarks.v2_report.verify_frozen", lambda *args: None)
    with _dispatch_lock(cfg.output_dir / "jev_openrouter"):
        with pytest.raises(RuntimeError, match="dispatch lock"):
            build_v2_report(cfg)
    few = fewshot_config(tmp_path / "few", rows(0))
    with _dispatch_lock(few.output_dir / "prior"):
        with pytest.raises(RuntimeError, match="dispatch lock"):
            run_fewshot(few, "prior")
