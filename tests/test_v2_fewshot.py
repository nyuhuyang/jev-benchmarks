from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
        if backend == "jev_openrouter":  # one test row retried, one failed after 3 attempts
            predictions[12] = replace(predictions[12], dispatch_attempts=2)
            predictions[15] = replace(
                predictions[15],
                error="RuntimeError: HTTP 503",
                probabilities=(),
                dispatch_attempts=3,
            )
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
    # Latency lives only in latency.csv (Amendment 6), never in metric rows.
    assert prior_rows and jev_rows and not any("latency" in key for key in metrics[0])
    payload = json.loads(json_path.read_text())
    few = [row for row in payload["headline"] if row["right"] == "prior"]
    assert {(row["set"], row["metric"], row["condition"]) for row in few} == {
        (set_name, metric, condition)
        for set_name in ("primary", "secondary")
        for metric, condition in (
            ("accuracy", "A_raw"),
            ("brier", "A_raw"),
            ("brier", "B_scaled"),
            ("test_coverage", "A_raw"),
            ("test_coverage", "B_scaled"),
        )
    }
    # Primary-set zero-shot accuracy/Brier rows are the C1 estimates, carrying the Holm decision.
    confirmatory = {row["id"]: row for row in payload["confirmatory"]}
    reused = [row for row in payload["headline"] if row.get("reuses")]
    assert {row["reuses"] for row in reused} == {"C1-accuracy", "C1-brier-A", "C1-brier-B"}
    for row in reused:
        source = confirmatory[row["reuses"]]
        assert row["difference"] == source["difference"]
        assert row["holm_reject_005"] == source["holm_reject_005"] and row["holm_k"] == 3
    with (json_path.parent / "latency.csv").open() as handle:
        latency = list(csv.DictReader(handle))
    assert {row["backend"] for row in latency} == {"jev_openrouter", "qwen_logit"}
    assert {row["workload"] for row in latency} == {"test-reference"}
    jev_latency = next(row for row in latency if row["backend"] == "jev_openrouter")
    assert float(jev_latency["retried_share"]) == pytest.approx(0.5)
    coverage = [row for row in payload["headline"] if row["metric"] == "test_coverage"]
    assert coverage and all(
        {"left_coverage", "right_coverage", "left_selective_error", "right_no_feasible"} <= set(row)
        for row in coverage
    )
    assert "Coverage detail" in json_path.with_suffix(".md").read_text()
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


def test_coverage_bootstrap_counts_no_feasible_threshold_as_zero() -> None:
    from jev_benchmarks.v2_metrics import joint_paired_bootstrap

    manifest = report_examples()
    confident = [report_prediction("a", row) for row in manifest]
    unsure = [
        replace(report_prediction("b", row), probabilities=(0.5, 0.5), predicted_index=0)
        for row in manifest
    ]
    split = {
        "cal": {"agnews": [row for row in confident if row.split == "calibration"]},
        "test": {"agnews": [row for row in confident if row.split == "test"]},
    }
    weak = {
        "cal": {"agnews": [row for row in unsure if row.split == "calibration"]},
        "test": {"agnews": [row for row in unsure if row.split == "test"]},
    }
    result = joint_paired_bootstrap(
        split["test"],
        weak["test"],
        split["cal"],
        weak["cal"],
        metric="test_coverage",
        condition="A_raw",
        resamples=20,
        seed=1,
    )
    assert result["difference"] == pytest.approx(-1.0)


def test_jev_prediction_records_dispatch_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    from jev_benchmarks.adapters.jev_openrouter import JevOpenRouterBackend

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    replies = [
        (429, {}, {"error": "limited"}),
        (
            200,
            {},
            {
                "model": "snap",
                "usage": {"cost": 0.001},
                "answers": {"label": {"probabilities": {"label_000": 0.2, "label_001": 0.8}}},
            },
        ),
    ]
    jev = JevOpenRouterBackend("m", {}, transport=lambda *_: replies.pop(0), sleep=lambda _: None)
    row = replace(report_examples()[0], instructions="Pick?")
    assert jev.predict("run", row).dispatch_attempts == 2


def test_all_failure_resamples_are_defined_or_skipped() -> None:
    from jev_benchmarks.v2_metrics import joint_paired_bootstrap, score_v2

    manifest = report_examples()
    good = [report_prediction("a", row) for row in manifest]
    failed = [replace(row, error="x", probabilities=()) for row in good]
    empty = score_v2([row for row in failed if row.split == "test"], threshold=0.5)
    assert empty["test_coverage"] == 0.0 and empty["ece"] is None and empty["brier"] == 2.0
    mostly_failed = [row if row.example_id == "id:0" else failed[i] for i, row in enumerate(good)]
    split = lambda rows, name: {"agnews": [row for row in rows if row.split == name]}  # noqa: E731
    result = joint_paired_bootstrap(
        split(good, "test"),
        split(mostly_failed, "test"),
        split(good, "calibration"),
        split(good, "calibration"),
        metric="ece",
        condition="A_raw",
        resamples=50,
        seed=1,
    )
    assert 0 < result["resamples_used"] < 50


def test_newest_attempt_is_chosen_numerically(tmp_path: Path) -> None:
    cfg = report_config(tmp_path)
    manifest = report_examples()
    for number, snapshot in ((9, "old"), (10, "new")):
        rows = [
            replace(report_prediction("qwen_logit", row), model_resolved=snapshot)
            for row in manifest
        ]
        write_jsonl(
            cfg.output_dir / "qwen_logit" / f"attempt-{number}" / "predictions.jsonl",
            [row.to_dict() for row in rows],
        )
    assert _select_attempt(cfg, "qwen_logit", manifest)[0].name == "attempt-10"


def test_failed_jev_call_keeps_its_attempt_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.v2_runner import run_v2_backend

    cfg = report_config(tmp_path)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in report_examples()])
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)

    class Jev:
        last_dispatch_attempts = 3

        def predict(self, experiment_id, row):
            raise RuntimeError("OpenRouter HTTP 503 after retries")

        def close(self):
            return None

    output = run_v2_backend(
        cfg, "jev_openrouter", split="calibration", backend_factory=lambda *_: Jev()
    )
    assert {row["dispatch_attempts"] for row in read_jsonl(output)} == {3}


def test_score_dataset_with_no_successful_calls_takes_vector_penalties() -> None:
    from jev_benchmarks.v2_metrics import score_v2

    rows = [
        replace(report_prediction("a", row), question_type="score", error="x", probabilities=())
        for row in report_examples()
    ]
    scores = score_v2(rows, threshold=0.5)
    assert scores["brier"] == 2.0 and scores["test_coverage"] == 0.0


def test_dispatch_history_spans_earlier_failed_invocations(tmp_path: Path) -> None:
    cfg = report_config(tmp_path)
    manifest = report_examples()
    rows = [report_prediction("qwen_logit", row) for row in manifest]
    failed = replace(rows[4], error="RuntimeError: HTTP 503", probabilities=(), dispatch_attempts=3)
    retried = replace(rows[4], dispatch_attempts=1)
    write_jsonl(
        cfg.output_dir / "qwen_logit" / "attempt-1" / "predictions.jsonl",
        [row.to_dict() for row in [*rows[:4], failed, retried, *rows[5:]]],
    )
    _, selected, _ = _select_attempt(cfg, "qwen_logit", manifest)
    assert (
        next(
            row for row in selected if row.example_id == rows[4].example_id and row.split == "test"
        ).dispatch_attempts
        == 4
    )


def test_condition_b_draws_without_a_fittable_temperature_are_kept() -> None:
    from jev_benchmarks.v2_metrics import joint_paired_bootstrap

    manifest = report_examples()
    good = [report_prediction("a", row) for row in manifest]
    sparse = [
        row
        if row.split == "test" or row.example_id == "id:0"
        else replace(row, error="x", probabilities=())
        for row in good
    ]
    split = lambda rows, name: {"agnews": [row for row in rows if row.split == name]}  # noqa: E731
    result = joint_paired_bootstrap(
        split(good, "test"),
        split(sparse, "test"),
        split(good, "calibration"),
        split(sparse, "calibration"),
        metric="brier",
        condition="B_scaled",
        resamples=50,
        seed=1,
    )
    assert result["resamples_used"] == 50


def test_complete_all_failure_local_attempt_is_scored_but_jev_is_not(tmp_path: Path) -> None:
    cfg = report_config(tmp_path)
    manifest = report_examples()
    for backend, repeats in (("qwen_logit", 1), ("jev_openrouter", 3)):
        rows = [
            replace(report_prediction(backend, row, repeat), model_resolved="unknown", error="x")
            for row in manifest
            for repeat in range(repeats)
        ]
        write_jsonl(
            cfg.output_dir / backend / "attempt-1" / "predictions.jsonl",
            [row.to_dict() for row in rows],
        )
    assert _select_attempt(cfg, "qwen_logit", manifest)[0].name == "attempt-1"
    with pytest.raises(ValueError, match="single-snapshot"):
        _select_attempt(cfg, "jev_openrouter", manifest)


def test_unavailable_condition_b_is_reported_not_raised() -> None:
    from jev_benchmarks.v2_metrics import joint_paired_bootstrap

    manifest = report_examples()
    good = [report_prediction("a", row) for row in manifest]
    no_cal = [
        row if row.split == "test" else replace(row, error="x", probabilities=()) for row in good
    ]
    split = lambda rows, name: {"agnews": [row for row in rows if row.split == name]}  # noqa: E731
    result = joint_paired_bootstrap(
        split(good, "test"),
        split(no_cal, "test"),
        split(good, "calibration"),
        split(no_cal, "calibration"),
        metric="brier",
        condition="B_scaled",
        resamples=10,
        seed=1,
    )
    assert result["difference"] is None and result["p_two_sided"] is None
    assert "unavailable" in result


def test_runner_times_every_call_at_its_own_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.v2_runner import run_v2_backend

    cfg = report_config(tmp_path)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in report_examples()])
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)

    class Backend:
        def predict(self, experiment_id, row):
            return replace(report_prediction("jev_openrouter", row), latency_seconds=99.0)

        def close(self):
            return None

    output = run_v2_backend(
        cfg, "jev_openrouter", split="calibration", backend_factory=lambda *_: Backend()
    )
    assert all(row["latency_seconds"] < 99.0 for row in read_jsonl(output))


def test_latency_reference_row_survives_when_every_call_was_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = report_config(tmp_path)
    protocol = tmp_path / "docs" / "PROTOCOL-v2.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol", encoding="utf-8")
    manifest = report_examples()
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in manifest])
    monkeypatch.setattr("jev_benchmarks.v2_report.verify_frozen", lambda *args: None)
    for backend, model in cfg.raw["models"].items():
        predictions = [
            replace(report_prediction(backend, row, repeat), dispatch_attempts=2)
            if backend == "jev_openrouter"
            else report_prediction(backend, row, repeat)
            for row in manifest
            for repeat in range(model["repeats"])
        ]
        attempt = cfg.output_dir / backend / "attempt-1"
        write_jsonl(attempt / "predictions.jsonl", [row.to_dict() for row in predictions])
    json_path, _ = build_v2_report(cfg)
    with (json_path.parent / "latency.csv").open() as handle:
        jev = next(row for row in csv.DictReader(handle) if row["backend"] == "jev_openrouter")
    assert (
        jev["n"] == "0" and jev["latency_p50_seconds"] == "" and float(jev["retried_share"]) == 1.0
    )


def test_missing_cost_response_keeps_its_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from jev_benchmarks.adapters.jev_openrouter import JevOpenRouterBackend, JevResponseError

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    body = {"model": "snap-b", "usage": {}, "answers": {}}
    jev = JevOpenRouterBackend("m", {}, transport=lambda *_: (200, {}, body))
    with pytest.raises(JevResponseError) as error:
        jev.predict("run", replace(report_examples()[0], instructions="Pick?"))
    assert error.value.model_resolved == "snap-b"


def test_cost_paused_attempt_is_not_reported(tmp_path: Path) -> None:
    cfg = report_config(tmp_path)
    manifest = report_examples()
    attempt = cfg.output_dir / "qwen_logit" / "attempt-1"
    write_jsonl(
        attempt / "predictions.jsonl",
        [report_prediction("qwen_logit", row).to_dict() for row in manifest],
    )
    (attempt / "status.json").write_text('{"state":"cost_paused"}', encoding="utf-8")
    with pytest.raises(ValueError, match="single-snapshot"):
        _select_attempt(cfg, "qwen_logit", manifest)


def test_incomplete_permutations_and_all_failure_latency_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = report_config(tmp_path)
    protocol = tmp_path / "docs" / "PROTOCOL-v2.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol", encoding="utf-8")
    base = report_examples()
    permutations = [
        replace(row, split="permutation", permutation_id=f"order-{i}")
        for row in base[4:6]
        for i in range(4)
    ]
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in base + permutations])
    monkeypatch.setattr("jev_benchmarks.v2_report.verify_frozen", lambda *args: None)
    for backend, model in cfg.raw["models"].items():
        rows = [
            report_prediction(backend, row, repeat)
            for row in base + (permutations[:3] if backend == "jev_openrouter" else [])
            for repeat in range(model["repeats"])
        ]
        if backend == "qwen_logit":  # every local call failed
            rows = [
                replace(row, error="x", probabilities=(), model_resolved="unknown") for row in rows
            ]
        write_jsonl(
            cfg.output_dir / backend / "attempt-1" / "predictions.jsonl",
            [row.to_dict() for row in rows],
        )
    json_path, _ = build_v2_report(cfg)
    with (json_path.parent / "flip.csv").open() as handle:
        flips = {row["backend"]: row for row in csv.DictReader(handle)}
    assert flips["jev_openrouter"]["unavailable"] == "permutation run incomplete: 3/8"
    assert flips["qwen_logit"]["unavailable"] == "permutation run incomplete: 0/8"
    with (json_path.parent / "latency.csv").open() as handle:
        qwen = next(row for row in csv.DictReader(handle) if row["backend"] == "qwen_logit")
    assert qwen["n"] == "0" and qwen["dispatched"] == "4"


def test_cost_pause_outranks_snapshot_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.adapters.jev_openrouter import JevResponseError
    from jev_benchmarks.v2_runner import _attempt_dir, run_v2_backend

    cfg = report_config(tmp_path)
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in report_examples()])
    monkeypatch.setattr("jev_benchmarks.v2_runner.verify_frozen", lambda *args: None)

    class Jev:
        budget = SimpleNamespace(paused=False)
        calls = 0

        def predict(self, experiment_id, row):
            Jev.calls += 1
            if Jev.calls == 1:
                return report_prediction("jev_openrouter", row)
            Jev.budget.paused = True
            raise JevResponseError("response missing a valid usage.cost", "snapshot-2")

        def close(self):
            return None

    with pytest.raises(RuntimeError, match="snapshot changed"):
        run_v2_backend(cfg, "jev_openrouter", split="calibration", backend_factory=lambda *_: Jev())
    with pytest.raises(RuntimeError, match="operator review"):
        _attempt_dir(cfg.output_dir / "jev_openrouter")


def test_prevalence_uses_the_grouped_representative_pool() -> None:
    from jev_benchmarks.data import _prevalence, representative_pool

    base = report_examples()[0]
    rows = [replace(base, example_id=f"a{i}", target_index=0) for i in range(3)] + [
        replace(base, example_id="b0", target_index=1)
    ]
    groups = {"a0": "prompt-a", "a1": "prompt-a", "a2": "prompt-a", "b0": "prompt-b"}
    pool, merged = representative_pool(rows, 1, groups)
    assert merged == 2
    assert _prevalence([row for _, row in pool]) == {"0": 0.5, "1": 0.5}
    assert _prevalence(rows) == {"0": 0.75, "1": 0.25}


def test_manifest_summary_is_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jev_benchmarks import data

    cfg = report_config(tmp_path)
    cfg.raw["schema_version"] = 2
    summaries = iter(
        [{"agnews": {"counts": 1}}, {"agnews": {"counts": 1}}, {"agnews": {"counts": 2}}]
    )
    monkeypatch.setattr(
        data, "load_v2_examples", lambda config: (report_examples(), next(summaries))
    )
    data.prepare_manifest(cfg)
    data.prepare_manifest(cfg)  # identical summary: accepted
    with pytest.raises(RuntimeError, match="different summary"):
        data.prepare_manifest(cfg)


def test_report_orders_like_for_like_first_with_per_dataset_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = report_config(tmp_path)
    protocol = tmp_path / "docs" / "PROTOCOL-v2.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol", encoding="utf-8")
    manifest = report_examples()
    write_jsonl(cfg.output_dir / "manifest.jsonl", [row.to_dict() for row in manifest])
    monkeypatch.setattr("jev_benchmarks.v2_report.verify_frozen", lambda *args: None)
    for backend, model in cfg.raw["models"].items():
        write_jsonl(
            cfg.output_dir / backend / "attempt-1" / "predictions.jsonl",
            [
                report_prediction(backend, row, repeat).to_dict()
                for row in manifest
                for repeat in range(model["repeats"])
            ],
        )
    json_path, md_path = build_v2_report(cfg)
    payload = json.loads(json_path.read_text())
    order = [(row["metric"], row["condition"]) for row in payload["headline"][:3]]
    assert order == [("accuracy", "A_raw"), ("brier", "B_scaled"), ("test_coverage", "B_scaled")]
    # Argmax accuracy is T-invariant: no B-scaled accuracy row for a choice-only set.
    assert ("accuracy", "B_scaled") not in {
        (row["metric"], row["condition"]) for row in payload["headline"]
    }
    parents = {row["parent"] for row in payload["per_dataset"]}
    assert {"C1-accuracy", "C1-brier-A", "C1-brier-B"} <= parents
    ids = [row["id"] for row in payload["per_dataset"]]
    assert len(ids) == len(set(ids))  # reused C1 headline rows are not emitted twice
    text = md_path.read_text()
    assert "class-balanced test items" in text and "Per-dataset paired differences" in text
