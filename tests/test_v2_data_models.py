from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jev_benchmarks.config import BenchmarkConfig, load_config
from jev_benchmarks.data import (
    _split_candidates,
    _v2_candidates,
    build_derived_suites,
    length_exclusion,
    load_v2_examples,
    make_length_counters,
    normalized_text_hash,
    prepare_manifest,
)
from jev_benchmarks.io import read_jsonl
from jev_benchmarks.models import Example, Prediction
from jev_benchmarks.runner import _validate_prediction


def example(index: int, text: str | None = None, dataset: str = "agnews") -> Example:
    text = text or f"sample {index}"
    return Example(
        dataset,
        "topic",
        f"{dataset}:{index}",
        text,
        hashlib.sha256(text.encode()).hexdigest(),
        ("a", "b"),
        index % 2,
    )


def config(tmp_path: Path) -> BenchmarkConfig:
    raw = {
        "schema_version": 2,
        "experiment_id": "v2fake",
        "seed": 20260923,
        "output_dir": str(tmp_path / "runs"),
        "report_dir": str(tmp_path / "reports"),
        "dataset": {
            "split_counts": {
                "default": {"pilot": 1, "calibration": 2, "test": 3},
                "massive": {"pilot": 1, "calibration": 2, "test": 2},
            },
            "permutation_items": 2,
            "latency_items": 2,
            "datasets": [
                {
                    "name": "agnews",
                    "kind": "btzsc",
                    "task": "topic",
                    "repository": "fake",
                    "revision": "abc",
                    "configuration": "agnews",
                    "source_split": "test",
                },
                {
                    "name": "sms_spam",
                    "kind": "sms",
                    "task": "spam",
                    "repository": "fake",
                    "revision": "abc",
                    "source_split": "train",
                    "labels": ["false", "true"],
                },
                {
                    "name": "massive_en",
                    "kind": "massive",
                    "task": "intent",
                    "repository": "fake",
                    "revision": "abc",
                    "locale": "en-US",
                    "configuration": "en-US",
                    "source_split": "test",
                    "labels": ["one", "two"],
                },
                {
                    "name": "massive_zh",
                    "kind": "massive",
                    "task": "intent",
                    "repository": "fake",
                    "revision": "abc",
                    "locale": "zh-CN",
                    "configuration": "zh-CN",
                    "source_split": "test",
                    "labels": ["one", "two"],
                },
            ],
        },
        "length_rule": {"required": False, "context_budgets": {"jev_openrouter": 1000}},
        "models": {"jev_openrouter": {"model_id": "jev", "datasets": ["agnews"], "repeats": 3}},
        "questions": {"choice": "Choose", "noul": "Is true?", "score": "Rate"},
        "metrics": {"ece_bins": 10, "error_budget": 0.05, "bootstrap_resamples": 10},
    }
    path = tmp_path / "configs" / "v2.yaml"
    path.parent.mkdir()
    for spec in raw["dataset"]["datasets"]:
        spec["instructions"] = "Which single label best describes the input text?"
    path.write_text("schema_version: 2\n", encoding="utf-8")
    return BenchmarkConfig(raw, path)


class FakeBTZSC:
    def __init__(self) -> None:
        self.rows = [
            {"text": f"text {index}", "hypothesis": label, "labels": int(index % 2 == pos)}
            for index in range(10)
            for pos, label in enumerate(("a", "b"))
        ]
        self.rows += [{"text": "oos", "hypothesis": label, "labels": 0} for label in ("a", "b")]

    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]

    def __len__(self):
        return len(self.rows)


def fake_loader(repo, *, name, split, revision, cache_dir):
    assert revision == "abc" and cache_dir
    if name == "agnews":
        return FakeBTZSC()
    if name is None and split == "train":
        return [{"sms": f"sms {index}", "label": index % 2} for index in range(10)]
    return [
        {"id": index, "utt": f"{name} utterance {index}", "intent": index % 2}
        for index in range(10)
    ]


def test_normalization_and_common_length_rule() -> None:
    assert normalized_text_hash("\uff26\uff4f\uff4f,  BAR!") == normalized_text_hash("foo bar")
    candidates = [example(0, "short"), example(1, "much longer")]
    kept, counts = length_exclusion(
        candidates, {"a": lambda row: len(row.text), "b": lambda row: 1}, {"a": 5, "b": 2}
    )
    assert kept == candidates[:1]
    assert counts == {"a": 1, "b": 0}
    with pytest.raises(ValueError, match="counter"):
        length_exclusion(candidates, {"a": lambda row: len(row.text)}, {"b": 1})


def test_shape_dispatch_and_old_jsonl_defaults() -> None:
    old = {
        "dataset": "d",
        "task": "t",
        "example_id": "id",
        "text": "x",
        "text_sha256": "h",
        "labels": ["one", "two"],
        "target_index": 1,
    }
    assert Example.from_dict(old).question_type == "choice"
    scalar = Prediction(
        "e",
        "m",
        "r",
        "s",
        "d",
        "id",
        1,
        1,
        ("1", "2", "3"),
        (),
        0.1,
        question_type="score",
        expected_score=2.5,
    )
    assert _validate_prediction(scalar) == scalar
    with pytest.raises(ValueError, match="expected_score"):
        _validate_prediction(replace(scalar, expected_score=4))
    vector = replace(scalar, probabilities=(0.2, 0.3, 0.49), expected_score=None)
    validated = _validate_prediction(vector)
    assert validated.raw_probabilities == vector.probabilities
    assert validated.probability_sum_raw == pytest.approx(0.99)
    assert validated.expected_score == pytest.approx((0.2 + 0.6 + 1.47) / 0.99)
    assert Prediction.from_dict(validated.to_dict()) == validated


def test_v2_loaders_split_groups_and_suites(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    # MASSIVE rows use a common cross-locale ID. The helper must assign every selected ID
    # to the same split in both locales.
    rows, summary = load_v2_examples(cfg, loader=fake_loader)
    assert len([row for row in rows if row.dataset == "agnews" and row.split == "test"]) == 3
    assert (
        len([row for row in rows if row.dataset == "sms_spam" and row.split == "calibration"]) == 2
    )
    assert (
        len([row for row in rows if row.dataset == "agnews" and row.split == "permutation"]) == 16
    )
    assert len([row for row in rows if row.dataset == "agnews" and row.split == "latency"]) == 4
    massive = [
        row
        for row in rows
        if row.dataset.startswith("massive") and row.split in {"pilot", "calibration", "test"}
    ]
    by_id: dict[str, set[str]] = {}
    for row in massive:
        by_id.setdefault(row.example_id.split(":")[1], set()).add(row.split)
    assert all(len(value) == 1 for value in by_id.values())
    assert cast(dict[str, Any], summary["agnews"])["source_split"] == "test"
    assert cast(dict[str, Any], summary["agnews"])["merged_groups"] == 0
    assert (
        len(
            {
                row.example_id
                for row in rows
                if row.dataset == "agnews" and row.split in {"pilot", "calibration", "test"}
            }
        )
        == 6
    )


def test_gliner_overflow_excludes_only_gliner_dataset(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cfg.raw["dataset"]["datasets"] = cfg.raw["dataset"]["datasets"][:1]
    cfg.raw["length_rule"]["context_budgets"]["gliner"] = 5
    counters = {"agnews": {"jev_openrouter": lambda row: 1, "gliner": lambda row: 6}}
    rows, summary = load_v2_examples(cfg, loader=fake_loader, counters=counters)
    selected = [row for row in rows if row.split in {"pilot", "calibration", "test"}]
    assert len(selected) == 6
    assert cast(dict[str, Any], summary["agnews"])["excluded_datasets"] == ["gliner"]
    assert cast(dict[str, Any], summary["agnews"])["length_excluded"]["jev_openrouter"] == 0


def test_duplicate_group_never_crosses_and_manifest_refuses_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [example(0, "A!"), example(1, "a"), *[example(index) for index in range(2, 10)]]
    selected, merged = _split_candidates(candidates, {"pilot": 1, "calibration": 2, "test": 3}, 3)
    assert merged == 1
    assert len(selected) == 6
    cfg = config(tmp_path)
    monkeypatch.setattr(
        "jev_benchmarks.data.load_v2_examples",
        lambda config: (selected, {"agnews": {"merged_groups": merged}}),
    )
    path = prepare_manifest(cfg)
    assert len(read_jsonl(path)) == 6
    assert prepare_manifest(cfg) == path
    monkeypatch.setattr("jev_benchmarks.data.load_v2_examples", lambda config: ([example(999)], {}))
    with pytest.raises(RuntimeError, match="overwrite"):
        prepare_manifest(cfg)


def test_high_cardinality_head_default_suite() -> None:
    row = replace(
        example(0, dataset="banking77"),
        labels=tuple(f"label {index}" for index in range(21)),
        split="test",
    )
    suite = build_derived_suites([row], 20260923, permutation_count=1, latency_count=0)
    assert sum(item.permutation_id == "head-default" for item in suite) == 1
    assert sum(item.permutation_id.startswith("order-") for item in suite) == 8


def test_dataset_types_and_revision_refusal(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    sms = cfg.raw["dataset"]["datasets"][1]
    assert _v2_candidates([{"sms": "x", "label": 1}], sms)[0].question_type == "noul"
    civil = {**sms, "kind": "civil_comments", "labels": ["false", "true"]}
    assert _v2_candidates([{"text": "x", "toxicity": 0.5}], civil)[0].target_index == 1
    yelp = {**sms, "kind": "yelp", "labels": ["1", "2", "3", "4", "5"]}
    assert _v2_candidates([{"text": "x", "label": 4}], yelp)[0].question_type == "score"
    bad = {**sms, "revision": "TODO-PIN-SMS"}
    cfg.raw["dataset"]["datasets"] = [bad]
    with pytest.raises(ValueError, match="TODO-PIN"):
        load_v2_examples(cfg, loader=fake_loader)
    with pytest.raises(ValueError, match="unknown dataset kind"):
        _v2_candidates([{"text": "x"}], {**sms, "kind": "other"})


def test_length_counter_factory_uses_full_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(tmp_path)
    cfg.raw["length_rule"].update(
        {
            "qwen_tokenizer_path": "qwen",
            "laya_tokenizer_paths": {"base": "laya", "multilingual": "laya", "typed": "laya"},
        }
    )
    cfg.raw["models"]["laya_base"] = {
        "model_id": "laya",
        "datasets": ["agnews"],
        "head_max_len": {"agnews": 200},
    }
    cfg.raw["models"]["qwen_logit"] = {
        "model_id": "qwen",
        "datasets": ["agnews"],
        "system_prompt": "Choose",
        "user_template": "{text}\n{options}",
    }

    class Tokenizer:
        mask_token = "[MASK]"

        def __call__(self, value, **kwargs):
            return {"input_ids": value.split()}

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"] + messages[1]["content"]

    common = SimpleNamespace(
        render_options=lambda q: (
            list(q["crit"].values()) if isinstance(q["crit"], dict) else q["crit"]
        ),
        serialize_state=lambda state: json.dumps(state),
    )
    monkeypatch.setitem(sys.modules, "laya.common", common)

    def to_internal(q):
        return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}

    counters = make_length_counters(
        cfg, tokenizer_loader=lambda path: Tokenizer(), laya_to_internal=to_internal
    )
    assert set(counters["agnews"]) == {"jev_openrouter", "laya_base", "qwen_logit"}
    assert all(counter(example(0)) > 0 for counter in counters["agnews"].values())
    cfg.raw["models"]["laya_base"]["head_max_len"]["agnews"] = "TO-FILL-AFTER-PROBE"
    with pytest.raises(ValueError, match="head_max_len"):
        make_length_counters(
            cfg, tokenizer_loader=lambda path: Tokenizer(), laya_to_internal=to_internal
        )


def test_laya_prepare_and_runtime_share_borderline_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_benchmarks.adapters.laya import LayaBackend, laya_request_tokens

    cfg = config(tmp_path)
    cfg.raw["length_rule"].update(
        {
            "qwen_tokenizer_path": "qwen",
            "laya_tokenizer_paths": {"base": "laya", "multilingual": "laya", "typed": "laya"},
            "context_budgets": {"jev_openrouter": 1000, "laya_base": 100},
        }
    )
    cfg.raw["models"]["laya_base"] = {
        "model_id": "laya",
        "datasets": ["agnews"],
        "head_max_len": {"agnews": 100},
    }

    class Tokenizer:
        mask_token = "[MASK]"

        def __call__(self, value, **kwargs):
            return {"input_ids": value.split()}

    def to_internal(q):
        return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}

    common = SimpleNamespace(
        render_options=lambda q: list(q["crit"].values()),
        serialize_state=lambda state: json.dumps(state),
    )
    monkeypatch.setitem(sys.modules, "laya.common", common)
    row = replace(example(0), instructions="Which single label best describes the input text?")
    tok = Tokenizer()
    question = LayaBackend.question_for(row, {})
    sizes, instruction_tokens, _, total = laya_request_tokens(question, row.text, tok, to_internal)
    head = sum(sizes) + instruction_tokens
    cfg.raw["models"]["laya_base"]["head_max_len"]["agnews"] = head
    counter = make_length_counters(
        cfg, tokenizer_loader=lambda path: tok, laya_to_internal=to_internal
    )["agnews"]["laya_base"]
    agent = SimpleNamespace(
        tok=tok,
        cfg={"max_len": 100},
        device="cpu",
        _to_internal=to_internal,
    )
    backend = LayaBackend("m", "r", "p", "english", {"agnews": head}, {}, agent=agent)
    assert counter(row) == total
    backend._assert_untruncated(row, question)
    cfg.raw["models"]["laya_base"]["head_max_len"]["agnews"] = head - 1
    excluded = make_length_counters(
        cfg, tokenizer_loader=lambda path: tok, laya_to_internal=to_internal
    )["agnews"]["laya_base"]
    backend.head_max_len["agnews"] = head - 1
    assert excluded(row) == 101
    with pytest.raises(ValueError, match="head"):
        backend._assert_untruncated(row, question)


def test_v2_config_loads_template_and_rejects_bad_split(tmp_path: Path) -> None:
    import yaml

    cfg = config(tmp_path)
    cfg.path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    assert load_config(cfg.path).raw["schema_version"] == 2
    cfg.raw["dataset"]["split_counts"]["default"]["pilot"] = 0
    cfg.path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    with pytest.raises(ValueError, match="split counts"):
        load_config(cfg.path)


def test_ultrafeedback_candidates_group_completions_and_skip_non_numeric() -> None:
    from jev_benchmarks.data import _split_candidates, ultrafeedback_candidates

    def completion(response: str, rating: str) -> dict[str, object]:
        return {"response": response, "annotations": {"helpfulness": {"Rating": rating}}}

    rows = [
        {
            "instruction": f"Q{index}?",
            "completions": [
                completion(f"a{index}", "5"),
                completion(f"b{index}", "N/A"),
                completion(f"c{index}", str(1 + index % 5)),
            ],
        }
        for index in range(40)
    ]
    spec = {
        "name": "uf",
        "task": "helpfulness_score",
        "labels": ["1", "2", "3", "4", "5"],
        "instructions": "How helpful?",
    }
    examples, groups = ultrafeedback_candidates(rows, spec)
    assert len(examples) == 80  # the N/A completion of every prompt is dropped
    assert all(row.question_type == "score" and 0 <= row.target_index < 5 for row in examples)
    assert examples[0].text.startswith("Question: Q0?\n\nResponse: a0")
    selected, merged = _split_candidates(
        examples, {"pilot": 5, "calibration": 10, "test": 20}, 7, mass_groups=groups
    )
    assert merged == 40
    by_split: dict[str, set[str]] = {}
    for row in selected:
        by_split.setdefault(row.split, set()).add(groups[row.example_id])
    assert not by_split["pilot"] & by_split["test"]
    assert not by_split["calibration"] & by_split["test"]
    assert len({groups[row.example_id] for row in selected}) == len(selected)
