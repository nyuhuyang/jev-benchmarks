from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jev_benchmarks import probe
from jev_benchmarks.adapters import laya as laya_module
from jev_benchmarks.config import BenchmarkConfig
from jev_benchmarks.models import Example, Prediction


class Tokenizer:
    mask_token = "[MASK]"

    def __call__(self, value, **kwargs):
        # BPE-like: a space joins the next character, as in Qwen (" A" is one token).
        return {"input_ids": re.findall(r" ?\S", value)}

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + messages[1]["content"]


@pytest.fixture(autouse=True)
def fake_laya_renderers(monkeypatch: pytest.MonkeyPatch) -> None:
    class Agent:
        @staticmethod
        def _to_internal(question):
            return {
                "t": question["type"],
                "ins": question["instructions"],
                "crit": question["criteria"],
            }

    common = SimpleNamespace(
        render_options=lambda q: (
            list(q["crit"].values()) if isinstance(q["crit"], dict) else q["crit"]
        ),
        serialize_state=lambda state: json.dumps(state),
    )
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(Agent=Agent, common=common))
    monkeypatch.setitem(sys.modules, "laya.common", common)


def config(tmp_path: Path) -> BenchmarkConfig:
    path = tmp_path / "configs" / "v2.yaml"
    path.parent.mkdir()
    path.write_text("fake: true", encoding="utf-8")
    raw = {
        "schema_version": 2,
        "experiment_id": "probe",
        "seed": 1,
        "output_dir": str(tmp_path / "runs"),
        "dataset": {
            "datasets": [
                {
                    "name": "agnews",
                    "kind": "btzsc",
                    "task": "topic",
                    "repository": "fake",
                    "revision": "rev",
                    "instructions": "Which single label best describes the input text?",
                }
            ]
        },
        "length_rule": {
            "qwen_tokenizer_path": "qwen",
            "laya_tokenizer_paths": {"base": "base", "multilingual": "multi", "typed": "typed"},
            "context_budgets": {
                "jev_openrouter": 32000,
                "laya_base": 512,
                "laya_multilingual": 1024,
                "laya_typed": 1024,
                "qwen_logit": 32768,
            },
        },
        "models": {
            "jev_openrouter": {"model_id": "jev", "datasets": ["agnews"]},
            "laya_base": {
                "model_id": "laya",
                "revision": "rev",
                "local_repo": "local",
                "checkpoint": "english",
                "shipped_head_max_len": 192,
                "datasets": ["agnews"],
                "head_max_len": {"agnews": "TO-FILL-AFTER-PROBE"},
            },
            "laya_multilingual": {
                "model_id": "laya",
                "revision": "rev",
                "local_repo": "local",
                "checkpoint": "multilingual",
                "shipped_head_max_len": 256,
                "datasets": [],
                "head_max_len": {},
            },
            "laya_typed": {
                "model_id": "laya",
                "revision": "rev",
                "local_repo": "local",
                "checkpoint": "typed-decisions",
                "shipped_head_max_len": 256,
                "datasets": [],
                "head_max_len": {},
            },
            "qwen_logit": {
                "model_id": "qwen",
                "revision": "rev",
                "local_path": "local",
                "datasets": ["agnews"],
                "system_prompt": "Choose",
                "user_template": "{text} {options}",
            },
        },
        "questions": {"choice": "Choose", "noul": "True?", "score": "Rate"},
    }
    return BenchmarkConfig(raw, path)


def test_option_head_and_rendered_qwen_suffixes() -> None:
    row = Example("agnews", "topic", "id", "synthetic", "h", ("a", "b"), 0)
    assert probe._option_head(row, Tokenizer(), "Choose") is not None
    assert (
        probe._option_head(
            Example("d", "t", "id", "x", "h", ("a" * 100, "b"), 0), Tokenizer(), "Choose"
        )
        is None
    )
    mode = probe._qwen_mode(
        row, Tokenizer(), {"system_prompt": "s", "user_template": "{text} {options}"}
    )
    assert mode["mode"] == "letter"
    large = Example("d", "t", "id", "x", "h", tuple(str(i) for i in range(30)), 0)
    assert (
        probe._qwen_mode(
            large, Tokenizer(), {"system_prompt": "s", "user_template": "{text} {options}"}
        )["mode"]
        == "two_digit_joint"
    )

    class MergingTokenizer(Tokenizer):
        def __call__(self, value, **kwargs):
            ids = super().__call__(value)["input_ids"]
            if value.endswith(" A"):
                ids[-2:] = [ord("A")]
            return {"input_ids": ids}

    assert (
        probe._qwen_mode(
            row,
            MergingTokenizer(),
            {"system_prompt": "s", "user_template": "{text} {options}"},
        )["mode"]
        == "full_sequence"
    )


def test_probe_static_and_synthetic_are_dataset_text_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    probe_path = tmp_path / "configs" / "probe-v2.yaml"
    probe_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "configs/v2.yaml",
                "model_revisions": {"laya": "rev", "qwen": "rev"},
                "synthetic_items": {
                    "choice": "Invented parcel text.",
                    "noul": "Invented statement.",
                    "score": "Invented review.",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "_probe_tagged", lambda path: None)
    monkeypatch.setattr(probe, "_laya_source_audit", lambda: {"manual_review_required": True})
    monkeypatch.setattr(probe, "load_config", lambda path: cfg)

    class Rows:
        def __len__(self):
            return 4

        def __getitem__(self, key):
            rows = [
                {"text": "PRIVATE BENCHMARK TEXT", "hypothesis": "a", "labels": 1},
                {"text": "PRIVATE BENCHMARK TEXT", "hypothesis": "b", "labels": 0},
                {"text": "other", "hypothesis": "a", "labels": 0},
                {"text": "other", "hypothesis": "b", "labels": 1},
            ]
            return [row[key] for row in rows] if isinstance(key, str) else rows[key]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: Rows())
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())),
    )
    output = probe.run_probe(probe_path, static_only=True)
    result = json.loads(output.read_text())
    assert result["datasets"]["agnews"]["qwen"]["mode"] == "letter"
    assert result["datasets"]["agnews"]["laya_head_max_len"]["laya_base"] > 0
    assert "PRIVATE BENCHMARK TEXT" not in output.read_text()
    assert result["synthetic_calls"] == {}

    class FakeLaya(laya_module.LayaBackend):
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, experiment_id, example):
            assert example.text == "Invented parcel text."
            return Prediction(
                "e",
                "laya",
                "m",
                "s",
                example.dataset,
                example.example_id,
                0,
                0,
                example.labels,
                (0.9, 0.1),
                0.1,
                device="cpu",
            )

        def close(self):
            return None

    class FakeQwen(FakeLaya):
        pass

    monkeypatch.setattr(laya_module, "LayaBackend", FakeLaya)
    monkeypatch.setitem(
        sys.modules,
        "jev_benchmarks.adapters.qwen_logit",
        SimpleNamespace(QwenLogitBackend=FakeQwen),
    )
    result = json.loads(probe.run_probe(probe_path).read_text())
    assert result["synthetic_calls"]["laya_base"]["agnews"]["vector_length"] == 2
    assert result["synthetic_calls"]["qwen_logit_duplicate"]["same_vector"] is True


def test_probe_revision_guard_and_source_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(tmp_path)
    path = tmp_path / "configs" / "probe-v2.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_config": "configs/v2.yaml",
                "model_revisions": {"laya": "different", "qwen": "rev"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "_probe_tagged", lambda path: None)
    monkeypatch.setattr(probe, "load_config", lambda path: cfg)
    with pytest.raises(ValueError, match="revisions differ"):
        probe.run_probe(path)
    source = tmp_path / "agent.py"
    source.write_text("allow_patterns = ['model.safetensors']", encoding="utf-8")
    monkeypatch.setattr(
        probe.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(source))
    )
    assert probe._laya_source_audit()["remote_python_execution_marker"] is False
