from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .config import BenchmarkConfig, load_config
from .data import _v2_candidates, _v2_rows, make_length_counters
from .io import write_json
from .models import Example


def _probe_tagged(path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    tagged = subprocess.run(
        ["git", "show", f"v2-probe:{path.resolve().relative_to(root)}"],
        check=True,
        capture_output=True,
    ).stdout
    if tagged != path.read_bytes():
        raise RuntimeError("probe config differs from v2-probe tag")


def _laya_source_audit() -> dict[str, Any]:
    spec = importlib.util.find_spec("laya.agent")
    if spec is None or spec.origin is None:
        raise RuntimeError("pinned laya package source is unavailable")
    source = Path(spec.origin).read_text(encoding="utf-8")
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "checkpoint_patterns": [
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        ],
        "remote_python_execution_marker": "trust_remote_code=True" in source,
        "manual_review_required": True,
    }


def _option_head(
    example: Example,
    tokenizer: Any,
    instructions: str,
    to_internal: Any = None,
) -> int | None:
    from .adapters.laya import LayaBackend, laya_request_tokens

    if to_internal is None:
        from importlib import import_module

        to_internal = import_module("laya").Agent._to_internal
    question = LayaBackend.question_for(replace(example, instructions=instructions), {})
    sizes, instruction_length, _, _ = laya_request_tokens(
        question, example.text, tokenizer, to_internal
    )
    if any(length > 49 for length in sizes):
        return None
    return sum(sizes) + instruction_length


def _qwen_mode(example: Example, tokenizer: Any, model: dict[str, Any]) -> dict[str, Any]:
    ids = (
        [chr(65 + index) for index in range(len(example.labels))]
        if len(example.labels) <= 26
        else [f"{index + 1:02d}" for index in range(len(example.labels))]
    )
    options = "\n".join(
        f"{label_id}) {label}" for label_id, label in zip(ids, example.labels, strict=True)
    )
    user = model["user_template"].format(
        text="Synthetic text only.", options=options, question=example.instructions
    )
    prompt = (
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": model["system_prompt"]},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "Answer: "
    )
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    suffixes = []
    compatible = True
    for label_id in ids:
        complete = tokenizer(prompt + label_id, add_special_tokens=False)["input_ids"]
        if complete[: len(prefix)] != prefix:
            compatible = False
        common = 0
        while common < min(len(prefix), len(complete)) and prefix[common] == complete[common]:
            common += 1
        suffixes.append(complete[common:])
    sizes = {len(suffix) for suffix in suffixes}
    mode = (
        "letter"
        if compatible and len(example.labels) <= 26 and sizes == {1}
        else "two_digit_joint"
        if compatible and len(example.labels) > 26 and sizes == {2}
        else "full_sequence"
    )
    return {
        "mode": mode,
        "suffix_lengths": [len(suffix) for suffix in suffixes],
        "suffix_sha256": hashlib.sha256(json.dumps(suffixes).encode()).hexdigest(),
    }


def run_probe(path: Path, *, static_only: bool = False) -> Path:
    """Run only pinned synthetic calls; no benchmark text is sent to a model."""
    _probe_tagged(path)
    probe = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_path = path.parent.parent / probe["base_config"]
    _probe_tagged(base_path)
    config = load_config(base_path)
    if config.raw.get("schema_version") != 2:
        raise ValueError("probe base must be a v2 config")
    if "datasets" in probe:
        by_name = {row["name"]: row for row in probe["datasets"]}
        for row in config.raw["dataset"]["datasets"]:
            if by_name[row["name"]]["instructions"] != row["instructions"]:
                raise ValueError("probe dataset instructions differ from v2 config")
    for name in ("laya_base", "laya_multilingual", "laya_typed", "qwen_logit"):
        if str(config.raw["models"][name]["revision"]).startswith("TODO-PIN"):
            raise ValueError(f"{name} revision needs TODO-PIN resolution")
    if (
        probe["model_revisions"]["laya"] != config.raw["models"]["laya_base"]["revision"]
        or probe["model_revisions"]["qwen"] != config.raw["models"]["qwen_logit"]["revision"]
    ):
        raise ValueError("probe model revisions differ from frozen v2 config")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer_paths = config.raw["length_rule"]
    if str(tokenizer_paths["qwen_tokenizer_path"]).startswith("TODO-PIN"):
        raise ValueError("tokenizer paths need TODO-PIN resolution")
    qwen_tok = AutoTokenizer.from_pretrained(
        tokenizer_paths["qwen_tokenizer_path"], local_files_only=True, trust_remote_code=False
    )
    laya_toks = {
        backend: AutoTokenizer.from_pretrained(
            tokenizer_paths["laya_tokenizer_paths"][key],
            local_files_only=True,
            trust_remote_code=False,
        )
        for backend, key in (
            ("laya_base", "base"),
            ("laya_multilingual", "multilingual"),
            ("laya_typed", "typed"),
        )
    }
    result: dict[str, Any] = {
        "probe_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "laya_package_source": _laya_source_audit(),
        "datasets": {},
        "synthetic_calls": {},
    }
    synthetic: dict[str, Example] = {}
    source_examples: dict[str, list[Example]] = {}
    for spec in config.raw["dataset"]["datasets"]:
        rows = _v2_rows(load_dataset, spec, str(config.path.parent.parent / "results" / "cache"))
        candidates = _v2_candidates(rows, spec)
        source_examples[spec["name"]] = candidates
        if not candidates:
            raise ValueError(f"{spec['name']} has no eligible source rows")
        sample = candidates[0]
        synthetic_text = probe["synthetic_items"][sample.question_type]
        synthetic[spec["name"]] = Example(
            sample.dataset,
            sample.task,
            f"synthetic:{sample.dataset}",
            synthetic_text,
            hashlib.sha256(synthetic_text.encode()).hexdigest(),
            sample.labels,
            0,
            question_type=sample.question_type,
            locale=sample.locale,
            instructions=sample.instructions,
        )
        dataset_result: dict[str, Any] = {
            "source_rows": len(candidates),
            "qwen": _qwen_mode(sample, qwen_tok, config.raw["models"]["qwen_logit"]),
            "laya_head_max_len": {},
        }
        for backend, tokenizer in laya_toks.items():
            if sample.dataset in config.raw["models"][backend]["datasets"]:
                minimum = _option_head(sample, tokenizer, sample.instructions)
                budget = config.raw["length_rule"]["context_budgets"][backend]
                dataset_result["laya_head_max_len"][backend] = (
                    minimum if minimum is not None and minimum < budget else None
                )
        result["datasets"][sample.dataset] = dataset_result
    adjusted = deepcopy(config.raw)
    for backend in laya_toks:
        assigned = []
        for name in adjusted["models"][backend]["datasets"]:
            minimum = result["datasets"][name]["laya_head_max_len"].get(backend)
            if minimum is not None:
                adjusted["models"][backend]["head_max_len"][name] = minimum
                assigned.append(name)
        adjusted["models"][backend]["datasets"] = assigned
    counts = make_length_counters(BenchmarkConfig(adjusted, config.path))
    for name, candidates in source_examples.items():
        budgets = adjusted["length_rule"]["context_budgets"]
        excluded = {backend: 0 for backend in counts[name]}
        common = 0
        for candidate in candidates:
            over = [
                backend
                for backend, counter in counts[name].items()
                if counter(candidate) > budgets[backend]
            ]
            if any(backend != "gliner" for backend in over):
                common += 1
                for backend in over:
                    excluded[backend] += 1
        result["datasets"][name]["full_request_length_excluded"] = common
        result["datasets"][name]["length_excluded_by_backend"] = excluded
        result["datasets"][name]["excluded_datasets"] = (
            ["gliner"] if excluded.get("gliner", 0) else []
        )
    if not static_only:
        from .adapters.laya import LayaBackend

        for backend in laya_toks:
            model = config.raw["models"][backend]
            if not model["datasets"]:
                continue
            heads = {
                name: result["datasets"][name]["laya_head_max_len"].get(backend)
                for name in model["datasets"]
            }
            excluded = [name for name, value in heads.items() if value is None]
            heads = {name: value for name, value in heads.items() if value is not None}
            if not heads:
                result["synthetic_calls"][backend] = {"excluded_datasets": excluded}
                continue
            adapter = LayaBackend(
                model["model_id"],
                model["revision"],
                model["local_repo"],
                model["checkpoint"],
                heads,
                config.raw.get("questions", {}),
            )
            try:
                result["synthetic_calls"][backend] = {
                    name: {
                        "vector_length": len(prediction.probabilities),
                        "expected_score_only": prediction.expected_score is not None
                        and not prediction.probabilities,
                        "device": prediction.device,
                    }
                    for name in heads
                    for prediction in [adapter.predict(config.experiment_id, synthetic[name])]
                }
                result["synthetic_calls"][backend]["excluded_datasets"] = excluded
            finally:
                adapter.close()
        from .adapters.qwen_logit import QwenLogitBackend

        qwen = config.raw["models"]["qwen_logit"]
        score_modes = {name: value["qwen"]["mode"] for name, value in result["datasets"].items()}
        adapter = QwenLogitBackend(
            qwen["model_id"],
            qwen["revision"],
            qwen["local_path"],
            qwen["system_prompt"],
            qwen["user_template"],
            score_modes,
        )
        try:
            sample = synthetic["agnews"]
            first = adapter.predict(config.experiment_id, sample)
            second = adapter.predict(config.experiment_id, sample)
            result["synthetic_calls"]["qwen_logit_duplicate"] = {
                "same_vector": first.probabilities == second.probabilities,
                "device": first.device,
            }
        finally:
            adapter.close()
    output = config.output_dir / "probe-results.json"
    write_json(output, result)
    return output
