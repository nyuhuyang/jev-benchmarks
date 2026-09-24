from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .io import read_jsonl, write_jsonl
from .models import Example


def _class_count(texts: list[str]) -> int:
    first = texts[0]
    for index in range(1, len(texts)):
        if texts[index] != first:
            return index
    raise ValueError("could not infer class count from repeated BTZSC texts")


def _balanced_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    chosen: list[int] = []
    classes = sorted(by_class)
    while len(chosen) < min(limit, len(targets)):
        made_progress = False
        for class_id in classes:
            if by_class[class_id] and len(chosen) < limit:
                chosen.append(by_class[class_id].pop())
                made_progress = True
        if not made_progress:
            break
    return sorted(chosen)


def load_examples(config: BenchmarkConfig) -> list[Example]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "dataset preparation requires the 'data' extra: "
            "install with `pip install 'jev-benchmarks[data]'`"
        ) from exc

    spec = config.raw["dataset"]
    output: list[Example] = []
    for dataset_offset, dataset_spec in enumerate(spec["datasets"]):
        name = dataset_spec["name"]
        rows = load_dataset(
            spec["repository"],
            name=name,
            split="test",
            revision=spec["revision"],
            cache_dir=str(config.output_dir.parent / "cache"),
        )
        binary = [int(value) for value in rows["labels"]]
        texts = [str(value) for value in rows["text"]]
        n_classes = _class_count(texts)
        total = len(rows) // n_classes
        labels = tuple(str(rows[index]["hypothesis"]) for index in range(n_classes))
        valid_sample_indices: list[int] = []
        targets: list[int] = []
        for sample_index in range(total):
            offset = sample_index * n_classes
            values = binary[offset : offset + n_classes]
            if sum(values) == 1:
                valid_sample_indices.append(sample_index)
                targets.append(values.index(1))
        selected_positions = _balanced_indices(
            targets,
            int(spec["samples_per_dataset"]),
            config.seed + dataset_offset,
        )
        for position in selected_positions:
            sample_index = valid_sample_indices[position]
            offset = sample_index * n_classes
            text = texts[offset]
            output.append(
                Example(
                    dataset=name,
                    task=str(dataset_spec["task"]),
                    example_id=f"{name}:{sample_index}",
                    text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    labels=labels,
                    target_index=targets[position],
                )
            )
    return output


def prepare_manifest(config: BenchmarkConfig) -> Path:
    if config.raw.get("schema_version") == 2:
        examples, summary = load_v2_examples(config)
        path = config.output_dir / "manifest.jsonl"
        rows = [example.to_dict() for example in examples]
        existing = read_jsonl(path)
        if existing and existing != rows:
            raise RuntimeError(f"refusing to overwrite a different manifest at {path}")
        if not existing:
            write_jsonl(path, rows)
        from .io import write_json

        write_json(config.output_dir / "manifest-summary.json", summary)
        return path
    examples = load_examples(config)
    path = config.output_dir / "manifest.jsonl"
    rows = [example.to_dict() for example in examples]
    existing = read_jsonl(path)
    if existing and existing != rows:
        raise RuntimeError(
            f"refusing to overwrite a different manifest at {path}; "
            "use a new experiment_id/output_dir"
        )
    if not existing:
        write_jsonl(path, rows)
    return path


def normalized_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = "".join(
        char for char in normalized if not unicodedata.category(char).startswith("P")
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def length_exclusion(
    examples: Sequence[Example],
    counters: Mapping[str, Callable[[Example], int]],
    budgets: Mapping[str, int],
) -> tuple[list[Example], dict[str, int]]:
    """Apply the frozen any-contender full-request length rule before sampling."""
    if set(counters) != set(budgets) or not counters:
        raise ValueError("each assigned contender needs a counter and context budget")
    kept: list[Example] = []
    excluded = {name: 0 for name in counters}
    for example in examples:
        over = [name for name, count in counters.items() if count(example) > budgets[name]]
        for name in over:
            excluded[name] += 1
        if not over:
            kept.append(example)
    return kept, excluded


def _v2_rows(load_dataset: Callable[..., object], spec: dict[str, object], cache: str) -> object:
    revision = str(spec["revision"])
    if revision.startswith("TODO-PIN"):
        raise ValueError(f"dataset {spec['name']} revision needs TODO-PIN resolution")
    extra: dict[str, object] = {}
    if spec.get("data_files"):
        extra["data_files"] = [str(value) for value in spec["data_files"]]  # type: ignore[union-attr]
    return load_dataset(
        str(spec["repository"]),
        name=spec.get("configuration"),
        split=str(spec.get("source_split", "train")),
        revision=revision,
        cache_dir=cache,
        **extra,
    )


def ultrafeedback_candidates(
    rows: Any, spec: dict[str, object]
) -> tuple[list[Example], dict[str, str]]:
    """One score item per (prompt, completion) with a numeric helpfulness rating.

    Completions of one prompt share a group so they can never straddle splits.
    """
    name = str(spec["name"])
    labels = tuple(str(label) for label in spec["labels"])  # type: ignore[union-attr]
    aspect = str(spec.get("aspect", "helpfulness"))
    output: list[Example] = []
    groups: dict[str, str] = {}
    for index, row in enumerate(rows):
        question = str(row["instruction"])
        group = normalized_text_hash(question)
        for position, completion in enumerate(row["completions"]):
            rating = str(completion["annotations"][aspect]["Rating"]).strip()
            if not rating.isdigit() or not 1 <= int(rating) <= len(labels):
                continue
            text = f"Question: {question}\n\nResponse: {completion['response']}"
            example_id = f"{name}:{index}:{position}"
            output.append(
                Example(
                    name,
                    str(spec["task"]),
                    example_id,
                    text,
                    hashlib.sha256(text.encode()).hexdigest(),
                    labels,
                    int(rating) - 1,
                    question_type="score",
                    instructions=str(spec.get("instructions", "")),
                )
            )
            groups[example_id] = group
    return output, groups


def _v2_candidates(rows: Any, spec: dict[str, object]) -> list[Example]:
    name = str(spec["name"])
    kind = str(spec["kind"])
    labels = tuple(str(label) for label in spec.get("labels", ()))  # type: ignore[arg-type]
    locale = spec.get("locale")
    output: list[Example] = []
    if kind == "btzsc":
        texts = [str(value) for value in rows["text"]]  # type: ignore[index]
        binary = [int(value) for value in rows["labels"]]  # type: ignore[index]
        count = _class_count(texts)
        source_labels = tuple(str(rows[index]["hypothesis"]) for index in range(count))  # type: ignore[index]
        if not labels:
            labels = source_labels
        if labels != source_labels:
            raise ValueError(f"{name}: complete BTZSC label universe differs from config")
        for sample in range(len(texts) // count):
            offset = sample * count
            values = binary[offset : offset + count]
            if sum(values) != 1:
                continue
            text = texts[offset]
            output.append(
                Example(
                    name,
                    str(spec["task"]),
                    f"{name}:{sample}",
                    text,
                    hashlib.sha256(text.encode()).hexdigest(),
                    labels,
                    values.index(1),
                    instructions=str(spec.get("instructions", "")),
                )
            )
        return output
    if kind == "ultrafeedback":
        return ultrafeedback_candidates(rows, spec)[0]
    if kind == "massive" and not labels:
        labels = tuple(str(value).replace("_", " ") for value in rows.features["intent"].names)  # type: ignore[union-attr]
    for index, row in enumerate(rows):  # type: ignore[union-attr]
        if kind == "sms":
            text = str(row["sms"])
            target = int(row["label"])
            question_type = "noul"
        elif kind == "civil_comments":
            text = str(row["text"])
            target = int(float(row["toxicity"]) >= 0.5)
            question_type = "noul"
        elif kind == "yelp":
            text = str(row["text"])
            target = int(row["label"])
            question_type = "score"
        elif kind == "massive":
            text = str(row["utt"])
            target = int(row["intent"])
            question_type = "choice"
        else:
            raise ValueError(f"unknown dataset kind: {kind}")
        if not 0 <= target < len(labels):
            raise ValueError(f"{name}: target outside full label universe")
        source_id = str(row.get("id", index))
        output.append(
            Example(
                name,
                str(spec["task"]),
                f"{name}:{source_id}",
                text,
                hashlib.sha256(text.encode()).hexdigest(),
                labels,
                target,
                question_type=question_type,
                locale=str(locale) if locale else None,
                instructions=str(spec.get("instructions", "")),
            )
        )
    return output


def _split_candidates(
    candidates: Sequence[Example],
    counts: Mapping[str, int],
    seed: int,
    *,
    mass_groups: Mapping[str, str] | None = None,
    assignments: dict[str, str] | None = None,
) -> tuple[list[Example], int]:
    by_group: dict[str, list[Example]] = defaultdict(list)
    for row in candidates:
        group = mass_groups[row.example_id] if mass_groups else normalized_text_hash(row.text)
        by_group[group].append(row)
    merged = sum(len(group) - 1 for group in by_group.values())

    # One representative per content group keeps requested counts exact and prevents leakage.
    # The representative is chosen by a seeded hash rather than list position, so a group's
    # first member (e.g. the first completion of a prompt) is not systematically favoured.
    def representative(rows: list[Example]) -> Example:
        return min(
            rows, key=lambda row: hashlib.sha256(f"{seed}:{row.example_id}".encode()).hexdigest()
        )

    pool = [(group, representative(rows)) for group, rows in sorted(by_group.items())]
    output: list[Example] = []
    for offset, (split, count) in enumerate(counts.items()):
        eligible = [
            (group, row)
            for group, row in pool
            if assignments is None or assignments.get(group, split) == split
        ]
        if len(eligible) < count:
            raise ValueError(f"{split}: only {len(eligible)} disjoint groups for {count} items")
        positions = _balanced_indices(
            [row.target_index for _, row in eligible], count, seed + offset
        )
        selected = {eligible[index][0] for index in positions}
        output.extend(replace(row, split=split) for group, row in eligible if group in selected)
        if assignments is not None:
            assignments.update({group: split for group in selected})
        pool = [(group, row) for group, row in pool if group not in selected]
    return output, merged


def build_derived_suites(
    examples: Sequence[Example],
    seed: int,
    permutation_count: int = 100,
    latency_count: int = 100,
) -> list[Example]:
    output: list[Example] = []
    for dataset in ("agnews", "emotiondair", "banking77"):
        pool = [row for row in examples if row.split == "test" and row.dataset == dataset]
        for index in _balanced_indices(
            [row.target_index for row in pool], min(permutation_count, len(pool)), seed
        ):
            row = pool[index]
            orders = [tuple(range(len(row.labels)))]
            rng = random.Random(f"{seed}:{row.example_id}")
            for _ in range(100):
                order = list(range(len(row.labels)))
                rng.shuffle(order)
                candidate = tuple(order)
                if candidate not in orders:
                    orders.append(candidate)
                if len(orders) == 4:
                    break
            while len(orders) < 4:
                # Fixture-sized label sets cannot yield four unique orders.
                orders.append(orders[-1])
            for order_index, order in enumerate(orders):
                for mode in ("stable", "positional"):
                    output.append(
                        replace(
                            row,
                            split="permutation",
                            permutation_id=f"order-{order_index}",
                            letter_mode=mode,
                            option_order=order,
                        )
                    )
    for row in examples:
        if row.split == "test" and len(row.labels) > 20:
            output.append(
                replace(
                    row, split="permutation", permutation_id="head-default", letter_mode="stable"
                )
            )
    pool = [row for row in examples if row.split == "test" and row.dataset == "agnews"]
    for index in _balanced_indices(
        [row.target_index for row in pool], min(latency_count, len(pool)), seed
    ):
        for workload in ("latency-1", "latency-10"):
            output.append(replace(pool[index], split="latency", permutation_id=workload))
    return output


def load_v2_examples(
    config: BenchmarkConfig,
    *,
    loader: Callable[..., object] | None = None,
    counters: Mapping[str, Mapping[str, Callable[[Example], int]]] | None = None,
) -> tuple[list[Example], dict[str, object]]:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    assert loader is not None
    if counters is None and config.raw["length_rule"]["required"]:
        counters = make_length_counters(config)
    counts = config.raw["dataset"]["split_counts"]
    output: list[Example] = []
    summaries: dict[str, object] = {}
    massive_groups: dict[str, str] = {}
    massive_assignments: dict[str, str] = {}
    for spec in config.raw["dataset"]["datasets"]:
        name = str(spec["name"])
        rows = _v2_rows(loader, spec, str(config.path.parent.parent / "results" / "cache"))
        content_groups: dict[str, str] | None = None
        if spec["kind"] == "ultrafeedback":
            candidates, content_groups = ultrafeedback_candidates(rows, spec)
        else:
            candidates = _v2_candidates(rows, spec)
        assigned: Mapping[str, Callable[[Example], int]] = {}
        budgets: dict[str, int] = {}
        if spec["kind"] == "massive":
            for row, source in zip(candidates, rows, strict=True):  # type: ignore[union-attr]
                massive_groups[row.example_id] = str(source["id"])
        exclusions: dict[str, int] = {}
        if counters is not None:
            assigned = counters[name]
            budgets = {
                key: int(config.raw["length_rule"]["context_budgets"][key]) for key in assigned
            }
            primary = {key: value for key, value in assigned.items() if key != "gliner"}
            candidates, exclusions = length_exclusion(
                candidates, primary, {key: budgets[key] for key in primary}
            )
            if "gliner" in assigned:
                exclusions["gliner"] = sum(
                    assigned["gliner"](row) > budgets["gliner"] for row in candidates
                )
        selected, merged = _split_candidates(
            candidates,
            counts["massive"] if spec["kind"] == "massive" else counts["default"],
            config.seed,
            mass_groups=massive_groups if spec["kind"] == "massive" else content_groups,
            assignments=massive_assignments if spec["kind"] == "massive" else None,
        )
        output.extend(selected)
        lengths = {
            "text_chars": [len(row.text) for row in selected],
            **(
                {
                    backend: [counter(row) for row in selected]
                    for backend, counter in counters[name].items()
                }
                if counters is not None
                else {}
            ),
        }

        def distribution(values: list[int]) -> dict[str, int | float]:
            ordered = sorted(values)
            return {
                "min": ordered[0],
                "median": ordered[len(ordered) // 2],
                "p95": ordered[int(0.95 * (len(ordered) - 1))],
                "max": ordered[-1],
            }

        summaries[name] = {
            "merged_groups": merged,
            "length_excluded": exclusions,
            "excluded_datasets": ["gliner"]
            if counters is not None
            and "gliner" in assigned
            and any(assigned["gliner"](row) > budgets["gliner"] for row in selected)
            else [],
            "retained_length_distribution": {
                key: distribution(value) for key, value in lengths.items()
            },
            "source_split": spec.get("source_split", "train"),
            "counts": {
                split: sum(row.split == split for row in selected)
                for split in ("pilot", "calibration", "test")
            },
        }
    # Cross-locale MASSIVE IDs must never bridge splits. Reject a selection that does.
    group_split: dict[str, str] = {}
    for row in output:
        if row.example_id in massive_groups:
            group = massive_groups[row.example_id]
            prior = group_split.setdefault(group, row.split)
            if prior != row.split:
                raise ValueError("MASSIVE cross-locale utterance crosses splits")
    group_counts: dict[str, int] = defaultdict(int)
    for row in output:
        if row.example_id in massive_groups:
            group_counts[massive_groups[row.example_id]] += 1
    for name in summaries:
        if name.startswith("massive_"):
            summary = summaries[name]
            assert isinstance(summary, dict)
            summary["cross_locale_shared_groups"] = sum(
                count > 1 for count in group_counts.values()
            )
    output.extend(
        build_derived_suites(
            output,
            config.seed,
            int(config.raw["dataset"].get("permutation_items", 100)),
            int(config.raw["dataset"].get("latency_items", 100)),
        )
    )
    return output, summaries


def load_local_tokenizer(path: str) -> Any:
    """Load a pinned local tokenizer without network access or remote code.

    Some checkpoints (GLiNER2.5's mDeBERTa) store ``extra_special_tokens`` as a list, which
    current Transformers rejects. The list is passed as ``additional_special_tokens`` instead,
    so those markers still tokenize as single tokens and length counts stay faithful. The
    pinned files on disk are never rewritten.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    config_path = Path(path) / "tokenizer_config.json"
    if config_path.exists():
        extra = json.loads(config_path.read_text(encoding="utf-8")).get("extra_special_tokens")
        if isinstance(extra, list):
            kwargs["extra_special_tokens"] = {}
            kwargs["additional_special_tokens"] = [str(token) for token in extra]
    return AutoTokenizer.from_pretrained(path, **kwargs)


def make_length_counters(
    config: BenchmarkConfig,
    *,
    tokenizer_loader: Callable[[str], Any] | None = None,
    laya_to_internal: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, dict[str, Callable[[Example], int]]]:
    """Load pinned local tokenizers only; model weights are never touched by prepare."""
    if tokenizer_loader is None:
        tokenizer_loader = load_local_tokenizer

    rule = config.raw["length_rule"]
    tokenizer_paths = {
        "qwen_logit": rule["qwen_tokenizer_path"],
        "laya_base": rule["laya_tokenizer_paths"]["base"],
        "laya_multilingual": rule["laya_tokenizer_paths"]["multilingual"],
        "laya_typed": rule["laya_tokenizer_paths"]["typed"],
    }
    if "gliner" in config.raw["models"]:
        tokenizer_paths["gliner"] = rule["gliner_tokenizer_path"]
    if any(str(value).startswith("TODO-PIN") for value in tokenizer_paths.values()):
        raise ValueError("length tokenizers need TODO-PIN resolution")
    tokenizers = {name: tokenizer_loader(str(path)) for name, path in tokenizer_paths.items()}
    output: dict[str, dict[str, Callable[[Example], int]]] = {}
    for spec in config.raw["dataset"]["datasets"]:
        name = str(spec["name"])
        assigned: dict[str, Callable[[Example], int]] = {}
        for backend, model in config.raw["models"].items():
            if name not in model["datasets"]:
                continue
            if backend == "jev_openrouter":
                from .adapters.jev_openrouter import JevOpenRouterBackend

                probe = JevOpenRouterBackend(model["model_id"], config.raw.get("questions", {}))
                assigned[backend] = lambda row, probe=probe: len(probe.request_body(row))
            elif backend.startswith("laya"):
                from .adapters.laya import LayaBackend, laya_request_tokens

                tokenizer = tokenizers[backend]
                head_limit = model["head_max_len"][name]
                if isinstance(head_limit, str):
                    raise ValueError("Laya head_max_len needs TO-FILL-AFTER-PROBE resolution")

                def laya_count(
                    row: Example,
                    *,
                    tokenizer: Any = tokenizer,
                    head_limit: int = head_limit,
                    backend: str = backend,
                ) -> int:
                    to_internal = laya_to_internal
                    if to_internal is None:
                        from importlib import import_module

                        to_internal = import_module("laya").Agent._to_internal
                    question = LayaBackend.question_for(row, config.raw.get("questions", {}))
                    option_sizes, instructions, _, total = laya_request_tokens(
                        question, row.text, tokenizer, to_internal
                    )
                    if (
                        any(size > 49 for size in option_sizes)
                        or sum(option_sizes) + instructions > head_limit
                    ):
                        return int(rule["context_budgets"][backend]) + 1
                    return total

                assigned[backend] = laya_count
            elif backend == "qwen_logit":
                tok = tokenizers[backend]

                def qwen_count(row: Example, *, tok: Any = tok, model: dict = model) -> int:
                    ids = (
                        [chr(65 + index) for index in range(len(row.labels))]
                        if len(row.labels) <= 26
                        else [f"{index + 1:02d}" for index in range(len(row.labels))]
                    )
                    options = "\n".join(
                        f"{option_id}) {label}"
                        for option_id, label in zip(ids, row.labels, strict=True)
                    )
                    user = model["user_template"].format(
                        text=row.text, options=options, question=row.instructions
                    )
                    prompt = tok.apply_chat_template(
                        [
                            {"role": "system", "content": model["system_prompt"]},
                            {"role": "user", "content": user},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    return max(
                        len(
                            tok(prompt + "Answer: " + option_id, add_special_tokens=False)[
                                "input_ids"
                            ]
                        )
                        for option_id in ids
                    )

                assigned[backend] = qwen_count
            elif backend == "gliner":
                tok = tokenizers[backend]

                def gliner_count(row: Example, *, tok: Any = tok) -> int:
                    return (
                        len(tok(row.text, add_special_tokens=False)["input_ids"])
                        + len(tok(row.instructions, add_special_tokens=False)["input_ids"])
                        + sum(
                            len(tok(label, add_special_tokens=False)["input_ids"])
                            for label in row.labels
                        )
                    )

                assigned[backend] = gliner_count
        output[name] = assigned
    return output
