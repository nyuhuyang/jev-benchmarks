from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfig
from .fewshot import FEWSHOT_BACKENDS
from .io import read_jsonl, runtime_metadata, scrubbed_json, sha256_file
from .models import Example, Prediction
from .v2_metrics import (
    condition_b,
    excess_order_bootstrap,
    fit_temperature,
    flip_summary,
    holm_decisions,
    joint_paired_bootstrap,
    mean_of_repeats,
    reliability_bins,
    repeat_summary,
    rounding_parity,
    score_v2,
    select_threshold,
)
from .v2_runner import prediction_key, verify_frozen


def _gliner_exclusions(config: BenchmarkConfig) -> list[str]:
    path = config.output_dir / "manifest-summary.json"
    if not path.exists():
        return []
    summary = json.loads(path.read_text(encoding="utf-8"))
    return [
        name
        for name, details in summary.items()
        if "gliner" in details.get("excluded_datasets", [])
    ]


def vector_datasets(
    left: dict[str, dict[str, list[Prediction]]],
    right: dict[str, dict[str, list[Prediction]]],
    names: list[str],
) -> list[str]:
    """Datasets where both sides have full distributions; scalar-only outputs have no T."""
    return [
        name
        for name in names
        if fit_temperature(left[name].get("calibration", [])) is not None
        and fit_temperature(right[name].get("calibration", [])) is not None
    ]


def _csv(path: Path, rows: list[dict[str, Any]], secret: str | None) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            scrubbed_json(row, secret)
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _select_attempt(
    config: BenchmarkConfig, backend: str, manifest: list[Example]
) -> tuple[Path, list[Prediction]]:
    excluded = set(_gliner_exclusions(config)) if backend == "gliner" else set()
    expected = {
        (*prediction_key(example)[:4], repeat)
        for example in manifest
        if example.split in {"calibration", "test"}
        and example.dataset in config.raw["models"][backend]["datasets"]
        and example.dataset not in excluded
        for repeat in range(int(config.raw["models"][backend]["repeats"]))
    }
    contract = {
        (row.split, row.example_id, row.permutation_id, row.letter_mode): row for row in manifest
    }
    for path in sorted((config.output_dir / backend).glob("attempt-*"), reverse=True):
        rows = [Prediction.from_dict(row) for row in read_jsonl(path / "predictions.jsonl")]
        for row in rows:
            source = contract.get(prediction_key(row)[:4])
            if source is None or (
                row.dataset != source.dataset
                or row.target_index != source.target_index
                or row.labels != source.labels
                or row.question_type != source.question_type
            ):
                raise ValueError(f"{backend} prediction contract mismatch: {row.example_id}")
        status = path / "status.json"
        if (
            status.exists()
            and json.loads(status.read_text(encoding="utf-8"))["state"] == "snapshot_changed"
        ):
            continue
        latest: dict[tuple[str, str, str, str, int], Prediction] = {}
        for row in rows:
            latest[prediction_key(row)] = row
        # Every call that received a response (success or unscorable) must share one snapshot;
        # calls with no response ("unknown") produced no output and are scored as failures.
        resolved = {
            row.model_resolved for row in latest.values() if row.model_resolved != "unknown"
        }
        if expected.issubset(latest) and len(resolved) == 1:
            return path, list(latest.values())
    raise ValueError(f"no complete single-snapshot attempt for {backend}")


def build_v2_report(config: BenchmarkConfig) -> tuple[Path, Path]:
    manifest_path = config.output_dir / "manifest.jsonl"
    verify_frozen(config, manifest_path)
    manifest = [Example.from_dict(row) for row in read_jsonl(manifest_path)]
    report_dir = config.path.parent.parent / config.raw["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    secret = os.environ.get("OPENROUTER_API_KEY")
    attempts: dict[str, str] = {}
    snapshots: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    predictions: dict[str, list[Prediction]] = {}
    for backend in config.raw["models"]:
        path, rows = _select_attempt(config, backend, manifest)
        attempts[backend] = path.name
        snapshots[backend] = next(row.model_resolved for row in rows if row.error is None)
        prediction_hashes[backend] = sha256_file(path / "predictions.jsonl")
        predictions[backend] = rows
    metric_rows: list[dict[str, Any]] = []
    gliner_excluded = _gliner_exclusions(config) if "gliner" in config.raw["models"] else []
    metric_rows.extend(
        {
            "backend": "gliner",
            "dataset": dataset,
            "condition": condition,
            "excluded": "GLiNER request length",
            "accuracy": None,
            "brier": None,
        }
        for dataset in gliner_excluded
        for condition in ("A_raw", "B_scaled")
    )
    reliability_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    temperature_rows: list[dict[str, Any]] = []
    by_backend: dict[str, dict[str, dict[str, list[Prediction]]]] = {}
    for backend, rows in predictions.items():
        grouped: dict[str, dict[str, list[Prediction]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row.repeat_index == 0:
                grouped[row.dataset][row.split].append(row)
        by_backend[backend] = grouped
        for dataset, splits in grouped.items():
            calibration, test = splits.get("calibration", []), splits.get("test", [])
            if not test:
                continue
            temperature = fit_temperature(calibration)
            temperature_rows.append(
                {
                    "backend": backend,
                    "dataset": dataset,
                    "temperature": temperature,
                    "condition_b_available": temperature is not None,
                }
            )
            for condition, cal_rows, test_rows in (
                ("A_raw", calibration, test),
                ("B_scaled", condition_b(calibration, temperature), condition_b(test, temperature)),
            ):
                if not test_rows:
                    continue
                threshold = select_threshold(cal_rows, float(config.raw["metrics"]["error_budget"]))
                scores = score_v2(
                    test_rows, threshold=threshold, ece_bins=int(config.raw["metrics"]["ece_bins"])
                )
                metric_rows.append(
                    {"backend": backend, "dataset": dataset, "condition": condition, **scores}
                )
                for bin_row in reliability_bins(test_rows, int(config.raw["metrics"]["ece_bins"])):
                    reliability_rows.append(
                        {"backend": backend, "dataset": dataset, "condition": condition, **bin_row}
                    )
                if condition == "A_raw" and backend != "jev_openrouter":
                    metric_rows.append(
                        {
                            "backend": backend,
                            "dataset": dataset,
                            "condition": "A_rounded_2dp",
                            **score_v2(rounding_parity(test_rows), threshold=threshold),
                        }
                    )
            for workload in ("latency-1", "latency-10"):
                timed = [
                    row
                    for row in splits.get("latency", [])
                    if row.permutation_id == workload and row.error is None
                ]
                if timed:
                    divisor = 10 if workload == "latency-10" else 1
                    model_times = [
                        row.model_latency_seconds
                        for row in timed
                        if row.model_latency_seconds is not None
                    ]
                    latency_rows.append(
                        {
                            "backend": backend,
                            "dataset": dataset,
                            "workload": workload,
                            "n": len(timed),
                            "latency_p50_seconds": float(
                                np.quantile([row.latency_seconds for row in timed], 0.5)
                            ),
                            "latency_p95_seconds": float(
                                np.quantile([row.latency_seconds for row in timed], 0.95)
                            ),
                            "per_question_p50_seconds": float(
                                np.quantile([row.latency_seconds / divisor for row in timed], 0.5)
                            ),
                            "per_question_p95_seconds": float(
                                np.quantile([row.latency_seconds / divisor for row in timed], 0.95)
                            ),
                            "model_p50_seconds": float(np.quantile(model_times, 0.5))
                            if model_times
                            else None,
                            "model_p95_seconds": float(np.quantile(model_times, 0.95))
                            if model_times
                            else None,
                        }
                    )
            permutation = splits.get("permutation", [])
            order_rows = [row for row in permutation if row.permutation_id.startswith("order-")]
            if permutation:
                head_default = [
                    row
                    for row in permutation
                    if row.permutation_id == "head-default" and row.repeat_index == 0
                ]
                if head_default and backend.startswith("laya"):
                    metric_rows.append(
                        {
                            "backend": backend,
                            "dataset": dataset,
                            "condition": "A_as_shipped_head_exploratory",
                            "head_max_len_used": head_default[0].head_max_len_used,
                            **score_v2(head_default),
                        }
                    )
                modes = ("stable", "positional") if backend == "qwen_logit" else ("stable",)
                for mode in modes if order_rows else ():
                    flip_rows.append(
                        {
                            "backend": backend,
                            "dataset": dataset,
                            "letter_mode": mode,
                            **flip_summary(
                                [
                                    row
                                    for row in order_rows
                                    if row.letter_mode == mode and row.repeat_index == 0
                                ]
                            ),
                        }
                    )
            repeats = [row for row in rows if row.dataset == dataset and row.split == "test"]
            if len(repeats) > len(test):
                repeat_rows.append(
                    {"backend": backend, "dataset": dataset, **repeat_summary(repeats)}
                )
                metric_rows.append(
                    {
                        "backend": backend,
                        "dataset": dataset,
                        "condition": "A_mean_of_3_exploratory",
                        **score_v2(mean_of_repeats(repeats)),
                    }
                )
                if order_rows:
                    flip_rows.append(
                        {
                            "backend": backend,
                            "dataset": dataset,
                            "letter_mode": "stable_excess_over_repeat",
                            **excess_order_bootstrap(
                                order_rows,
                                repeats,
                                resamples=int(config.raw["metrics"]["bootstrap_resamples"]),
                                seed=config.seed,
                            ),
                        }
                    )
    family = config.raw["confirmatory_family"]
    k = int(family["k"])
    if k not in (3, 9) or len(family["tests"]) != k:
        raise ValueError("confirmatory family must be frozen with k=3 or k=9")
    pairwise: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for test in family["tests"]:
        left, right = test["left"], test["right"]
        datasets = test["datasets"]
        result = joint_paired_bootstrap(
            {name: by_backend[left][name]["test"] for name in datasets},
            {name: by_backend[right][name]["test"] for name in datasets},
            {name: by_backend[left][name]["calibration"] for name in datasets},
            {name: by_backend[right][name]["calibration"] for name in datasets},
            metric=test["metric"],
            condition=test["condition"],
            resamples=int(config.raw["metrics"]["bootstrap_resamples"]),
            seed=config.seed,
        )
        p_values[test["id"]] = result["p_two_sided"]
        pairwise.append(
            {
                "id": test["id"],
                "left": left,
                "right": right,
                "datasets": datasets,
                "metric": test["metric"],
                "condition": test["condition"],
                "family": "confirmatory",
                **result,
            }
        )
    decisions = holm_decisions(p_values, k)
    for row in pairwise:
        row["holm_reject_005"] = decisions[row["id"]]
        row["holm_k"] = k
    confirmatory_rows = list(pairwise)
    resamples = int(config.raw["metrics"]["bootstrap_resamples"])
    for backend, datasets in by_backend.items():
        for name, splits in datasets.items():
            if fit_temperature(splits.get("calibration", [])) is None or not splits.get("test"):
                continue
            for metric in ("ece", "brier", "nll"):
                result = joint_paired_bootstrap(
                    {name: splits["test"]},
                    {name: splits["test"]},
                    {name: splits["calibration"]},
                    {name: splits["calibration"]},
                    metric=metric,
                    condition="A_vs_B",
                    left_condition="A_raw",
                    right_condition="B_scaled",
                    resamples=resamples,
                    seed=config.seed,
                )
                pairwise.append(
                    {
                        "id": f"{backend}-{name}-A-vs-B-{metric}",
                        "left": backend,
                        "right": backend,
                        "datasets": [name],
                        "metric": metric,
                        "condition": "A_vs_B",
                        "family": "descriptive",
                        **result,
                    }
                )
    for left, right in (
        ("jev_openrouter", "qwen_logit"),
        ("jev_openrouter", "laya_base"),
        ("qwen_logit", "laya_base"),
    ):
        if left not in by_backend or right not in by_backend:
            continue
        shared = sorted(set(by_backend[left]) & set(by_backend[right]))
        shared = [
            name
            for name in shared
            if by_backend[left][name].get("test")
            and by_backend[right][name].get("test")
            and fit_temperature(by_backend[right][name].get("calibration", [])) is not None
        ]
        if shared:
            result = joint_paired_bootstrap(
                {name: by_backend[left][name]["test"] for name in shared},
                {name: by_backend[right][name]["test"] for name in shared},
                {name: by_backend[left][name]["calibration"] for name in shared},
                {name: by_backend[right][name]["calibration"] for name in shared},
                metric="brier",
                condition="A_vs_B",
                left_condition="A_raw",
                right_condition="B_scaled",
                resamples=resamples,
                seed=config.seed,
            )
            pairwise.append(
                {
                    "id": f"{left}-A-vs-{right}-B",
                    "left": left,
                    "right": right,
                    "datasets": shared,
                    "metric": "brier",
                    "condition": "A_vs_B",
                    "family": "descriptive",
                    **result,
                }
            )
    for few in sorted(FEWSHOT_BACKENDS & set(by_backend)):
        if "jev_openrouter" not in by_backend:
            continue
        jev = by_backend["jev_openrouter"]
        common = sorted(set(jev) & set(by_backend[few]))
        vectors = vector_datasets(jev, by_backend[few], common)
        for metric, condition in (("accuracy", "A_raw"), ("brier", "A_raw"), ("brier", "B_scaled")):
            shared = common if metric == "accuracy" else vectors
            if not shared:
                continue
            result = joint_paired_bootstrap(
                {name: by_backend["jev_openrouter"][name]["test"] for name in shared},
                {name: by_backend[few][name]["test"] for name in shared},
                {name: by_backend["jev_openrouter"][name]["calibration"] for name in shared},
                {name: by_backend[few][name]["calibration"] for name in shared},
                metric=metric,
                condition=condition,
                resamples=resamples,
                seed=config.seed,
            )
            pairwise.append(
                {
                    "id": f"jev_openrouter-vs-{few}-{metric}-{condition}",
                    "left": "jev_openrouter",
                    "right": few,
                    "datasets": shared,
                    "metric": metric,
                    "condition": condition,
                    "family": "descriptive_few_label",
                    **result,
                }
            )
    if k == 9:
        shared = sorted(set(by_backend["jev_openrouter"]) & set(by_backend["qwen_logit"]))
        frozen = set(family["tests"][0]["datasets"])
        if set(shared) - frozen:
            result = joint_paired_bootstrap(
                {name: by_backend["jev_openrouter"][name]["test"] for name in shared},
                {name: by_backend["qwen_logit"][name]["test"] for name in shared},
                {name: by_backend["jev_openrouter"][name]["calibration"] for name in shared},
                {name: by_backend["qwen_logit"][name]["calibration"] for name in shared},
                metric="accuracy",
                condition="A_raw",
                resamples=resamples,
                seed=config.seed,
            )
            pairwise.append(
                {
                    "id": "broader-C1-descriptive",
                    "left": "jev_openrouter",
                    "right": "qwen_logit",
                    "datasets": shared,
                    "metric": "accuracy",
                    "condition": "A_raw",
                    "family": "descriptive",
                    **result,
                }
            )
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "laya"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        import torch

        mps_available: bool | None = bool(torch.backends.mps.is_available())
    except ImportError:
        mps_available = None
    for row in metric_rows:
        if row["backend"] in FEWSHOT_BACKENDS:
            # No common single-call operation: training/batch time is not latency.
            for key in [key for key in row if "latency" in key]:
                del row[key]
    outputs = {
        "metrics.csv": metric_rows,
        "pairwise_ci.csv": pairwise,
        "reliability_bins.csv": reliability_rows,
        "latency.csv": latency_rows,
        "flip.csv": flip_rows,
        "repeat.csv": repeat_rows,
        "temperature.csv": temperature_rows,
    }
    for filename, rows in outputs.items():
        _csv(report_dir / filename, rows, secret)
    payload = {
        "schema_version": 2,
        "experiment_id": config.experiment_id,
        "confirmatory_holm_k": k,
        "attempts": attempts,
        "snapshots": snapshots,
        "config_sha256": sha256_file(config.path),
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_sha256": prediction_hashes,
        "protocol_sha256": sha256_file(
            config.path.parent.parent / config.raw["frozen"]["protocol_file"]
        ),
        "runtime": {
            **runtime_metadata(),
            "packages": packages,
            "mps_available": mps_available,
            "devices": {
                backend: sorted({row.device for row in rows if row.device})
                for backend, rows in predictions.items()
            },
        },
        "confirmatory": confirmatory_rows,
        "descriptive_intervals": pairwise[len(confirmatory_rows) :],
        "metrics": metric_rows,
        "excluded_datasets": {"gliner": gliner_excluded},
        "limitations": [
            "Public benchmark contamination is possible.",
            "Hosted latency includes an OpenRouter hop.",
            "Length exclusion favors shorter texts.",
            "Few-label contenders use 200 in-distribution labels per dataset; not zero-shot. "
            "Their bootstrap refits temperature but not the classifier, so intervals understate "
            "training variance.",
        ],
    }
    public_path = config.path.parent.parent / "docs" / "public-results.csv"
    public_rows: list[dict[str, str]] = []
    if public_path.exists():
        with public_path.open(encoding="utf-8") as handle:
            public_rows = list(csv.DictReader(handle))
    payload["public_results"] = public_rows
    json_path = report_dir / "v2.json"
    json_path.write_text(scrubbed_json(payload, secret) + "\n", encoding="utf-8")
    markdown_path = report_dir / "v2.md"
    lines = [
        "# V2 benchmark",
        "",
        f"Frozen Holm family: k={k}.",
        "",
        "Primary contrasts use shared items and include failures.",
        "",
        "| test | difference | 95% CI | p | Holm reject |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in confirmatory_rows:
        lines.append(
            f"| {row['id']} | {row['difference']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['p_two_sided']:.4f} | {row['holm_reject_005']} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive intervals",
            "",
            "These unadjusted intervals do not support winner claims.",
            "",
            "| analysis | difference | 95% CI |",
            "| --- | ---: | --- |",
        ]
    )
    for row in pairwise[len(confirmatory_rows) :]:
        lines.append(
            f"| {row['id']} | {row['difference']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    if public_rows:
        lines.extend(
            [
                "",
                "## Relation to public results",
                "",
                "Same-run controlled re-check; public values use other protocols and samples.",
                "",
                "| source | contenders | task | metric | values | protocol |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| [{row['source']}]({row['url']}) | {row['contenders']} | {row['task']} | "
            f"{row['metric']} | {row['values']} | {row['protocol_note']} |"
            for row in public_rows
        )
    lines.extend(
        [
            "",
            "Public benchmark contamination is possible. Hosted latency includes "
            "an OpenRouter hop. The common length rule favors shorter text.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashed = [*outputs, "v2.json", "v2.md"]
    (report_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(report_dir / filename)}  {filename}\n" for filename in hashed),
        encoding="utf-8",
    )
    return json_path, markdown_path
