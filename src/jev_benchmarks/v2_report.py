from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.jev_openrouter import MISSING_MODEL
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
from .v2_runner import _dispatch_lock, prediction_key, verify_frozen

# Like-for-like reading first (Amendment 7 C002): accuracy and B-vs-B, then A-vs-A supplements.
HEADLINE_METRICS = (
    ("accuracy", "A_raw"),
    ("brier", "B_scaled"),
    ("test_coverage", "B_scaled"),
    ("brier", "A_raw"),
    ("test_coverage", "A_raw"),
)


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


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


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
) -> tuple[Path, list[Prediction], str]:
    """Newest complete single-snapshot attempt, its rows, and the sha256 of the exact bytes read."""
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
    attempts = sorted(
        (config.output_dir / backend).glob("attempt-*"),
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    for path in attempts:
        file = path / "predictions.jsonl"
        data = file.read_bytes() if file.exists() else b""
        rows = [
            Prediction.from_dict(json.loads(line)) for line in data.decode().splitlines() if line
        ]
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
        if status.exists() and json.loads(status.read_text(encoding="utf-8"))["state"] in {
            "snapshot_changed",
            "cost_paused",
            "cost_exhausted",
        }:
            # A paused or exhausted attempt needs operator resolution before it can be reported.
            continue
        latest: dict[tuple[str, str, str, str, int], Prediction] = {}
        dispatches: dict[tuple[str, str, str, str, int], int] = defaultdict(int)
        for row in rows:
            latest[prediction_key(row)] = row
            dispatches[prediction_key(row)] += row.dispatch_attempts or 0
        # Earlier failed invocations of a key count toward its dispatch history.
        for key, row in latest.items():
            if dispatches[key] > (row.dispatch_attempts or 0):
                latest[key] = replace(row, dispatch_attempts=dispatches[key])
        # Every call that received a response (success, unscorable or later retried) must share
        # one snapshot; calls with no response ("unknown") produced no output and score as failures.
        resolved = {row.model_resolved for row in rows if row.model_resolved != "unknown"}
        # Jev needs one known snapshot; a complete all-failure local attempt (no resolved model)
        # is accepted so its failures are scored, with the pinned revision as provenance.
        known = len(resolved) == 1 or (not resolved and backend != "jev_openrouter")
        if expected.issubset(latest) and known and MISSING_MODEL not in resolved:
            return path, list(latest.values()), hashlib.sha256(data).hexdigest()
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
        # Read, select and hash under the dispatch lock: no run may append meanwhile.
        with _dispatch_lock(config.output_dir / backend):
            path, rows, digest = _select_attempt(config, backend, manifest)
        attempts[backend] = path.name
        model = config.raw["models"][backend]
        pinned = f"{model.get('model_id', backend)}@{model.get('revision', 'unpinned')}"
        snapshots[backend] = next(
            (row.model_resolved for row in rows if row.model_resolved != "unknown"), pinned
        )
        prediction_hashes[backend] = digest
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
            done = [row for row in test if row.error is None]
            first_try = [row for row in done if (row.dispatch_attempts or 1) == 1]
            # Every dispatched item counts toward the retry share, including final failures.
            dispatched = [
                row
                for row in test
                # Local backends dispatch every item once; Jev failures carry their attempt count.
                if row.error is None or row.dispatch_attempts or backend != "jev_openrouter"
            ]
            if not splits.get("latency") and backend not in FEWSHOT_BACKENDS and dispatched:
                # Amendment 6 reference: first-attempt successes on the test split only.
                model_times = [
                    row.model_latency_seconds
                    for row in first_try
                    if row.model_latency_seconds is not None
                ]
                latency_rows.append(
                    {
                        "backend": backend,
                        "dataset": dataset,
                        "workload": "test-reference",
                        "n": len(first_try),
                        "dispatched": len(dispatched),
                        "retried_share": sum((row.dispatch_attempts or 1) > 1 for row in dispatched)
                        / len(dispatched),
                        # n = 0 (every item retried or failed) keeps the row with null quantiles.
                        "latency_p50_seconds": float(
                            np.quantile([row.latency_seconds for row in first_try], 0.5)
                        )
                        if first_try
                        else None,
                        "latency_p95_seconds": float(
                            np.quantile([row.latency_seconds for row in first_try], 0.95)
                        )
                        if first_try
                        else None,
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
            # Order effects need the complete frozen permutation run; otherwise unavailable.
            expected_orders = {
                (row.example_id, row.permutation_id, row.letter_mode)
                for row in manifest
                if row.split == "permutation"
                and row.dataset == dataset
                and row.permutation_id.startswith("order-")
                and (row.letter_mode != "positional" or backend == "qwen_logit")
            }
            observed_orders = {
                (row.example_id, row.permutation_id, row.letter_mode) for row in order_rows
            }
            if backend not in FEWSHOT_BACKENDS and not expected_orders <= observed_orders:
                flip_rows.append(
                    {
                        "backend": backend,
                        "dataset": dataset,
                        "unavailable": "permutation run incomplete: "
                        f"{len(expected_orders & observed_orders)}/{len(expected_orders)}",
                    }
                )
                order_rows = []
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
        # No Holm rejection from an undefined estimate or a reduced (success-conditioned) draw set.
        p_value = result["p_two_sided"]
        full = result["resamples_used"] == int(config.raw["metrics"]["bootstrap_resamples"])
        p_values[test["id"]] = float(p_value) if p_value is not None and full else 1.0
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
    # Amendment 6 headline: Jev's zero-shot gap to Qwen and its gap to each few-label contender,
    # side by side on identical items. Estimation only; primary-set zero-shot rows are C1.
    budget = float(config.raw["metrics"]["error_budget"])
    c1 = {
        (row["metric"], row["condition"]): row
        for row in confirmatory_rows
        if row["left"] == "jev_openrouter" and row["right"] == "qwen_logit"
    }
    primary = list(family["tests"][0]["datasets"])
    headline: list[dict[str, Any]] = []
    jev = by_backend.get("jev_openrouter", {})
    for right in ["qwen_logit", *sorted(FEWSHOT_BACKENDS & set(by_backend))]:
        if not jev or right not in by_backend:
            continue
        common = sorted(set(jev) & set(by_backend[right]))
        vectors = vector_datasets(jev, by_backend[right], common)
        gap = "zero_shot" if right == "qwen_logit" else "few_label"
        for set_name, names in (
            ("primary", [name for name in primary if name in common]),
            ("secondary", common),
        ):
            for metric, condition in HEADLINE_METRICS:
                shared = names if metric == "accuracy" else [n for n in names if n in vectors]
                if not shared:
                    continue
                base = {
                    "left": "jev_openrouter",
                    "right": right,
                    "datasets": shared,
                    "metric": metric,
                    "condition": condition,
                    "family": "headline_estimation",
                    "set": set_name,
                    "gap": gap,
                }
                reused = c1.get((metric, condition))
                if gap == "zero_shot" and set_name == "primary" and reused and shared == primary:
                    headline.append({**reused, **base, "reuses": reused["id"]})
                    continue
                result = joint_paired_bootstrap(
                    {name: jev[name]["test"] for name in shared},
                    {name: by_backend[right][name]["test"] for name in shared},
                    {name: jev[name]["calibration"] for name in shared},
                    {name: by_backend[right][name]["calibration"] for name in shared},
                    metric=metric,
                    condition=condition,
                    resamples=resamples,
                    seed=config.seed,
                    error_budget=budget,
                )
                headline.append(
                    {"id": f"headline-{set_name}-{right}-{metric}-{condition}", **base, **result}
                )
    # Each side's realized coverage, selective error and no-feasible count beside coverage rows.
    by_key = {
        (row["backend"], row["dataset"], row["condition"]): row
        for row in metric_rows
        if "condition" in row
    }
    for row in headline:
        if row["metric"] != "test_coverage":
            continue
        for prefix in ("left", "right"):
            cells = [
                by_key.get((row[prefix], name, row["condition"]), {}) for name in row["datasets"]
            ]
            coverage = [c["test_coverage"] for c in cells if c.get("test_coverage") is not None]
            errors = [
                c["test_selective_error"]
                for c in cells
                if c.get("test_selective_error") is not None
            ]
            row[f"{prefix}_coverage"] = float(np.mean(coverage)) if coverage else None
            row[f"{prefix}_selective_error"] = float(np.mean(errors)) if errors else None
            row[f"{prefix}_no_feasible"] = sum(bool(c.get("no_feasible_threshold")) for c in cells)
    # Per-dataset paired differences beside every pooled confirmatory and headline effect.
    per_dataset: list[dict[str, Any]] = []
    memo: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    parents: dict[str, dict[str, Any]] = {}
    for row in [*confirmatory_rows, *headline]:
        parents.setdefault(row["id"], row)  # reused C1 headline rows share the C1 parent
    for parent in parents.values():
        left, right = parent["left"], parent["right"]
        for name in parent["datasets"]:
            key = (left, right, name, parent["metric"], parent["condition"])
            if key not in memo:
                memo[key] = joint_paired_bootstrap(
                    {name: by_backend[left][name]["test"]},
                    {name: by_backend[right][name]["test"]},
                    {name: by_backend[left][name]["calibration"]},
                    {name: by_backend[right][name]["calibration"]},
                    metric=parent["metric"],
                    condition=parent["condition"],
                    resamples=resamples,
                    seed=config.seed,
                    error_budget=budget,
                )
            per_dataset.append(
                {
                    "id": f"{parent['id']}:{name}",
                    "parent": parent["id"],
                    "left": left,
                    "right": right,
                    "dataset": name,
                    "metric": parent["metric"],
                    "condition": parent["condition"],
                    "family": "per_dataset",
                    **memo[key],
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
        # Latency is published only as the latency.csv reference (Amendment 6).
        for key in [key for key in row if "latency" in key]:
            del row[key]
    outputs = {
        "metrics.csv": metric_rows,
        "pairwise_ci.csv": [*headline, *pairwise, *per_dataset],
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
        "headline": headline,
        "per_dataset": per_dataset,
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
            "Few-label thresholds are chosen on out-of-fold predictions and applied to an all-200 "
            "refit, so realized selective error can drift from the budget; it is reported beside "
            "coverage.",
            "Latency is a deployment-specific reference (first-attempt test-split calls).",
        ],
    }
    absent_classes: dict[str, Any] = {}
    for backend in sorted(FEWSHOT_BACKENDS & set(attempts)):
        metadata = config.output_dir / backend / attempts[backend] / "fewshot-metadata.json"
        if metadata.exists():
            absent_classes[backend] = json.loads(metadata.read_text(encoding="utf-8"))[
                "absent_class_held_out_rows"
            ]
    payload["few_label_absent_class_held_out_rows"] = absent_classes
    public_path = config.path.parent.parent / "docs" / "public-results.csv"
    public_rows: list[dict[str, str]] = []
    if public_path.exists():
        with public_path.open(encoding="utf-8") as handle:
            public_rows = list(csv.DictReader(handle))
    payload["public_results"] = public_rows
    payload["public_results_sha256"] = sha256_file(public_path) if public_path.exists() else None
    json_path = report_dir / "v2.json"
    json_path.write_text(scrubbed_json(payload, secret) + "\n", encoding="utf-8")
    markdown_path = report_dir / "v2.md"
    lines = [
        "# V2 benchmark",
        "",
        "## Headline: how much of zero-shot Jev's advantage is left",
        "",
        "Differences are contender minus Jev on identical items: a negative accuracy or coverage "
        "difference, or a positive Brier difference, means Jev leads. Estimation only, unadjusted; "
        "primary-set zero-shot accuracy/Brier rows are the C1 confirmatory estimates. "
        "The like-for-like reading is accuracy plus B-vs-B; A-vs-A rows are supplements.",
        "",
        "**Estimand:** every value describes class-balanced test items (candidate-pool "
        "prevalence is in manifest-summary.json), not deployment traffic.",
        "",
        "| set | contender | gap | metric | condition | difference | 95% CI | Holm (C1) |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in headline:
        lines.append(
            f"| {row['set']} | {row['right']} | {row['gap']} | {row['metric']} | "
            f"{row['condition']} | {_num(row['difference'])} | "
            f"[{_num(row['ci95_low'])}, {_num(row['ci95_high'])}] | "
            f"{row.get('holm_reject_005', '')} |"
        )
    coverage_rows = [row for row in headline if row["metric"] == "test_coverage"]
    if coverage_rows:
        lines += [
            "",
            "Coverage detail (mean over datasets; selective error is realized on test):",
            "",
            "| set | contender | condition | Jev coverage | contender coverage | "
            "Jev selective error | contender selective error | no-feasible (Jev/contender) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        lines.extend(
            f"| {row['set']} | {row['right']} | {row['condition']} | "
            f"{_num(row['left_coverage'])} | {_num(row['right_coverage'])} | "
            f"{_num(row['left_selective_error'])} | {_num(row['right_selective_error'])} | "
            f"{row['left_no_feasible']}/{row['right_no_feasible']} |"
            for row in coverage_rows
        )
    lines += [
        "",
        "## Confirmatory family",
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
            f"| {row['id']} | {_num(row['difference'])} | "
            f"[{_num(row['ci95_low'])}, {_num(row['ci95_high'])}] | "
            f"{_num(row['p_two_sided'])} | {row['holm_reject_005']} |"
        )
    lines += [
        "",
        "## Per-dataset paired differences",
        "",
        "Each pooled effect above is an equal-weight average over its datasets. Brier spans "
        "[0, 2] for every K, but its chance baseline (uniform prediction: 1 - 1/K) differs by "
        "K, so read pooled Brier with these rows.",
        "",
        "| parent | dataset | difference | 95% CI |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {row['parent']} | {row['dataset']} | {_num(row['difference'])} | "
        f"[{_num(row['ci95_low'])}, {_num(row['ci95_high'])}] |"
        for row in per_dataset
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
            f"| {row['id']} | {_num(row['difference'])} | "
            f"[{_num(row['ci95_low'])}, {_num(row['ci95_high'])}] |"
        )
    if absent_classes:
        lines.extend(
            ["", "## Few-label arm: held-out rows whose class was absent from training", ""]
        )
        lines.extend(
            f"- {backend}: "
            + ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
            for backend, counts in absent_classes.items()
        )
    if public_rows:
        lines.extend(
            [
                "",
                "## Relation to public results",
                "",
                "Same-run controlled re-check on class-balanced test items; public values use "
                "other protocols and the sampling distributions listed here.",
                "",
                "| source | contenders | task | sampling | metric | values | protocol |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| [{row['source']}]({row['url']}) | {row['contenders']} | {row['task']} | "
            f"{row.get('sampling', 'not stated')} | {row['metric']} | {row['values']} | "
            f"{row['protocol_note']} |"
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
