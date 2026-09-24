from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfig
from .io import read_jsonl, write_json
from .metrics import _macro_f1_all_classes
from .models import Example, Prediction
from .runner import _validate_prediction
from .v2_metrics import score_v2


def compare_anchor(
    rows: Sequence[Prediction], published: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset].append(row)
    if set(grouped) != set(published):
        raise ValueError("anchor datasets differ from published pilot")
    output: dict[str, dict[str, float]] = {}
    for dataset, predictions in grouped.items():
        if any(row.error is not None for row in predictions):
            raise ValueError(f"{dataset}: anchor prediction failed")
        current = score_v2(predictions)
        targets = np.asarray([row.target_index for row in predictions], dtype=int)
        guesses = np.asarray([row.predicted_index for row in predictions], dtype=int)
        old_f1 = _macro_f1_all_classes(targets, guesses, len(predictions[0].labels))
        values = {
            "accuracy": float(current["accuracy"]),
            "macro_f1_all_classes": old_f1,
            "macro_f1_targets_predictions": float(current["macro_f1"]),
            "brier": float(current["brier"]),
        }
        reference = published[dataset]
        if len(predictions) != reference["n"]:
            raise ValueError(f"{dataset}: anchor count mismatch")
        for current_key, published_key, tolerance in (
            ("accuracy", "accuracy", 1e-9),
            ("macro_f1_all_classes", "macro_f1", 1e-6),
            ("brier", "brier", 1e-6),
        ):
            if abs(values[current_key] - float(reference[published_key])) > tolerance:
                raise ValueError(f"{dataset}: anchor {current_key} differs from published pilot")
        output[dataset] = values
    return output


def run_anchor(
    config: BenchmarkConfig,
    *,
    backend_factory: Callable[[BenchmarkConfig], Any] | None = None,
) -> Path:
    if config.raw.get("schema_version") == 2:
        raise ValueError("anchor requires the upstream pilot-v1 config")
    manifest = config.output_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError("anchor requires the existing pilot-v1 manifest")
    examples = [Example.from_dict(row) for row in read_jsonl(manifest)]
    if backend_factory is None:
        from .config import load_config
        from .v2_runner import LocalProcessBackend, local_runtime

        # GLiNER is built from the frozen pilot-v1 model spec, inside the minimal-environment
        # worker whose runtime (pinned HF_HOME, offline, no credentials) comes from v2.yaml.
        runtime = local_runtime(load_config(config.path.parent / "v2.yaml"))
        backend = LocalProcessBackend(
            config, "gliner", config.output_dir / "anchor-worker", runtime=runtime
        )
    else:
        backend = backend_factory(config)
    rows: list[Prediction] = []
    try:
        for example in examples:
            v2_example = replace(example, instructions="")
            prediction = _validate_prediction(backend.predict(config.experiment_id, v2_example))
            if prediction.example_id != example.example_id or prediction.labels != example.labels:
                raise ValueError("anchor prediction differs from manifest")
            rows.append(prediction)
    finally:
        backend.close()
    report = json.loads(
        (config.path.parent.parent / "results/reports/btzsc-pilot-v1.json").read_text(
            encoding="utf-8"
        )
    )
    comparison = compare_anchor(rows, report["results"]["gliner"])
    output = config.output_dir / "anchor-v2-comparison.json"
    write_json(output, comparison)
    return output
