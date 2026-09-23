# Architecture

`jev-benchmarks` separates experiment definition, data selection, inference, validation, and analysis
so a model call can never silently rewrite the evaluation contract.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Load and validate the YAML experiment contract; resolve artifact paths. |
| `data.py` | Load a pinned BTZSC revision, reject targetless rows, sample deterministically, write the manifest. |
| `adapters/` | Translate a common `Example` into backend-specific calls and return `Prediction`. |
| `runner.py` | Resume append-only runs, validate probability vectors, retain failures and usage. |
| `metrics.py` | Compute discrimination, calibration, selective-risk, latency, controls, and paired intervals. |
| `report.py` | Verify manifest/prediction identity and produce hashed JSON and Markdown summaries. |
| `io.py` | Atomic JSONL writes, durable appends, artifact hashing, and runtime metadata. |
| `v2_runner.py`, `worker.py` | Verify the preregistered hash record, isolate local inference processes, and write split-aware attempt logs. |
| `v2_metrics.py`, `v2_report.py` | Derive temperature-scaled B from stored A, run failure-aware analysis, and emit aggregate CSVs and hashes. |
| `anchor.py` | Re-run the pilot-v1 GLiNER manifest through v2 validation and compare both macro-F1 definitions with the published pilot. |

## Artifact flow

### Manifest

`prepare` writes one JSON object per selected example:

```json
{
  "dataset": "agnews",
  "task": "topic",
  "example_id": "agnews:9",
  "text": "...",
  "text_sha256": "...",
  "labels": ["..."],
  "target_index": 0
}
```

The ordered `labels` tuple is part of the evaluation contract. If an existing manifest differs from
the deterministic regeneration, preparation fails and requires a new experiment directory.

### Predictions

Each backend writes an append-only JSONL record containing requested and resolved model identities,
the complete probability vector, target and predicted indices, latency, optional token usage, the raw
probability sum, and any error. Successful example IDs are skipped on resume. Failed calls remain in
the log and are retried; reporting selects the most recent record for each example.

### Report

Before scoring, report generation checks that every backend covers exactly the manifest IDs and that
each target and ordered label tuple matches the manifest. The JSON report includes:

- schema and package versions;
- the complete experiment config and protocol revision;
- config, manifest, and prediction-log SHA-256 hashes;
- resolved model identities;
- runtime metadata;
- per-model/per-dataset metrics;
- paired confidence intervals.

## Probability validation

Vectors must have one finite value in `[0, 1]` for every label. Sums within `0.02` of one are treated
as serialization rounding and normalized; the pre-normalization sum is retained. Larger deviations
are failures.

The tolerance addresses APIs that expose rounded probabilities. It is not a general repair for
unnormalized scores, logits, or independent one-vs-rest probabilities.

## Selective coverage

Coverage at an error budget is computed only at thresholds realizable from reported confidence
values. Every example tied at a threshold is accepted together. This avoids optimistic coverage that
could only be achieved by splitting identical scores.

The pilot chooses and evaluates the threshold on one slice. A confirmatory experiment should select
the threshold on validation data and report risk once on a disjoint test set.

## Extension points

Backends implement the `Backend` protocol in `adapters/base.py`. Dataset loaders emit the common
`Example` type. The current CLI lists supported backend names explicitly so adding a provider is a
reviewed API change rather than dynamic code loading.

Heavy dependencies are optional:

- `data`: Hugging Face datasets;
- `gliner`: GLiNER2, PyTorch, Transformers, tokenizer dependencies;
- `jev`: the TypeSafe SDK;
- `benchmark`: the complete stack.
- `openrouter`: stdlib hosted adapter; `laya` and `qwen`: local runtimes with lazy imports.

Imports occur only when the corresponding command/backend is selected, so metrics and report tooling
remain lightweight.

## V2 path

The [v2 draft protocol](PROTOCOL-v2.md) defines choice, noul and score questions. `prepare` applies
the common full-request length rule before class-balanced sampling, groups normalized text (or
MASSIVE cross-locale IDs), and writes disjoint pilot, calibration and test rows plus derived
permutation and latency suites. The permutation split also holds separately labelled Laya
as-shipped-head rows on K > 20 tasks. `manifest-summary.json` records merged groups and length
exclusions. GLiNER is descriptive; its overflow excludes that backend from a dataset without
changing the primary shared sample.

`run --split` for a v2 config requires an existing manifest and a hash record committed at the
preregistered tag. Records live under `results/runs/<experiment>/<backend>/attempt-<n>/` and resume
by split, example ID, permutation ID, letter mode and repeat index. A changed Jev snapshot ends the
attempt. Local backends run in a separate process with only PATH, HOME, HF_HOME and offline HF
settings; this is process-environment isolation, not an OS sandbox. Worker stderr is retained in
each attempt directory.

`report` selects a complete single-snapshot attempt per backend. It fits temperatures on calibration
vectors, derives B without a second inference call, chooses selective thresholds on calibration,
and scores held-out test rows with failure penalties. The output is `v2.json`, `v2.md`, seven tidy
aggregate CSVs and `SHA256SUMS` under the configured report directory. Probe-dependent family and
pin placeholders must be filled before the v2 preregistration tag.
