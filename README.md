# jev-benchmarks

[![CI](https://github.com/AbdelStark/jev-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/jev-benchmarks/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Probability-aware evaluation for typed decision models.

`jev-benchmarks` measures more than whether a model selects the right label. It evaluates whether the
reported probabilities are calibrated enough to support automation, how much work can be accepted at
a fixed error budget, what resources each decision uses, and how long it takes end to end.

The first study compares [TypeSafe Jev](https://typesafe.ai/) with
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1) on their shared
capability: zero-shot, single-label text classification with per-label probabilities. It uses three
conditions from [BTZSC](https://huggingface.co/datasets/btzsc/btzsc), fixed model and dataset
revisions, identical examples and label descriptions, a uniform negative control, and paired
target-stratified bootstrap intervals.

## Pilot result

Three hundred held-out examples, 100 per condition:

| Dataset | Labels | Jev accuracy | GLiNER2.5 accuracy | Jev − GLiNER 95% CI | Jev coverage at ≤5% error | GLiNER coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG News | 4 | **0.910** | 0.700 | **[+0.130, +0.290]** | **0.830** | 0.240 |
| Banking77/BTZSC | 72 | **0.870** | 0.610 | **[+0.220, +0.300]** | **0.860** | 0.270 |
| DAIR Emotion | 6 | 0.480 | 0.440 | [−0.070, +0.150] | 0.000 | **0.020** |

The result is deliberately mixed. Jev has a clear accuracy and Brier-score advantage on AG News and
Banking77/BTZSC. On DAIR Emotion, the accuracy difference is unresolved and Jev is substantially
worse calibrated: Brier `0.846` versus `0.668`, NLL `5.588` versus `1.381`, and zero probability on
the true label for 16% of examples.

Latency is deployment-specific. GLiNER2.5 ran locally on an Apple M4 Max CPU; Jev was called as a
hosted service from France. GLiNER was faster on the 4- and 6-label tasks (~44 ms p50 versus
236–256 ms), while Jev was slightly faster on the 72-label condition (246 ms versus 296 ms p50).

Read the [full result](results/reports/btzsc-pilot-v1.md), inspect the
[machine-readable metrics](results/reports/btzsc-pilot-v1.json), or start with the frozen
[evaluation protocol](docs/PROTOCOL.md).

## What is measured

| Dimension | Metrics |
| --- | --- |
| Discrimination | Accuracy, macro-F1 |
| Probability quality | Multiclass Brier score, negative log likelihood, top-label ECE |
| Selective automation | Maximum threshold-realizable coverage at a fixed empirical error budget |
| Systems behavior | p50/p95 end-to-end latency, input tokens, failures, resolved model identity |
| Uncertainty | Paired, target-stratified bootstrap confidence intervals |
| Integrity | Probability-vector validation, artifact hashes, pinned revisions, uniform control |

Failures remain explicit. Accuracy and macro-F1 count them as incorrect; probability-only metrics
are computed over valid vectors and report the valid count separately. Rounded vectors may be
renormalized only when their raw sum is within a narrow declared tolerance, and the raw sum is kept.

## Quickstart

Requirements: Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AbdelStark/jev-benchmarks.git
cd jev-benchmarks
uv sync --extra benchmark --dev
```

Prepare the exact evaluation manifest without calling either model:

```bash
uv run jev-bench prepare --config configs/pilot-v1.yaml
```

Run one backend at a time. JSONL output is append-only and successful examples are skipped on
resume:

```bash
uv run jev-bench run --config configs/pilot-v1.yaml --backend gliner

export TYPESAFE_API_KEY=...
uv run jev-bench run --config configs/pilot-v1.yaml --backend jev
```

Build the JSON and Markdown reports:

```bash
uv run jev-bench report --config configs/pilot-v1.yaml
```

The Jev SDK also honors `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL`. Never commit credentials.
GLiNER downloads the pinned checkpoint on first use.

### V2 code path

[The v2 draft protocol](docs/PROTOCOL-v2.md) covers choice, binary noul and five-level score
questions across Jev via OpenRouter, three local Laya checkpoints and local Qwen3-1.7B logits.
GLiNER2.5 runs on all nine datasets as a descriptive contender and is excluded from the
confirmatory family.
`configs/v2.yaml` and `configs/probe-v2.yaml` are templates. A human operator must pin all
`TODO-PIN` revisions and paths, complete the synthetic capability checks, fill every
`TO-FILL-AFTER-PROBE` field, prepare the manifest, record hashes in `configs/v2-freeze.json`, and
create the `v2-preregistered` tag before test inference.

After pinning the external artifacts, commit both configs under `v2-probe`. The probe command
checks tokenization and the installed Laya package source; without `--static-only`, it also makes
synthetic local-model calls and writes an ignored `probe-results.json`:

```bash
uv run jev-bench probe --config configs/probe-v2.yaml --static-only
uv run jev-bench probe --config configs/probe-v2.yaml
```

Once revisions and probe fields are complete, prepare the manifest, fill
`configs/v2-freeze.json`, and commit the frozen artifacts under `v2-preregistered`:

```bash
uv sync --extra benchmark --extra laya --extra qwen --dev
uv run jev-bench prepare --config configs/v2.yaml
```

Before the v2 pilot, run the GLiNER harness anchor on the existing pilot-v1 manifest. It fails on
any accuracy, Brier, or historical macro-F1 mismatch and writes a comparison beside the raw runs:

```bash
uv run jev-bench anchor --config configs/pilot-v1.yaml
```

Then run named splits and backends:

```bash
uv run jev-bench run --config configs/v2.yaml --backend jev_openrouter --split pilot
uv run jev-bench run --config configs/v2.yaml --backend laya_base --split calibration
uv run jev-bench run --config configs/v2.yaml --backend qwen_logit --split test
uv run jev-bench run --config configs/v2.yaml --backend qwen_logit --split permutation
uv run jev-bench fewshot --config configs/v2.yaml --backend qwen_probe
uv run jev-bench fewshot --config configs/v2.yaml --backend tfidf_lr
uv run jev-bench fewshot --config configs/v2.yaml --backend prior
uv run jev-bench report --config configs/v2.yaml
```

GLiNER runs only as the anchor above, not as a v2 backend. The `fewshot` contenders are trained on
the 200 calibration labels per dataset (cross-fitted) and are not zero-shot. There is no latency
split; the report gives a test-split latency reference.

The v2 runner refuses a missing or hash-mismatched manifest, uses a separate attempt namespace for
each snapshot run, and runs local backends with a minimal offline process environment. Reports use
complete single-snapshot attempts, derive condition B from stored A probabilities, and write only
aggregate CSVs and hashes under `results/reports/`. The OpenRouter key is read only from the
environment at dispatch.

### Minimal and backend-specific installs

The package keeps heavyweight ML runtimes optional:

```bash
uv sync --dev                # metrics, reporting, tests
uv sync --extra data         # BTZSC preparation
uv sync --extra gliner       # local GLiNER inference
uv sync --extra jev          # hosted Jev inference
uv sync --extra benchmark    # complete v1 benchmark stack
uv sync --extra openrouter   # hosted v2 adapter (stdlib HTTP)
uv sync --extra laya         # local Laya runtime
uv sync --extra qwen         # local Qwen runtime
```

## Reproducibility contract

The pilot is anchored by:

- dataset revision `fef2a2ac62b69c58670047dddf045c53d7c3cb5e`;
- GLiNER checkpoint revision `235cf92d6d4318da9bfca0d08975c8fa7250d13b`;
- deterministic class-balanced sampling with seed `20260917`;
- protocol tag `pilot-v1-preregistered`, created before inference;
- resolved Jev model `jev-1.13.0`, recorded from every successful response;
- SHA-256 hashes for the config, manifest, and append-only prediction logs;
- exact metric configuration and 2,000 bootstrap resamples.

Raw examples and predictions live under ignored `results/runs/`. Aggregate results under
`results/reports/` contain no example text or credentials. Re-running `prepare` refuses to overwrite
a different manifest in the same experiment directory.

## How the pipeline fits together

```text
YAML experiment contract
        │
        ▼
Pinned BTZSC revision ──► deterministic manifest.jsonl
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Jev Choice adapter       GLiNER classifier
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                    validated prediction JSONL
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
      discrimination       calibration          risk / latency
             └────────────────────┬────────────────────┘
                                  ▼
                   JSON report + Markdown report
```

See [Architecture](docs/ARCHITECTURE.md) for module boundaries, artifact schemas, resume semantics,
and extension points.

## Adding a backend or benchmark

A backend implements three operations: `warmup`, `predict`, and `close`, returning the common
`Prediction` record. Dataset adapters produce `Example` records with an ordered label tuple and a
single target index. New comparisons should:

1. define the scientific question and frozen config;
2. compare only capabilities with equivalent output contracts;
3. pin dataset and model revisions;
4. add a fixture-level contract test that requires no network or accelerator;
5. preserve raw probabilities and failures before summarizing results.

Open an issue before adding a large benchmark family so dataset licensing, contamination risk, and
the primary metric can be reviewed first.

## Limitations

- This is a 300-example pilot, not a leaderboard or a universal model ranking.
- Public benchmark data may have appeared in either model's training data.
- The 5% error-budget thresholds are selected and evaluated on the same pilot slice; they are
  descriptive, not deployment thresholds.
- Most Banking77/BTZSC classes have one or two sampled examples, so per-class estimates are unstable.
- Hosted Jev latency and local CPU GLiNER latency are operational measurements, not normalized model
  throughput.
- The pinned BTZSC Banking77 configuration exposes 72 hypotheses and 200 rows with no positive
  candidate. Those rows are excluded rather than assigned a fabricated class.
- NLL is sensitive to exact zero probabilities. This is intentional: assigning zero mass to the
  observed outcome should be penalized strongly.

## Development

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

The test suite uses deterministic fixtures and fake provider/model boundaries; CI never requires API
keys or downloads a checkpoint. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md).

## Citation and references

If this repository supports published work, cite the software using [CITATION.cff](CITATION.cff).
The benchmark builds on:

- Ilias Aarab, *BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders,
  Embedding Models, Rerankers and LLMs*, ICLR 2026.
- Urchade Zaratiana et al., *GLiNER2: An Efficient Multi-Task Information Extraction System with
  Schema-Driven Interface*, EMNLP 2025 Demo.
- TypeSafe's official [Jev introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  and [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python).

Licensed under the [Apache License 2.0](LICENSE).
