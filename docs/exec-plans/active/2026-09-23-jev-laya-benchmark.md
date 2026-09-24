# Plan: Jev vs Laya vs local LLM-logit decision benchmark (v2)
_Locked via claudex-loop — by Claude + Dr. Yang Hu, 2026-09-23_

Review log: `docs/exec-plans/active/2026-09-23-jev-laya-benchmark.review-log.md`
Recon/handoff record: `docs/exec-plans/active/2026-09-23-jev-laya-benchmark.handoff.md`
Study pointer plan: `study/docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md`

## Goal

Measure, in one frozen protocol on identical items, how three routes to "typed decisions with probabilities" compare:

1. **Hosted Jev 1.13**, via OpenRouter.
2. **Open-weights Laya**, three checkpoints run locally.
3. **A small local LLM** (Qwen3-1.7B) reading final-position logits of option-label tokens and softmax-normalizing them — the approach the user originally asked about.

Coverage: all three question types (choice / noul / score), on discrimination, probability quality, selective automation, order sensitivity, run-to-run variance, and latency. Each contender is scored both as-returned and after the same post-hoc temperature scaling.

Deliverables:
- aggregate reports in this repo;
- a Chinese R Markdown essay in `study/` whose every number is computed from vendored aggregate CSVs.

This is a **shared-capability comparison, not a claim about overall intelligence**.

## Approach

### P0 — Environment, pinning, capability probes (no benchmark-item inference)

1. `uv` venv with Python 3.12 (`.python-version` is already 3.12). New optional extras, all lazily imported:
   - `openrouter`: stdlib `urllib`, or `httpx` if needed;
   - `laya`: `laya==0.3.9`, torch, transformers;
   - `qwen`: torch, transformers;
   - `data`: already exists.
2. **Local artifact locations**, outside the git repository (AGENTS.md: never commit checkpoints or caches). Both are created at P0 and referenced from the config as absolute paths:
   - `experiments/models/hf-cache/` — `HF_HOME` for the Laya and Qwen weights (~6 GB: Laya's three subfolders ≈ 2.3 GB, Qwen3-1.7B ≈ 3.5 GB). Downloaded once at the pinned revisions; afterwards every local-model process runs with `HF_HUB_OFFLINE=1`.
   - `experiments/models/sandbox-home/` — the empty `HOME` passed to local-model processes via `env -i`, with no shell profile and no credentials.
   - Jev has no local artifacts.
   - Dataset downloads stay in the repo's gitignored `results/cache/`.
3. Record `torch.backends.mps.is_available()`, the torch/transformers/laya versions, and the hardware (M1 Pro 16 GB) in runtime metadata.
4. **Pin first.** Every external artifact is pinned by immutable revision (see the pinning list at the end of P0) and committed **before** any capability check, so the probes run against the frozen artifacts (R2#5).
5. **Capability checks before the protocol freeze**, results recorded in `docs/PROTOCOL-v2.md` (R2#5):
   - **Static checks (no model calls):** the tokenizer-only checks below and the package-source read.
   - **Synthetic probe calls:** the only model calls allowed before `v2-preregistered`. They run on a committed `configs/probe-v2.yaml` tagged `v2-probe` and use **only synthetic, non-dataset strings**, so no benchmark item is ever seen.

   The checks:
   - **Synthetic probe:** Laya's output shape for `score` (per-level distribution vs expected value only) and `noul` (P(true)) (#3). The shared score metrics are fixed regardless of the result (R2#8):
     - MAE of the expected level;
     - exact-level accuracy of `round(expected level)`, with ties rounded up.

     Every contender is scored with these same two rules. Per-level Brier, NLL and ECE for `score` are **non-shared** and reported only for contenders exposing distributions.
   - **Scalar-only path (R3#1):** if a contender returns only an expected level, its `Prediction` carries `expected_score` with an empty `probabilities`. A separate validator checks that `expected_score` is finite and within [min level, max level]. Condition **B is marked unavailable** for that contender × score dataset; no scalar calibration method is attempted. For distribution-output contenders, `expected_score = Σ level·p`.
   - **Static (tokenizer counting), input length (R3#2):** for every contender's tokenizer and context budget, the full encoded request — item text + question + all options — must fit. **Frozen common exclusion rule:** an item that would be truncated by *any* contender assigned to its dataset is excluded from that dataset's pool for **all** contenders, before sampling, and never truncated.
     - The binding constraint is Laya English: 512 tokens total, option head 192–256.
     - Excluded counts and the resulting length distribution are reported. The bias toward shorter texts, notably in Civil Comments and Yelp, is stated as a limitation.
     - A per-call assertion re-checks that no text or option was truncated.
   - **Static (tokenizer counting), options:** for every Laya checkpoint × choice dataset, the smallest `head_max_len` at which **every complete option description** is encoded without truncation (#5). **Synthetic probe:** confirm that this setting actually runs. A per-call assertion in the adapter enforces it at run time.
   - **Static:** Qwen rendered-prompt tokenization (#14): for the exact chat template, `Answer: ` prefix and every option ID of every dataset, the suffix token sequence of each complete ID. Single-token IDs → next-token scoring; otherwise full sequence log-probability scoring for that dataset. Recorded per dataset.
   - **Static:** supply-chain read of the pinned `laya` package source (#12): which files it loads or executes from the HF repo. The finding is recorded; any remote `.py` import is a STOP for the Laya track.
6. **Confirmatory dataset sets (R2#1, R3#4).** From the probe results, and **before** `v2-preregistered`, `PROTOCOL-v2.md` records:
   - the three-way set: English datasets where Jev, Qwen and Laya-base all produce valid full probability vectors under the length rule, with Laya untruncated. With ≥ 3 datasets, C1–C3 all use this set (R4#4);
   - otherwise C2/C3 leave the confirmatory family and C1 uses the Jev ∩ Qwen set;
   - the resulting exact list of confirmatory tests and the Holm denominator k.

The pinning list referenced in item 3 — every external artifact is pinned by immutable revision in the config before any inference:
   - HF model revisions: `convaiinnovations/laya` (all three subfolders) and `Qwen/Qwen3-1.7B`;
   - HF dataset revisions: `btzsc/btzsc` (already pinned `fef2a2ac…`), SMS Spam, Civil Comments, Yelp Review Full, MASSIVE.

### P1 — Data and protocol freeze

1. Extend `data.py` with loaders for each dataset, all producing `Example` with a new `question_type` ∈ {`choice`, `noul`, `score`}:
   - **choice:** BTZSC `agnews`, `emotiondair`, `banking77` (72 hypotheses; out-of-scope rows excluded exactly as in upstream pilot-v1). MASSIVE intent `en-US`, `zh-CN`, `km-KH`, with English intent descriptions as options.
   - **noul:** SMS Spam and Civil Comments (CC0; binary toxicity = `toxicity ≥ 0.5`). ToxicChat was replaced: it is CC-BY-NC, and hosted inference would send its text to OpenRouter/TypeSafe (#13). Labels are the ordered pair (`false`, `true`); target is the index.
   - **score:** Yelp Review Full. Labels are the ordered levels 1★–5★; target is the level index.
2. Per dataset, draw three **disjoint** ID sets with seed `20260923`, class-balanced via the existing `_balanced_indices`:
   - `pilot`: 30 items; plumbing and timing only, never reported as results;
   - `calibration`: 200 items; used only to fit temperatures;
   - `test`: 300 items, or 100 for each MASSIVE locale.

   **Class universe (R2#9):** the full label set of each dataset is always offered as options. **Macro-F1 denominator = labels present in (targets ∪ predictions) of the evaluated sample**, recomputed in every bootstrap resample. A label that is predicted but absent from targets contributes F1 = 0; a label absent from both is excluded. The upstream `_macro_f1`, which assigns 0 to every absent class, is replaced to match, with a unit test. Per-class support is reported. For Banking77 (~4 per class) and MASSIVE (<2 per class), macro-F1 is **exploratory**; accuracy and Brier are the reported metrics for those slices (#6).

   Where a dataset exposes only one split (BTZSC test; SMS Spam train), the three sets are carved from that split by disjoint IDs. This is recorded per dataset.

   **Content disjointness (R2#6):**
   - Before sampling, items are grouped by a normalized-text hash (NFKC, lowercase, whitespace-collapsed, punctuation-stripped). Each group is assigned to exactly one split, so exact and trivial near-duplicates cannot straddle calibration and test.
   - MASSIVE parallel utterances are grouped by their cross-locale utterance ID, so a translated item never lands in one locale's calibration and another locale's test.
   - The number of groups merged is reported per dataset.
3. Derived suites, built from the test manifest:
   - **permutation:** 100 test items per BTZSC choice dataset × 3 fixed option-order permutations, plus the identity order. Two estimands (#9):
     - **order-only:** option identifiers stay attached to their labels while the listing order changes — Jev's stable IDs, Laya's dict keys, and Qwen with letters bound to labels (e.g. `C) Sports` may be listed first);
     - **order + label-token:** Qwen with positional letters (A is always first). Reported separately and never compared with the order-only rates;
   - **latency:** 100 AG News items. **Cross-contender comparison uses only the common operation**: one item, one choice question, one call. Scaling workloads are backend-specific and compared only within a backend (#10):
     - Jev and Laya: 1 vs 10 questions sharing one state (ten paraphrased choice questions over the same label set, fixed in the config);
     - Qwen: batch size 1 vs 10 independent prompts.
4. Write `docs/PROTOCOL-v2.md`: hypotheses, primary and secondary metrics, conditions, exclusions, analysis plan, and the list of exploratory analyses. Commit the protocol, configs and manifests' SHA-256, then create git tag `v2-preregistered` **before any test-slice inference**. `prepare` refuses to overwrite a differing manifest (existing behaviour).

### P2 — Adapters and runner (TDD with fake boundaries)

1. Extend `models.py` without breaking existing JSONL.
   - `Example` gains `question_type`, `split`, `permutation_id` (default `identity`) and `locale` (default `None`).
   - `Prediction` gains `question_type`, `split`, `condition` (`A_raw`), `repeat_index` (default 0), `permutation_id`, `raw_probabilities` (as returned, before any renormalization), `expected_score` (score type; the only output for scalar-only contenders, R3#1), `model_latency_seconds` (local only), `device`, `cost_usd` and `resolved_checkpoint`. The runner dispatches validation by `question_type` and output shape.

   All new fields are defaulted so upstream records and tests still load.
2. **`jev_openrouter` adapter:**
   - Calls `POST https://openrouter.ai/api/alpha/decisions` with `{model: "typesafe/jev-1.13", state, questions}`.
   - Option IDs are stable per label (`label_000…` keyed to label identity, not position). A permutation changes only insertion order.
   - `criteria` format: object for choice, array for score, `{"true": …, "false": …}` for noul.
   - Records the resolved `model` (e.g. `jev-1.13-20260917`), `usage.cost` and `usage.input_tokens`.
   - Reads `OPENROUTER_API_KEY` from the environment only.
   - Serial requests for the latency suite; at most 4 concurrent otherwise.
   - Retries only 429/5xx, with backoff and `retry-after`; every attempt is logged.
   - Budget guard (#11):
     - Before dispatch, atomically reserve a conservative per-attempt cost from the **full serialized request body**: UTF-8 bytes (an upper bound on tokens) × list price × 1.5, with a minimum reservation (R2#7).
     - At most 4 attempts in flight, so outstanding liability is bounded by 4 reservations.
     - After a response, the reservation is settled to `usage.cost`. A **2xx** response lacking `usage.cost`, a timeout or a transport error keeps its full reservation and pauses dispatch pending operator review.
     - A 429/5xx without `usage.cost` keeps its full reservation, which stays counted against the cap, and is retried within the attempt limit. A non-retryable 4xx without cost keeps its reservation and fails that item. Unbilled errors can therefore only over-count spend, never under-count it, so outstanding liability stays below the cap (review A5-F3).
     - **Refuse to dispatch** any request whose reservation would push reserved + settled spend above **$2.00** (R2#7).
3. **`laya` adapter:**
   - Loads the pinned `laya` package and pinned HF revision, with `trust_remote_code=False` and safetensors only. The repository's `.py` files are **not executed** unless the pinned `laya` package imports them; verify by reading the package before P3, and record the finding.
   - Checkpoints: `laya` (English base) for English datasets; `laya-multilingual` via `Router` for MASSIVE zh/km (and en for parity); `laya-typed-decisions` for English datasets, labelled out-of-domain.
   - Device MPS → CPU fallback; the device is recorded.
   - `model_latency_seconds` is timed around the forward pass with `torch.mps.synchronize()`.
   - Primary `head_max_len` = the P0 no-truncation value per dataset. The as-shipped default is an exploratory condition on K > 20 datasets. If no feasible no-truncation setting runs, Laya is excluded from that dataset (reported N/A) (#5).
4. **`qwen_logit` adapter:**
   - `Qwen/Qwen3-1.7B`, pinned revision, bf16/fp16 on MPS, `enable_thinking=False`.
   - One fixed chat template (in the config, not tuned per dataset), listing options as `A) …`, `B) …`. The assistant turn is prefilled with `Answer: `.
   - Reads next-token logits for the option-label tokens only and softmaxes over the K labels.
   - Token scoring follows the P0 rendered-prompt check (#14). **K ≤ 26:** single-letter labels A–Z, when each is a single suffix token in the exact rendered prompt.
   - **K > 26 (Banking77 72, MASSIVE 60):** zero-padded two-digit IDs `01…K`, with exact joint probability
     $P(d_1d_2) = P(d_1)\,P(d_2 \mid d_1)$
     from one prefix pass plus one batched pass over the distinct $d_1$ values (reusing the KV cache), renormalized over the K valid IDs. Used only if the P0 check shows every ID renders as exactly two suffix tokens; otherwise full sequence log-probability scoring over each complete ID.
   - noul uses options A/B (`true`/`false` in a fixed order). score uses levels A–E in ascending order.
5. **Runner changes:**
   - Runs a named split: `pilot`, `calibration`, `test`, `permutation` or `latency`.
   - Supports `repeats` (Jev = 3; local = 1 — deterministic, verified by one duplicate call on the pilot).
   - Refuses to run unless the config and manifest hashes match those recorded under tag `v2-preregistered`. `run` never generates a missing manifest (unlike the current runner) (#4).
   - Every run writes into its own **attempt namespace** `results/runs/<experiment>/<backend>/attempt-<n>/` (R4#2). A Jev snapshot change starts a new attempt, so no earlier row is reused. Reports select exactly one complete attempt per backend: the single-snapshot attempt that covers the full shared slice. The chosen attempt ID and resolved snapshot are recorded in the report.
   - Within an attempt, records are keyed by `(split, example_id, permutation_id, letter_mode, repeat_index)`. Only condition A is ever inferred; condition B is a pure derivation from stored A probabilities plus the frozen T, never a second model call (#4).
   - Failures remain in the denominator (upstream contract).
   - The existing `abs_tol=0.02` renormalization rule stays. `raw_probabilities` and `probability_sum_raw` are kept.
6. **Secret hygiene:**
   - The runner refuses to write any row, log line or report whose serialized form contains the API key.
   - Local backends (Laya, Qwen) run in a separate process launched with an explicit minimal environment: `env -i PATH HOME=<scratch home> HF_HOME=<pinned cache> HF_HUB_OFFLINE=1` after a one-time download. No `OPENROUTER_API_KEY`, no inherited shell profile, and `trust_remote_code=False` (#12).
   - Residual risk, accepted and stated: this is process-environment isolation, not an OS sandbox. Code from the inspected Apache-2.0 package could still read files by absolute path.
   - No key in configs, notebooks, Rmd or git.
7. Deterministic contract tests for each new adapter, with fake HTTP/model boundaries: request shape, criteria encoding per question type, permutation keeps IDs stable, two-digit joint-probability math on a toy logit table, budget abort, key-scrub refusal, resume keys with repeats. No network in tests.

### P3 — Pilot (30 per condition, disjoint from test)

1. Run all contenders on `pilot`. Check:
   - every output validates against its shape-specific validator (full vector, or `expected_score` for scalar-only score outputs, R4#1);
   - tokenizer assertions hold;
   - wall-clock per contender is estimated;
   - Jev cost per call matches expectation.
2. Pilot numbers are written to `results/reports/pilot-v2.*`, labelled PILOT, and never used to change prompts, thresholds or conditions. Any protocol change after the pilot requires a new protocol revision and tag; the pilot slice stays excluded from the reported test slice.

### P4 — Calibration fit + main run

1. Run every contender on `calibration`. For each contender × dataset **that returns a full probability vector** (scalar-only score outputs are skipped, R4#1), fit one temperature $T$ by minimizing NLL on floored probabilities (bounded scalar search over $T \in [0.05, 20]$):

   $$\tilde p_i \propto \exp\!\big(\log \max(p_i, \varepsilon)/T\big), \qquad \varepsilon = 0.005$$

   Condition **B** = test probabilities transformed with the frozen $T$. Jev is included.
   - **"Calibration built in" is judged on the test set, not by $T$** (#8): A-vs-B differences in ECE, Brier and NLL, plus reliability diagrams, with intervals from a joint bootstrap that resamples the calibration set, **refits $T$** and resamples the test set.
   - **Selective-automation threshold** (#1): the confidence threshold meeting the 5% error budget is chosen on the calibration split. Realized coverage and error are reported on test, including a "no feasible threshold" outcome. The upstream same-slice coverage is kept only as a labelled descriptive number.
2. Run `test`, `permutation` and `latency`. Local latency uses 5 excluded warm-ups; Jev uses none (the upstream billing-accounting rationale).

### P5 — Metrics and report

All metrics are reported for conditions A and B, per contender × dataset. For scalar-only score outputs, only condition A and the two shared score metrics are reported; vector metrics and B are marked unavailable (R4#1).

- **Discrimination:** accuracy and macro-F1 (primary), Brier (primary).
- **Probability quality:**
  - NLL computed on ε-floored probabilities (replacing the upstream `1e-12` clip, which is kept as `nll_clip12` for comparability), plus the share of items whose raw true-label probability is 0;
  - top-label ECE with 10 bins, where confidence = max probability. Jev's own `confidence` field is logged but not used; its documented formula is `clamp((K·p_max−1)/(K−1),0,1)`;
  - coverage at a 5% error budget, with the threshold chosen on calibration and applied to test (see P4).
- **Score type:** additionally MAE of the expected level, and quadratic-weighted kappa.
- **Robustness:**
  - flip rate across the 4 orders (argmax changes relative to identity). For Jev, the **identical-order repeat flip rate** (repeats 0 vs 1 vs 2 at identity order) is reported alongside it, together with the excess order effect = permutation flip rate − repeat flip rate, with a paired bootstrap CI (R2#10);
  - Jev repeat variance: argmax agreement across 3 repeats, and mean absolute probability difference.

  Primary Jev metrics use `repeat_index = 0`. Mean-of-3 is secondary.
- **Rounding-parity sensitivity:** Laya/Qwen probabilities rounded to 2 decimals, renormalized, and all metrics recomputed.
- **Latency:** p50/p95 end-to-end for all contenders; model-only for local contenders; per-question latency at 1 vs 10 questions per call.
- **Failure handling (#2, R2#2):** primary comparisons use **every shared item**.
  - A failed call counts as incorrect for accuracy and macro-F1.
  - It receives the **worst-case penalty** for probability metrics: Brier = 2 (the maximum), NLL = −log ε with ε = 0.005. This way a failure can never score better than any answer.
  - **Failure rate** per contender × dataset is reported as its own outcome.
  - Metrics on successful calls only are labelled *conditional*. `paired_bootstrap` gains the failure-aware version, and the upstream valid-only version is kept as the conditional variant.
- **Primary contrasts and multiplicity (#7, R2#3, R2#4, R3#4):** **up to 9 confirmatory tests. The exact family and the Holm denominator are frozen in `PROTOCOL-v2.md` after P0, before `v2-preregistered`.**
  - Each contrast averages over its own frozen dataset set:
    - **When C2/C3 are in the family, all three contrasts use the same three-way-intersection dataset set**, so the effects refer to identical items and can support a joint ranking (R4#4). The broader Jev ∩ Qwen comparison is reported separately as descriptive;
    - only when C2/C3 drop out does C1 use the Jev ∩ Qwen set.
  - If C2 and C3 are dropped (fewer than 3 common datasets), the family is C1 × 3 metrics = 3 tests, and Holm runs across 3.
  - Contrasts: C1 Jev vs Qwen, C2 Jev vs Laya-base, C3 Qwen vs Laya-base.
  - Each contrast is tested on three metrics, always policy-matched:
    - accuracy — temperature scaling cannot change the argmax, so this one test covers both conditions;
    - Brier under **A-vs-A**;
    - Brier under **B-vs-B**.
  - Every metric is the equal-weight average over that contrast's frozen dataset set.
  - The mixed-policy contrasts (Jev-A vs Qwen-B, and similar) are reported separately as descriptive "as-deployed" comparisons.
  - **Intervals:** a dataset-stratified paired bootstrap with 2,000 resamples. Each resample redraws the test set **and** the calibration set and refits every condition-B temperature, so the confirmatory intervals carry the uncertainty of fitting T.
  - **Holm decisions:** two-sided bootstrap p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)), floored at 1/2000. Holm step-down across the **frozen family of k tests** (k = 9 or 3, as recorded in `PROTOCOL-v2.md`) at family-wise α = 0.05. The report states k (R4#3). Both adjusted decisions and unadjusted CIs are reported.
  - Everything else — per-dataset results, other metrics, multilingual, permutation, latency, mixed-policy contrasts — is **descriptive**, with unadjusted 95% CIs, and is never phrased as a winner claim.
- **Exploratory, labelled:** Laya **as-shipped default** `head_max_len` on K > 20 datasets (the primary Laya setting is the P0 no-truncation value, #5); mean-of-3 Jev; rounding parity; macro-F1 on Banking77 and MASSIVE.
- **Output:**
  - `results/reports/v2.json` and `v2.md`;
  - tidy aggregate CSVs (no example text) for R: `metrics.csv`, `pairwise_ci.csv`, `reliability_bins.csv`, `latency.csv`, `flip.csv`, `repeat.csv`, `temperature.csv`;
  - `SHA256SUMS`.

### P6 — Study Rmd

1. Vendor the aggregate CSVs and `SHA256SUMS` into `study/docs/data-external/jev-laya-bench/`. The Rmd verifies the hashes in its setup chunk and stops on mismatch.
2. Write `study/jev_laya_benchmark_zh.Rmd` following `study/CLAUDE.md`:
   - Chinese prose, English figure labels, `ft_show()` tables, `code_folding: hide` + `toc_float`.
   - **Every number comes from `reg()`** (`kind = "source"`, with the run id and CSV hash in `note`).
   - Sections: question; three routes; protocol; results by question type; calibration A vs B; order sensitivity; Jev non-determinism; latency (with the OpenRouter-hop caveat); limitations (contamination, OpenRouter vs direct API, pilot scale, Laya typed-decisions out-of-domain, the Jev/Laya/Qwen contrasts are zero-shot with no fine-tuning; the Amendment 5 few-label arm (`qwen_probe`, `tfidf_lr`, `prior`) is supervised on 200 calibration labels and labelled as such (review A5-R2-F3)); answers to wiki Q165/Q166/Q167/Q174/Q183/Q189/Q190/Q191/Q192.
3. Render: `rm -rf jev_laya_benchmark_zh_files`, parse-check all chunks, then render in the background. Gates: G0 render freshness, figures base64-embedded, zero `Execution halted`, every `REG$key` consumed after its `reg()`. Never publish.
4. Add a row to `study/docs/PLANS.md`.

### Amendment 6 — scope cut and headline question (user-approved 2026-09-24)

Source: the approved assessment's value analysis. Public work already covers zero-shot accuracy rankings, hosted-vs-local latency and GLiNER. The study's distinct value is one question:

> **When a local model gets the same ~200 labels, how much of zero-shot Jev's advantage is left — in accuracy, in calibration (A-vs-A, B-vs-B), and in how much traffic can be automated at a 5% error budget?**

**Unchanged:** the frozen confirmatory family (C1–C3 zero-shot, k = 9, AG News / Emotion / SMS / Civil), the few-label arm (Amendment 5, descriptive), the budget and provenance rules, and all metrics.

**Cuts:**
- **Latency suite removed** (`dataset.latency_items: 0`): no `latency-1`/`latency-10` rows. The only latency reported is each contender's per-call wall time p50/p95 on the ordinary test split, labelled as a deployment-specific reference (hosted Jev includes an OpenRouter hop). The few-label arm reports none (Amendment 5).
- **Permutation suite on AG News only** (`dataset.permutation_datasets: [agnews]`): 100 items × 4 orders; Jev's excess order effect over repeat noise and Qwen's two letter modes are still reported there. The Laya as-shipped-head exploratory rows on K > 20 datasets (MASSIVE) are kept.
- **`laya_typed` removed** from the v2 models: the checkpoint is out of domain for these datasets and not in the confirmatory family.
- **GLiNER removed from the v2 runs:** it remains only as the P3.0 harness anchor, run on the upstream pilot-v1 config and manifest.

**Probe validity (review A6-F1).**
- The `v2-probe` record (tag `v2-probe`, commit `8720bb5`, synthetic strings only) remains the capability check for the retained contenders: `laya_base`, `laya_multilingual` and `qwen_logit`.
- Removing contenders can only shrink the common length exclusion. `laya_typed` has a 1,024-token budget and the same tokenizer as `laya_base` (512), so its exclusions are a subset of base's.
- Final exclusion counts come from `jev-bench prepare` on the Amendment 6 config and are reported from `manifest-summary.json`.
- `run_probe` now follows the configured Laya backends. A rerun needs a new probe tag, because the current config no longer matches `v2-probe`.

**Anchor isolation (review A6-F2).**
- The pilot-v1 config has no `local_runtime`, so the P3.0 anchor loads GLiNER in-process.
- Before loading, it removes `OPENROUTER_API_KEY` and `TYPESAFE_API_KEY` from the environment and sets `HF_HUB_OFFLINE=1`, so it uses only the pinned cache.
- A contract test checks that the key is absent and the hub is offline when the backend is constructed.

**Reporting order (report and study Rmd):**
1. The headline question: few-label contenders versus zero-shot Jev (accuracy, Brier A/B, coverage at 5% error).
2. The confirmatory zero-shot family (C1–C3).
3. Calibration A vs B and selective automation per contender.
4. Secondary: multilingual (MASSIVE en/zh/km), yes/no and score datasets, AG News order sensitivity.
5. The relation-to-public-results table and limitations.

**Tasks (Amendment 6):**
- [ ] `build_derived_suites` takes `permutation_datasets` (default unchanged); `load_v2_examples` passes `dataset.permutation_datasets`; contract test.
- [ ] `configs/v2.yaml`: `permutation_datasets: [agnews]`, `latency_items: 0`; remove the `laya_typed` and `gliner` models.
- [ ] `PROTOCOL-v2.md`: Amendment 6 section; latency, permutation and contender text updated.
- [ ] Report/Rmd section order as above (P5/P6).
- [x] `run_probe` follows the configured Laya backends (A6-F1).
- [x] The anchor drops credentials and forces offline before loading GLiNER, with a contract test (A6-F2).

### Amendment 5 — few-label arm, confirmatory set, framing (user-approved 2026-09-24)

Source: the cross-provider-approved assessment `docs/exec-plans/active/2026-09-24-benchmark-assessment.md` (sha `851394f6…`). The user accepted P2, P3 and C1 from it.

**A. Few-label arm (P2) — descriptive, never in C1–C3.** It answers the practical question the zero-shot comparison leaves open: does ~200 in-distribution labels plus a small local model match zero-shot Jev?
- Three new contenders:
  - `qwen_probe`: logistic regression on Qwen3-1.7B hidden states;
  - `tfidf_lr`: logistic regression on character TF-IDF;
  - `prior`: calibration-label class frequencies, a Brier/accuracy reference.
- **Labels:** only the 200 `calibration` items of each dataset (each MASSIVE locale separately). Pilot and test labels are never used for fitting. Calibration and test are already content-group disjoint (P1).
- **`qwen_probe` features:**
  - model: the pinned Qwen3-1.7B;
  - prompt: the zero-shot prompt (identity order, stable IDs, same prefill);
  - position: the last prompt token, whose next token is the option ID;
  - layer: output of decoder block 18 of 28 (`hidden_states[18]`), cast to fp32. The layer is fixed a priori from AnyJev's public report (block 18/28 for Qwen3-1.7B) and is not tuned.
  - Extraction runs in the same minimal-environment local worker, after `v2-preregistered`, on `calibration` and `test` only.
  - Features are saved as ignored artifacts `results/runs/<experiment>/qwen_probe/attempt-<n>/features-{calibration,test}.npz` (example IDs + float32 matrix). Their SHA-256 goes into the attempt metadata.
- **`tfidf_lr` features:** `char_wb` 2–4-gram TF-IDF with sublinear tf. The vectorizer is fitted **inside each cross-fitting training fold**, together with the classifier, and refitted on all 200 calibration texts for test prediction. Held-out fold texts never shape their own features (review A5-F2). Character n-grams are used so zh/km need no word segmenter.
- **Classifier (both):**
  - multinomial logistic regression, L2, `C = 1.0`, `lbfgs`, `max_iter = 2000`, `random_state = seed`;
  - Qwen features are standardized with statistics from the training fold;
  - all hyperparameters are frozen; there is no search.
  - Classes absent from a training fold get probability 0 in the full label vector (the ε floor applies to NLL and T fitting, as for Jev).
  - Score (UltraFeedback) is nominal 5-class with `expected_score = Σ level·p`, as for GLiNER. noul is binary.
- **Cross-fitting:**
  - unstratified 5-fold split, `KFold(n_splits=5, shuffle=True, random_state=20260923)`, over the 200 calibration items for **every** dataset. Stratification is infeasible: Banking77 and MASSIVE calibration classes have 2–4 items each (review A5-F1). Out-of-fold probabilities are the contender's `calibration` predictions.
  - A class absent from a training fold gets probability 0 on that fold's held-out rows. Its ε-floored contribution enters NLL and the temperature fit, and the absent-class count per dataset is reported;
  - one refit on all 200 gives the `test` predictions.
  - Condition A is the classifier probability. Condition B is the existing temperature fit on those out-of-fold predictions, so the A/B, selective-threshold and bootstrap machinery is reused unchanged.
  - Stated limitation: the bootstrap refits T on resampled out-of-fold predictions but does not retrain the classifier, so few-label intervals understate training variance.
  - `prior` uses training-fold frequencies for out-of-fold calibration rows and all-200 frequencies for test.
- **Reporting:**
  - pairwise with Jev (A-vs-A, B-vs-B, accuracy), unadjusted 95% CIs, labelled "uses 200 in-distribution labels; not zero-shot";
  - excluded from the permutation and latency suites. Their prediction rows carry no latency fields in reports, since training and batch prediction time is not the common single-call operation (review A5-F5).
- **Dependency:** `scikit-learn` in a new lazily imported `fewshot` extra, also added to `benchmark`.

**B. Confirmatory family (P3).**
- The three-way English set is AG News, DAIR Emotion, SMS Spam and Civil Comments:
  - Banking77: Laya N/A (the 72 complete option descriptions need a ~1,280-token head);
  - UltraFeedback: descriptive only, because its GPT-4 labels favour LLM-family contenders.
- With 4 datasets (≥ 3), C1–C3 stay in the family: `k = 9` (C1–C3 × {accuracy, Brier A-vs-A, Brier B-vs-B}).
- The exact test list is copied into `configs/v2.yaml` once the synthetic probe confirms that Laya-base returns full noul vectors. If it does not, the plan's own C1-only fallback (`k = 3`) applies.

**C. Framing (C1).**
- The report and the study Rmd call this a **same-run controlled re-check**, never a first result.
- They include a "relation to public results" table (elcronos, open-alternative-jev, AnyJev, nibzard, Anthus, upstream pilot) with protocol differences.
- The assessment's public-evidence priors are recorded in `PROTOCOL-v2.md` as non-binding expectations; the tests remain two-sided.
- Accuracy differences smaller than the §13 sensitivity table's detectable range are reported as "unresolved".

**D. Budget ledger hardening (review A5-R2-F1, A5-R2-F2).** These are defects in the existing Jev dispatch path, fixed before any hosted call.
- **One experiment-level ledger:** `results/runs/<experiment>/jev_openrouter/budget-ledger.jsonl`.
  - Every HTTP dispatch gets a fresh unique transaction ID (`uuid4`).
  - Before dispatch, `{"event": "reserved", "txn": ID, "amount": A}` is appended and flushed with `fsync`.
  - After the response, exactly one closing record is appended: `{"event": "settled", "txn": ID, "cost": C}` (C finite ≥ 0), or `{"event": "retained", "txn": ID}` (no valid cost).
- **Reconstruction on start:** liability = Σ `cost` over `settled` + Σ `amount` over every reservation **not closed by `settled`**, i.e. both `retained` and unclosed reservations count at the full reservation.
  - Only a valid finite-cost settlement replaces a reservation.
  - The runner refuses to start if the ledger has a duplicate transaction ID, a closing record without a reservation, or two closing records for one ID.
  - A crash between dispatch and response can therefore only over-count spend (review A5-R3-F1, A5-R3-F2).
- **Single dispatcher:** the Jev run holds an exclusive non-blocking `fcntl.flock` on `jev_openrouter/budget.lock` for its whole lifetime. A second concurrent Jev process refuses to start.
- **Accounting validation:** `usage` must be an object and `usage.cost` a finite number ≥ 0 (bool excluded). Otherwise a 2xx response is treated as missing its cost: the reservation is retained, the attempt is logged, and dispatch pauses. Non-2xx responses follow the rules above. A malformed value never reaches `Budget.settle`.
- **Contract tests (fake transport):** an exception between reserve and response is counted on restart; a retained 429 reservation still counts after restart; duplicate, unmatched and double-closed transactions refuse to start; a concurrent lock is refused; `usage: null`, cost `"NaN"`, `NaN`, `-1`, `"0.1"` and `true` all pause without settling; the ledger survives the key-scrub writer.

**Tasks (Amendment 5):**
- [x] `fewshot` extra + `uv.lock`; lazily imported `src/jev_benchmarks/fewshot.py` (feature extraction via the local worker, cross-fitting, prior, prediction writing into attempt namespaces).
- [x] CLI `jev-bench fewshot --config configs/v2.yaml --backend {qwen_probe,tfidf_lr,prior}`; refuses without the frozen manifest, like `run`.
- [x] Report: include the few-label contenders as descriptive rows and pairwise-vs-Jev intervals; add the relation-to-public-results table source.
- [x] Contract tests with fakes: no test label reaches fitting; out-of-fold coverage is exact; absent-class zero mapping; a Banking77-like case (72 classes, 2–3 items each) and a MASSIVE-like case run; TF-IDF vocabulary is per-fold (a held-out-only token has no feature); deterministic output for a fixed seed; features hash recorded; prior frequencies; latency fields absent for few-label rows.
- [x] `configs/v2.yaml`: few-label contender entries, and the confirmatory family (k=9) filled from the probe
- [x] `PROTOCOL-v2.md`: few-label section, confirmatory set, priors, framing.
- [x] Budget ledger hardening (D) in `jev_openrouter.py` and `v2_runner.py`, with its contract tests.

### Amendment 4 — score dataset swap (user-approved 2026-09-23)

Yelp Review Full is replaced by **UltraFeedback helpfulness** (`openbmb/UltraFeedback` @ `40b4365`, MIT; `truthful_qa.jsonl` + `false_qa.jsonl`).
- Candidates are prompt–completion pairs with a numeric 1–5 helpfulness rating; N/A ratings are dropped.
- All completions of a prompt form one group. The splitter keeps **one seeded-hash representative completion per prompt**, so the frozen estimand is the helpfulness of one randomly chosen completion per prompt. Items are then independent across prompts, as the bootstrap assumes. Other completions are not used (review A5-F4).
- Labels are the UltraFeedback rubric levels.
- The Yelp terms restrict third-party disclosure, and hosted Jev receives the text (same reasoning as the ToxicChat swap, #13).
- Limitation: the ratings are GPT-4 annotations.

### Amendment 3 — GLiNER2.5 descriptive contender (user-requested 2026-09-23)

- `fastino/gliner2.5-multi-v1` at the upstream-pinned revision `235cf92d6d4318da9bfca0d08975c8fa7250d13b`, CPU, batch size 1. It reuses the upstream `gliner` adapter through the v2 local-process isolation.
- Question types:
  - choice: native;
  - noul: 2-class;
  - score: 5-class **nominal**, labelled not ordinal-aware, with `expected_score = Σ level·p`;
  - MASSIVE en/zh/km included.
- Conditions A and B.
- **Descriptive only:** never in C1–C3, and k is unchanged.
- Length rule: if GLiNER alone overflows, GLiNER is excluded from that dataset (reported N/A). It never shrinks the shared pool.
- **P3.0 harness anchor (before the v2 pilot):** re-run GLiNER on the upstream pilot-v1 manifest with the v2 code, and compare with `results/reports/btzsc-pilot-v1.json` under both the old (all-classes) and new (targets ∪ predictions) macro-F1 definitions. Deterministic CPU results must match within tolerance. A mismatch blocks the pilot.

## Key decisions & tradeoffs

- **Q1 — Jev access through OpenRouter**, not TypeSafe direct (no open registration). This adds an OpenRouter hop to latency; the alpha endpoint may change. The resolved snapshot ID is logged on every call.
- **Q2 — Five contenders, one LLM size (1.7B).** Kev, Decider, GLiNER and larger Qwen models are out.
- **Q3 — Fork the upstream harness** at `0d610cc` rather than writing a new one: this keeps its metrics, bootstrap, manifest discipline and tests. `nibzard/decision-model-benchmark` has no license: ideas only.
- **Q4 — n = 300 per test set, 3 Jev repeats, pilot first.** This is a trade-off between CI width (±~5–6 pp accuracy) and M1 wall-clock time.
- **Q5 — Symmetric conditions A/B for all contenders**, including Jev. Confirmatory Brier contrasts are policy-matched: A-vs-A and B-vs-B (R2#3). No fine-tuning: no CUDA locally, and fine-tuning on a benchmark train split is the contamination already flagged in the wiki.
- **Q6 — Zero/rounding policy:**
  - raw values kept;
  - ε = 0.005 floor only for NLL and temperature fitting;
  - zero-true-label rate and infinite-NLL share reported;
  - rounding-parity sensitivity analysis.
- **Qwen label encoding:** exact two-token joint probability for K > 26 instead of lossy first-digit scoring. This costs one extra batched pass.
- **Permutation semantics:** Jev keeps option IDs tied to label identity (tests position only). Qwen letters are positional (tests position and letter prior jointly). The asymmetry is stated in the protocol.

## Toolchain

- Claude side, for P6: `rmd-fig-loop` and `rmd-table-loop` (study Rmd figures and flextables must be visually self-checked after knit).
- Later, outside this plan's build: `llm-wiki-ingest` for the final results.
- Codex side: no matching skills installed.

## Assumptions

- M1 Pro 16 GB, 75 GB free; system Python 3.14, venv Python 3.12; uv 0.7.13; R 4.6.1 with rmarkdown/flextable/showtext/data.table/ggplot2 (recon).
- OpenRouter lists `typesafe/jev-1.13` with modality `text->decisions`, context 32k, prompt $4.2e-8/token and completion $0.
- A live probe (2026-09-23) returned per-option probabilities at 2 decimals with exact zeros, and `confidence`; cost was $0.0000177 per 422-token call, latency ~260 ms. Two identical calls differed (0.22→0.20).
- Jev docs: `jev-1.13.0` is the only model; limits are 64k per request, 32k for state plus longest question, 255 choice options, 1,200 rpm; confidence formula as above.
- Laya: PyPI 0.3.9, HF `convaiinnovations/laya` Apache-2.0 with 3 subfolders plus `.py` files. The option head uses a 192–256-token budget; base ECE is 0.466 raw.
- Upstream harness: Python 3.11–3.13, choice-only, append-only runs, failures in the denominator, a TypeSafe-SDK Jev adapter (kept for reference, unused).
- Study rules: `reg()` for every number, vendored data with hashes, Chinese prose / English labels, never publish; plans live in `docs/exec-plans/active/`.

## Risks / open questions

- **BTZSC has only a test split**, so calibration and pilot items come from the same distribution as test, disjoint by ID. This is fine for temperature scaling but must be stated.
- **ModernBERT on MPS** may be unsupported or slow (e.g. attention kernels). The fallback is CPU; the documented CPU cost is ~193–464 ms per call, and Flowtivity measured much worse on a weak VPS. The pilot decides whether n = 300 is feasible for three Laya checkpoints.
- **Qwen3 tokenizer:** the digit-splitting and letter-token assumptions are unverified until the P2 assertions run.
- **Laya score/noul outputs:** resolved by the P0 capability check before the freeze (#3), not discovered mid-run.
- **OpenRouter `alpha` endpoint drift or rate limits** mid-run. The resolved model ID and timestamps are logged. **Confirmatory results require every confirmatory Jev call (calibration + test, all confirmatory datasets) to come from one resolved snapshot (R3#3).** On a snapshot change the runner stops. The complete shared calibration + test slice is then re-run against a single snapshot (cost ≈ $0.1), and the earlier rows are kept as a labelled, non-confirmatory record.
- **Data transfer:** hosted Jev receives the raw text of every item sent to it (via OpenRouter to TypeSafe). Every dataset in the hosted track is CC0, Apache-2.0, CC-BY or research-permissive, and ToxicChat was dropped for this reason. Raw text is never committed; only aggregates leave `results/runs/`.
- **Contamination:** public checkpoints may have seen AG News, Yelp, SMS Spam and MASSIVE. Results measure benchmark performance, not uncontaminated generalization.
- **The 10-questions-per-call latency suite** uses paraphrased questions, not independent decisions. It measures per-question marginal latency only.

## Out of scope

- Fine-tuning any model; Kaggle runs.
- Kev, Decider, GLiNER, larger Qwen models, TypeSafe direct API.
- Browser-agent and phishing batteries.
- Publishing HTML, and pushing to `upstream` (its push URL is disabled). The public fork `origin` = `github.com/nyuhuyang/jev-benchmarks` receives code, configs, docs and aggregate reports only after the human commit gate. Raw runs, caches, checkpoints and licensed raw text never leave the gitignored paths (AGENTS.md).
- Writing to the wiki during the build. Wiki ingest happens afterwards, as a separate task.
