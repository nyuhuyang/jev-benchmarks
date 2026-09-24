# Plan Review Log: Jev vs Laya vs local LLM-logit decision benchmark (v2)

Phases 0–1 (recon + interrogation) complete; plan locked with the user on 2026-09-23. MAX_ROUNDS=5. Reviewer: Codex (codex-cli 0.156.0, ChatGPT auth; model = CLI default). Plan: `docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md`.

Locked load-bearing decisions: Q1 Jev via OpenRouter; Q2 five contenders; Q3 fork upstream @0d610cc; Q4 datasets/n/repeats/pilot; Q5 symmetric raw vs temperature-scaled conditions; Q6 zero/rounding policy. The cosmetic batch was accepted by the user ("直接写Plan").

## Round 1 — Codex

The plan needs revision before implementation. The following issues could change the reported comparisons or allow an incomplete run to appear valid.

1. **The 5% coverage result leaks test labels.** The plan reports coverage at an empirical error budget, while the current metric chooses the best threshold on the same evaluated rows ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:124), [metrics.py](src/jev_benchmarks/metrics.py:48)). The architecture explicitly says this is only descriptive on a pilot slice ([ARCHITECTURE.md](docs/ARCHITECTURE.md:68)).  
   **Fix:** Select thresholds on calibration data and report their realized coverage and error on test, including a no-feasible-threshold outcome.

2. **“Failures remain in the denominator” is false for paired intervals.** `paired_bootstrap` keeps only examples valid for both models, which can favor an unreliable contender ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:90), [metrics.py](src/jev_benchmarks/metrics.py:126)). Brier and NLL also use valid rows only.  
   **Fix:** Predefine failure-aware primary comparisons on every shared item, and label probability metrics on successful calls as conditional results.

3. **The shared-capability claim conflicts with the proposed Laya fallback.** The plan allows Laya score probabilities to become N/A, yet still promises raw and calibrated probability metrics for every contender and dataset ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:16), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:186)).  
   **Fix:** Verify per-level probability output before freezing the score comparison; otherwise restrict the common score analysis to metrics all contenders can produce.

4. **The resume key can mix splits and conditions.** Calibration and test IDs can come from one source split, but the proposed key omits `split` and `condition` ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:49), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:89)). The current runner resumes by `example_id` alone and can generate a missing manifest during `run` ([runner.py](src/jev_benchmarks/runner.py:58)).  
   **Fix:** Bind every run to verified config and manifest hashes; key records by split, example, permutation, repeat, and model condition, with calibrated B derived from A rather than inferred again.

5. **Laya may never see all high-cardinality options.** The plan states a 192–256-token option-head budget but schedules 72 Banking77 and 60 MASSIVE descriptions, with a longer head tested only as exploratory on Banking77 ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:41), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:76), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:177)). A valid probability vector would not reveal truncated descriptions.  
   **Fix:** Assert, for every inference, that the encoded input contains every complete option; choose a feasible common setting before freezing or exclude that task.

6. **The high-cardinality primary metric will be sparse.** A balanced 300-item Banking77 test has roughly four examples per class; a 100-item, 60-intent MASSIVE locale has fewer than two on average ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:44)). Per-class F1 and class-stratified bootstrap intervals will be unstable, and some classes may be absent.  
   **Fix:** Specify the class universe and absent-class rule, report per-class support, and either increase these slices or downgrade their macro-F1 comparisons to exploratory.

7. **Nominal 95% intervals cannot support all proposed winner claims.** “CI excludes zero” is applied across multiple contenders, datasets, conditions, and primary metrics without a comparison hierarchy ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:133)).  
   **Fix:** Predeclare a small set of primary contrasts and a multiplicity rule; label the remaining intervals descriptive or exploratory.

8. **Temperature \(T\) is an inadequate test of “calibration built in.”** A bootstrap CI for fitted \(T\) alone neither establishes calibration nor measures whether scaling helps on independent test items ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:109)). Test intervals also need to reflect uncertainty from fitting \(T\).  
   **Fix:** Judge calibration by prespecified test-set scoring and reliability results, and refit \(T\) within calibration resamples when estimating A–B uncertainty.

9. **The permutation flip rates measure different interventions.** Jev retains label IDs while Qwen reassigns positional letters, so Qwen’s result mixes order sensitivity with token-label preference ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:163)). Stating the asymmetry does not make the rates directly comparable.  
   **Fix:** Report these as distinct estimands, or add a Qwen condition with stable option identifiers to isolate order.

10. **The latency workload is underspecified for Qwen.** Ten questions sharing one state is defined for Jev and Laya, while Qwen’s next-token scoring yields one answer per prompt ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:52), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:132)). “Per-question latency at 1 vs 10 per call” therefore has no common operation.  
   **Fix:** Define separate, reproducible request and batch workloads for each API, and compare only equivalent workloads.

11. **The $2 guard is racy and incomplete.** Four concurrent requests plus retries can incur charges before `usage.cost` arrives; a timeout may have been billed without returning usage ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:68)).  
   **Fix:** Reserve a conservative per-attempt cost atomically before dispatch, cap in-flight liability, and stop on missing cost data.

12. **The secret boundary is weaker than the plan claims.** Removing one environment variable does not isolate installed model code from other credentials or files; the plan acknowledges Python files in the model repository and leaves their execution dependent on package behavior ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:72), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:92)).  
   **Fix:** Inspect the pinned package before pilot and run local model loading in a process with an explicit minimal environment and restricted credential access.

13. **The ToxicChat data statement contradicts hosted inference.** “No raw text leaves `results/runs/`” cannot hold when ToxicChat examples are posted to OpenRouter ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:62), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:188)).  
   **Fix:** State the external transfer accurately and resolve its data-use conditions before including ToxicChat in hosted calls.

14. **Token assertions do not establish Qwen scoring correctness.** Checking digits as isolated tokens misses context-dependent tokenization after the actual chat template and `Answer: ` prefix ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:79)).  
   **Fix:** Verify the token suffix for every complete option ID in the exact rendered prompt, then use full sequence log probabilities whenever the two-token assumption fails.

VERDICT: REVISE
### Claude's response

Accepted 13, partially accepted 1 (#12). All changes are tagged `(#n)` in the plan.

| # | Disposition | Change |
|---|---|---|
| 1 | Accept | 5%-error threshold chosen on calibration, realized coverage and error reported on test (with "no feasible threshold"); upstream same-slice coverage kept only as a labelled descriptive number. |
| 2 | Accept | Failure-aware primary comparisons on every shared item: a failure counts as incorrect and as uniform for Brier/NLL. The valid-only bootstrap becomes a labelled *conditional* variant. |
| 3 | Accept | P0 capability check (before freeze) of Laya score/noul outputs; if missing, the shared score analysis is restricted to argmax metrics and probability metrics are labelled non-shared. |
| 4 | Accept | Runs bound to config/manifest hashes under the `v2-preregistered` tag; `run` never generates a manifest; key `(split, example_id, permutation_id, letter_mode, repeat_index)`; B derived from stored A, never re-inferred. |
| 5 | Accept | P0 measures the no-truncation `head_max_len` per checkpoint × dataset, and a per-call assertion enforces it. That value is the primary Laya setting; the as-shipped default becomes exploratory; infeasible → N/A. |
| 6 | Accept | Class-universe and absent-class rule declared; per-class support reported; macro-F1 on Banking77/MASSIVE downgraded to exploratory. |
| 7 | Accept | Six confirmatory tests (3 contrasts × {accuracy, Brier}, averaged over six English datasets) with dataset-stratified paired bootstrap and Holm adjustment; everything else is descriptive. |
| 8 | Accept | Built-in calibration judged by test A-vs-B ECE/Brier/NLL and reliability diagrams, with a joint bootstrap that refits T inside calibration resamples; T alone is not evidence. |
| 9 | Accept | Two estimands: order-only (stable IDs; adds a Qwen letters-bound-to-labels condition) vs order + label-token (Qwen positional letters), never compared across. |
| 10 | Accept | Cross-contender latency uses only the common single-question call; scaling workloads are backend-specific and compared within a backend. |
| 11 | Accept | Atomic pre-dispatch reservation (tokens × price × 3), ≤4 in flight, reservation retained and dispatch paused on missing cost or timeout. |
| 12 | Partial | Adopted: package source read before pilot (any remote `.py` import = STOP), local models in a separate `env -i` process with scratch HOME, offline HF cache, no key. **Rejected: a full OS sandbox** (`sandbox-exec`/container) — disproportionate for an inspected Apache-2.0 package on a single-user research box. The residual risk (absolute-path file reads) is stated explicitly in the plan instead. |
| 13 | Accept | ToxicChat (CC-BY-NC) replaced by Civil Comments (CC0, toxicity ≥ 0.5). The data-transfer statement is corrected: hosted Jev receives item text. **This changes a Q4 dataset choice; flagged to the user at sign-off.** |
| 14 | Accept | Tokenization verified on the exact rendered prompt for every complete option ID, with a fallback to full sequence log-probability scoring per dataset. |

## Round 2 — Codex

Several Round 1 fixes are now incorporated, including the held-out coverage threshold, split-aware resume key, and separate permutation estimands. The revised confirmatory analysis still has material contradictions.

1. **Two confirmatory contrasts may be impossible.** C2 and C3 require Laya-base-B Brier on all six English datasets, including Yelp and Banking77 ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:153)). The plan also permits missing Yelp probabilities and exclusion from Banking77 when no feasible head length runs ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:34), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:91)).  
   **Fix:** Make those capabilities a prerequisite for the six frozen tests, or predefine a valid reduced dataset set and contrasts before freeze.

2. **Uniform failure imputation rewards some failures.** A uniform forecast has Brier score \(1-1/K\), better than many confidently wrong forecasts, so a failed call can improve a contender’s primary Brier and NLL ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:152)).  
   **Fix:** Use a prespecified worst-case penalty for failed probability forecasts, or make failure rate a separate primary outcome and label Brier/NLL conditional on success.

3. **The primary contrasts compare different calibration policies.** Jev-A is tested against Qwen-B and Laya-B, although B is available for Jev too; the “deployed” rationale does not establish why only local contenders receive fitted temperatures ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:153)).  
   **Fix:** Use A-versus-A and B-versus-B as the primary route comparisons, with the proposed mixed-policy contrasts reported separately.

4. **The confirmatory intervals omit temperature-fit uncertainty.** P4 specifies a joint calibration/test bootstrap for A–B changes, but P5 specifies only a dataset-stratified paired bootstrap for the primary contrasts involving Qwen-B and Laya-B ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:131), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:156)).  
   **Fix:** Refit each B temperature inside calibration resamples for the confirmatory intervals too, and specify how Holm-adjusted decisions are computed.

5. **Capability probes are placed before the inference freeze.** P0 says “no inference,” yet checking actual Laya score outputs and whether a longer head *runs* may require model calls; artifact revisions are pinned only after those checks ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:26), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:34)). This conflicts with the repository rule to freeze configs before model inference ([AGENTS.md](AGENTS.md:5)).  
   **Fix:** Specify static checks where possible; pin and record a separate probe config before any necessary synthetic model calls.

6. **Disjoint IDs do not guarantee disjoint content.** The split rule excludes an ID from two sets but does not group duplicate texts or translated versions, allowing near-identical content into calibration and test ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:49)). The existing loader hashes text but samples by index ([data.py](src/jev_benchmarks/data.py:74)).  
   **Fix:** Audit duplicate and near-duplicate text across sets, assign duplicate groups to one split, and report any retained cross-locale overlap.

7. **The cost reservation omits most of some requests.** It estimates tokens from manifest text length, while a 72-option request also includes descriptions, instructions, and potentially ten questions; it then aborts only *after* reserved plus settled spend exceeds $2 ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:81)).  
   **Fix:** Bound cost from the full serialized request and reject dispatch when adding its reservation would exceed the remaining budget.

8. **The score fallback assumes an argmax that may not exist.** If Laya returns only an expected score, the proposed fallback of “accuracy, MAE of the argmax level” cannot be computed from that output ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:35)).  
   **Fix:** Define the shared scalar-score metric and any scalar-to-class decision rule before freeze, or exclude Laya from argmax comparisons.

9. **The revised macro-F1 definition is self-contradictory and differs from the code.** It says to average over all labels, then exclude absent labels unless predicted; the current implementation assigns zero to every absent class ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:54), [metrics.py](src/jev_benchmarks/metrics.py:13)).  
   **Fix:** Specify one fixed denominator and absent-class rule, including for bootstrap resamples, and update the scorer to match it.

10. **Jev’s permutation flip rate still mixes order effects with run noise.** The plan observes three Jev repeats and reports order flips relative to identity, but an identity-versus-permutation flip can occur on identical repeated calls ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:145)).  
    **Fix:** Report an identical-order repeat flip baseline beside the permutation rate and estimate the excess order effect.

VERDICT: REVISE
### Claude's response

Accepted all 10. Changes are tagged `(R2#n)` in the plan. #1, #3 and #9 were contradictions introduced by my Round 1 revision.

| # | Disposition | Change |
|---|---|---|
| 1 | Accept | P0 step 5: the confirmatory dataset set is declared from the probe results *before* `v2-preregistered`. It contains only English datasets where all three primary contenders give valid full vectors (Laya untruncated). With fewer than 3 datasets, C2/C3 leave the confirmatory family. |
| 2 | Accept | A failure gets the worst-case penalty (Brier 2, NLL −log ε), so it can never score better than an answer; failure rate is reported as its own outcome. |
| 3 | Accept | Confirmatory Brier is policy-matched (A-vs-A and B-vs-B). Accuracy is tested once because it is temperature-invariant. The family is 9 tests; mixed-policy "as-deployed" contrasts are descriptive. |
| 4 | Accept | The confirmatory bootstrap redraws the calibration set and refits T in every resample. The Holm procedure is specified: bootstrap two-sided p floored at 1/2000, step-down at α = 0.05. |
| 5 | Accept | Pins are committed first. Static checks need no model calls. Synthetic-string probes run on a committed `configs/probe-v2.yaml` tagged `v2-probe`, and no benchmark item is seen before `v2-preregistered`. |
| 6 | Accept | Normalized-text-hash grouping, with each group assigned to one split. MASSIVE is grouped by cross-locale utterance ID. Merged group counts are reported. |
| 7 | Accept | The reservation is computed from the full serialized request bytes (× price × 1.5). Dispatch is refused if the reservation would exceed the $2 budget. |
| 8 | Accept | Shared score metrics are fixed for everyone: MAE of the expected level and accuracy of round(expected level). Per-level probability metrics are non-shared. |
| 9 | Accept | Macro-F1 denominator = labels in targets ∪ predictions, recomputed per resample. The upstream scorer is replaced to match, with a unit test. |
| 10 | Accept | Jev's identical-order repeat flip rate is reported beside the permutation flip rate, and the excess order effect gets a paired CI. |

## Round 3 — Codex

The Round 2 responses are mostly reflected in the plan. Four material gaps remain:

1. **The scalar score fallback has no runnable path.** P0 permits Laya to return only an expected score, but the planned `Prediction` and runner still require a full probability vector, and P4 still fits a temperature for every contender and dataset ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:40), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:84), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:142)).  
   **Fix:** Define a scalar score prediction and validation path; mark temperature-scaled B unavailable for scalar-only outputs or specify a separate scalar calibration method before freeze.

2. **The truncation check ignores the example text.** It verifies complete option descriptions, but Laya’s stated 512-token context can still truncate a long Civil Comments or Yelp item; contenders would then receive different text while passing the per-call assertion ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:45), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:227)).  
   **Fix:** Assert that both the complete item text and all options survive encoding, with a frozen common truncation or exclusion rule for overlength items.

3. **A Jev snapshot change still breaks the primary comparison.** Logging a changed resolved model and “splitting” the run does not yield the planned paired, full-slice result if different test items were answered by different Jev snapshots ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:237), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:173)).  
   **Fix:** Abort confirmatory reporting on a revision change and rerun the complete shared slice against one revision, or predeclare snapshot-specific analyses with adequate sample sizes.

4. **The conditional confirmatory family is inconsistent.** P0 can drop C2/C3 when fewer than three common datasets remain, while P5 still specifies exactly nine tests and Holm adjustment across nine ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:48), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:173)).  
   **Fix:** Freeze the resulting test family and its Holm denominator explicitly after P0, including which datasets C1 uses if Laya is excluded.

VERDICT: REVISE
### Claude's response

Accepted all 4 (tagged `R3#n`).

| # | Disposition | Change |
|---|---|---|
| 1 | Accept | Scalar-only score path: `expected_score` field, a separate validator, and condition B marked unavailable, with no scalar calibration. Distribution contenders derive `expected_score = Σ level·p`. |
| 2 | Accept | Static full-request length check for every contender's tokenizer. Frozen common exclusion rule: any item that any assigned contender would truncate is excluded for all, before sampling. Exclusion counts and the short-text bias are reported, and a per-call assertion re-checks. |
| 3 | Accept | Confirmatory Jev calls must all come from one resolved snapshot. On change, stop and re-run the whole shared calibration + test slice on one snapshot (~$0.1); the old rows are kept as non-confirmatory. |
| 4 | Accept | Per-contrast frozen dataset sets (C1 = Jev ∩ Qwen valid; C2/C3 = all three). The exact test family and Holm denominator k are written into `PROTOCOL-v2.md` before `v2-preregistered`; the family falls back to C1 × 3 if Laya drops out. |

## Round 4 — Codex

All four Round 3 fixes appear in the plan, but three are not carried through consistently, and the new dataset rule limits the stated comparison.

1. **The scalar score path still conflicts with later phases.** P0 marks B unavailable for scalar-only output, while P3 requires every probability vector to validate, P4 fits a temperature for every contender × dataset, and P5 promises A/B metrics for all ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:45), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:141), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:150)).  
   **Fix:** Make pilot validation, calibration, and reporting explicitly skip vector and B analyses for scalar-only score outputs.

2. **A snapshot rerun would reuse completed records.** The new rule requires rerunning Jev’s entire calibration and test slice after a snapshot change, but the resume key contains neither snapshot nor run-attempt ID, so earlier successful rows would be skipped ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:129), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:249)).  
   **Fix:** Give each snapshot attempt a separate run namespace and make reports select one complete attempt.

3. **The Holm denominator is still hard-coded to nine.** P0 and P5 allow the family to fall back to three tests, but the decision rule still says to adjust across nine ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:53), [plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:194)).  
   **Fix:** Apply Holm across the frozen family of \(k=3\) or \(k=9\) tests and report that \(k\).

4. **The three contrasts can use different datasets.** C1 uses the Jev–Qwen intersection while C2/C3 use the three-way intersection, so their effect sizes cannot jointly establish a three-model ranking on identical items ([plan](docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md:181)).  
   **Fix:** Use the three-way intersection for any joint ranking; report the broader C1 comparison separately.

VERDICT: REVISE
### Claude's response

Accepted all 4 (tagged `R4#n`). All four are propagation gaps left by my Round 3 edits.

| # | Disposition | Change |
|---|---|---|
| 1 | Accept | Scalar-only score outputs are carried through P3 (shape-specific validator), P4 (no T fit) and P5 (A plus the two shared metrics only; B and vector metrics unavailable). |
| 2 | Accept | Per-backend attempt namespaces. A snapshot change starts a new attempt, and reports select one complete single-snapshot attempt, recording the attempt ID and snapshot. |
| 3 | Accept | Holm runs across the frozen family k (9 or 3), recorded in `PROTOCOL-v2.md` and stated in the report. |
| 4 | Accept | When C2/C3 are confirmatory, C1–C3 all use the three-way intersection (identical items, joint ranking valid); the broader Jev ∩ Qwen C1 is descriptive. |

## Round 5 — Codex

The four Round 4 fixes are incorporated: scalar-only scores bypass vector calibration, snapshot reruns use separate attempt namespaces, Holm uses the frozen family size, and three-way contrasts use identical datasets. I found no remaining material plan blocker. The stated sampling and hosted-model limitations still need to be reported with the results.

VERDICT: APPROVED
## Resolution

APPROVED at Round 5 of MAX_ROUNDS=5 (Codex thread retained in the session scratch dir until the build decision). Findings per round: 14 → 10 → 4 → 4 → 0. Dispositions: 31 accepted, 1 partially accepted (R1#12: no OS sandbox; residual risk stated). Awaiting user sign-off and a build decision.

## Post-approval amendment (user-requested, 2026-09-23)

P0 item 2 added: local artifact paths `experiments/models/hf-cache/` (HF_HOME, offline after the pinned download) and `experiments/models/sandbox-home/` (empty HOME for `env -i` local-model processes), both outside git. Subsequent P0 items were renumbered. This concretizes paths already implied by the approved plan (R1#12); no methodological change, so no re-review. Both directories were created.

## Post-approval amendment 2 (user-requested, 2026-09-23)

The user chose a **public GitHub fork**: `github.com/nyuhuyang/jev-benchmarks`, forked from AbdelStark at `0d610cc`, set as local `origin`; `upstream` push stays disabled. Before publishing, the handoff doc was scrubbed of a session-resume ID and the note on where the key is stored. The plan's Out-of-scope line was updated: pushes to `origin` happen only after the human commit gate, and raw runs, caches, checkpoints and licensed text never leave gitignored paths.

## Post-approval amendment 3 (user-requested, 2026-09-23) — applied during Act 3 verification

**Add GLiNER2.5 as a descriptive contender and harness-validation anchor.** Not in the confirmatory family; k is unchanged.

Plan edits:

1. **Goal / contenders:** add a 4th route, "zero-shot encoder without calibration training": `fastino/gliner2.5-multi-v1` at the revision upstream pinned (`235cf92d6d4318da9bfca0d08975c8fa7250d13b`), CPU, batch size 1, using the existing upstream `gliner` adapter unchanged.
2. **Question-type mapping:**
   - choice: native single-label classification;
   - noul: 2-class classification (`false`, `true`);
   - score: **5-class nominal** classification. GLiNER has no ordinal notion, so its score results are labelled "nominal — not ordinal-aware". `expected_score = Σ level·p` is still computed for the shared score metrics.
   - multilingual: runs on MASSIVE en/zh/km (it is the `-multi` checkpoint).
3. **Conditions:** A and B like every other contender (its probabilities are full vectors, so T-scaling applies).
4. **Harness-validation anchor (new P3.0, before the v2 pilot):** re-run GLiNER on the **upstream pilot-v1 manifest** (the same 300 BTZSC items, `configs/pilot-v1.yaml`) with the v2 code, and compare with `results/reports/btzsc-pilot-v1.json`. Accuracy/macro-F1/Brier must match to within the deterministic tolerance, since GLiNER on CPU is deterministic; the macro-F1 definition change (R2#9) is accounted for by recomputing the upstream number under both definitions. A mismatch blocks the v2 pilot. This costs nothing and does not involve Jev.
5. **Length rule (R3#2):** GLiNER's tokenizer and context are added to the static full-request length check. If GLiNER is the binding constraint on any dataset, it is excluded from that dataset (reported N/A) instead of shrinking the shared pool, so the confirmatory pools stay determined by the primary contenders.
6. **Reporting:** GLiNER appears in all descriptive tables and figures. It never appears in C1–C3.
7. **Budget/time:** CPU only; roughly +10–20 min total wall-clock. No API spend.

Rationale: the adapter and tests already exist (zero new code); it gives a free, deterministic reproduction check of the forked pipeline against the published upstream numbers; and it adds an "encoder, no RLCD" reference point next to Laya's "encoder + RLCD".

## Act 3 — Build

Builder: Codex (`gpt-6-sol`, `codex exec --yolo`), launched with `OPENROUTER_API_KEY` and `TYPESAFE_API_KEY` unset. Base commit `bbb7859`. PROOF_CMD: `uv sync --extra benchmark --dev && uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`. Scope: code only (P1 code, P2, P4/P5 analysis code); no inference, downloads, tags or commits.

### Round 1 — Codex build

Codex report: 30 files changed or added (new adapters `jev_openrouter`/`laya`/`qwen_logit`; `probe`, `v2_runner`, `worker`, `v2_metrics`, `v2_report`; data loaders; configs; `PROTOCOL-v2.md` draft; 6 new test files). Proof passed (70 tests, 90.15% coverage). Reported deviations: UTF-8 bytes as the Jev token bound; Laya 0.3.9 has no trust_remote_code argument.

### Claude's verdict

Proof re-run independently: pass (pytest exit 0, pyright 0 errors, ruff clean, lock check OK). Scope respected (no changes under `docs/exec-plans/` or `results/reports/`, nothing committed). The full diff was read. **REVISE — 3 critical, 1 high, 3 medium/low findings; tests passed because the fakes encoded the same wrong assumptions:**

1. **Critical:** the Jev adapter parses `choices` and noul `probabilities`, but the live API returns `answers`, and noul answers carry only `noul: p`. Every real call would fail after being billed.
2. **Critical:** 429/5xx responses carry no `usage.cost`, so the code pauses dispatch instead of retrying.
3. **Critical:** noul and score questions are content-free (generic instruction, criteria `{"true":"true","false":"false"}`), and the Qwen prompt carries no question at all. Per-dataset instructions and option descriptions are required, identical across contenders.
4. **High:** the prepare-time Laya length counter differs from the runtime truncation assertion, which would turn truncations into penalized failures.
5. **Medium:** score-type primary accuracy used argmax instead of the shared round(expected) rule; scalar-only contenders got failure penalties instead of N/A; the score MAE penalty was hard-coded at 4.
6. **Medium:** `repeat_summary`/`flip_summary` groupings could mix splits and repeats.
7. **Low:** worker stderr was discarded.

Accepted deviation: serial Jev dispatch. Also folded into this fix round: Amendment 3 (GLiNER contender plus the P3.0 anchor command).

### Round 2 — Codex fix round 1 (interrupted)

Codex resumed the same session (`01a0cf99-…`) with the F1–F9 fix list and exited 1 on a **ChatGPT usage limit** (`You've hit your usage limit … try again at 7:48 PM`) while editing tests. No final report was written. State at interruption: F1–F9 were implemented with tests; the only remaining defect was one unformatted test file (`ruff format --check` failed).

### Claude takeover (trivial remainder) and verdict

- **Takeover:** Claude ran `ruff format tests/test_v2_runner_report.py` (mechanical). No logic was changed by Claude. The alternative, waiting ~4 h for the quota reset, was not worth it for a formatting fix.
- **Verification:** Claude read the fix diff for F1 (live `answers` shape, noul `(1−p, p)`, score order mapping), F2 (4xx/5xx without cost keep the reservation, do not pause, and are retried; only a 2xx without cost or a timeout pauses), F3 (per-dataset instructions and descriptive noul options shared by Jev, Laya and Qwen; Qwen `{question}`), F4 (one shared `laya_request_tokens` helper used by prepare and runtime), F5–F6 (metrics), F7 (`gliner_v2` adapter, `anchor` command with both macro-F1 definitions, confirmatory-family guard), F8 (worker stderr log) and F9 (PROTOCOL deviations). Every fix has tests.
- **Proof, run by Claude:** exit 0 — ruff clean, format clean (54 files), pyright 0 errors, pytest pass, coverage 90.14%.
- **Noted, not blocking:**
  - ~~GLiNER's API accepts label descriptions but no instruction text~~ **Corrected at the commit gate:** `gliner2.classification.ClassificationSchema.task(..., instruction=None)` exists, and the built adapter passes the shared instruction. GLiNER therefore receives the same instruction as the other contenders. Open for P0: `gliner2` also exposes `ordinal()`, so GLiNER score could be run ordinal-aware instead of nominal (Amendment 3). Decide before the freeze.
  - The anchor tolerance (Brier 1e-6) may be too tight across CPUs (M4 Max → M1 Pro float differences). If P3.0 fails only on Brier at the 1e-6 level, loosen it with a logged reason.
  - The anchor needs the pilot-v1 manifest, which is regenerated by `prepare` (gitignored); this is an operator step.
- **Status:** build accepted pending the human diff gate. An optional independent cross-provider inspection via agy/Gemini 3.1 Pro was offered.

### Jev adapter live smoke (user-requested, 2026-09-23, pre-probe, synthetic strings only)

This ran the built `JevOpenRouterBackend` through `_validate_prediction` against OpenRouter `/api/alpha/decisions`. It used hand-written synthetic items only (no benchmark data, so preregistration is unaffected), a $0.01 budget, and an attempt log in the session scratch dir (not the repo).

| Case | Resolved | Result | Latency | Cost |
|---|---|---|---|---|
| choice (4 options) | jev-1.13-20260917 | correct, p = 1.00, confidence 1 | 0.40 s | $0.0000157 |
| choice, permuted order (3,1,0,2) | same | correct; probabilities mapped back to label identity | 0.29 s | $0.0000157 |
| noul (spam) | same | correct, P(true) = 0.99 | **10.38 s** | $0.0000146 |
| score (Yelp 1–5) | same | correct, expected level 1.0, confidence 0.99 | 0.24 s | $0.0000143 |
| latency-10 (10 questions / 1 call) | same | correct | 0.23 s total | $0.0000506 |

Budget settled $0.00011; nothing paused; 5 attempt rows; key-in-log check: False. **Findings:**
- The end-to-end parsing and validation path works for all three types and for permutation mapping.
- One 10.4 s outlier (noul) shows the hosted tail latency is real; p95 must be reported, and a 30 s timeout is adequate.
- The easy synthetic items give saturated 0/1 probabilities, so they are not informative for calibration.

### Commit gate

The user approved commit + push to the fork after the Jev live smoke. The cross-provider inspection was **not performed**: Gemini had 0% quota; Opus 4.6 via agy hit its individual quota after reading the diff and produced no findings; GPT-OSS returned 503 with no capacity. This is an explicit, logged gap. Pre-commit scans passed: no API key, no personal paths, no raw runs or caches.

## Amendment 5 (2026-09-24) — plan review
- Trigger: user accepted P2 (few-label arm), P3 (confirmatory set = AG News/Emotion/SMS/Civil, UltraFeedback descriptive), C1 (framing as same-run controlled re-check) from the cross-provider-approved assessment (sha 851394f6…).
- Reviewer: Codex via claudex-loop runner `review`, fresh session (earlier rounds 1–5 predate runner result files), CLI default model. Fallback: same-provider-on-unavailable.

### A5 round 1 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-r1/claudex-4cpc5ko_/result.json (session 01a0d408-68ea-79a0-bb32-f9d7efaf0666, cross_provider, plan sha d56e79c0…)
- Findings: F1 high (stratified 5-fold infeasible for Banking77/MASSIVE), F2 medium (TF-IDF fit leakage across folds), F3 medium (budget pause text vs code), F4 medium (UltraFeedback one-completion-per-prompt undeclared), F5 low (few-label latency fields).
  Round 1 dispositions (host = Claude):
  - F1 ACCEPTED. The fold policy is now unstratified KFold(5, shuffle=True, random_state=20260923) for every dataset, with the reason (Banking77/MASSIVE calibration classes have 2–4 items). The absent-class zero rule and its reporting are stated. Banking77-like and MASSIVE-like contract cases were added to the tasks.
  - F2 ACCEPTED. TF-IDF is fitted inside each training fold together with the classifier and refitted on all 200 for test; a per-fold vocabulary contract test was added.
  - F3 PARTLY ACCEPTED: plan text fixed, code unchanged. The plan text contradicted the implemented and intended behaviour. The budget-guard text now reads: pause on 2xx-without-cost, timeout or transport error; 429/5xx-without-cost keep the full reservation (counted against the cap) and retry; non-retryable 4xx keep the reservation and fail the item. Rationale: pausing on every 429 would halt runs on routine rate limits. Retaining reservations means unbilled errors can only over-count spend, so liability stays within the $2 cap. The code (jev_openrouter.py:175-197) already implements this.
  - F4 ACCEPTED as "freeze the estimand". Amendment 4 and PROTOCOL now state one seeded-hash representative completion per prompt (independent items for the bootstrap); other completions are unused.
  - F5 ACCEPTED. Few-label rows carry no latency fields in reports; added to the contract tests.
  - Limitation noted: the study directory is outside this repo (../../study), which is expected.

### A5 round 2 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-r2/claudex-urs877gb/result.json (same session, cross_provider, plan sha 83d198ad…). Prior F1–F5 confirmed resolved.
- New findings: R2-F1 high (budget liability not durable across crash/concurrent processes), R2-F2 high (usage/cost not validated; NaN corrupts the cap), R2-F3 medium (P6 "zero-shot only" contradicts the few-label arm).
  Round 2 dispositions (host = Claude):
  - R2-F1 ACCEPTED (verified: v2_runner.py:70-74 rebuilds from post-response http-attempts only; Budget is in-memory). Added Amendment 5 section D: an fsync'd experiment-level budget ledger with reserved-before-dispatch and settled records; on start, unmatched reservations count in full; a single dispatcher via a non-blocking fcntl.flock for the process lifetime (a second process refuses); crash and concurrency contract tests; a task added to the checklist.
  - R2-F2 ACCEPTED. Section D requires usage to be an object and cost a finite number >= 0 (bool excluded); malformed accounting on 2xx retains the reservation, logs and pauses, and never reaches Budget.settle. Fake-response cases: usage null, "NaN", NaN, -1, "0.1", true.
  - R2-F3 ACCEPTED. The P6 limitation text now says the Jev/Laya/Qwen contrasts are zero-shot and the few-label arm is supervised on 200 calibration labels; PROTOCOL's closing sentence was updated the same way.

### A5 round 3 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-r3/claudex-dph_codr/result.json (same session, cross_provider, plan sha 2eefdd97…). Prior R2 findings confirmed addressed.
- Findings: R3-F1 high (retained reservation dropped on reconstruction), R3-F2 medium (no unique per-dispatch transaction ID).
  Round 3 dispositions (host = Claude):
  - R3-F1 ACCEPTED. Reconstruction now reads: liability = sum of settled costs + the full amount of every reservation not closed by a valid `settled` record, so `retained` and unclosed reservations both count in full. Test added: a retained 429 still counts after restart.
  - R3-F2 ACCEPTED. Every HTTP dispatch gets a uuid4 transaction ID, carried on `reserved`, `settled` and `retained` records. Startup refuses on a duplicate ID, a closing record without a reservation, or a double close; tests added.

### A5 round 4 — Codex — APPROVED
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-r4/claudex-17bphvzs/result.json (same session 01a0d408-…, cross_provider; observed model not reported). Plan sha 9ee02f224d3e3a7eefbcb37a502847446dc622c11320f7a2c9b20ea513963905; runner `check` matches.
- Limitations: ledger and few-label arm are planned, not implemented; the code needs a fresh Codex inspection after the build.

### A5 build inspection 1 — Codex (fresh) — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-inspect1/claudex-pw_pjcom/result.json (cross_provider; builder claude; base 0ee429b, head 10eb90c)
- R1 high: the few-label vs Jev report could crash on scalar-only score output → ACCEPTED: `vector_datasets` limits Brier A/B comparisons to datasets with full distributions on both sides; accuracy uses all shared datasets; test added.
- R2 high: a Jev 2xx with a changed model that failed parsing was written as "unknown" → ACCEPTED: `JevResponseError` carries the serving snapshot; the runner records it and stops on a snapshot change; attempt selection requires one known snapshot across every call that received a response. Calls with no response (timeouts, transport errors) stay failures without provenance, because they produced no output. Tests added.
- R3 medium: ledger replay did not validate numbers → ACCEPTED: amounts and costs must be finite non-negative JSON numbers (bool excluded), else refuse; tests added.
- R4 medium: the report did not include public results → ACCEPTED: v2.json `public_results` and a v2.md "Relation to public results" table from docs/public-results.csv; test added.

### A5 build inspection 2 — Codex (fresh) — REVISE (inspection budget 2/2 used)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-inspect2/claudex-7rxt7ik8/result.json (cross_provider; base 0ee429b)
- R1 high: a snapshot change was missed when the returned prediction failed validation → FIXED: the runner keeps the raw prediction's snapshot before validation.
- R2 high: selection checked only the latest row per key, so a retry could hide an earlier response from another snapshot; restore used successful rows only → FIXED: selection and restore use every response-bearing row.
- R3 medium: attempt state and pending work were computed before the Jev lock → FIXED: a per-backend `dispatch.lock` (flock) is held across attempt selection, pending computation and dispatch.
- R4 medium: absent-class counts were not reported → FIXED: v2.json `few_label_absent_class_held_out_rows` and a v2.md section.
- Regression tests added for each. **These fixes are NOT covered by a Codex inspection** (budget exhausted); another fresh inspection needs user approval.

### A5 build inspection 3 — Codex (fresh, user-approved extra round) — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a5-inspect3/claudex-dmjnjksz/result.json (cross_provider; base 0ee429b; head 72a468b)
- R1 high: a billed 2xx without `model` was recorded as "unknown" and ignored by provenance → FIXED: `JevResponseError` carries `response-without-model`; attempt selection rejects any attempt containing it.
- R2 high: the report selected rows and hashed the file at different times → FIXED: selection runs under the backend's dispatch lock, and `prediction_sha256` is the hash of the exact bytes parsed.
- R3 medium: the few-label run took no lock and wrote features non-atomically → FIXED: `run_fewshot` holds the per-backend dispatch lock; features are written to `.tmp` and published with `replace`.
- R4 medium: the Qwen `latency-10` batch repeated one prompt → N/A: Amendment 6 (user-approved) removes the latency suite, so no latency-10 rows exist.
- Regression tests added for R1–R3. These fixes are not yet re-inspected.

## Amendment 6 (2026-09-24) — plan review
- Trigger: user approved the Amendment 6 scope cut and headline question. Reviewer: the same Codex plan-review session, resumed from A5 round 4.

### A6 round 1 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r1/claudex-rezrxkpc/result.json (resumed A5 plan-review session; cross_provider; plan sha 008aeba6…)
- A6-F1 medium (probe hard-codes laya_typed; probe validity after the cut), A6-F2 medium (anchor loads GLiNER without credential scrubbing).
  Amendment 6 round 1 dispositions (host = Claude):
  - A6-F1 ACCEPTED. The plan now states that the v2-probe record (8720bb5) stays the capability check for the retained contenders, and that laya_typed's exclusions are a subset of laya_base's (same tokenizer, larger budget), so removing contenders only shrinks the common exclusion. Final counts come from `prepare` and the manifest summary. run_probe now iterates the configured Laya backends (code done), and a rerun needs a new probe tag.
  - A6-F2 ACCEPTED. run_anchor now removes OPENROUTER_API_KEY and TYPESAFE_API_KEY and sets HF_HUB_OFFLINE=1 before constructing GLiNERV2Backend; a contract test asserts both at construction. The minimal-env worker is not used because the frozen pilot-v1 config has no local_runtime, and editing it would change the anchor's input.

### A6 round 2 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r2/claudex-jmz3ufts/result.json (cross_provider; plan sha 73626691…). A6-F1 and A6-F2 confirmed resolved.
  Amendment 6 round 2 disposition (host = Claude):
  - A6-R2-F1 ACCEPTED. Added the Amendment 6 task: the README v2 workflow removes the `--backend gliner` v2 run (GLiNER only via the pilot-v1 anchor command), adds the three `jev-bench fewshot` commands, and lists no latency split. It will be done in the Amendment 6 build together with the config, derived-suite and protocol tasks.

### A6 round 3 — Codex — FAILED (provider_unavailable)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r3/claudex-f6ekt8aj/result.json — Codex usage limit ("try again at Sep 27th, 2026 11:31 AM"). fallback_eligible=true.
- Policy same-provider-on-unavailable → fresh Claude reviewer; assurance=degraded_same_provider.

### A6 round 3 — Claude fallback (degraded_same_provider) — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r3-fallback/claudex-1fe1_3q8/result.json (fresh Claude session; assurance degraded_same_provider; fallback from /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r3/claudex-f6ekt8aj/result.json; plan sha 06feb388…)
- Findings: F1 medium (headline estimand undefined), F2 medium (anchor HF_HOME not pinned; deny-list isolation), F3 low (stale latency text; latency reference undefined), F4 low (probe tag hard-coded; typed tokenizer loaded unconditionally; CLI choice), F5 low (few-label threshold transfer).
  Fallback round 3 dispositions (host = Claude):
  - A6-R3-F1 ACCEPTED. Added a "Headline estimands" block. The primary set is the 4 confirmatory datasets, the secondary set is all shared datasets (vector metrics only where both sides have distributions). "Advantage left" is reported as side-by-side paired differences on identical items (Jev − qwen_logit and Jev − X), with no ratio. Metrics: accuracy, Brier A/B, and coverage at 5% under A and B. The coverage bootstrap re-selects each side's threshold on the resampled calibration set (refitting T for B), and no feasible threshold counts as 0, with flags reported. Estimation only, unadjusted, no winner claims. Code task added.
  - A6-R3-F2 ACCEPTED (stronger option). The anchor will run through the minimal-env LocalProcessBackend worker with the v2 local_runtime (pinned HF_HOME, offline, scratch HOME, no inherited credentials). The runtime environment is not part of the frozen pilot-v1 input. The earlier in-process scrub task is marked superseded; code task and contract test added.
  - A6-R3-F3 ACCEPTED. The superseded base-plan latency passages are listed. The latency reference is test-split per-call p50/p95 (workload `test-reference` in latency.csv), with one unrecorded warm-up per invocation for local backends and serial Jev dispatch in the runner loop. It is placed as item 5 of the reporting order.
  - A6-R3-F4 ACCEPTED. Probe tag read from the probe config (default v2-probe); laya_base checks guarded; make_length_counters loads only configured Laya tokenizers; laya_typed removed from the CLI run choices; a prepare-counter test on the A6 config.
  - A6-R3-F5 ACCEPTED. The threshold-transfer limitation is stated, and realized selective error is reported next to coverage.

### A6 round 4 — Claude fallback (degraded_same_provider) — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r4-fallback/claudex-7hul8n0h/result.json (resumed fallback session; plan sha 038bbcb0…). Round-3 findings confirmed reflected.
  Fallback round 4 dispositions (host = Claude):
  - A6-R4-F1 ACCEPTED. The interface is now explicit: LocalProcessBackend takes `runtime` as a separate mapping (defaulting to config.raw["local_runtime"]). The anchor passes the v2 runtime together with the pilot-v1 config path, so GLiNER is built from pilot-v1's models.gliner, and v2.yaml needs no gliner entry. Two tests: an environment capture, and an in-process worker.main on a pilot-v1 config with the GLiNER constructor monkeypatched at import level, asserting the pilot-v1 model_id/revision/device. Verified: the pinned cache holds the GLiNER model.safetensors, not just its tokenizer.
  - A6-R4-F2 ACCEPTED. local_runtime paths resolve against the repo root, like output_dir (config.py:24-26). The runner refuses to start a worker if hf_home is missing. The test asserts HF_HOME equals the resolved absolute pinned cache.
  - A6-R4-F3 ACCEPTED. Jev predictions record dispatch_attempts. The latency reference uses first-attempt-success rows, and the retried share is reported separately.
  - A6-R4-F4 ACCEPTED. The primary-set zero-shot-gap rows for accuracy, Brier A and Brier B are the C1 estimates (identical numbers) and carry the C1 Holm decision and k=9. Only the coverage and few-label-gap rows are new estimation-only quantities.
  This is round 5 of 5.

### A6 round 5 — Claude fallback (degraded_same_provider) — APPROVED
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-r5-fallback/claudex-bavnnzw5/result.json (plan sha e9fb95818243f989e4dfecc9f02f1847e270be1fb96add6c36153a2597a4f568); runner check: "Approval matches the current plan; assurance=degraded_same_provider".
- Assurance note: Codex was unavailable (usage limit until 2026-09-27 11:31); rounds 3–5 were fresh-then-resumed Claude fallback reviews, not cross-provider.
- Minor ambiguity noted by the reviewer: how `anchor` locates configs/v2.yaml → resolved in the build as the sibling `configs/v2.yaml` of the pilot-v1 config.

### A6 build inspection 1 — Codex (fresh; Codex available again) — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-inspect1/claudex-akold8j6/result.json (cross_provider; base ff5aeb9; head 524e382)
- R1 high: all-failure bootstrap resamples crashed on missing coverage/ECE → FIXED: score_v2 returns Brier 2, coverage 0, ECE None and the no-feasible flag for all-failure samples; the bootstrap skips undefined draws and reports resamples_used; an undefined confirmatory p counts as no rejection.
- R2 medium: attempts sorted lexically → FIXED: numeric sort.
- R3 medium: latency in metrics.csv differed from the latency.csv reference → FIXED: latency fields removed from every metric row.
- R4 medium: the attempt count was lost on failed Jev calls → FIXED: the backend exposes last_dispatch_attempts, the runner stores it on failure rows, and retried_share covers every dispatched item.
- R5 medium: sign convention ambiguous → FIXED: contender − Jev everywhere (same as C1), stated in the plan, PROTOCOL and v2.md.
- R6 medium: realized selective error missing next to coverage → FIXED: left/right coverage, selective error and no-feasible counts on headline coverage rows, plus a v2.md table.
- R7 low: stale README/PROTOCOL intro text → FIXED.
- Regression tests added.

### A6 build inspection 2 — Codex (fresh) — REVISE (inspection budget 2/2 used)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-inspect2/claudex-7wl73nf_/result.json (cross_provider; base ff5aeb9; head 2e830f7)
- R1 high: an all-failure score dataset was treated as scalar-only → FIXED: scalar-only requires a successful scalar call; otherwise vector penalties apply.
- R2 medium: dispatch history was lost across invocations → FIXED: attempt selection sums dispatch_attempts across every row of a key.
- R3 medium: condition-B draws without a fittable T were dropped (success-conditioned) → FIXED: such draws use T = 1; confirmatory tests with an undefined estimate or fewer than the frozen usable draws get no Holm rejection. Rules written into PROTOCOL.
- Regression tests added. **These fixes are not covered by a Codex inspection** (budget exhausted).

### A6 build inspection 3 — user-approved extra round (2026-09-24)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-inspect3/claudex-1xesxkpj/result.json (cross_provider; base ff5aeb9; head 3d11c10) — REVISE
- R1 high: a complete all-failure local attempt was unselectable → FIXED: accepted with the pinned checkpoint as provenance; Jev still needs one known snapshot.
- R2 high: an unavailable condition B raised and stopped the report → FIXED: the bootstrap returns an "unavailable" result with no estimate and no Holm rejection; A results are still published.
- R3 medium: the local latency boundary excluded prompt preparation and IPC → FIXED: the runner times every call at its own boundary; model-only time stays separate.
- R4 medium: the latency row was dropped when every Jev call retried → FIXED: the row is kept with n=0, null quantiles and the retry share.
- Rules added to PROTOCOL; regression tests added. Not yet re-inspected.

### A6 build inspection 4 — user-approved (claudex-loop 2.2.1; CCFA research workflow enabled by user)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a6-inspect4/claudex-uclpyrs_/result.json (cross_provider; base ff5aeb9; head 35568e0) — REVISE
- R1 high: a 2xx without valid cost lost its model ID; cost_paused attempts could be reported → FIXED.
- R2 medium: order effects reported from absent or partial permutation runs → FIXED: marked unavailable unless complete.
- R3 medium: all-failure local attempts had no latency row → FIXED.

## Amendment 7 (2026-09-24) — pre-freeze CCFA research gate
- User enabled the CCFA research workflow ("继续Codex审阅，启用ccfa 论文审阅"). The host ran ccf-humanization → ccf-common → ccf-paper-reviewer (scientific, generic-7) on PROTOCOL-v2 @35568e0: ccfa-review-reports/protocol-v2-scientific-review.md (validator PASS; 7 findings; borderline 6/10; confidence 4).
- Per claudex-loop research-skills: the Codex review is not a substitute for the CCFA specialist; CCFA criteria are copied into the plan's Amendment 7 acceptance criteria.
- Reviewer: fresh Codex plan review (runner 2.2.1), CLI default model; fallback same-provider-on-unavailable.

### A7 round 1 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-r1/claudex-lh6rgnes/result.json (cross_provider; plan sha 47bdcf1f…)
- Findings: F1 high (cost pause lost on snapshot change), F2–F6 medium (summary not frozen; prevalence population; per-dataset rows missing; headline order; sampling notes).
  Amendment 7 round 1 dispositions (host = Claude). All accepted and implemented, with regression tests (suite passes, coverage 91.30%):
  - F1: in the runner's error path, a snapshot change during a cost pause writes state `cost_paused` (with old/new snapshot), so `_attempt_dir` blocks new attempts until an operator resolves it. Test: missing-cost JevResponseError from a new snapshot → cost_paused and _attempt_dir refuses.
  - F2: prepare_manifest refuses to overwrite a differing manifest-summary.json. The freeze record gains manifest_summary_sha256, and verify_frozen checks it when present. Test: identical summary accepted, differing summary refused.
  - F3: a new representative_pool helper is shared by the splitter and the prevalence computation. pool_class_prevalence covers the grouped representative pool; raw_candidate_class_prevalence is kept separately. Test: 3 completions of one prompt vs 1 of another give pool {0.5, 0.5} and raw {0.75, 0.25}.
  - F4: per-dataset paired-difference rows (family per_dataset, memoized) for every pooled confirmatory and headline effect, in pairwise_ci.csv, v2.json and a v2.md table. Test asserts C1 parents are present.
  - F5: HEADLINE_METRICS order is accuracy, B Brier, B coverage, A Brier, A coverage. Test asserts the order.
  - F6: a balanced-estimand note beside the headline; a `sampling` column in docs/public-results.csv, shown in the v2.md public table (unknown sources marked "not stated"). Test asserts the note.
  Plan and PROTOCOL text updated accordingly; freeze template updated.

### A7 round 2 — Codex — REVISE
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-r2/claudex-uchhilc7/result.json (cross_provider; plan sha 3cac0cf3…). Round-1 F1–F6 confirmed addressed.
  Amendment 7 round 2 dispositions (host = Claude). All accepted, with tests (suite passes, coverage 91.32%):
  - A7R2-F1: per-dataset rows iterate unique parent IDs, so reused C1 headline rows share the C1 per-dataset rows; the test asserts unique per-dataset IDs.
  - A7R2-F2: docs/public-results.csv is bound by public_results_sha256 in the freeze record; verify_frozen checks it (test: an edited file is refused); v2.json records public_results_sha256.
  - A7R2-F3: corrected. The sum-of-squares Brier spans [0, 2] for all K; the chance baseline 1 − 1/K and typical values differ. PROTOCOL, the v2.md note and the CCFA review's C005 are updated (C005 marked as corrected by this finding).
  The CCFA review records C001–C007 as resolved at f659375 and still passes validation.

### A7 round 3 — Codex — APPROVED
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-r3/claudex-j5venmy1/result.json (cross_provider; plan sha e4056cfaa5b743e04e12cfa3d135b7f568ab239b7bdab794559d743c1490efeb). No findings.

### A7 build inspection 1 — Codex (fresh), base 35568e0
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-inspect1/claudex-21pbf8b0/result.json (cross_provider; base 35568e0; head 8e76aef) — REVISE
- F1 medium: verify_frozen accepted records missing the new hashes → FIXED: all six hashes required; missing or TO-FILL values refused; each artifact checked (test covers every key and every artifact).
- F2 medium: score accuracy is not temperature-invariant → FIXED: the secondary headline adds B-scaled accuracy rows for score datasets only; the protocol limits the invariance claim to choice/noul.

### A7 build inspection 2 — Codex (fresh), base 35568e0
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-inspect2/claudex-btkxz3g6/result.json (cross_provider; base 35568e0; head 67be976) — REVISE (A7 inspection budget 2/2 used)
- R1 high: a cost pause could be lost in a crash between the prediction and status writes, and was not rebuilt on restart → FIXED: status is written before the row; the ledger records pause=true and Budget rebuilds paused until an operator appends pause_cleared; the report accepts only status=active attempts.
- R2 medium: all-failed order calls reported as stable → FIXED: only successful order calls count toward completeness.
- R3 medium: MASSIVE prevalence mis-described → FIXED (documentation): stated as pre-assignment prevalence for later locales.
- Regression tests added. **These fixes are not covered by a Codex inspection** (budget exhausted).

### A7 build inspection 3 — user-approved extra round, focused on 8369186 (base 67be976)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-inspect3/claudex-4laqkgwy/result.json (cross_provider; base 67be976; head 8369186) — REVISE
- A7I3-F1 high: a stale "active" status could hide a pause after a crash in a later split → FIXED: the runner marks the attempt "running" before dispatch; the report refuses every Jev attempt while ledger_paused(); only "active" attempts are selectable.
- A7I3-F2 medium: pause_cleared never unblocked new runs → FIXED: _attempt_dir blocks cost_paused only while the ledger is still paused; after pause_cleared it starts a new attempt and the paused attempt stays ineligible. cost_exhausted still blocks.
- A7I3-F3 medium: REJECTED with evidence. v2_report.py:204 groups only repeat_index == 0 rows, so order_rows never includes repeat 1 and a successful repeat 1 cannot complete an order. A regression test (repeat 0 fails on every order call, repeats 1–2 succeed → "permutation run incomplete: 0/8") confirms the behaviour.
- Tests added; suite passes (coverage 91.49%). Not yet re-inspected.

### A7 build inspection 4 — user-approved, focused on 484dc0c (base 8369186)
- result: /private/tmp/claude-501/-Users-yanghu-Documents-AI-Workspace-experiments-jev-benchmarks/74a33ab8-1c21-42b2-9bff-14105b4725cf/scratchpad/claudex-a7-inspect4/claudex-mm5h8585/result.json (cross_provider; base 8369186; head 484dc0c) — REVISE
- A7I4-F1 high: a crash-left "running" attempt could be reused after pause_cleared → FIXED: ledger records carry the attempt name; paused_attempts() marks every attempt that ever received a pausing retention, and it is never reused (_attempt_dir starts a new one) or reported (_select_attempt skips it).
- A7I4-F2 medium: a complete attempt left "running" by a crash could never be reported → FIXED: a rerun with no pending work finalizes it to "active" (safe, because _attempt_dir never returns a pause-exposed attempt).
- Tests added; suite passes (coverage 91.50%). Not yet re-inspected.

## Freeze (2026-09-24) — user-authorized
- User: "推送到 fork（7 个提交），然后进入冻结：运行 prepare，填入冻结记录需要的 6 个哈希，打 v2-preregistered 标签，接着跑 GLiNER 基准复现检查和 pilot。" The last fixes (e26a9a2) were frozen without a further Codex inspection, by user choice.
- P3.0 anchor: the pilot-v1 manifest was regenerated and written in v1 seven-field format; sha256 equals the upstream ec064c52…. GLiNER needed an offline refs/main → 235cf92 (gliner2 fetches encoder_config at main). Comparison PASS; exact accuracy, Brier/F1 within 1e-6.
- prepare: 5,570 rows (test 2,100 / calibration 1,800 / permutation 1,400 / pilot 270). Length exclusions: Civil 233, UltraFeedback 201. Emotion surprise test = 28 (availability-limited).

## P3 pilot (2026-09-24)
- All four zero-shot contenders ran on the frozen pilot split after v2-preregistered (verify_frozen OK). 0 failures in every contender × dataset cell; vector lengths match label counts; Jev: 810 calls, one snapshot (typesafe/jev-1.13-20260917), settled $0.032196, 0 retained.
- Aggregate: results/reports/pilot-v2.{json,md} (PILOT: plumbing/timing only, no accuracy). Qwen two-digit scoring on Banking77 is slow (p50 2.1 s, p95 8.1 s).
