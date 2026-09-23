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
