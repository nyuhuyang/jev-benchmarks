# Jev–Laya–Qwen benchmark v2 protocol

Status: **preregistered** under git tag `v2-preregistered`, before any benchmark inference. The
freeze record `configs/v2-freeze.json` holds the sha256 of the config, manifest, manifest summary,
this protocol, `docs/public-results.csv` and `results/reports/probe-v2.json`, and every run and
report checks them. The P3.0 GLiNER anchor reproduced the upstream pilot-v1 exactly
(`results/reports/anchor-v2-comparison.json`).

## Question and hypotheses

The study compares hosted Jev 1.13 through OpenRouter, local Laya base and multilingual
checkpoints, and local Qwen3-1.7B logit scoring on the same typed decisions. Amendment 6 removed
the typed-decision checkpoint and the v2 GLiNER runs. It measures shared
capability, not overall intelligence. The confirmatory questions are pairwise differences in
accuracy and Brier score across a frozen common English dataset set. The null for each contrast
is a zero mean difference. Direction is not prespecified. GLiNER2.5 appears only as the P3.0
harness anchor on the upstream pilot-v1 manifest; it is never in C1–C3 or the Holm family.

## Artifacts and capability gate

The immutable model and dataset revisions and the local snapshot paths are in `configs/v2.yaml`:
- Laya `aa8c91c`;
- Qwen3-1.7B `70d244c`;
- GLiNER `235cf92`;
- datasets: BTZSC `fef2a2a`, SMS Spam `cae486f`, Civil Comments `f2970eb`, MASSIVE `ed58ac4`,
  UltraFeedback `40b4365`.

The pinned synthetic probe ran at tag `v2-probe` (commit `8720bb5`) on synthetic strings only. Its
aggregate record (shapes, lengths, counts and hashes; no dataset text) is committed as
`results/reports/probe-v2.json`, and its sha256 is part of the freeze record:
- **Laya output shapes:** noul returns a full two-way vector and score a full five-level
  distribution, so condition B is available for every Laya track. Choice returns the full label
  vector. All tracks ran on MPS.
- **Laya option head:** the primary head is `max(shipped head, sum(options) + max(instruction,
  16))`:
  - English base: 192 on all its datasets;
  - typed-decisions: 256;
  - multilingual on MASSIVE: 591.
  - Banking77 is N/A for both English tracks (a ~1,280-token head is needed).
  - The pinned snapshot JSON files were unchanged after loading.
- **Full-request length exclusions** (whole candidate pools):
  - Civil Comments: 233 of 1,804,874 (Laya base 512 budget);
  - UltraFeedback: 201 of 12,600;
  - none elsewhere.
- **Qwen rendered suffixes:** single-token `" A"`-style IDs after `"Answer:"` (`letter` mode) on
  AG News, Emotion, SMS Spam, Civil Comments and UltraFeedback; two-token IDs after `"Answer: "`
  (`two_digit_joint`) on Banking77 and MASSIVE. A duplicate call returned an identical vector on
  MPS.
- **Laya package source:** only `rl_agent_config.json`, `model.safetensors`, `tokenizer/*` and
  `encoder/*` are loaded; no remote Python is executed (source sha256 `d8945fed…`).

**Confirmatory family (frozen, k = 9).** The three-way set is AG News, DAIR Emotion, SMS Spam and
Civil Comments. The tests are C1 Jev–Qwen, C2 Jev–Laya-base and C3 Qwen–Laya-base, each × {accuracy
A_raw, Brier A_raw, Brier B_scaled}. The same list is in `configs/v2.yaml`
(`confirmatory_family`).

## Data and exclusions

BTZSC AG News, DAIR Emotion and Banking77 are choice tasks. SMS Spam and Civil Comments are
binary noul tasks, with toxicity `>= 0.5` positive for Civil Comments. UltraFeedback helpfulness (MIT; `truthful_qa` + `false_qa` sources; prompts grouped with all their completions; one seeded-hash representative completion per prompt is sampled, so the estimand is one randomly chosen completion per prompt) is a
five-level score task. MASSIVE en-US, zh-CN and km-KH are choice intent tasks with English option
descriptions. The full label universe is offered on every item. BTZSC out-of-scope rows with no
unique positive hypothesis are excluded. Public benchmark contamination is possible.

Pilot 30, calibration 200 and test 300 items are sampled per dataset, except MASSIVE test 100 per
locale, with seed 20260923 and class-balanced selection **within class availability** (a class
with too few candidates, e.g. DAIR Emotion *surprise* with 66, is exhausted and the remaining
slots go to other classes; the per-split class counts are recorded in `manifest-summary.json`).

**Estimand.** Every reported metric describes the class-balanced sample distribution, not the
natural base rate. Brier, ECE, temperature fitting and coverage at the 5% error budget all depend
on class prevalence, and the natural rates differ sharply (for example 13.4% spam in SMS Spam and
8.0% toxic in Civil Comments). Coverage therefore means "share of balanced test items automated",
not deployment traffic. The class prevalence of each dataset's sampled population (one representative per content group)
is recorded in `manifest-summary.json` as `pool_class_prevalence`, with raw-candidate prevalence
kept separately. For MASSIVE zh-CN and km-KH this is the pre-assignment prevalence: cross-locale
split assignments made for earlier locales further restrict which groups each split can draw.
Order effects are computed only when every required order call succeeded; otherwise they are
reported as unavailable with the number of successful calls. The summary is part of the freeze record (`manifest_summary_sha256`), as is
`docs/public-results.csv` (`public_results_sha256`), which the report checks and records.
Comparisons with public results list each source's sampling distribution. The three sets are disjoint. Single-source
BTZSC and SMS Spam splits are carved into those sets. NFKC/lowercase/whitespace-collapsed/
punctuation-stripped text hashes group duplicates before selection; MASSIVE parallel utterances
share a group by cross-locale ID. Merged-group and length-exclusion counts are reported. Any item
whose complete request exceeds any primary contender's context or Laya option head is excluded
from every primary contender on that dataset before sampling. If GLiNER alone overflows a selected
item, GLiNER is excluded from that dataset and reported N/A; it never shrinks the shared primary
pool. Primary inference never truncates. The
separately labelled Laya as-shipped-head exploratory condition may truncate options. This
favors shorter Civil Comments and UltraFeedback texts.

## Conditions and operations

Condition A is the returned probability vector. Condition B applies one temperature per contender
and dataset, fit on calibration NLL with `max(p, 0.005)` and bounded `T` in `[0.05, 20]`. B is a
derivation from stored A, never a new call. Scalar-only score output carries a bounded expected
level and has A only; B and vector metrics are unavailable. Shared score rules are MAE of expected
level and exact-level accuracy after round-half-up. Per-level probability metrics for score are
reported only where distributions exist.

Pilot items are for plumbing/timing and never support result claims. Jev has three repeats; local
backends have one after a duplicate pilot check. Primary Jev scores use repeat 0; mean-of-three is
secondary. A Jev snapshot change stops the attempt. Confirmatory Jev calibration and test records
must be from one complete single-snapshot attempt.

The permutation suite (Amendment 6: AG News only) uses 100 test items, identity plus three fixed
orders. Order-only keeps option IDs bound to semantic labels; Qwen also has a separate positional
letter mode estimating order plus label-token effects. These rates are never compared across
estimands. There is no latency suite (Amendment 6). The latency reference is each zero-shot
contender's per-call wall time and local model time, p50/p95, on first-attempt-successful test
calls, with the retried share reported. Local backends make one unrecorded warm-up per run and
Jev calls are serial. It is deployment-specific: hosted latency includes an OpenRouter hop.

## Outcomes and failure policy

The confirmatory outcomes are accuracy and Brier. Macro-F1 is a secondary, descriptive outcome
(no inference). Macro-F1
averages only labels in targets union predictions on the evaluated sample, recalculated inside
every bootstrap resample; a predicted class absent from targets has F1 zero. Banking77 and MASSIVE
macro-F1 is exploratory. Per-class support is reported. Failed calls remain in every shared-item
denominator, count as incorrect, and receive Brier 2 and NLL `-log(0.005)`. Failure rate is an
outcome. Success-only metrics are labelled conditional.

Secondary outcomes: epsilon-floored NLL, historical 1e-12-clipped NLL, true-label zero share,
top-label ECE with 10 bins, reliability bins, calibration-chosen threshold coverage and realized
test error at 5% calibration risk, including no feasible threshold; score MAE and quadratic-weighted
kappa; order flip rates, Jev repeat flip baseline and excess order effect; Jev repeat agreement and
mean absolute probability difference; two-decimal rounding parity; p50/p95 wall and local model
latency. Same-slice selective coverage is descriptive only.

## Confirmatory inference

C1 compares Jev with Qwen, C2 Jev with Laya-base, and C3 Qwen with Laya-base. When C2/C3 remain,
all three use the same frozen three-way English set. Otherwise C1 uses the frozen Jev–Qwen set.
Each contrast has one accuracy test (argmax is unchanged by temperature), Brier A-versus-A, and
Brier B-versus-B. The resulting `k` is 9 or 3, as frozen above. Mixed-policy comparisons are
descriptive.

Each contrast is the equal-weight average of its frozen dataset effects. The sum-of-squares Brier
spans [0, 2] for every K, but its chance baseline (uniform prediction, 1 − 1/K) and typical values
differ across the K = 4, 6, 2, 2 datasets, so the pooled Brier effect mixes different baselines.
Per-dataset differences are reported beside every pooled effect. A dataset-stratified
paired bootstrap uses 2,000 resamples. Each resample redraws calibration and test items and refits
both B temperatures. Unadjusted 95% percentile intervals are reported. Two-sided bootstrap
`p = 2 min(P(delta* <= 0), P(delta* >= 0))`, floored at 1/2000. Holm step-down runs across the
frozen `k` tests at family-wise alpha 0.05. The report states `k`, unadjusted intervals and Holm
decisions. Other comparisons are descriptive, with unadjusted intervals and no winner claims.

## Exploratory analyses and limitations

Laya as-shipped `head_max_len` on K > 20, mean-of-three Jev, rounding parity, Banking77 and MASSIVE
macro-F1, multilingual slices, AG News permutation modes and mixed-policy contrasts are
exploratory or descriptive. Limitations include public-data contamination, the class-balanced
estimand (see Data and exclusions), smaller rare-class supports, length-selection bias,
hosted-network versus local-hardware latency, and OpenRouter versus a direct TypeSafe call. Process environment
isolation for local backends is not an OS sandbox; inspected package code can still read an
absolute-path file. No fine-tuning or benchmark-specific prompt tuning is performed; the Jev/Laya/Qwen contrasts are zero-shot. The Amendment 5 few-label arm is supervised on the 200 calibration labels per dataset and is reported as such.

## Amendment 5: few-label arm, confirmatory set, framing

**Framing.** This study is a same-run controlled re-check of public Jev, Laya and local-logit
results, not a first result. `docs/public-results.csv` lists the public numbers it is compared
against, with their protocols. Accuracy differences below the detectable range (about 4–6
percentage points for plausible paired disagreement rates, Holm k=9) are reported as unresolved.

**Non-binding expectations from public evidence** (not hypotheses; tests stay two-sided):
- zero-shot Jev leads Laya-base and Qwen3-1.7B logit scoring;
- the order of Laya-base and Qwen3-1.7B is undecided;
- raw Qwen3-1.7B calibration is worst, and temperature scaling narrows its gap the most;
- with about 200 labels, a small local classifier can approach or pass zero-shot Jev.

**Confirmatory set.** The three-way English datasets are AG News, DAIR Emotion, SMS Spam and
Civil Comments. Laya is N/A on Banking77, and UltraFeedback is descriptive because its GPT-4
labels favour LLM-family contenders. C1–C3 × {accuracy, Brier A-vs-A, Brier B-vs-B} gives k = 9.
The test list goes into `configs/v2.yaml` once the synthetic probe confirms that Laya-base returns
full noul vectors; otherwise the C1-only family with k = 3 applies.

**Few-label arm (descriptive; supervised on 200 calibration labels per dataset; not zero-shot).**
- `qwen_probe` uses the output of Qwen3-1.7B decoder block 18 of 28 at the last prompt position of
  the zero-shot prompt. The layer was fixed a priori from AnyJev's public report.
- `tfidf_lr` uses `char_wb` 2–4-gram TF-IDF, fitted inside each training fold.
- `prior` predicts training-label class frequencies.
- The classifier is L2 multinomial logistic regression (`C = 1`, `lbfgs`, 2000 iterations), with
  no hyperparameter search.
- Cross-fitting uses `KFold(5, shuffle=True, random_state=20260923)` over the calibration items.
  The out-of-fold probabilities are the calibration predictions for temperature, threshold and
  bootstrap, and one refit on all 200 predicts test.
- A class absent from a training fold gets probability 0; the count is reported.
- Test labels are never used for fitting.
- These contenders are compared with Jev with unadjusted intervals. They report no latency and are
  outside the permutation and latency suites.
- The bootstrap refits T but not the classifier, so their intervals understate training variance.

## Amendment 6: scope cut and headline estimands

The headline question is how much of zero-shot Jev's advantage is left when a local model gets the
same ~200 labels. On identical test items, two paired gaps are reported side by side. Both are
contender minus Jev, the same sign as C1: a negative accuracy or coverage difference, or a positive
Brier difference, means Jev leads.
- the zero-shot gap, Jev vs `qwen_logit`;
- the few-label gap, Jev vs `qwen_probe`, `tfidf_lr` and `prior`.

**Like-for-like reading.** Under condition B, Jev's temperature is fitted on the same 200
calibration labels that train the few-label contenders, so **B-vs-B (Brier and coverage) plus
accuracy** answers the headline question. Accuracy is unchanged by temperature, so A and B coincide
for choice and noul datasets (argmax is T-invariant). Score accuracy rounds the expected level,
which T can move, so score datasets also get a B-scaled accuracy row. A-vs-A contrasts raw
zero-shot output with supervised output; it is reported as a supplement, and the report and Rmd
present B-vs-B first.

The metrics are accuracy, Brier A/B and coverage at 5% error under A/B. The primary set is the 4
confirmatory datasets; the secondary set is every shared dataset. The coverage bootstrap
re-selects each side's threshold on the resampled calibration set, and no feasible threshold
counts as coverage 0. On the primary set, the zero-shot accuracy and Brier rows are the C1
estimates, carrying the C1 Holm decision. Every other headline row is estimation only, with
unadjusted intervals and no winner claims.

Removed from v2: the latency suite, `laya_typed` (out of domain), and GLiNER runs. GLiNER remains
only as the P3.0 anchor, run through the minimal-environment worker with the pilot-v1 model spec
and the v2 runtime. The `v2-probe` record remains the capability check for the retained
contenders. Final length exclusions come from `prepare` on the Amendment 6 config.

Few-label thresholds are chosen on out-of-fold predictions and applied to an all-200 refit, so
realized selective error can drift from 5%; it is reported beside coverage.

Bootstrap failure rules (build inspection):
- A resampled calibration set with no successful vectors cannot fit T, so that draw uses T = 1
  (B = A for the draw); failures never drop draws.
- A score dataset with no successful call takes the vector failure penalties (Brier 2, NLL
  −log 0.005, coverage 0). Scalar-only handling needs a successful scalar call.
- A confirmatory test whose estimate is undefined, or whose usable draws are fewer than the frozen
  2,000, is reported without a Holm rejection.
- Jev dispatch history is summed across every invocation of a prediction key: earlier failed
  attempts count in the retry share and exclude the key from the first-attempt latency reference.
- If the observed calibration set of a contender has no successful vectors, its condition-B
  contrasts are reported as unavailable (no estimate, no Holm rejection); condition-A results and
  failure counts are still published.
- A complete local attempt in which every call failed is scored with the failure penalties, with the
  pinned checkpoint as provenance; Jev still needs one known snapshot.
- A Jev cost pause (a 2xx without a valid cost) is written to the attempt status before the
  prediction row and rebuilt from the budget ledger on restart. It stays in force until an
  operator reviews it and appends `{"event": "pause_cleared"}` to the ledger; the next run then
  starts a new attempt, and the paused attempt stays ineligible. A run marks its attempt `running`
  before dispatch and `active` only after it finishes cleanly. Only `active` attempts are
  eligible for the report, and no Jev attempt is reportable while the ledger holds a pause.
  Ledger records carry their attempt name, so an attempt that ever received a pausing retention is
  never reused or reported, even after `pause_cleared`. A complete attempt left `running` by a
  crash after its last row is finalized to `active` on rerun.
- Wall latency is measured by the runner around each call for every backend (including prompt
  preparation and local worker IPC); adapters keep model-only time separately. The latency
  reference row is published even when no call succeeded on its first attempt (n = 0, null
  quantiles), so the retry share is always reported.

## Deviations from the locked plan

- Jev's prepare-time token bound uses serialized UTF-8 request bytes, a conservative bound where
  the OpenRouter tokenizer is unavailable locally.
- Laya 0.3.9 has no `trust_remote_code` argument. Local package source inspection and the
  minimal process environment are used instead.
- Jev dispatch is serial, within the plan's maximum of four concurrent calls. This gives cleaner
  latency measurement and remains within the expected full-run time at about 0.3 seconds per call.
- The user-requested Amendment 3 added GLiNER2.5 as a descriptive contender and a pilot-v1 anchor
  command; Amendment 6 later removed the v2 GLiNER runs, so only the anchor remains.
- Amendment 4 (user-approved): Yelp Review Full was replaced by UltraFeedback helpfulness. The Yelp dataset terms restrict disclosure to third parties, and hosted Jev would receive the review text. UltraFeedback is MIT-licensed. Its 1–5 helpfulness ratings are GPT-4 annotations, not human labels, which is a stated limitation.
- The Laya package source review (0.3.9 at HF revision `aa8c91c`) passed. The package loads only `rl_agent_config.json`, `model.safetensors` (safetensors), `tokenizer/` and `encoder/`. The repository's `rl_agent_api.py`, `rl_common.py` and `email_utils.py` are not referenced by the package, and no remote code is executed.
- Group representatives are chosen by a seeded hash of the example ID rather than list position.
- Qwen prefill: letter IDs are scored as `" A"` after `"Answer:"` and two-digit IDs as `"01"` after
  `"Answer: "`. Qwen merges a space into a following letter but splits digits, so with the plan's
  single `"Answer: "` prefill the letter IDs were not prefix-compatible suffixes and fell back to
  K-pass full-sequence scoring; the split prefill keeps the plan's letter and two-digit modes.
- Laya primary `head_max_len` = `max(shipped head, smallest untruncated head)`, where the smallest
  untruncated head is `sum(options) + max(instruction, 16)`: `laya.common.build_sequence` trims
  every option once `head - sum(options) < 16`. The plan's bare `sum(options) + instruction` value
  silently truncated options (found in review before any benchmark inference); a contract test
  checks the rule against the pinned `build_sequence`.
- Laya-base/typed are N/A on Banking77: its 72 complete option descriptions need a 1,280-token
  head, above both the 512 and the 1,024 budgets. The Laya as-shipped-head exploratory condition on
  Banking77 is dropped: head 192 would expose under 15% of the option tokens. It remains on MASSIVE
  (multilingual checkpoint, 60 labels).
- The MASSIVE primary head is above the multilingual checkpoint's shipped head of 256 and
  therefore outside its training configuration; this is stated as a limitation.
- Pinned Laya snapshot JSON files were hashed before and after the synthetic probe and were
  unchanged, so `_fix_tokenizer_config` did not mutate the pinned artifacts.
- GLiNER score items use nominal five-class probabilities (descriptive only), not `ordinal()`.
- gliner2 2.0.0 fetches `encoder_config/config.json` at revision `main` instead of the pinned
  revision. Local backends run offline, so the pinned cache carries `refs/main` pointing at
  `235cf92`, and "main" can only resolve to the pinned snapshot.
- The pilot-v1 anchor manifest was regenerated from the pinned config and written in its original
  seven-field v1 format; its sha256 equals the upstream report's `manifest_sha256` (`ec064c52…`).
- Dataset loads that use `data_files` (MASSIVE, UltraFeedback) need Hub metadata resolution;
  `HF_HUB_OFFLINE=1` fails for them, so preparation runs online with pinned revisions.

