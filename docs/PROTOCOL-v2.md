# Jev–Laya–Qwen benchmark v2 protocol

Status: **draft, not preregistered**. Fill every `TO-FILL-AFTER-PROBE` field after the pinned
synthetic capability checks, then freeze this document, the configs and manifest hashes under
`v2-preregistered` before test inference.

## Question and hypotheses

The study compares hosted Jev 1.13 through OpenRouter, local Laya base/multilingual/typed-decision
checkpoints and local Qwen3-1.7B logit scoring on the same typed decisions. It measures shared
capability, not overall intelligence. The confirmatory questions are pairwise differences in
accuracy and Brier score across a frozen common English dataset set. The null for each contrast
is a zero mean difference. Direction is not prespecified. GLiNER2.5 is a descriptive contender on
all nine datasets, including nominal five-class score probabilities. It is never in C1–C3 or the
Holm family.

## Artifacts and capability gate

The immutable model and dataset revisions are in `configs/v2.yaml`. Exact revisions and local
snapshot paths: **TODO-PIN**. Synthetic probe observations for Laya score/noul output shapes,
complete option head sizes, full-request token lengths, Qwen rendered suffixes and the Laya
package-source review: **TO-FILL-AFTER-PROBE**. Any remote `.py` import by Laya is a stop for that
track. The confirmatory three-way dataset set is **TO-FILL-AFTER-PROBE**. If fewer than three
English datasets qualify, C2/C3 leave the family and C1 uses the Jev–Qwen intersection. Exact
confirmatory test IDs, their dataset sets and Holm denominator `k` (9 or 3):
**TO-FILL-AFTER-PROBE**. The frozen list must also be copied into `configs/v2.yaml`.

## Data and exclusions

BTZSC AG News, DAIR Emotion and Banking77 are choice tasks. SMS Spam and Civil Comments are
binary noul tasks, with toxicity `>= 0.5` positive for Civil Comments. UltraFeedback helpfulness (MIT; `truthful_qa` + `false_qa` sources; prompts grouped with all their completions; one seeded-hash representative completion per prompt is sampled, so the estimand is one randomly chosen completion per prompt) is a
five-level score task. MASSIVE en-US, zh-CN and km-KH are choice intent tasks with English option
descriptions. The full label universe is offered on every item. BTZSC out-of-scope rows with no
unique positive hypothesis are excluded. Public benchmark contamination is possible.

Pilot 30, calibration 200 and test 300 items are sampled per dataset, except MASSIVE test 100 per
locale, with seed 20260923 and class-balanced selection. The three sets are disjoint. Single-source
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

The permutation suite uses 100 test items per BTZSC choice dataset, identity plus three fixed
orders. Order-only keeps option IDs bound to semantic labels; Qwen also has a separate positional
letter mode estimating order plus label-token effects. These rates are never compared across
estimands. The latency suite uses 100 AG News items. Cross-contender latency uses one item, one
choice question, one call. Scaling is within backend: Jev/Laya 1 versus 10 paraphrased questions
sharing state, Qwen 1 versus 10 independent prompts. Local latency excludes five warm-ups; Jev
excludes none. Hosted latency includes an OpenRouter hop.

## Outcomes and failure policy

Primary discrimination and probability outcomes are accuracy, macro-F1 and Brier. Macro-F1
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

Each contrast is the equal-weight average of its frozen dataset effects. A dataset-stratified
paired bootstrap uses 2,000 resamples. Each resample redraws calibration and test items and refits
both B temperatures. Unadjusted 95% percentile intervals are reported. Two-sided bootstrap
`p = 2 min(P(delta* <= 0), P(delta* >= 0))`, floored at 1/2000. Holm step-down runs across the
frozen `k` tests at family-wise alpha 0.05. The report states `k`, unadjusted intervals and Holm
decisions. Other comparisons are descriptive, with unadjusted intervals and no winner claims.

## Exploratory analyses and limitations

Laya as-shipped `head_max_len` on K > 20, mean-of-three Jev, rounding parity, Banking77 and MASSIVE
macro-F1, multilingual slices, permutation modes, latency scaling and mixed-policy contrasts are
exploratory or descriptive. Limitations include public-data contamination, smaller rare-class
supports, length-selection bias, hosted-network versus local-hardware latency, OpenRouter versus a
direct TypeSafe call, and the out-of-domain typed-decision checkpoint. Process environment
isolation for local backends is not an OS sandbox; inspected package code can still read an
absolute-path file. No fine-tuning or benchmark-specific prompt tuning is performed; the Jev/Laya/Qwen contrasts are zero-shot. The Amendment 5 few-label arm is supervised on the 200 calibration labels per dataset and is reported as such.

## Deviations from the locked plan

- Jev's prepare-time token bound uses serialized UTF-8 request bytes, a conservative bound where
  the OpenRouter tokenizer is unavailable locally.
- Laya 0.3.9 has no `trust_remote_code` argument. Local package source inspection and the
  minimal process environment are used instead.
- Jev dispatch is serial, within the plan's maximum of four concurrent calls. This gives cleaner
  latency measurement and remains within the expected full-run time at about 0.3 seconds per call.
- The user-requested Amendment 3 adds GLiNER2.5 as a descriptive contender and a pilot-v1 anchor
  command. Its dataset overflow never changes the primary common pool.
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
- Dataset loads that use `data_files` (MASSIVE, UltraFeedback) need Hub metadata resolution;
  `HF_HUB_OFFLINE=1` fails for them, so preparation runs online with pinned revisions.

