# Handoff — Jev vs Laya local benchmark (claudex-loop, mid-Phase 1)

> **Superseded 2026-09-23** by `2026-09-23-jev-laya-benchmark.md` (locked plan) and its `.review-log.md`. Kept as the recon record. Paths below that mention `docs/claudex/` now live in `docs/exec-plans/active/`.

Origin: a claudex-loop planning session, 2026-09-22/23.

## Status
- Fork decision executed: local clone at `experiments/jev-benchmarks` (no GitHub fork created).
- Phase 0 recon: done. Research gate: `none`. Reviewer: Codex 0.156.0 (logged in), `timeout 600` available.
- Phase 1: load-bearing Q1–Q6 **locked**. The cosmetic batch was **proposed and awaits a veto-or-accept**. PLAN.md is **not written yet**.
- Next: accept the cosmetic batch → clone the fork → write `docs/claudex/PLAN.md` and `docs/claudex/PLAN-REVIEW-LOG.md` → Phase 2 Codex review (MAX_ROUNDS=5) → user sign-off → Phase 3 build in this repo (`experiments/jev-benchmarks/`).

## Locked decisions
| # | Decision |
|---|---|
| Q1 | Jev via **OpenRouter** `POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`; log the resolved snapshot (`jev-1.13-20260917`). Key: `OPENROUTER_API_KEY` from the environment; read at runtime only, never written or printed. |
| Q2 | 5 contenders: Jev 1.13; Laya English base; Laya multilingual + Router (non-English slice); Laya typed-decisions; **Qwen3-1.7B logit scoring** (raw + temperature-scaled). Plus majority and random baselines. Kev, Decider and GLiNER are out. |
| Q3 | Fork `AbdelStark/jev-benchmarks` (Apache-2.0) at a pinned commit, keeping LICENSE and attribution. Add adapters `jev_openrouter`, `laya`, `qwen_logit`; add noul/score question types, a permutation suite, and a single-vs-batched latency suite. `nibzard/decision-model-benchmark` has no license: ideas only, no code. |
| Q4 | choice: BTZSC AG News / DAIR Emotion / Banking77, 300 each. noul: SMS Spam 300 and ToxicChat 300 (CC-BY-NC, local only). score: Yelp Review Full 300. Multilingual: MASSIVE en/zh/km, 100 each. Permutation: 100 per choice set × 3 shuffles. Latency: AG News 100 at 1 vs 10 questions per call. Jev: 3 repeats per item. **Pilot first at 30 per condition.** |
| Q5 | Every contender gets two conditions: **A as-returned** and **B post-hoc temperature scaling** fitted on a disjoint 200-item calibration split per dataset (Jev included). Laya typed-decisions is labelled "fine-tuned on TypeSafe's 4 workflows, out-of-domain here". Exploratory, pre-registered: Laya `head_max_len=512` on Banking77. No fine-tuning (no CUDA; avoids train-split contamination). |
| Q6 | Keep Jev's raw 2-decimal probabilities and raw sum. Report the **zero-probability-on-true-label rate**. For NLL and temperature fitting, floor at ε=0.005 and renormalize; also report the share of infinite NLL. Primary metrics: accuracy, macro-F1, Brier. **Rounding-parity sensitivity analysis**: round Laya/Qwen to 2 decimals and recompute. Renormalize only within a narrow declared tolerance; otherwise count as a failure. |

## Cosmetic batch (proposed; accept unless vetoed)
1. Python 3.12 via `uv`; torch/laya as optional dependencies.
2. Seed `20260923`.
3. Latency: 5 warm-ups excluded; report p50/p95. Local models: model-only and end-to-end. Jev: end-to-end only.
4. Jev concurrency: serial for the latency suite, ≤4 parallel otherwise (limit 1,200 rpm).
5. Laya on MPS, falling back to CPU; record the device.
6. Qwen labels: A–Z when K≤26. When K>26, two-digit IDs 01–K with exact joint probability P(d1)·P(d2|d1) from ~2 forward passes, then normalized over the K options.
7. Key from env only; scrub logs; refuse to write output that contains the key.
8. Rmd `study/jev_laya_benchmark_zh.Rmd`: Chinese prose, English figure labels, every number via `reg()`, reading vendored CSVs in `study/docs/data-external/jev-laya-bench/` with a SHA-256 manifest. Never publish HTML.
9. `PLAN_FILE=experiments/jev-benchmarks/docs/claudex/PLAN.md`; review log beside it; `MAX_ROUNDS=5`.

## Layout
- Code, configs, runs: `experiments/jev-benchmarks/` — **cloned 2026-09-23 at upstream `0d610cc53e79bcbec691312b0c4adb4a0e371642`**; remote renamed `upstream`, push URL set to `DISABLED`. Claudex artifacts live in `docs/claudex/`. Raw JSONL goes in gitignored `results/runs/`; aggregates in `results/reports/`.
- study: the Rmd above plus a pointer plan at `study/docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md` (follow `study/CLAUDE.md`: `reg()`, never hand-typed numbers, `rm -rf <name>_files` before rendering).
- wiki: nothing written during the benchmark. Ingest the final results via `llm-wiki-ingest` afterwards.

## Recon facts (verified 2026-09-23)
- Hardware: M1 Pro, 16 GB, 75 GB free. System Python 3.14; uv 0.7.13; R 4.6.1 with rmarkdown/flextable/showtext/data.table/ggplot2.
- Jev request shape: `{model, state, questions}`. `criteria` is **required**: an object for choice (option→description), an array for score (levels), an object for noul (e.g. `{"true":…, "false":…}`). The response includes per-option probabilities (2 decimals, zeros occur), `confidence`, and usage cost (~$0.0000177 per 422-token call). ~260 ms end to end. **Non-deterministic** between identical calls (0.22→0.20).
- Jev confidence (TypeSafe docs): `clamp((K·p_max − 1)/(K − 1), 0, 1)` — a deterministic transform of p_max, not independent information.
- Jev limits (docs): 64k tokens per request; 32k for state + longest question; up to 255 choice options (256 → 400); 1,200 rpm. Only one model exists (`jev-1.13.0`; `jev-latest`/`jev-preview` are aliases). The docs' jaggedness list covers literal reading, math, dates, indirection, large irrelevant state, adversarial content, and contradictory criteria.
- Laya: PyPI `laya` 0.3.9 (py≥3.10; torch, transformers). HF `convaiinnovations/laya` (Apache-2.0) has 3 subfolders and ships `.py` files — **treat as a supply-chain risk: isolated venv, no secrets**. The English option head shares a 192–256-token budget; the context is 512. Base weights are uncalibrated (ECE 0.466 raw).
- Prior art: AbdelStark pilot (n=100/condition; Jev AG News 0.910, Banking77/BTZSC 0.870, DAIR 0.480; 16% zero probability on DAIR). nibzard (Jev Banking77 76.3%, 13% flip under permutation, ECE 0.246, 255-choice cap). Neither tested Laya or a local LLM-logit baseline.

## Background (wiki references, do not copy)
`knowledge_base/wiki/syntheses/2026-09-22-system-one-decision-layer-jev-vs-local-logit-scoring.md` (Layers 1–3 plus a 6-step local recipe); entities `Jev.md`, `Laya.md`, `localdecide.md`; concepts `Logit-Based Option Scoring.md`, `Probability Calibration.md`, `Option Order Sensitivity.md`; open questions Q164–Q195 in `wiki/QUESTIONS.md` (this benchmark targets Q165, Q166, Q167, Q174, Q183, Q184, Q189, Q190, Q191, Q192).
