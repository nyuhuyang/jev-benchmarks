from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np

from .metrics import _macro_f1
from .models import Prediction

EPSILON = 0.005


def _shared_index(row: Prediction) -> int:
    if row.error is not None:
        return -1
    if row.question_type == "score":
        if row.expected_score is None:
            return -1
        return max(0, min(len(row.labels) - 1, int(row.expected_score + 0.5) - 1))
    return row.predicted_index


def _vector_rows(rows: Sequence[Prediction]) -> list[Prediction]:
    return [row for row in rows if row.error is None and bool(row.probabilities)]


def transform_temperature(probabilities: Sequence[float], temperature: float) -> tuple[float, ...]:
    logits = np.log(np.maximum(np.asarray(probabilities, dtype=float), EPSILON)) / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    return tuple(float(value) for value in weights / np.sum(weights))


def fit_temperature(rows: Sequence[Prediction]) -> float | None:
    valid = _vector_rows(rows)
    if not valid:
        return None
    log_probabilities = np.log(
        np.maximum(np.asarray([row.probabilities for row in valid], dtype=float), EPSILON)
    )
    targets = np.asarray([row.target_index for row in valid], dtype=int)
    indices = np.arange(len(valid))

    def loss(value: float) -> float:
        logits = log_probabilities / value
        peak = np.max(logits, axis=1)
        log_normalizer = peak + np.log(np.exp(logits - peak[:, None]).sum(axis=1))
        return float(np.mean(log_normalizer - logits[indices, targets]))

    # Bounded golden-section scalar search with explicit endpoint comparison.
    left, right = 0.05, 20.0
    ratio = (math.sqrt(5) - 1) / 2
    low = right - ratio * (right - left)
    high = left + ratio * (right - left)
    for _ in range(60):
        if loss(low) < loss(high):
            right, high = high, low
            low = right - ratio * (right - left)
        else:
            left, low = low, high
            high = left + ratio * (right - left)
    candidates = (0.05, (left + right) / 2, 20.0)
    return min(candidates, key=loss)


def condition_b(rows: Sequence[Prediction], temperature: float | None) -> list[Prediction]:
    if temperature is None:
        return []
    output = []
    for row in rows:
        probabilities = (
            transform_temperature(row.probabilities, temperature)
            if row.error is None and row.probabilities
            else ()
        )
        output.append(
            replace(
                row,
                condition="B_scaled",
                probabilities=probabilities,
                predicted_index=(
                    max(range(len(probabilities)), key=probabilities.__getitem__)
                    if probabilities
                    else -1
                ),
                expected_score=(
                    sum((index + 1) * value for index, value in enumerate(probabilities))
                    if probabilities and row.question_type == "score"
                    else row.expected_score
                ),
            )
        )
    return output


def select_threshold(calibration: Sequence[Prediction], error_budget: float = 0.05) -> float | None:
    valid = _vector_rows(calibration)
    if not valid:
        return None
    thresholds = sorted({max(row.probabilities) for row in valid}, reverse=True)
    feasible: float | None = None
    accepted_max = 0
    for threshold in thresholds:
        accepted = [row for row in valid if max(row.probabilities) >= threshold]
        risk = sum(_shared_index(row) != row.target_index for row in accepted) / len(accepted)
        if risk <= error_budget and len(accepted) > accepted_max:
            feasible, accepted_max = threshold, len(accepted)
    return feasible


def reliability_bins(rows: Sequence[Prediction], count: int = 10) -> list[dict[str, float | int]]:
    valid = _vector_rows(rows)
    output: list[dict[str, float | int]] = []
    for index in range(count):
        lower, upper = index / count, (index + 1) / count
        members = [
            row
            for row in valid
            if lower <= max(row.probabilities)
            and (max(row.probabilities) < upper or index == count - 1)
        ]
        output.append(
            {
                "bin": index,
                "n": len(members),
                "lower": lower,
                "upper": upper,
                "mean_confidence": sum(max(row.probabilities) for row in members) / len(members)
                if members
                else 0.0,
                "accuracy": sum(_shared_index(row) == row.target_index for row in members)
                / len(members)
                if members
                else 0.0,
            }
        )
    return output


def _weighted_kappa(rows: Sequence[Prediction]) -> float | None:
    if not rows:
        return None
    k = len(rows[0].labels)
    observed = np.zeros((k, k), dtype=float)
    for row in rows:
        observed[row.target_index, row.predicted_index] += 1
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    indices = np.arange(k)
    weights = ((indices[:, None] - indices[None, :]) / max(k - 1, 1)) ** 2
    denominator = float(np.sum(weights * expected))
    return 1 - float(np.sum(weights * observed)) / denominator if denominator else None


def score_v2(
    rows: Sequence[Prediction], *, threshold: float | None = None, ece_bins: int = 10
) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    valid = _vector_rows(rows)
    scalar = [
        row
        for row in rows
        if row.error is None and row.question_type == "score" and row.expected_score is not None
    ]
    success = [row for row in rows if row.error is None]
    targets = np.array([row.target_index for row in rows], dtype=int)
    argmax_predicted = np.array(
        [row.predicted_index if row.error is None else -1 for row in rows], dtype=int
    )
    score_question = rows[0].question_type == "score"

    predicted = (
        np.array([_shared_index(row) for row in rows], dtype=int)
        if score_question
        else argmax_predicted
    )
    result: dict[str, Any] = {
        "n": n,
        "valid": len(success),
        "failures": n - len(success),
        "failure_rate": (n - len(success)) / n,
        "accuracy": float(np.mean(targets == predicted)),
        "macro_f1": _macro_f1(targets, predicted, len(rows[0].labels)),
        "per_class_support": {
            str(index): int(np.sum(targets == index)) for index in range(len(rows[0].labels))
        },
    }
    if score_question:
        result["accuracy_argmax"] = float(np.mean(targets == argmax_predicted)) if valid else None
    if success:
        success_targets = np.asarray([row.target_index for row in success], dtype=int)
        success_predicted = np.asarray(
            [_shared_index(row) for row in success],
            dtype=int,
        )
        result["accuracy_conditional"] = float(np.mean(success_targets == success_predicted))
        result["macro_f1_conditional"] = _macro_f1(
            success_targets, success_predicted, len(rows[0].labels)
        )
    if score_question:
        errors = [abs(cast(float, row.expected_score) - (row.target_index + 1)) for row in scalar]
        result["score_mae"] = (sum(errors) + (len(rows[0].labels) - 1) * (n - len(scalar))) / n
        result["score_round_accuracy"] = result["accuracy"]
        result["quadratic_weighted_kappa_conditional"] = _weighted_kappa(
            [
                replace(
                    row,
                    predicted_index=min(
                        len(row.labels) - 1, int(cast(float, row.expected_score) + 0.5) - 1
                    ),
                )
                for row in scalar
            ]
        )
    if not valid:
        if not score_question or any(row.probabilities for row in rows):
            result["brier"] = 2.0
            result["nll"] = -math.log(EPSILON)
        else:
            result.update(
                {
                    key: None
                    for key in (
                        "brier",
                        "nll",
                        "ece",
                        "calibration_threshold",
                        "test_coverage",
                        "test_selective_error",
                        "no_feasible_threshold",
                        "descriptive_same_slice_coverage",
                    )
                }
            )
        return result
    brier_values = [
        sum(
            (value - int(index == row.target_index)) ** 2
            for index, value in enumerate(row.probabilities)
        )
        for row in valid
    ]
    true_probs = [row.probabilities[row.target_index] for row in valid]
    nll_values = [-math.log(max(value, EPSILON)) for value in true_probs]
    failures = n - len(valid)
    result.update(
        {
            "brier": (sum(brier_values) + 2 * failures) / n,
            "nll": (sum(nll_values) - failures * math.log(EPSILON)) / n,
            "brier_conditional": sum(brier_values) / len(valid),
            "nll_conditional": sum(nll_values) / len(valid),
            "nll_clip12_conditional": -sum(math.log(max(value, 1e-12)) for value in true_probs)
            / len(valid),
            "true_label_zero_rate": sum(value == 0 for value in true_probs) / len(valid),
            "infinite_nll_share": sum(value == 0 for value in true_probs) / len(valid),
            "mean_confidence": sum(max(row.probabilities) for row in valid) / len(valid),
            "ece": sum(
                bin_row["n"] / len(valid) * abs(bin_row["mean_confidence"] - bin_row["accuracy"])
                for bin_row in reliability_bins(valid, ece_bins)
            ),
        }
    )
    accepted = [
        row for row in valid if threshold is not None and max(row.probabilities) >= threshold
    ]
    result["calibration_threshold"] = threshold
    result["test_coverage"] = len(accepted) / n
    result["test_selective_error"] = (
        sum(_shared_index(row) != row.target_index for row in accepted) / len(accepted)
        if accepted
        else None
    )
    result["no_feasible_threshold"] = threshold is None
    descriptive_threshold = select_threshold(valid)
    result["descriptive_same_slice_coverage"] = (
        sum(max(row.probabilities) >= descriptive_threshold for row in valid) / n
        if descriptive_threshold is not None
        else 0.0
    )
    result["latency_p50_seconds"] = float(
        np.quantile([row.latency_seconds for row in success], 0.5)
    )
    result["latency_p95_seconds"] = float(
        np.quantile([row.latency_seconds for row in success], 0.95)
    )
    local_times = [
        row.model_latency_seconds for row in success if row.model_latency_seconds is not None
    ]
    if local_times:
        result["model_latency_p50_seconds"] = float(np.quantile(local_times, 0.5))
        result["model_latency_p95_seconds"] = float(np.quantile(local_times, 0.95))
    return result


def rounding_parity(rows: Sequence[Prediction]) -> list[Prediction]:
    output = []
    for row in rows:
        rounded = tuple(round(value, 2) for value in row.probabilities)
        total = sum(rounded)
        probabilities = tuple(value / total for value in rounded) if total else rounded
        output.append(
            replace(
                row,
                probabilities=probabilities,
                predicted_index=(
                    max(range(len(probabilities)), key=probabilities.__getitem__)
                    if probabilities
                    else -1
                ),
                expected_score=(
                    sum((index + 1) * value for index, value in enumerate(probabilities))
                    if probabilities and row.question_type == "score"
                    else row.expected_score
                ),
            )
        )
    return output


def flip_summary(rows: Sequence[Prediction]) -> dict[str, float | int]:
    grouped: dict[tuple[str, str, str], list[Prediction]] = defaultdict(list)
    for row in rows:
        if row.repeat_index == 0:
            grouped[(row.split, row.example_id, row.letter_mode)].append(row)
    flips = total = 0
    for group in grouped.values():
        identity = next((row for row in group if row.permutation_id == "order-0"), None)
        if identity is None:
            continue
        for row in group:
            if row.permutation_id != "order-0":
                total += 1
                flips += row.predicted_index != identity.predicted_index
    return {"n": total, "flip_rate": flips / total if total else 0.0}


def repeat_summary(rows: Sequence[Prediction]) -> dict[str, float | int]:
    grouped: dict[tuple[str, str, str, str], list[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[(row.split, row.example_id, row.permutation_id, row.letter_mode)].append(row)
    flips = total = 0
    differences: list[float] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: row.repeat_index)
        if len(ordered) < 2:
            continue
        for row in ordered[1:]:
            total += 1
            flips += row.predicted_index != ordered[0].predicted_index
            if row.probabilities and ordered[0].probabilities:
                differences.append(
                    sum(
                        abs(a - b)
                        for a, b in zip(row.probabilities, ordered[0].probabilities, strict=True)
                    )
                    / len(row.probabilities)
                )
    return {
        "n": total,
        "flip_rate": flips / total if total else 0.0,
        "argmax_agreement": 1 - flips / total if total else 0.0,
        "mean_abs_probability_difference": sum(differences) / len(differences)
        if differences
        else 0.0,
    }


def mean_of_repeats(rows: Sequence[Prediction]) -> list[Prediction]:
    grouped: dict[tuple[str, str], list[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[(row.split, row.example_id)].append(row)
    output = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: row.repeat_index)
        first = ordered[0]
        if any(row.error is not None or not row.probabilities for row in ordered):
            output.append(
                replace(first, error="repeat failure", predicted_index=-1, probabilities=())
            )
            continue
        probabilities = tuple(
            sum(row.probabilities[index] for row in ordered) / len(ordered)
            for index in range(len(first.labels))
        )
        output.append(
            replace(
                first,
                probabilities=probabilities,
                predicted_index=max(range(len(probabilities)), key=probabilities.__getitem__),
                expected_score=(
                    sum((index + 1) * value for index, value in enumerate(probabilities))
                    if first.question_type == "score"
                    else first.expected_score
                ),
                condition="A_mean_of_repeats",
            )
        )
    return output


def excess_order_bootstrap(
    permutation: Sequence[Prediction],
    repeats: Sequence[Prediction],
    *,
    resamples: int = 2000,
    seed: int = 20260923,
) -> dict[str, float]:
    by_id: dict[str, dict[str, list[Prediction]]] = defaultdict(lambda: defaultdict(list))
    for row in permutation:
        if row.repeat_index == 0 and row.letter_mode == "stable":
            by_id[row.example_id]["order"].append(row)
    for row in repeats:
        by_id[row.example_id]["repeat"].append(row)
    effects = []
    for groups in by_id.values():
        identity = next((row for row in groups["order"] if row.permutation_id == "order-0"), None)
        variants = [row for row in groups["order"] if row.permutation_id != "order-0"]
        repeat = sorted(groups["repeat"], key=lambda row: row.repeat_index)
        if identity is None or not variants or len(repeat) < 2:
            continue
        order_rate = sum(row.predicted_index != identity.predicted_index for row in variants) / len(
            variants
        )
        baseline = sum(row.predicted_index != repeat[0].predicted_index for row in repeat[1:]) / (
            len(repeat) - 1
        )
        effects.append(order_rate - baseline)
    if not effects:
        return {"n": 0, "excess_order_effect": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(rng.choice(effects, size=len(effects), replace=True)))
        for _ in range(resamples)
    ]
    return {
        "n": len(effects),
        "excess_order_effect": float(np.mean(effects)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _paired_rows(
    left: Sequence[Prediction], right: Sequence[Prediction]
) -> tuple[list[Prediction], list[Prediction]]:
    lmap = {row.example_id: row for row in left}
    rmap = {row.example_id: row for row in right}
    if set(lmap) != set(rmap):
        raise ValueError("paired example IDs differ")
    ids = sorted(lmap)
    for key in ids:
        if lmap[key].target_index != rmap[key].target_index or lmap[key].labels != rmap[key].labels:
            raise ValueError("paired example contract mismatch")
    return [lmap[key] for key in ids], [rmap[key] for key in ids]


def joint_paired_bootstrap(
    left_test: Mapping[str, Sequence[Prediction]],
    right_test: Mapping[str, Sequence[Prediction]],
    left_cal: Mapping[str, Sequence[Prediction]],
    right_cal: Mapping[str, Sequence[Prediction]],
    *,
    metric: str,
    condition: str,
    left_condition: str | None = None,
    right_condition: str | None = None,
    resamples: int = 2000,
    seed: int = 20260923,
) -> dict[str, float]:
    if set(left_test) != set(right_test):
        raise ValueError("dataset sets differ")
    datasets = sorted(left_test)
    pairs = {name: _paired_rows(left_test[name], right_test[name]) for name in datasets}
    left_policy = left_condition or condition
    right_policy = right_condition or condition
    if left_policy not in {"A_raw", "B_scaled"} or right_policy not in {"A_raw", "B_scaled"}:
        raise ValueError("unknown condition policy")
    if left_policy == "B_scaled" or right_policy == "B_scaled":
        if any(
            (left_policy == "B_scaled" and fit_temperature(left_cal[name]) is None)
            or (right_policy == "B_scaled" and fit_temperature(right_cal[name]) is None)
            for name in datasets
        ):
            raise ValueError("condition B unavailable without full calibration vectors")

    def difference(
        sample: Mapping[str, tuple[Sequence[Prediction], Sequence[Prediction]]],
        calibrations: Mapping[str, tuple[Sequence[Prediction], Sequence[Prediction]]],
    ) -> float:
        values = []
        for name in datasets:
            left, right = sample[name]
            if left_policy == "B_scaled":
                left = condition_b(left, fit_temperature(calibrations[name][0]))
            if right_policy == "B_scaled":
                right = condition_b(right, fit_temperature(calibrations[name][1]))
            values.append(float(score_v2(right)[metric] - score_v2(left)[metric]))
        return sum(values) / len(values)

    cal_pairs = {name: _paired_rows(left_cal[name], right_cal[name]) for name in datasets}
    observed = difference(pairs, cal_pairs)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        test_sample: dict[str, tuple[list[Prediction], list[Prediction]]] = {}
        cal_sample: dict[str, tuple[list[Prediction], list[Prediction]]] = {}
        for name, source, destination in ((name, pairs, test_sample) for name in datasets):
            left, right = source[name]
            indices = rng.integers(0, len(left), size=len(left))
            destination[name] = (
                [left[int(index)] for index in indices],
                [right[int(index)] for index in indices],
            )
        for name in datasets:
            left, right = cal_pairs[name]
            indices = rng.integers(0, len(left), size=len(left))
            cal_sample[name] = (
                [left[int(index)] for index in indices],
                [right[int(index)] for index in indices],
            )
        differences.append(difference(test_sample, cal_sample))
    nonpositive = sum(value <= 0 for value in differences) / resamples
    nonnegative = sum(value >= 0 for value in differences) / resamples
    return {
        "difference": observed,
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "p_two_sided": max(1 / 2000, min(1.0, 2 * min(nonpositive, nonnegative))),
    }


def holm_decisions(p_values: Mapping[str, float], k: int, alpha: float = 0.05) -> dict[str, bool]:
    if len(p_values) != k or k not in (3, 9):
        raise ValueError("Holm family must match frozen k of 3 or 9")
    ordered = sorted(p_values, key=p_values.__getitem__)
    rejected = True
    decisions: dict[str, bool] = {}
    for index, name in enumerate(ordered):
        rejected = rejected and p_values[name] <= alpha / (k - index)
        decisions[name] = rejected
    return decisions
