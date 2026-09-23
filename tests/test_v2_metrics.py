from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from jev_benchmarks.metrics import _macro_f1
from jev_benchmarks.models import Prediction
from jev_benchmarks.v2_metrics import (
    EPSILON,
    condition_b,
    excess_order_bootstrap,
    fit_temperature,
    flip_summary,
    holm_decisions,
    joint_paired_bootstrap,
    mean_of_repeats,
    reliability_bins,
    repeat_summary,
    rounding_parity,
    score_v2,
    select_threshold,
    transform_temperature,
)


def prediction(
    index: int, target: int, vector: tuple[float, ...], dataset: str = "d"
) -> Prediction:
    return Prediction(
        "e",
        "b",
        "m",
        "snapshot",
        dataset,
        f"id:{index}",
        target,
        max(range(len(vector)), key=vector.__getitem__),
        tuple(str(i) for i in range(len(vector))),
        vector,
        0.1,
        model_latency_seconds=0.05,
    )


def test_macro_f1_denominator_and_failure_penalties() -> None:
    assert _macro_f1(np.array([0, 0]), np.array([0, 0]), 4) == 1.0
    assert _macro_f1(np.array([0]), np.array([1]), 4) == 0.0
    good = prediction(0, 0, (1.0, 0.0))
    failed = replace(
        prediction(1, 1, (0.8, 0.2)), probabilities=(), predicted_index=-1, error="timeout"
    )
    metrics = score_v2([good, failed])
    assert metrics["accuracy"] == 0.5
    assert metrics["brier"] == 1.0
    assert metrics["nll"] == pytest.approx(-math.log(EPSILON) / 2)
    assert metrics["brier_conditional"] == 0
    assert metrics["failure_rate"] == 0.5
    assert metrics["per_class_support"] == {"0": 1, "1": 1}
    all_failed = score_v2([failed])
    assert all_failed["brier"] == 2
    assert all_failed["nll"] == pytest.approx(-math.log(EPSILON))


def test_temperature_fit_condition_b_and_zero_policy() -> None:
    calibration = [prediction(i, i % 2, (0.9, 0.1) if i % 2 else (0.1, 0.9)) for i in range(6)]
    temperature = fit_temperature(calibration)
    assert temperature is not None and 1 < temperature <= 20
    assert fit_temperature([]) is None
    transformed = condition_b(calibration, temperature)
    assert transformed[0].condition == "B_scaled"
    assert transformed[0].predicted_index == calibration[0].predicted_index
    assert condition_b(calibration, None) == []
    assert sum(transform_temperature((0.0, 1.0), 1.0)) == pytest.approx(1)
    raw = prediction(0, 0, (0.0, 1.0))
    scores = score_v2([raw])
    assert scores["true_label_zero_rate"] == 1
    assert scores["nll"] == pytest.approx(-math.log(EPSILON))
    assert scores["nll_clip12_conditional"] == pytest.approx(-math.log(1e-12))


def test_threshold_is_calibration_only_and_ties_stay_together() -> None:
    perfect = prediction(0, 0, (0.9, 0.1))
    wrong = prediction(1, 1, (0.9, 0.1))
    assert select_threshold([perfect, wrong]) is None
    assert score_v2([perfect, wrong], threshold=None)["no_feasible_threshold"] is True
    threshold = select_threshold([perfect, prediction(2, 1, (0.1, 0.9))])
    assert threshold == 0.9
    scores = score_v2([perfect, wrong], threshold=threshold)
    assert scores["test_coverage"] == 1
    assert scores["test_selective_error"] == 0.5
    assert len(reliability_bins([perfect, wrong], 10)) == 10
    assert sum(row["n"] for row in reliability_bins([perfect, wrong])) == 2


def test_scalar_score_shared_rules_and_rounding_parity() -> None:
    scalar = Prediction(
        "e",
        "b",
        "m",
        "s",
        "score",
        "id",
        2,
        2,
        ("1", "2", "3", "4", "5"),
        (),
        0.1,
        question_type="score",
        expected_score=2.5,
    )
    scores = score_v2([scalar])
    assert scores["score_mae"] == 0.5
    assert scores["score_round_accuracy"] == 1
    assert scores["brier"] is None
    assert scores["nll"] is None and scores["ece"] is None
    assert scores["calibration_threshold"] is None
    assert scores["quadratic_weighted_kappa_conditional"] is None
    vector = replace(
        scalar, probabilities=(0.0, 0.0, 0.494, 0.506, 0.0), expected_score=3.506, predicted_index=3
    )
    assert score_v2([vector])["score_mae"] == pytest.approx(0.506)
    assert score_v2([vector])["accuracy"] == 0.0
    assert score_v2([vector])["accuracy_argmax"] == 0.0
    rounded_wins = replace(
        vector,
        target_index=2,
        probabilities=(0.0, 0.0, 0.4, 0.35, 0.25),
        expected_score=3.85,
        predicted_index=2,
    )
    assert score_v2([rounded_wins])["accuracy"] == 0.0
    assert score_v2([rounded_wins])["accuracy_argmax"] == 1.0
    failed = replace(scalar, error="failure", expected_score=None)
    assert score_v2([failed])["score_mae"] == 4.0
    three_level = replace(failed, labels=("1", "2", "3"), target_index=1)
    assert score_v2([three_level])["score_mae"] == 2.0
    rounded = rounding_parity([vector])[0]
    assert sum(rounded.probabilities) == pytest.approx(1)
    assert rounded.probabilities != vector.probabilities


def test_flip_repeat_and_excess_effect() -> None:
    identity = replace(prediction(0, 0, (0.7, 0.3)), permutation_id="order-0")
    order_rows = [identity]
    for index in range(1, 4):
        order_rows.append(
            replace(
                identity, permutation_id=f"order-{index}", predicted_index=1 if index == 1 else 0
            )
        )
    repeats = [
        replace(
            identity,
            permutation_id="identity",
            repeat_index=index,
            predicted_index=1 if index == 2 else 0,
        )
        for index in range(3)
    ]
    assert flip_summary(order_rows)["flip_rate"] == pytest.approx(1 / 3)
    assert repeat_summary(repeats)["flip_rate"] == 0.5
    result = excess_order_bootstrap(order_rows, repeats, resamples=10)
    assert result["excess_order_effect"] == pytest.approx(1 / 3 - 0.5)
    assert result["ci95_low"] == result["ci95_high"]
    averaged = mean_of_repeats(repeats)[0]
    assert averaged.condition == "A_mean_of_repeats"
    assert averaged.probabilities == pytest.approx(repeats[0].probabilities)
    failed = replace(repeats[1], error="timeout", probabilities=())
    assert mean_of_repeats([repeats[0], failed])[0].error == "repeat failure"


def test_flip_and_repeat_group_boundaries() -> None:
    base = replace(prediction(0, 0, (0.8, 0.2)), permutation_id="order-0")
    variant = replace(base, permutation_id="order-1", predicted_index=1)
    assert flip_summary([base, variant])["n"] == 1
    assert flip_summary([base, replace(variant, repeat_index=1)])["n"] == 0
    assert flip_summary([base, replace(variant, split="pilot")])["n"] == 0
    assert flip_summary([base, replace(variant, letter_mode="positional")])["n"] == 0
    assert repeat_summary([base, replace(base, repeat_index=1, predicted_index=1)])["n"] == 1
    assert repeat_summary([base, replace(base, split="pilot", repeat_index=1)])["n"] == 0
    assert repeat_summary([base, replace(variant, repeat_index=1)])["n"] == 0
    assert repeat_summary([base, replace(base, letter_mode="positional", repeat_index=1)])["n"] == 0


def test_joint_bootstrap_refits_temperature_and_holm() -> None:
    left_test = [prediction(i, i % 2, (0.7, 0.3)) for i in range(4)]
    right_test = [prediction(i, i % 2, (0.9, 0.1) if i % 2 == 0 else (0.1, 0.9)) for i in range(4)]
    left_cal = [replace(row, example_id=f"cal:{i}") for i, row in enumerate(left_test)]
    right_cal = [replace(row, example_id=f"cal:{i}") for i, row in enumerate(right_test)]
    result = joint_paired_bootstrap(
        {"d": left_test},
        {"d": right_test},
        {"d": left_cal},
        {"d": right_cal},
        metric="brier",
        condition="B_scaled",
        resamples=20,
        seed=3,
    )
    assert result["p_two_sided"] >= 1 / 2000
    assert "ci95_low" in result
    raw = joint_paired_bootstrap(
        {"d": left_test},
        {"d": right_test},
        {"d": left_cal},
        {"d": right_cal},
        metric="accuracy",
        condition="A_raw",
        resamples=10,
        seed=3,
    )
    assert raw["difference"] == 0.5
    with pytest.raises(ValueError, match="dataset sets differ"):
        joint_paired_bootstrap(
            {"d": left_test}, {}, {}, {}, metric="accuracy", condition="A_raw", resamples=2
        )
    with pytest.raises(ValueError, match="contract mismatch"):
        joint_paired_bootstrap(
            {"d": left_test},
            {"d": [replace(right_test[0], target_index=1), *right_test[1:]]},
            {"d": left_cal},
            {"d": right_cal},
            metric="accuracy",
            condition="A_raw",
            resamples=2,
        )
    assert holm_decisions({"a": 0.001, "b": 0.02, "c": 0.06}, 3) == {
        "a": True,
        "b": True,
        "c": False,
    }
    with pytest.raises(ValueError, match="frozen k"):
        holm_decisions({"a": 0.01}, 3)
