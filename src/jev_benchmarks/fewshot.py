"""Amendment 5 few-label arm: classifiers trained on the 200 calibration labels per dataset.

Out-of-fold calibration predictions stand in for the contender's ``calibration`` split, so the
report's temperature, threshold and bootstrap code applies unchanged. Test labels never reach a fit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfig
from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .models import Example, Prediction
from .runner import _validate_prediction

FEWSHOT_BACKENDS = frozenset({"qwen_probe", "tfidf_lr", "prior"})
FOLDS = 5


class PriorModel:
    """Predicts the training-label class frequencies for every row."""

    def fit(self, x: Any, y: np.ndarray) -> PriorModel:
        self.classes_, counts = np.unique(y, return_counts=True)
        self._frequencies = counts / counts.sum()
        return self

    def predict_proba(self, x: Sequence[Any]) -> np.ndarray:
        return np.tile(self._frequencies, (len(x), 1))


def make_model(backend: str, seed: int) -> Any:
    if backend == "prior":
        return PriorModel()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    if backend == "tfidf_lr":
        # Fitted inside each training fold, so held-out texts never shape their own features.
        return make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True), classifier
        )
    if backend == "qwen_probe":
        return make_pipeline(StandardScaler(), classifier)
    raise ValueError(f"unknown few-label backend: {backend}")


def _take(x: Any, index: np.ndarray) -> Any:
    return x[index] if isinstance(x, np.ndarray) else [x[i] for i in index]


def _fit_predict(model: Any, x: Any, y: np.ndarray, x_new: Any, classes: int) -> np.ndarray:
    """Full-width probabilities; a class absent from ``y`` gets probability 0."""
    output = np.zeros((len(x_new), classes))
    seen = np.unique(y)
    if len(seen) == 1:
        output[:, seen[0]] = 1.0
        return output
    model.fit(x, y)
    output[:, model.classes_] = model.predict_proba(x_new)
    return output


def cross_fit(
    x: Any,
    y: np.ndarray,
    x_test: Any,
    classes: int,
    new_model: Callable[[], Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Out-of-fold calibration probabilities, test probabilities from an all-data refit, and the
    number of held-out rows whose true class was absent from their training fold."""
    from sklearn.model_selection import KFold

    # ponytail: unstratified folds; Banking77/MASSIVE calibration classes have only 2-4 items.
    folds = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    out_of_fold = np.zeros((len(y), classes))
    absent = 0
    for train, held_out in folds.split(np.arange(len(y))):
        out_of_fold[held_out] = _fit_predict(
            new_model(), _take(x, train), y[train], _take(x, held_out), classes
        )
        absent += int(np.sum(~np.isin(y[held_out], y[train])))
    return out_of_fold, _fit_predict(new_model(), x, y, x_test, classes), absent


def _features(
    config: BenchmarkConfig, attempt: Path, rows: list[Example], extractor: Any
) -> dict[str, np.ndarray]:
    """Load or extract Qwen features for ``rows``, cached per split as ignored ``.npz`` files."""
    layer = int(config.raw["models"]["qwen_probe"]["layer"])
    by_id: dict[str, np.ndarray] = {}
    for split in ("calibration", "test"):
        path = attempt / f"features-{split}.npz"
        wanted = [row for row in rows if row.split == split]
        if not path.exists():
            if extractor is None:
                from .v2_runner import LocalProcessBackend

                extractor = LocalProcessBackend(config, "qwen_logit", attempt)
            matrix = np.asarray([extractor.features(row, layer) for row in wanted], np.float32)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, ids=np.asarray([row.example_id for row in wanted]), x=matrix)
            temporary.replace(path)  # publish atomically
        stored = np.load(path)
        by_id.update(zip(stored["ids"].tolist(), stored["x"], strict=True))
    if extractor is not None and hasattr(extractor, "close"):
        extractor.close()
    return by_id


def run_fewshot(config: BenchmarkConfig, backend: str, *, feature_extractor: Any = None) -> Path:
    from .v2_runner import _dispatch_lock

    if backend not in FEWSHOT_BACKENDS or backend not in config.raw["models"]:
        raise ValueError(f"unknown few-label backend: {backend}")
    # Attempt selection, feature extraction and publication run under the per-backend lock.
    with _dispatch_lock(config.output_dir / backend):
        return _run_fewshot(config, backend, feature_extractor)


def _run_fewshot(config: BenchmarkConfig, backend: str, feature_extractor: Any) -> Path:
    from .v2_runner import _attempt_dir, verify_frozen

    if backend not in FEWSHOT_BACKENDS or backend not in config.raw["models"]:
        raise ValueError(f"unknown few-label backend: {backend}")
    manifest = config.output_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError("few-label run requires an existing frozen manifest")
    verify_frozen(config, manifest)
    model = config.raw["models"][backend]
    rows = [
        Example.from_dict(row)
        for row in read_jsonl(manifest)
        if row.get("split") in {"calibration", "test"}
        and row.get("permutation_id", "identity") == "identity"
        and row.get("dataset") in model["datasets"]
    ]
    attempt = _attempt_dir(config.output_dir / backend)
    attempt.mkdir(parents=True, exist_ok=True)
    output = attempt / "predictions.jsonl"
    if output.exists():
        return output
    features = (
        _features(config, attempt, rows, feature_extractor) if backend == "qwen_probe" else {}
    )
    if backend == "qwen_probe":
        qwen = config.raw["models"]["qwen_logit"]
        resolved = f"{qwen['model_id']}@{qwen['revision']}:L{model['layer']}+logreg"
    else:
        resolved = {"tfidf_lr": "char_wb2-4+logreg", "prior": "calibration-frequency"}[backend]
    grouped: dict[str, dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda row: row.example_id):
        grouped[row.dataset][row.split].append(row)
    predictions: list[dict[str, Any]] = []
    absent: dict[str, int] = {}
    for dataset, splits in sorted(grouped.items()):
        calibration, test = splits["calibration"], splits["test"]
        classes = len(calibration[0].labels)

        def inputs(examples: list[Example]) -> Any:
            if backend == "qwen_probe":
                return np.stack([features[row.example_id] for row in examples])
            return [row.text for row in examples]

        out_of_fold, test_probabilities, absent[dataset] = cross_fit(
            inputs(calibration),
            np.asarray([row.target_index for row in calibration]),
            inputs(test),
            classes,
            lambda: make_model(backend, config.seed),
            config.seed,
        )
        for examples, matrix in ((calibration, out_of_fold), (test, test_probabilities)):
            for row, vector in zip(examples, matrix, strict=True):
                probabilities = tuple(float(value) for value in vector)
                prediction = Prediction(
                    config.experiment_id,
                    backend,
                    backend,
                    resolved,
                    row.dataset,
                    row.example_id,
                    row.target_index,
                    int(np.argmax(vector)),
                    row.labels,
                    probabilities,
                    0.0,  # no common single-call operation; reports drop latency for this arm
                    question_type=row.question_type,
                    split=row.split,
                    raw_probabilities=probabilities,
                    expected_score=sum(
                        (index + 1) * value for index, value in enumerate(probabilities)
                    )
                    if row.question_type == "score"
                    else None,
                    attempt_id=int(attempt.name.split("-")[-1]),
                )
                predictions.append(_validate_prediction(prediction).to_dict())
    write_json(
        attempt / "fewshot-metadata.json",
        {
            "backend": backend,
            "resolved": resolved,
            "folds": FOLDS,
            "seed": config.seed,
            "absent_class_held_out_rows": absent,
            "features_sha256": {
                path.name: sha256_file(path) for path in sorted(attempt.glob("features-*.npz"))
            },
            "manifest_sha256": sha256_file(manifest),
        },
    )
    write_jsonl(output, predictions)
    write_json(attempt / "status.json", {"state": "active", "snapshot": resolved})
    return output
