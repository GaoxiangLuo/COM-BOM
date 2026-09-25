"""Objective specifications for multi-objective exemplar search inside DSPy.

An :class:`Objective` reduces the per-example results produced by ``dspy.Evaluate``
(``[(example, prediction, score), ...]``) to one scalar. COM-BOM needs whole-set
aggregates (ECE is not decomposable per example), so objectives are defined on the
full result list rather than as per-example metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np

EvalResults = Sequence[tuple[Any, Any, Any]]


def score_to_float(score: Any) -> float:
    """Coerce a DSPy metric output (bool / number / Prediction with ``score``) to float."""
    if score is None:
        return 0.0
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def prediction_field(prediction: Any, name: str, default: float = 0.0) -> float:
    """Read a numeric field from a prediction, tolerating failed/empty predictions."""
    try:
        value = prediction[name]
    except (KeyError, TypeError, AttributeError, IndexError):
        try:
            value = getattr(prediction, name)
        except AttributeError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_calibration_error(confidence, correct, p: str = "2", beta: int = 10) -> float:
    """Equal-mass-binned calibration error (same definition as ``tasks/mmlupro.py``)."""
    confidence = np.array(confidence, dtype=float)
    correct = np.array(correct, dtype=float)
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if len(confidence) % beta != 0:
        bins.append([len(confidence) - len(confidence) % beta, len(confidence)])
    elif bins:
        bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0.0
    total_examples = len(confidence)
    for start, end in bins:
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples_in_bin = len(bin_confidence)
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in {"infty", "infinity", "max"}:
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")
    if p == "2":
        cerr = np.sqrt(cerr)
    return float(cerr)


@dataclass(frozen=True)
class Objective:
    """One scalar objective computed from a full ``dspy.Evaluate`` result list."""

    name: str
    goal: Literal["maximize", "minimize"]
    aggregate: Callable[[EvalResults], float]

    def __post_init__(self) -> None:
        """Validate the optimisation direction."""
        if self.goal not in {"maximize", "minimize"}:
            raise ValueError("goal must be 'maximize' or 'minimize'")

    @property
    def weight(self) -> float:
        """``+1`` for an objective to maximise, ``-1`` for one to minimise."""
        return 1.0 if self.goal == "maximize" else -1.0

    def __call__(self, results: EvalResults) -> float:
        """Aggregate per-example evaluation results into the scalar objective."""
        return float(self.aggregate(results))


def mean_score(results: EvalResults) -> float:
    """Average of the per-example metric values (accuracy for a 0/1 metric)."""
    if not results:
        return 0.0
    return float(np.mean([score_to_float(score) for _, _, score in results]))


def calibration_error(
    results: EvalResults,
    *,
    confidence_field: str = "confidence",
    p: str = "2",
    beta: int = 10,
) -> float:
    """ECE from ``prediction.<confidence_field>`` and the per-example metric as correctness."""
    if not results:
        return 0.0
    conf = [prediction_field(pred, confidence_field) for _, pred, _ in results]
    corr = [score_to_float(score) for _, _, score in results]
    return compute_calibration_error(conf, corr, p=p, beta=beta)


def accuracy_objective(name: str = "accuracy") -> Objective:
    """Mean per-example metric score, to be maximised."""
    return Objective(name, "maximize", mean_score)


def calibration_error_objective(
    name: str = "calibration_error",
    *,
    confidence_field: str = "confidence",
    p: str = "2",
    beta: int = 10,
) -> Objective:
    """Expected calibration error over the whole set, to be minimised.

    Reads ``confidence_field`` from each prediction; see :func:`calibration_error` for the
    binning scheme controlled by ``p`` and ``beta``.
    """

    def _aggregate(results: EvalResults) -> float:
        return calibration_error(results, confidence_field=confidence_field, p=p, beta=beta)

    return Objective(name, "minimize", _aggregate)


DEFAULT_OBJECTIVES = (accuracy_objective(), calibration_error_objective())
