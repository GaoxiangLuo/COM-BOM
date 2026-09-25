"""Program wrappers that expose a confidence signal to the calibration objective."""

from __future__ import annotations

import re
from typing import Callable

import dspy


def _default_normalize(value):
    """Lowercase, collapse whitespace, strip surrounding punctuation ("(A)." -> "a").

    Deliberately not ``dspy.evaluate.normalize_text``: that also drops English articles,
    which turns the answer "(A)" into an empty string.
    """
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text or None


class SelfConsistency(dspy.Module):
    """Sample ``n`` completions from one predictor, majority-vote a field, attach ``confidence``.

    The vote is ``dspy.majority`` over the normalized field; ``confidence`` is the fraction of
    samples in the majority cluster (COM-BOM Sec. 3.1.1(c)). The wrapped ``dspy.Predict`` is the
    only parameter, so any optimizer that assigns ``predictor.demos`` can tune it.
    """

    def __init__(
        self,
        signature,
        *,
        n: int = 16,
        temperature: float = 0.7,
        field: str | None = None,
        normalize: Callable | None = None,
        **config,
    ):
        """Wrap ``signature`` in a predictor that samples ``n`` completions."""
        super().__init__()
        self.predict = dspy.Predict(signature, n=n, temperature=temperature, **config)
        self.field = field
        self.normalize = normalize or _default_normalize

    def forward(self, **kwargs):
        """Sample ``n`` completions, majority-vote the target field and attach ``confidence``."""
        demos = kwargs.pop(
            "demos", None
        )  # optional per-call demo override (used by retrieval-style baselines)
        prediction = (
            self.predict(**kwargs) if demos is None else self.predict(demos=demos, **kwargs)
        )
        completions = prediction.completions
        field = self.field or list(self.predict.signature.output_fields)[-1]
        # dspy.majority: most common normalized value, ties to the earliest completion; when no
        # sample parses it returns the first completion, which gets confidence 0
        fields = dict(dspy.majority(completions, normalize=self.normalize, field=field).items())
        value = self.normalize(fields[field])
        if value is None:
            fields["confidence"] = 0.0
            return dspy.Prediction(**fields)
        self._trace_majority(fields)
        votes = sum(self.normalize(v) == value for v in completions[field])
        fields["confidence"] = votes / len(completions)
        return dspy.Prediction(**fields)

    def _trace_majority(self, fields):
        """Record the majority completion, not the first sample, as this predictor's trace output.

        ``dspy.Predict`` traces the whole ``n``-sample prediction, and DSPy's bootstrapping turns
        a trace into a demo through the prediction's first completion. That sample can disagree
        with the majority answer the metric accepted, producing demos with wrong answers.
        """
        trace = dspy.settings.trace
        if trace and trace[-1][0] is self.predict:
            predictor, inputs, _ = trace[-1]
            trace[-1] = (predictor, inputs, dspy.Prediction(**fields))
