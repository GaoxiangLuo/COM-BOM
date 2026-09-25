"""Completion-style few-shot adapter (classic "Question:/Answer:" template, no chat structure).

Why this exists: with DSPy's ChatAdapter (system field spec + demos as chat turns), some models
answer near-deterministically, so self-consistency confidence collapses to 1.0 and the calibration
objective degenerates. A plain completion template, as in the COM-BOM paper, keeps the answer
variation the method relies on.
"""

from __future__ import annotations

from typing import Any

from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature


def _get(values: Any, name: str):
    try:
        return values[name]
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


class PlainTextAdapter(Adapter):
    """Render demos and the query as one raw text prompt.

    Each demo is ``input_template`` per input field followed by ``output_template`` per output
    field; the query repeats the inputs and ends with ``query_suffix`` for the first output field
    (the completion cue). ``labels`` maps field names to their labels, e.g.
    ``{"question": "Question:", "answer": "The answer is:"}``.
    """

    def __init__(
        self,
        labels: dict[str, str] | None = None,
        *,
        header_sep: str = "\n",
        input_template: str = "{label}\n{value}\n",
        output_template: str = "{label} {value}\n",
        query_suffix: str = "{label} ",
        callbacks=None,
    ):
        """Build an adapter with the given field labels and templates."""
        super().__init__(callbacks=callbacks)
        self.labels = dict(labels or {})
        self.header_sep = header_sep
        self.input_template = input_template
        self.output_template = output_template
        self.query_suffix = query_suffix

    def _label(self, name: str) -> str:
        return self.labels.get(name) or f"{name.replace('_', ' ').capitalize()}:"

    def _render(self, signature: type[Signature], values: Any, *, with_outputs: bool) -> str:
        text = ""
        for name in signature.input_fields:
            value = _get(values, name)
            if value is not None:
                text += self.input_template.format(label=self._label(name), value=value)
        for name in signature.output_fields:
            if with_outputs:
                value = _get(values, name)
                if value is not None:
                    text += self.output_template.format(label=self._label(name), value=value)
            else:
                text += self.query_suffix.format(label=self._label(name))
                break
        return text

    def format(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Render demos and inputs as a single plain-text user message."""
        parts = []
        if signature.instructions:
            parts.append(signature.instructions + self.header_sep)
        for demo in demos:
            parts.append(self._render(signature, demo, with_outputs=True))
        parts.append(self._render(signature, inputs, with_outputs=False))
        return [{"role": "user", "content": "".join(parts)}]

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Split a completion into output fields on the field labels."""
        names = list(signature.output_fields)
        text = completion.strip()
        if len(names) == 1:
            return {names[0]: text}
        # Multiple output fields: split on the labels of the remaining fields, in order.
        values: dict[str, Any] = {}
        rest = text
        for i, name in enumerate(names):
            nxt = self._label(names[i + 1]) if i + 1 < len(names) else None
            if nxt and nxt in rest:
                head, rest = rest.split(nxt, 1)
            else:
                head, rest = rest, ""
            values[name] = head.strip()
        return values
