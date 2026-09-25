"""``CompletionLM``: raw-prompt client for an OpenAI-compatible ``/v1/completions`` endpoint.

``dspy.LM(model_type="text")`` joins messages and appends ``BEGIN RESPONSE:`` to the prompt, which
breaks completion-style templates. This legacy-contract ``BaseLM`` sends the adapter's text as-is
(so the prompt ends with the cue, e.g. ``The answer is: ``) and supports ``n``/``top_k``/``stop``.
"""

from __future__ import annotations

from typing import Any

from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from openai import OpenAI


class CompletionLM(BaseLM):
    """A ``dspy.BaseLM`` that calls the vLLM ``/v1/completions`` endpoint directly.

    ``dspy.LM(model_type="text")`` appends a ``BEGIN RESPONSE:`` preamble, which changes
    the prompt the model sees; this class sends the raw prompt instead, which is what the
    reference COM-BOM implementation does.
    """

    forward_contract = "legacy"

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "local_deployment",
        *,
        temperature: float = 0.7,
        max_tokens: int = 3,
        n: int = 1,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Configure the endpoint, model name and default sampling parameters."""
        super().__init__(
            model=model,
            model_type="text",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
            **kwargs,
        )
        self.api_base = api_base
        self.api_key = api_key
        self.n = n
        self.top_p = top_p
        self.top_k = top_k
        self.stop = list(stop) if stop else None
        self._client = OpenAI(
            base_url=api_base, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs: Any):
        """Send one completion request and return the raw OpenAI-style response."""
        kwargs = {**self.kwargs, **kwargs}
        if prompt is None:
            prompt = "".join(
                m["content"] for m in (messages or []) if isinstance(m.get("content"), str)
            )
        top_k = kwargs.get("top_k", self.top_k)
        request = {
            "model": self.model,
            "prompt": prompt,
            "n": kwargs.get("n", self.n),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": kwargs.get("stop", self.stop),
            "extra_body": {"top_k": top_k} if top_k is not None else None,
        }
        request = {k: v for k, v in request.items() if v is not None}
        response = self._client.completions.create(**request)
        choices = [
            dotdict(
                message=dotdict(
                    content=(c.text if c.text and c.text.strip() else " "), tool_calls=None
                ),
                finish_reason=c.finish_reason,
            )
            for c in response.choices
        ]
        usage = response.usage
        return dotdict(
            choices=choices,
            usage=dotdict(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            ),
            model=response.model,
        )
