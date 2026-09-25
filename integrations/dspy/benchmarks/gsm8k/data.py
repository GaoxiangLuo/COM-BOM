"""GSM8K loader identical to dspy.datasets.gsm8k.GSM8K, but pointing at the hub id 'openai/gsm8k'
(DSPy's legacy alias 'gsm8k' is rejected by current huggingface_hub). Splits: seed-0 shuffle of the
official train -> train[:200], dev = train[200:1200] (DSPy uses the first 300 of these; scripts slice [:n_val],
so n_val<=300 reproduces DSPy's dev exactly and n_val up to 1000 extends it); test = all 1319 official test items.
"""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset
from dspy.datasets.gsm8k import parse_integer_answer  # noqa: F401  (re-exported)


def _rows(split):
    out = []
    for ex in split:
        q = ex["question"]
        ans = ex["answer"].strip().split()
        assert ans[-2] == "####"
        out.append(
            {
                "question": q,
                "gold_reasoning": " ".join(ans[:-2]),
                "answer": str(int(ans[-1].replace(",", ""))),
            }
        )
    return out


class GSM8K:
    def __init__(self):
        ds = load_dataset("openai/gsm8k", "main")
        train, test = _rows(ds["train"]), _rows(ds["test"])
        random.Random(0).shuffle(train)
        random.Random(0).shuffle(test)
        mk = lambda rows: [dspy.Example(**r).with_inputs("question") for r in rows]
        self.train, self.dev, self.test = mk(train[:200]), mk(train[200:1200]), mk(test)
