"""Build the GSM8K exemplar pool with DSPy's BootstrapFewShot, as COMBOM(bootstrap=True) does.

The program, prompted with 16 labeled examples, answers each of the 32 training questions once; the answers
that pass the metric become the pool. Writes a JSON list of {question, reasoning, answer, train_index},
the format of configs/gsm8k/pool_dspy_bootstrap_llama31_8b.json.

    python bootstrap_pool.py --out pool.json
"""

from __future__ import annotations

import argparse
import json
import time

import _paths  # noqa: F401  (sets sys.path)
import pilot as P

from dspy_combom import bootstrap_pool

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()

_, questions, _, _ = P.load_data(0, 0)
index = {
    e.question: i for i, e in enumerate(questions, start=1)
}  # position in the GSM8K train split
P.configure(a.port, a.model)
t0 = time.time()
demos = bootstrap_pool(P.make_student(), questions, P.metric, teacher_demos=P.MAX_LABELED)
pool = [
    dict(
        question=d.question,
        reasoning=str(d.reasoning).strip(),
        answer=str(d.answer),
        train_index=index[d.question],
    )
    for d in demos
]
json.dump(pool, open(a.out, "w"), indent=1)
print(
    f"wrote {a.out}: {len(pool)} demos from {len(questions)} questions in {time.time() - t0:.0f}s",
    flush=True,
)
