"""Random design for tuning COM-BOM's GP on GSM8K (see the combom-gp-tuning skill).

Scores random subsets of the exemplar pool with COM-BOM's own evaluator, then re-scores a few of them to
measure evaluation noise (dspy_combom.design.collect_design). Resumes from --out if it exists.

    python collect_design.py --pool dspy --seed 0 --out design_seed0.pt
"""

from __future__ import annotations

import argparse

import _paths  # noqa: F401  (sets sys.path)
import pilot as P

from dspy_combom import COMBOM
from dspy_combom.design import collect_design

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--pool", choices=list(P.POOLS), default="dspy")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-iid", type=int, default=60)
ap.add_argument("--n-stratified", type=int, default=20)
ap.add_argument("--n-repeat", type=int, default=5)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--n-val", type=int, default=1000)
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--threads", type=int, default=64)
a = ap.parse_args()

format_demo, questions, valset, _ = P.load_data(a.n_val, 0)
P.configure(a.port, a.model)
opt = COMBOM(
    metric=P.metric, fixed_demos=[format_demo], num_threads=a.threads, max_errors=P.MAX_ERRORS
)
collect_design(
    opt,
    P.make_student(),
    trainset=P.load_pool(a.pool, questions),
    valset=valset,
    n_iid=a.n_iid,
    n_stratified=a.n_stratified,
    n_repeat=a.n_repeat,
    reps=a.reps,
    seed=a.seed,
    out=a.out,
)
