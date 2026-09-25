"""Random designs of exemplar subsets, for tuning COM-BOM's GP on a new task.

A design scores random subsets of the exemplar pool exactly as ``COMBOM`` would, and re-scores a
few of them to measure evaluation noise. ``tools/gp_tune.py`` cross-validates GP configurations on
it. Validate GP settings on a random design: BO trajectories cluster around good subsets and make
any GP look more predictive than it is.

Saved format (``torch.save``): ``x`` ``[N, pool]`` 0/1 subsets, ``y`` ``[N, m]`` objective values,
``repeat_x`` ``[R, pool]`` and ``repeat_y`` ``[R, reps, m]`` for the noise floor, plus ``names``,
``weights``, ``n_val`` and ``pool_size``.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import torch


def sample_design(pool_size: int, n_iid: int, n_stratified: int, seed: int = 0) -> torch.Tensor:
    """Draw ``n_iid`` half-probability subsets, then ``n_stratified`` size-stratified ones.

    In the first kind each example is included with probability 1/2; the second kind has a size
    drawn uniformly from ``1..pool_size``. Half-probability subsets all hold about
    ``pool_size / 2`` examples, so on their own they never show the GP how the objectives change
    with the number of demos.
    """
    g = torch.Generator().manual_seed(seed)
    iid = torch.randint(0, 2, (n_iid, pool_size), generator=g).double()
    strat = torch.zeros(n_stratified, pool_size, dtype=torch.float64)
    for i in range(n_stratified):
        k = int(torch.randint(1, pool_size + 1, (1,), generator=g))
        strat[i, torch.randperm(pool_size, generator=g)[:k]] = 1.0
    return torch.cat([iid, strat])


def signal_to_noise(design: dict) -> list[dict]:
    """Per objective: design spread, evaluation-noise sd from the repeats, and their ratio."""
    y, rep = design["y"], design.get("repeat_y")
    out = []
    for j, name in enumerate(design.get("names", [f"y{j}" for j in range(y.shape[1])])):
        spread = float(y[:, j].std())
        noise = (
            float(rep[..., j].std(dim=1).mean()) if rep is not None and len(rep) else float("nan")
        )
        out.append(
            {
                "objective": name,
                "design_sd": spread,
                "noise_sd": noise,
                "ratio": spread / noise if noise > 0 else float("inf"),
            }
        )
    return out


def collect_design(
    optimizer,
    student,
    *,
    trainset,
    valset,
    n_iid: int = 60,
    n_stratified: int = 20,
    n_repeat: int = 5,
    reps: int = 3,
    seed: int = 0,
    out: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Score a random design with ``optimizer``'s own evaluation (a configured ``COMBOM``).

    ``trainset`` is the exemplar pool and ``valset`` the set the objectives are measured on, as in
    ``COMBOM.compile``. The first ``n_repeat`` design points are re-scored until each has ``reps``
    evaluations. With ``out`` set, the design is saved after every evaluation and an interrupted
    run resumes from the file.
    """
    pool, valset = list(trainset), list(valset)
    evaluate = optimizer.evaluator(student, pool, valset)
    X = sample_design(len(pool), n_iid, n_stratified, seed)
    meta = {
        "names": [o.name for o in optimizer.objectives],
        "weights": [o.weight for o in optimizer.objectives],
        "n_val": len(valset),
        "pool_size": len(pool),
    }
    ys, rep_y = [], []
    if out and os.path.exists(out):
        prev = torch.load(out, weights_only=False)
        if torch.equal(prev["x"], X[: len(prev["y"])]):
            ys = prev["y"].tolist()
            rep_y = prev.get("repeat_y", torch.empty(0)).tolist()
            log(f"resuming {out}: {len(ys)} design points, {len(rep_y)} repeat sets done")

    def save(done_x):
        design = {"x": done_x, "y": torch.tensor(ys, dtype=torch.float64), **meta}
        if rep_y:
            design["repeat_x"] = X[: len(rep_y)]
            design["repeat_y"] = torch.tensor(rep_y, dtype=torch.float64)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            torch.save(design, out)
        return design

    t0 = time.time()
    for i in range(len(ys), len(X)):
        ys.append(evaluate(X[i : i + 1])[0].tolist())
        values = " ".join(f"{n}={v:.4f}" for n, v in zip(meta["names"], ys[-1], strict=False))
        minutes = (time.time() - t0) / 60
        log(f"[design {i + 1}/{len(X)}] n_demos={int(X[i].sum())} {values} ({minutes:.1f} min)")
        save(X[: i + 1])
    for i in range(len(rep_y), min(n_repeat, len(X))):
        runs = [ys[i]] + [evaluate(X[i : i + 1])[0].tolist() for _ in range(reps - 1)]
        rep_y.append(runs)
        log(f"[repeat {i + 1}/{n_repeat}] {[[round(v, 4) for v in r] for r in runs]}")
        save(X)
    design = save(X)
    if rep_y:
        for s in signal_to_noise(design):
            log(
                f"{s['objective']}: design sd {s['design_sd']:.4f}, noise sd {s['noise_sd']:.4f}, "
                f"signal/noise {s['ratio']:.2f}"
            )
    return design
