"""Summarize the GSM8K runs written by pilot.py: the README's results table.

Test columns are means over seeds; demos counts every example in the prompt. For COM-BOM, the reported
program is the Pareto-front point with the best validation accuracy - ECE. Validation hypervolume covers every program a run evaluated (squared percentage
points; reference accuracy 50 / ECE 25).

    python summarize.py [--runs DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

import _paths

REF = (0.50, 0.25)  # hypervolume reference point: accuracy, ECE


def hypervolume(points, ref=REF):
    """2-D hypervolume of (accuracy up, ECE down) points relative to `ref`, in squared percentage points."""
    area, best_ece = 0.0, ref[1]
    for acc, ece in sorted(points, key=lambda p: -p[0]):
        if acc > ref[0] and ece < best_ece:
            area += (acc - ref[0]) * (best_ece - ece)
            best_ece = ece
    return 1e4 * area


def validation_points(run, result):
    if "history" in result:  # COM-BOM
        return [tuple(y) for y in result["history"]["y"]]
    path = os.path.join(run, "evaluations.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path)]
    return [(r["accuracy"], r["calibration_error"]) for r in rows]


ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=_paths.RUNS_DIR)
a = ap.parse_args()
groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.runs, "*", "result.json"))):
    r = json.load(open(f))
    method = r["args"]["method"] + (f" ({r['args']['pool']} pool)" if "pareto_test" in r else "")
    if "pareto_test" in r:
        pick = max(
            r["pareto_test"], key=lambda p: p["val"]["accuracy"] - p["val"]["calibration_error"]
        )
        test, demos = pick["test"], pick["n_demos"] + 1  # + the fixed format demo
    else:
        test, demos = r["test"], r["n_demos"]
    pts = validation_points(os.path.dirname(f), r)
    groups[method].append(
        dict(test=test, demos=demos, hv=hypervolume(pts) if pts else None, seed=r["args"]["seed"])
    )

print(
    f"{'method':<24} {'seeds':>5} {'test acc':>9} {'test ECE':>9} {'acc-ECE':>8} {'val HV':>7}  demos"
)
for method, rows in groups.items():
    acc = st.mean(100 * x["test"]["accuracy"] for x in rows)
    ece = st.mean(100 * x["test"]["calibration_error"] for x in rows)
    hvs = [x["hv"] for x in rows if x["hv"] is not None]
    hv = f"{st.mean(hvs):7.0f}" if hvs else f"{'-':>7}"
    demos = sorted({x["demos"] for x in rows})
    print(f"{method:<24} {len(rows):>5} {acc:9.1f} {ece:9.1f} {acc - ece:8.1f} {hv}  {demos}")
