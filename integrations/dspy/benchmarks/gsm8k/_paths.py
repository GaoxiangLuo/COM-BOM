"""Paths for the GSM8K benchmark scripts.

Puts the repository root (for the release ``combom`` package), the integration root (for
``dspy_combom``) and its ``tools`` directory on ``sys.path``. Set ``COMBOM_RUNS_DIR`` to write run directories elsewhere; it
defaults to ``<repo>/runs/dspy-combom-gsm8k``, which is gitignored.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(INTEGRATION_ROOT, "..", ".."))
RUNS_DIR = os.environ.get("COMBOM_RUNS_DIR", os.path.join(REPO_ROOT, "runs", "dspy-combom-gsm8k"))

for _p in (REPO_ROOT, INTEGRATION_ROOT, os.path.join(INTEGRATION_ROOT, "tools"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
