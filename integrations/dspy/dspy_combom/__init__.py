"""DSPy adapter for COM-BOM (multi-objective Bayesian exemplar selection).

Layout:
- ``objectives``: whole-set objectives (accuracy, ECE) computed from ``dspy.Evaluate`` results.
- ``modules``: ``SelfConsistency`` wrapper that exposes a ``confidence`` field.
- ``bo_loop``: task-agnostic COM-BOM loop on top of the release ``combom`` package.
- ``acquisition``, ``gp``: Sobol-pinned qNEHVI and the config-driven GP used by ``bo_loop``.
- ``pool``: ``bootstrap_pool``, an exemplar pool built with DSPy's ``BootstrapFewShot``.
- ``teleprompter``: ``COMBOM``, a ``dspy.teleprompt.Teleprompter`` that searches demo subsets.
"""

from .adapters import PlainTextAdapter
from .bo_loop import COMBOMResult, run_combom
from .lm import CompletionLM
from .modules import SelfConsistency
from .objectives import (
    DEFAULT_OBJECTIVES,
    Objective,
    accuracy_objective,
    calibration_error_objective,
    compute_calibration_error,
)
from .pool import bootstrap_pool
from .teleprompter import COMBOM

__all__ = [
    "COMBOM",
    "COMBOMResult",
    "CompletionLM",
    "PlainTextAdapter",
    "DEFAULT_OBJECTIVES",
    "Objective",
    "SelfConsistency",
    "accuracy_objective",
    "bootstrap_pool",
    "calibration_error_objective",
    "compute_calibration_error",
    "run_combom",
]
