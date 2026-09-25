"""Exemplar pools built with DSPy's own bootstrapping.

``bootstrap_pool`` runs ``dspy.BootstrapFewShot`` over the training set, exactly as DSPy's few-shot
optimizers do, and returns only the demos the program wrote itself and got right. DSPy normally tops
a program up with gold labeled examples; those are dropped here, because COM-BOM searches over the
pool and decides itself how many demos a program gets.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import dspy

logger = logging.getLogger(__name__)


def bootstrap_pool(
    student,
    trainset: Sequence[Any],
    metric: Callable,
    *,
    teacher=None,
    teacher_demos: int = 16,
    max_errors: int | None = None,
) -> list:
    """Bootstrap one demo per solvable training example with ``dspy.BootstrapFewShot``.

    Every example in ``trainset`` is attempted once. The teacher (``student`` by default) is
    prompted with ``teacher_demos`` labeled examples, DSPy's default for ``max_labeled_demos``;
    an example becomes a demo when the teacher's prediction passes ``metric``. Only
    single-predictor programs are supported, since the pool is shared by every predictor.
    """
    if len(student.predictors()) != 1:
        raise NotImplementedError("bootstrap_pool supports programs with exactly one predictor.")
    trainset = list(trainset)
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=len(trainset),
        max_labeled_demos=teacher_demos,
        max_errors=max_errors,
    )
    program = optimizer.compile(student, teacher=teacher, trainset=trainset)
    pool = [demo for demo in program.predictors()[0].demos if demo.get("augmented")]
    logger.info("bootstrapped %d of %d training examples into the pool", len(pool), len(trainset))
    return pool
