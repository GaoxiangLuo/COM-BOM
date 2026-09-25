"""``COMBOM``: a DSPy optimizer that selects few-shot demos by multi-objective BO."""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import torch
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bo_loop import COMBOMResult, default_reference_point, hypervolume_contributions, run_combom
from .objectives import DEFAULT_OBJECTIVES, Objective
from .pool import bootstrap_pool

logger = logging.getLogger(__name__)


class COMBOM(Teleprompter):
    """Combinatorial multi-objective BO over which ``trainset`` examples become demos.

    Search space: one bit per ``trainset`` example (the exemplar pool). Every predictor in the
    program receives the same selected subset (optionally prefixed by ``fixed_demos``, e.g. the
    paper's format exemplar). Each trial materialises a program with those demos, evaluates it
    on ``valset`` with ``dspy.Evaluate`` and reduces the per-example results with the configured
    objectives (default: accuracy up, calibration error down).

    With ``bootstrap=True``, ``trainset`` is a set of labeled questions instead: the pool is built
    from them with DSPy's ``BootstrapFewShot`` (see ``dspy_combom.pool.bootstrap_pool``), keeping
    the demos the program writes itself and gets right.

    ``compile`` returns the Pareto point with the largest hypervolume contribution. The whole
    front is attached as ``program.pareto_programs`` (each with ``objectives``, ``x`` and a
    materialised ``program``), the full trajectory as ``program.combom_history``.
    """

    def __init__(
        self,
        metric: Callable,
        *,
        objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
        n_initial_pts: int = 20,
        num_trials: int = 200,
        fixed_demos: Sequence[Any] | None = None,
        bootstrap: bool = False,
        teacher_demos: int = 16,
        num_threads: int | None = None,
        max_errors: int | None = None,
        failure_score: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
        init_radius: int | None = None,
        fixed_noise_value: float = 1e-3,
        gp_noise_type: str = "fixed",
        gp_lengthscale_prior_scale: float = 1.73205,
        gp_config: dict | list[dict] | None = None,
        num_restarts: int = 20,
        raw_samples: int = 4096,
        max_batch_size: int = 128,
        ref_point: Sequence[float] | None = None,
        acqf_type: str = "qnehvi",
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ):
        """Configure the search: objectives, budget, GP settings and evaluation."""
        super().__init__()
        self.metric = metric
        self.objectives = tuple(objectives)
        if len(self.objectives) != 2:
            raise NotImplementedError(
                "COMBOM currently supports exactly two objectives: TrustRegionManager."
                "initialize_center and the hypervolume bookkeeping are written for m=2."
            )
        self.n_initial_pts = n_initial_pts
        self.num_trials = num_trials
        self.fixed_demos = list(fixed_demos or [])
        self.bootstrap = bootstrap
        self.teacher_demos = teacher_demos
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.failure_score = failure_score
        self.seed = seed
        self.device = device
        self.init_radius = init_radius
        self.fixed_noise_value = fixed_noise_value
        self.gp_noise_type = gp_noise_type
        self.gp_lengthscale_prior_scale = gp_lengthscale_prior_scale
        self.gp_config = gp_config
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.max_batch_size = max_batch_size
        self.ref_point = list(ref_point) if ref_point is not None else None
        self.acqf_type = acqf_type
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.trial_logs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ compile
    def compile(self, student, *, trainset, teacher=None, valset=None):
        """Search exemplar subsets of ``trainset`` and return the best program.

        ``trainset`` is the candidate exemplar pool (or, with ``bootstrap=True``, the labeled
        questions the pool is bootstrapped from) and ``valset`` the set the objectives are
        measured on; like DSPy's own optimizers, it defaults to ``trainset``, but a separate
        validation set avoids scoring programs on their own demos. ``teacher`` is used only for
        bootstrapping. The returned program also carries
        ``pareto_programs``, ``candidate_programs``, ``combom_history`` and ``combom_pool``.
        """
        if teacher is not None and not self.bootstrap:
            logger.warning("COMBOM uses a teacher only with bootstrap=True; ignoring `teacher`.")
        valset = list(valset) if valset is not None else list(trainset)
        if not valset:
            raise ValueError("valset must be non-empty; objectives are estimated on it.")
        if len(student.predictors()) == 0:
            raise ValueError("student has no predictors to attach demos to.")
        if self.bootstrap:
            pool = bootstrap_pool(
                student,
                trainset,
                self.metric,
                teacher=teacher,
                teacher_demos=self.teacher_demos,
                max_errors=self.max_errors,
            )
        else:
            pool = list(trainset)
        if len(pool) < 2:
            raise ValueError("the exemplar pool needs at least two examples.")

        self.trial_logs = []
        result = run_combom(
            self.evaluator(student, pool, valset),
            decision_dim=len(pool),
            weights=[obj.weight for obj in self.objectives],
            n_initial_pts=self.n_initial_pts,
            num_trials=self.num_trials,
            seed=self.seed,
            device=self.device,
            init_radius=self.init_radius,
            fixed_noise_value=self.fixed_noise_value,
            gp_noise_type=self.gp_noise_type,
            gp_lengthscale_prior_scale=self.gp_lengthscale_prior_scale,
            gp_config=self.gp_config,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            max_batch_size=self.max_batch_size,
            acqf_type=self.acqf_type,
            checkpoint_path=self.checkpoint_path,
            log=logger.info if self.verbose else (lambda *_: None),
        )
        return self._assemble(student, pool, result)

    def evaluator(self, student, pool, valset) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the function ``compile`` optimises: ``[b, len(pool)]`` 0/1 subsets to objectives.

        Each row is materialised as a program (``fixed_demos`` + the selected pool examples),
        evaluated on ``valset`` and reduced with the configured objectives. Trials are appended to
        ``trial_logs``. Used by ``dspy_combom.design`` to score random designs exactly as the
        optimizer would.
        """
        evaluate = Evaluate(
            devset=list(valset),
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=self.max_errors,
            failure_score=self.failure_score,
            display_progress=False,
            display_table=False,
        )
        names = [obj.name for obj in self.objectives]

        def evaluate_x(X: torch.Tensor) -> torch.Tensor:
            rows = []
            for x in X:
                program = self._materialize(student, pool, x)
                result = evaluate(program)
                values = [obj(result.results) for obj in self.objectives]
                rows.append(values)
                bits = self._bits(x)
                self.trial_logs.append(
                    {
                        "x": bits,
                        "n_demos": sum(bits),
                        "objectives": dict(zip(names, values, strict=False)),
                    }
                )
                if self.verbose:
                    logger.info(
                        "trial %d: demos=%s %s",
                        len(self.trial_logs),
                        [i for i, b in enumerate(bits) if b],
                        {k: round(v, 4) for k, v in zip(names, values, strict=False)},
                    )
            return torch.tensor(rows, dtype=torch.float64)

        return evaluate_x

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _bits(x: torch.Tensor) -> list[int]:
        return [int(b >= 0.5) for b in x.detach().cpu().flatten().tolist()]

    def _materialize(self, student, pool, x: torch.Tensor):
        program = student.reset_copy()
        demos = list(self.fixed_demos) + [pool[i] for i, b in enumerate(self._bits(x)) if b]
        for _, predictor in program.named_predictors():
            predictor.demos = list(demos)
        return program

    def _assemble(self, student, pool, result: COMBOMResult):
        names = [obj.name for obj in self.objectives]
        weighted = result.weighted_y
        mask = result.pareto_mask
        pareto_idx = torch.where(mask)[0]
        if self.ref_point is not None:
            ref = torch.as_tensor(self.ref_point, dtype=weighted.dtype, device=weighted.device)
            ref = ref * result.weights
        else:
            ref = default_reference_point(weighted)
        contributions = hypervolume_contributions(weighted[mask], ref)
        best_local = int(torch.argmax(contributions))

        pareto_programs = []
        for local_i, i in enumerate(pareto_idx.tolist()):
            program = self._materialize(student, pool, result.train_x[i])
            program._compiled = True  # keeps DSPy from walking these when saving the winner
            pareto_programs.append(
                {
                    "iteration": i,
                    "objectives": dict(zip(names, result.train_y[i].tolist(), strict=False)),
                    "x": self._bits(result.train_x[i]),
                    "hv_contribution": float(contributions[local_i]),
                    "program": program,
                }
            )

        best_program = pareto_programs[best_local]["program"]
        best_program.pareto_programs = pareto_programs
        best_program.combom_pool = list(pool)
        best_program.candidate_programs = [
            {"iteration": i, "objectives": dict(zip(names, y, strict=False)), "x": self._bits(x)}
            for i, (x, y) in enumerate(zip(result.train_x, result.train_y.tolist(), strict=False))
        ]
        best_program.combom_history = {
            "objectives": names,
            "x": [self._bits(x) for x in result.train_x],
            "y": result.train_y.cpu().tolist(),
            "pareto_indices": pareto_idx.tolist(),
            "best_index": int(pareto_idx[best_local]),
            "reference_point_weighted": ref.cpu().tolist(),
            "hypervolume": result.hypervolume(ref),
        }
        return best_program
