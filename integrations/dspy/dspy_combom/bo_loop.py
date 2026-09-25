"""Task-agnostic COM-BOM optimisation loop.

The inner loop of the repo's ``train.py`` with the MMLU-Pro task and YAML plumbing removed, so the
same loop can drive a DSPy program. It uses the release ``combom`` package unchanged for the GP,
the trust region and the discrete local search, and adds two pieces from this integration:
``acquisition`` (qNEHVI with the paper's Sobol sampler on current BoTorch) and ``gp`` (the
config-driven GP used for per-benchmark tuning).
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from botorch.models.model import ModelList
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated

from combom import TrustRegionManager, get_fitted_gp_model, optimize_acqf_discrete_local_search

from .acquisition import get_acquisition_function
from .gp import get_fitted_gp_model_from_config

EvaluateFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class COMBOMResult:
    """Full optimisation trajectory. ``train_y`` holds raw (not sign-flipped) values."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    weights: torch.Tensor

    @property
    def weighted_y(self) -> torch.Tensor:
        """Objectives signed so that larger is better on every dimension."""
        return self.train_y * self.weights

    @property
    def pareto_mask(self) -> torch.Tensor:
        """Boolean mask selecting the non-dominated evaluated points."""
        return is_non_dominated(self.weighted_y)

    def hypervolume(self, ref_point) -> float:
        """Dominated hypervolume of the whole trajectory w.r.t. a ref point in weighted units."""
        y = self.weighted_y
        ref = torch.as_tensor(ref_point, dtype=y.dtype, device=y.device)
        return DominatedPartitioning(ref_point=ref, Y=y).compute_hypervolume().item()


def default_reference_point(weighted_y: torch.Tensor) -> torch.Tensor:
    """Reference point rule shared with ``get_acquisition_function``."""
    low = weighted_y.min(dim=0)[0]
    return low - torch.abs(low) * 0.1


def hypervolume_contributions(
    weighted_pareto_y: torch.Tensor, ref_point: torch.Tensor
) -> torch.Tensor:
    """Per-point hypervolume contribution (two objectives), as in ``TrustRegionManager``."""
    n = weighted_pareto_y.shape[0]
    total = DominatedPartitioning(ref_point=ref_point, Y=weighted_pareto_y).compute_hypervolume()
    if n == 1:
        return total.reshape(1)
    contributions = []
    for i in range(n):
        keep = torch.ones(n, dtype=torch.bool, device=weighted_pareto_y.device)
        keep[i] = False
        hv_without = DominatedPartitioning(
            ref_point=ref_point, Y=weighted_pareto_y[keep]
        ).compute_hypervolume()
        contributions.append(total - hv_without)
    return torch.stack(contributions)


def _bo_step(
    train_x,
    train_y,
    weights,
    tkwargs,
    tr_manager,
    discrete_choices,
    n_objectives,
    gp_kwargs,
    acqf_kwargs,
    num_restarts,
    max_batch_size,
    raw_samples,
):
    """Fit the GPs, build the acquisition and run the trust-region local search.

    Returns ``(next_x, t_fit, t_acq)``.
    """
    t0 = time.time()
    gp_models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(n_objectives):
            gp_config = gp_kwargs.get("gp_config")
            if gp_config is not None:
                cfg = gp_config[j] if isinstance(gp_config, (list, tuple)) else gp_config
                gp_models.append(
                    get_fitted_gp_model_from_config(
                        train_x=train_x, train_y=train_y[:, j], train_yvar=None, config=cfg
                    )
                )
                continue
            gp_models.append(
                get_fitted_gp_model(
                    train_x=train_x,
                    train_y=train_y[:, j],
                    train_yvar=None,
                    prior_choice="dimscaled",
                    kernel_choice="categorical",
                    **gp_kwargs,
                    **tkwargs,
                )
            )
    model = ModelList(*gp_models)
    t_fit = time.time() - t0
    t0 = time.time()

    acqf, pareto_points = get_acquisition_function(
        model=model, train_x=train_x, train_y=train_y, weights=weights, **acqf_kwargs, **tkwargs
    )
    next_x, _ = optimize_acqf_discrete_local_search(
        acq_function=acqf,
        discrete_choices=discrete_choices,
        q=1,
        num_restarts=num_restarts,
        max_batch_size=max_batch_size,
        raw_samples=raw_samples,
        inequality_constraints=None,
        X_avoid=None,
        batch_initial_conditions=None,
        pareto_points=pareto_points,
        n_local_steps=tr_manager.current_radius,
        init_sample_in_tr=True,
        tr_manager=tr_manager,
    )
    t_acq = time.time() - t0
    del acqf, model, gp_models  # release GPU memory before the (long) LM evaluation
    if tkwargs["device"] != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return next_x, t_fit, t_acq


def run_combom(
    evaluate: EvaluateFn,
    *,
    decision_dim: int,
    weights,
    n_initial_pts: int,
    num_trials: int,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    init_radius: int | None = None,
    fixed_noise_value: float = 1e-3,
    gp_noise_type: str = "fixed",
    gp_lengthscale_prior_scale: float = 1.73205,
    gp_config: dict | list[dict] | None = None,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    max_batch_size: int = 128,
    initial_x: torch.Tensor | None = None,
    acqf_type: str = "qnehvi",
    checkpoint_path: str | None = None,
    log: Callable[[str], None] = print,
) -> COMBOMResult:
    """Run trust-region multi-objective BO over ``{0,1}^decision_dim``.

    Args:
        evaluate: maps a ``[b, d]`` 0/1 tensor to ``[b, m]`` raw objective values.
        decision_dim: size of the binary decision vector, i.e. the exemplar pool size.
        weights: ``+1`` for objectives to maximise, ``-1`` to minimise (length ``m``).
        n_initial_pts: random design points evaluated before BO starts.
        num_trials: total evaluations (initial + BO), matching ``train.py``.
        seed: seed for the random initial design and the acquisition optimiser.
        device: torch device for the surrogate; falls back to CPU if unavailable.
        dtype: torch dtype for the surrogate, double by default.
        init_radius: initial trust-region radius; defaults to ``min(10, decision_dim)``.
        fixed_noise_value: observation-noise variance when ``gp_noise_type='fixed'``.
        num_restarts: restarts for the discrete local search over the acquisition.
        raw_samples: raw candidates drawn before the local search starts.
        max_batch_size: batch size used when evaluating the acquisition.
        initial_x: optional explicit initial design (overrides the seeded random design).
        log: callable used for progress messages.
        gp_noise_type: ``"fixed"`` (paper: noise variance ``fixed_noise_value``), ``"standard"`` or
            ``"high"`` (noise inferred by marginal likelihood; use on noisy landscapes
            such as GSM8K).
        gp_lengthscale_prior_scale: LogNormal scale of the lengthscale prior (paper 1.73205).
            The release GP fixes it, so other values are served by the config-driven GP with
            the same noise model, which builds the identical model otherwise.
        gp_config: optional config dict for the config-driven CategoricalKernel GP (see
            ``dspy_combom.gp.GP_CONFIG_DEFAULTS``; keys: ard, ls_prior, ls_shift,
            ls_scale, ls_bounds, outputscale, mean, noise, restarts, known_noise_var,
            n_val). A list gives one config per objective. When set, it overrides
            ``gp_noise_type`` / ``fixed_noise_value`` / ``gp_lengthscale_prior_scale``.
        acqf_type: ``"qnehvi"`` (paper) or ``"qlognehvi"``.
        checkpoint_path: if given, the trajectory and trust-region state are saved there after
            every evaluation, and a run restarted with the same path resumes from it (the global
            RNG state is not restored, so a resumed run is statistically but not bitwise identical).
    """
    if gp_config is None and abs(gp_lengthscale_prior_scale - 1.73205) > 1e-9:
        noise = {"fixed": f"fixed:{fixed_noise_value}", "standard": "gamma", "high": "ln-2"}
        gp_config = {"noise": noise[gp_noise_type], "ls_scale": gp_lengthscale_prior_scale}
    if gp_config is not None:
        gp_kwargs = {"gp_config": gp_config}
    else:
        gp_kwargs = {
            "noise_type": gp_noise_type,
            "fixed_noise_value": fixed_noise_value if gp_noise_type == "fixed" else None,
        }
    acqf_kwargs = {"acqf_type": acqf_type}
    if device != "cpu":
        try:
            torch.zeros(1, device=device)
            torch.empty(int(256e6 // 8), dtype=torch.float64, device=device)  # ~256 MB probe
            torch.cuda.empty_cache()
        except (
            Exception
        ) as exc:  # CUDA context/allocation failure: run the BO on CPU instead of dying later
            log(
                f"[bo] WARNING: device {device!r} unavailable "
                f"({type(exc).__name__}); running the surrogate on CPU"
            )
            device = "cpu"
    tkwargs = {"dtype": dtype, "device": device}
    weights = torch.as_tensor(weights, **tkwargs)
    n_objectives = int(weights.numel())
    n_bo_iterations = num_trials - n_initial_pts
    if n_bo_iterations < 0:
        raise ValueError("num_trials must be >= n_initial_pts")

    # The acquisition's MC sampler and the local search draw from the global RNGs, as in train.py.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if initial_x is None:
        generator = torch.Generator().manual_seed(seed)
        initial_x = torch.randint(
            0, 2, (n_initial_pts, decision_dim), dtype=dtype, generator=generator
        )

    train_x = torch.empty((0, decision_dim), **tkwargs)
    train_y = torch.empty((0, n_objectives), **tkwargs)
    tr_manager: TrustRegionManager | None = None
    resumed = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_x = ck["x"].to(**tkwargs)
        train_y = ck["y"].to(**tkwargs)
        resumed = int(train_x.shape[0])
        if ck.get("tr") is not None:
            tr_manager = TrustRegionManager(
                name="tr_1",
                n_categorical=decision_dim,
                n_objectives=n_objectives,
                best_value=[0.0] * n_objectives,
            )
            for k, v in ck["tr"].items():
                setattr(tr_manager, k, v.to(**tkwargs) if isinstance(v, torch.Tensor) else v)
        log(f"[resume] {resumed} evaluations loaded from {checkpoint_path}")

    def _save() -> None:
        if not checkpoint_path:
            return
        tr = None
        if tr_manager is not None:
            tr = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in tr_manager.__dict__.items()
            }
        tmp = checkpoint_path + ".tmp"
        torch.save({"x": train_x.detach().cpu(), "y": train_y.detach().cpu(), "tr": tr}, tmp)
        os.replace(tmp, checkpoint_path)

    for i in range(resumed, initial_x.shape[0]):
        x = initial_x[i : i + 1].to(**tkwargs)
        y = evaluate(x).to(**tkwargs).reshape(1, n_objectives)
        train_x = torch.cat([train_x, x], dim=0)
        train_y = torch.cat([train_y, y], dim=0)
        _save()
        log(f"[init {i + 1}/{initial_x.shape[0]}] y={y.flatten().tolist()}")

    discrete_choices = torch.cat(
        [torch.zeros(decision_dim, 1, **tkwargs), torch.ones(decision_dim, 1, **tkwargs)], dim=1
    )

    first_bo = max(0, resumed - initial_x.shape[0])
    for itr in range(first_bo, n_bo_iterations):
        transformed_y = train_y * weights
        if tr_manager is None:
            best_value = [transformed_y[:, j].max().item() for j in range(n_objectives)]
            tr_manager = TrustRegionManager(
                name="tr_1",
                n_categorical=decision_dim,
                n_objectives=n_objectives,
                init_radius=min(init_radius if init_radius is not None else 10, decision_dim),
                best_value=best_value,
            )
            pareto_mask = is_non_dominated(transformed_y)
            if int(pareto_mask.sum()) < 2:
                # initialize_center's leave-one-out HV needs >= 2 non-dominated points.
                tr_manager.center = train_x[pareto_mask][0]
            else:
                tr_manager.initialize_center(train_x=train_x, train_y=transformed_y, **tkwargs)
        else:
            tr_manager.adjust_radius(new_x=train_x[-1], new_value=weights * train_y[-1])

        t0 = time.time()
        try:
            next_x, t_fit, t_acq = _bo_step(
                train_x,
                train_y,
                weights,
                tkwargs,
                tr_manager,
                discrete_choices,
                n_objectives,
                gp_kwargs,
                acqf_kwargs,
                num_restarts,
                max_batch_size,
                raw_samples,
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or device == "cpu":
                raise
            log(
                f"[bo {itr + 1}/{n_bo_iterations}] WARNING: CUDA OOM "
                f"({str(exc)[:80]}...); moving the surrogate to CPU for the rest of the run"
            )
            device = "cpu"
            tkwargs = {"dtype": dtype, "device": device}
            weights = weights.to(**tkwargs)
            train_x = train_x.to(**tkwargs)
            train_y = train_y.to(**tkwargs)
            discrete_choices = discrete_choices.to(**tkwargs)
            tr_manager.center = tr_manager.center.to(**tkwargs)
            torch.cuda.empty_cache()
            next_x, t_fit, t_acq = _bo_step(
                train_x,
                train_y,
                weights,
                tkwargs,
                tr_manager,
                discrete_choices,
                n_objectives,
                gp_kwargs,
                acqf_kwargs,
                num_restarts,
                max_batch_size,
                raw_samples,
            )
        next_x = next_x.to(**tkwargs)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
        next_y = evaluate(next_x).to(**tkwargs).reshape(1, n_objectives)
        t_eval = time.time() - t0
        train_x = torch.cat([train_x, next_x], dim=0)
        train_y = torch.cat([train_y, next_y], dim=0)
        _save()
        log(
            f"[bo {itr + 1}/{n_bo_iterations}] radius={tr_manager.current_radius} "
            f"y={next_y.flatten().tolist()} "
            f"timing: fit={t_fit:.1f}s acqf+search={t_acq:.1f}s eval={t_eval:.1f}s"
        )

    return COMBOMResult(train_x=train_x, train_y=train_y, weights=weights)
