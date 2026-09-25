"""qNEHVI acquisition for the DSPy integration.

The release ``combom.get_acquisition_function`` relies on BoTorch's default Monte Carlo sampler. For
a ``ModelList`` BoTorch <= 0.15 built one ``SobolQMCNormalSampler`` per objective, which is what the
paper's runs used; BoTorch >= 0.16 switched to IID normal samplers. This version pins the Sobol
samplers, drawing their seeds exactly as BoTorch 0.13 did, so the paper's acquisition is reproduced
on current BoTorch.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models.model import ModelList
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated


def get_acquisition_function(
    model: ModelList,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    disable_noisy: bool = False,
    acqf_type: Literal["qnehvi", "qlognehvi"] = "qnehvi",
    sampler=None,
    sampler_type: Literal["sobol", "iid", "botorch_default"] = "sobol",
    mc_samples: int = 128,
    **tkwargs: dict,
) -> Tuple[AcquisitionFunction, torch.Tensor]:
    """Construct a (noisy) EHVI acquisition function for binary search spaces.

    Args:
        model: Joint GP surrogate over all objectives.
        train_x: Previously evaluated points.
        train_y: Objective values corresponding to ``train_x``.
        weights: Optional tensor of objective weights (defaults to [1, -1]).
        disable_noisy: If ``True`` use deterministic qEHVI instead of qNEHVI.
        acqf_type: ``"qnehvi"`` (paper; BoTorch's legacy MC estimator) or ``"qlognehvi"``
            (BoTorch's recommended log-space variant, same interface).
        sampler: Optional ``MCSampler`` (e.g. a seeded ``SobolQMCNormalSampler``) for
            reproducible acquisition values across library versions.
        sampler_type: Used when ``sampler`` is ``None``. ``"sobol"`` (default) builds one
            ``SobolQMCNormalSampler`` per objective, which is what BoTorch <= 0.15 did by default
            for ``ModelList`` posteriors (and what the paper's runs used); BoTorch >= 0.16 switched
            the default to IID normal samplers. ``"botorch_default"`` defers to the
            installed version.
        mc_samples: MC sample count for the acquisition estimator (BoTorch default: 128).
        **tkwargs: Torch keyword arguments (dtype/device).

    Returns:
        Tuple ``(acqf, pareto_points)`` where ``acqf`` is the EHVI module and
        ``pareto_points`` are the current non-dominated points.
    """
    if weights is None:
        weights = torch.tensor([1.0, -1.0], **tkwargs)
    else:
        weights = weights.to(**tkwargs)

    # transform objectives into maximisation form for Pareto/ref point calculations
    transformed_train_y = train_y * weights

    # find pareto optimal points in the transformed (maximization) space
    pareto_points = train_x[is_non_dominated(transformed_train_y)].clone()

    # calculate the reference point in the transformed (maximization) space
    reference_point = (
        transformed_train_y.min(dim=0)[0] - torch.abs(transformed_train_y.min(dim=0)[0]) * 0.1
    )

    # Define the objective function using the original weights
    objective = WeightedMCMultiOutputObjective(weights=weights)

    if sampler is None and sampler_type != "botorch_default":
        n_obj = train_y.shape[-1]
        seeds = torch.randint(high=1_000_000, size=(n_obj,)).tolist()
        cls = SobolQMCNormalSampler if sampler_type == "sobol" else IIDNormalSampler
        sampler = ListSampler(
            *[cls(sample_shape=torch.Size([mc_samples]), seed=int(s)) for s in seeds]
        )

    # Initialize qNEHVI with the transformed reference point and the weighted objective
    if not disable_noisy:
        if acqf_type == "qlognehvi":
            from botorch.acquisition.multi_objective.logei import (
                qLogNoisyExpectedHypervolumeImprovement,
            )

            acqf_cls = qLogNoisyExpectedHypervolumeImprovement
        elif acqf_type == "qnehvi":
            acqf_cls = qNoisyExpectedHypervolumeImprovement
        else:
            raise ValueError(f"unknown acqf_type {acqf_type!r}")
        acqf = acqf_cls(
            X_baseline=train_x,
            model=model,
            ref_point=reference_point,
            objective=objective,
            sampler=sampler,
            cache_root=False,
        )
    else:
        with torch.no_grad():
            pred = model.posterior(train_x).mean
            pred = pred * weights
        partitioning = FastNondominatedPartitioning(
            ref_point=reference_point,
            Y=pred,
        )

        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference_point,
            objective=objective,
            partitioning=partitioning,
        )
    return acqf, pareto_points
