"""GP modelling utilities for binary exemplar-selection experiments."""

import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.kernels import CategoricalKernel
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import (
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

from .dimscaled_priors import (
    LogTransformedInterval,
    get_kernel_hp_priors,
)
from .tanimoto_kernel import TanimotoKernel


def get_fitted_gp_model(
    train_x,
    train_y,
    train_yvar,
    prior_choice="standard",
    kernel_choice="linear",
    ard=True,
    noise_type="high",
    fixed_noise_value=None,
    **tkwargs,
):
    """Fit a single-objective GP with categorical-aware kernels.

    Args:
        train_x: Input design points.
        train_y: Observed responses for the objective.
        train_yvar: Optional observation noise values.
        prior_choice: Either ``"standard"`` or ``"dimscaled"`` for kernel priors.
        kernel_choice: Name of kernel to use (e.g. ``"categorical"``, ``"tanimoto"``).
        ard: If ``True`` enable automatic relevance determination.
        noise_type: One of ``{"standard", "high", "fixed"}`` controlling likelihood.
        fixed_noise_value: Noise level to use when ``noise_type == "fixed"``.
        **tkwargs: Torch keyword arguments (dtype/device).

    Returns:
        A fitted :class:`botorch.models.SingleTaskGP` instance.
    """
    input_dim = train_x.shape[-1]
    if noise_type == "standard":
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
    elif noise_type == "high":
        likelihood = GaussianLikelihood(
            noise_prior=LogNormalPrior(loc=-2.0, scale=1.0),
            noise_constraint=LogTransformedInterval(1e-4, 1.0),
        )
    elif noise_type == "fixed":
        if fixed_noise_value is None:
            raise ValueError("fixed_noise_value must be provided if noise_type is 'fixed'")
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(fixed_noise_value - 1e-5))
        likelihood.noise = torch.tensor([fixed_noise_value])
        likelihood.raw_noise.requires_grad = False

    if prior_choice == "standard":
        if kernel_choice == "categorical":
            kernel = CategoricalKernel(ard_num_dims=input_dim if ard else None)
        elif kernel_choice == "tanimoto":
            kernel = TanimotoKernel()
        outputscale_prior = GammaPrior(torch.tensor(2.0, **tkwargs), torch.tensor(0.15, **tkwargs))
        outputscale_constraint = GreaterThan(1e-6)
        covar_module = ScaleKernel(
            base_kernel=kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
        )
    elif prior_choice == "dimscaled":
        (
            lengthscale_prior,
            lengthscale_constraint,
            outputscale_prior,
            outputscale_constraint,
        ) = get_kernel_hp_priors(ard_num_dims=input_dim, use_outputscale_prior=True)
        base_kernel = CategoricalKernel(
            ard_num_dims=input_dim if ard else None,
            lengthscale_constraint=lengthscale_constraint,
            lengthscale_prior=lengthscale_prior,
        )
        covar_module = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )
    gp_model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y.unsqueeze(-1),
        train_Yvar=train_yvar,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
        likelihood=likelihood,
    )
    mll = ExactMarginalLogLikelihood(model=gp_model, likelihood=gp_model.likelihood)
    fit_gpytorch_model(mll)
    return gp_model
