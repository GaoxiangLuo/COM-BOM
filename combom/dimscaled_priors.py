# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors
"""Custom priors and constraints used for dimscaled categorical kernels."""

import math

import torch
from gpytorch import settings
from gpytorch.constraints import Interval
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors import GammaPrior, LogNormalPrior


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. We don't want to do this when using log as an
    inverse transform. This class will skip this step and apply the log transform
    directly to the parameter values so we can optimize log(parameter) under the bound
    constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(self, lower_bound, upper_bound, initial_value=None):
        r"""Initialize LogTransformedInterval."""
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            (
                torch.tensor(initial_value).to(self.lower_bound)
                if initial_value is not None
                else None
            ),
        )

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a "
                    "derived class like GreaterThan or LessThan instead."
                )

    def transform(self, tensor):
        """Apply the log-space constraint when enforcement is enabled."""
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        """Map a constrained tensor back to the original parameter space."""
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)
        return tensor


def get_kernel_hp_priors(
    ard_num_dims: int,
    active_dims: list[int] | None = None,
    use_outputscale_prior: bool = False,
) -> RBFKernel | MaternKernel | ScaleKernel:
    """Define priors and constraints for kernel hyperparameters.

    Args:
        ard_num_dims: Number of input dimensions for the ARD kernel.
        active_dims: Optional subset of dimensions to apply the kernel to.
        use_outputscale_prior: Whether to include a Gamma prior on the outputscale.

    Returns:
        Tuple of priors and constraints for lengthscale and optional output scale.
    """
    lengthscale_prior = LogNormalPrior(loc=1.4 + math.log(ard_num_dims) * 0.5, scale=1.73205)
    lengthscale_constraint = LogTransformedInterval(0.1, 100.00, initial_value=1.0)

    outputscale_prior = GammaPrior(concentration=1.1, rate=0.2) if use_outputscale_prior else None
    outputscale_constraint = LogTransformedInterval(0.01, 100.0, initial_value=1.0)
    return (
        lengthscale_prior,
        lengthscale_constraint,
        outputscale_prior,
        outputscale_constraint,
    )
