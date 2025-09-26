import math

from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors import GammaPrior, LogNormalPrior


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
