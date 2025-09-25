"""Custom Tanimoto kernel used for binary/categorical similarity."""

import torch
from gpytorch.kernels import Kernel


def batch_tanimoto_sim(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Tanimoto similarity between two batched tensors across the last two dimensions.

    The ``eps`` argument ensures numerical stability if all-zero tensors are added.
    Tanimoto similarity is proportional to:

    :math:`(<x, y>) / (||x||^2 + ||y||^2 - <x, y>)`

    where x and y may be bit or count vectors or in set notation:

    :math:`|A \\cap B| / |A| + |B| - |A \\cap B|`

    Args:
        x1: ``[b x n x d]`` tensor where ``b`` is the batch dimension.
        x2: ``[b x m x d]`` tensor.
        eps: Small float added for numerical stability (default: ``1e-6``).

    Returns:
        Tensor denoting the Tanimoto similarity.
    """
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_norm = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2**2, dim=-1, keepdims=True)

    tan_similarity = (dot_prod + eps) / (
        eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
    )

    return tan_similarity.clamp_min_(0)


class TanimotoKernel(Kernel):
    r"""Compute a covariance matrix based on the Tanimoto kernel.

    The kernel operates on inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

     .. math::

        \begin{equation*}
        k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x'}) =
        \frac{\langle\mathbf{x}, \mathbf{x'}\rangle}{\lVert\mathbf{x}\rVert^2 +
        \lVert\mathbf{x'}\rVert^2 - \langle\mathbf{x}, \mathbf{x'}\rangle}
        \end{equation*}

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

    Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        """Initialise the kernel without adding extra hyperparameters."""
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """Evaluate the kernel matrix or its diagonal for ``x1`` and ``x2``."""
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
        return self.covar_dist(x1, x2, **params)

    def covar_dist(
        self,
        x1,
        x2,
        last_dim_is_batch=False,
        **params,
    ):
        r"""Compute the bit-vector similarity between each pair of points in ``x1`` and ``x2``.

        Args:
            x1: Tensor shaped ``n x d`` or ``b1 x ... x bk x n x d`` containing the first batch.
            x2: Tensor shaped ``m x d`` or ``b1 x ... x bk x m x d`` containing the second batch.
            last_dim_is_batch: Whether the last dimension should be interpreted as batch size.
            **params: Additional keyword parameters forwarded to :func:`batch_tanimoto_sim`.

        Returns:
            Tensor representing the kernel matrix between ``x1`` and ``x2``.
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        return batch_tanimoto_sim(x1, x2)
