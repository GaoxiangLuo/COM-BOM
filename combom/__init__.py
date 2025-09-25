"""Convenience imports for COM-BOM optimisation components."""

from .acquisition_functions import get_acquisition_function, optimize_acqf_discrete_local_search
from .gp_modeling import get_fitted_gp_model
from .trust_region import TrustRegionManager

__all__ = [
    "get_acquisition_function",
    "optimize_acqf_discrete_local_search",
    "get_fitted_gp_model",
    "TrustRegionManager",
]
