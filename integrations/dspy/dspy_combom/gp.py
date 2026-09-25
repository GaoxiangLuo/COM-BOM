"""Config-driven GP for per-benchmark tuning.

Same model family as the release ``combom.gp_modeling.get_fitted_gp_model`` (a ``SingleTaskGP`` with
a ``CategoricalKernel`` and ``Standardize``), with each hyperparameter choice exposed as a
dictionary entry: lengthscale prior, noise model, outputscale prior, mean and MAP restarts. The
default configuration reproduces the release GP used in the paper.
"""

import math

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.kernels import CategoricalKernel
from botorch.models.transforms import Standardize
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior


class TanimotoKernel(Kernel):
    """Tanimoto / Jaccard set similarity on binary vectors: k(x, x') = |x & x'| / |x | x'|.

    Treats an exemplar selection as a set. Standard kernel for binary fingerprints in
    molecular BO. Has no lengthscale.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        """Return the Tanimoto similarity matrix (or its diagonal)."""
        inter = x1 @ x2.transpose(-1, -2)
        n1 = x1.sum(-1, keepdim=True)
        n2 = x2.sum(-1, keepdim=True).transpose(-1, -2)
        res = inter / (n1 + n2 - inter).clamp_min(1e-9)
        return torch.diagonal(res, dim1=-1, dim2=-2) if diag else res


GP_CONFIG_DEFAULTS = dict(
    ard=True,
    # 'lognormal' (dim-scaled, Hvarfner et al. 2024), 'saas' (MAP-SAAS), 'none' (MLE)
    ls_prior="lognormal",
    ls_shift=0.0,  # added to the dim-scaled LogNormal location 1.4 + 0.5*log(d)
    ls_scale=1.73205,  # LogNormal scale (paper sqrt(3))
    ls_bounds="0.1,100",
    # 'paper' Gamma(1.1,0.2) | 'gamma2' Gamma(2,0.15) | 'noprior' | 'fixed1' (no ScaleKernel)
    outputscale="paper",
    mean="constant",  # or 'zero'
    noise="fixed:1e-3",  # 'fixed:<v>' | 'gamma' | 'ln<loc>' | 'known' | 'known_binom'
    restarts=1,  # MAP restarts from prior samples, best marginal likelihood kept
    kernel="categorical",  # or 'tanimoto' (set similarity; ignores the lengthscale settings)
    # observation-noise variance used by 'known' (float) / 'known_binom' fallback
    known_noise_var=None,
    n_val=None,  # sample count behind 'known_binom' (variance p(1-p)/n_val)
)


def build_gp_from_config(train_x, train_y, train_yvar=None, config=None):
    """Build an unfitted SingleTaskGP with a CategoricalKernel from a config dict.

    See :data:`GP_CONFIG_DEFAULTS` for the recognised keys.
    """
    from botorch.models.map_saas import add_saas_prior
    from gpytorch.means import ZeroMean

    cfg = dict(GP_CONFIG_DEFAULTS, **(config or {}))
    d = train_x.shape[-1]
    tk = dict(dtype=train_x.dtype, device=train_x.device)
    lo, hi = (float(v) for v in str(cfg["ls_bounds"]).split(","))
    lc = LogTransformedInterval(lo, hi, initial_value=1.0)
    ard_dims = d if cfg["ard"] else None
    if cfg.get("kernel", "categorical") == "tanimoto":
        base = TanimotoKernel()
    elif cfg["ls_prior"] == "lognormal":
        lp = LogNormalPrior(
            loc=1.4 + math.log(d) * 0.5 + float(cfg["ls_shift"]), scale=float(cfg["ls_scale"])
        )
        base = CategoricalKernel(
            ard_num_dims=ard_dims, lengthscale_prior=lp, lengthscale_constraint=lc
        )
    elif cfg["ls_prior"] == "saas":
        base = CategoricalKernel(ard_num_dims=ard_dims, lengthscale_constraint=lc)
        add_saas_prior(base, tau=None, log_scale=True)
    elif cfg["ls_prior"] == "none":
        base = CategoricalKernel(ard_num_dims=ard_dims, lengthscale_constraint=lc)
    else:
        raise ValueError(f"unknown ls_prior {cfg['ls_prior']!r}")
    oc = LogTransformedInterval(0.01, 100.0, initial_value=1.0)
    if cfg["outputscale"] == "paper":
        covar = ScaleKernel(base, outputscale_prior=GammaPrior(1.1, 0.2), outputscale_constraint=oc)
    elif cfg["outputscale"] == "gamma2":
        covar = ScaleKernel(
            base, outputscale_prior=GammaPrior(2.0, 0.15), outputscale_constraint=GreaterThan(1e-6)
        )
    elif cfg["outputscale"] == "noprior":
        covar = ScaleKernel(base, outputscale_constraint=oc)
    elif cfg["outputscale"] == "fixed1":
        covar = base
    else:
        raise ValueError(f"unknown outputscale {cfg['outputscale']!r}")
    nz = str(cfg["noise"])
    likelihood = None
    if nz.startswith("fixed:"):
        v = float(nz[6:])
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(max(v - 1e-5, 0.0)))
        likelihood.noise = torch.tensor([v], **tk)
        likelihood.raw_noise.requires_grad = False
    elif nz == "gamma":
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **tk), torch.tensor(10.0, **tk)),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
    elif nz.startswith("ln"):
        likelihood = GaussianLikelihood(
            noise_prior=LogNormalPrior(loc=float(nz[2:]), scale=1.0),
            noise_constraint=LogTransformedInterval(1e-4, 1.0),
        )
    elif nz in ("known", "known_binom"):
        if train_yvar is None:
            if nz == "known_binom" and cfg.get("n_val"):
                p = train_y.clamp(1e-3, 1 - 1e-3)
                train_yvar = p * (1 - p) / float(cfg["n_val"])
            elif cfg.get("known_noise_var") is not None:
                train_yvar = torch.full_like(train_y, float(cfg["known_noise_var"]))
            else:
                raise ValueError("noise='known' needs train_yvar, known_noise_var or n_val")
        train_yvar = train_yvar.to(train_x)
        if train_yvar.dim() == 1:
            train_yvar = train_yvar.unsqueeze(-1)
    else:
        raise ValueError(f"unknown noise {nz!r}")
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y.unsqueeze(-1) if train_y.dim() == 1 else train_y,
        train_Yvar=train_yvar,
        covar_module=covar,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    if cfg["mean"] == "zero":
        model.mean_module = ZeroMean()
    elif cfg["mean"] != "constant":
        raise ValueError(f"unknown mean {cfg['mean']!r}")
    return model


def fit_gp_with_restarts(model, restarts=1):
    """MAP fit of a GP.

    Extra restarts sample hyperparameters from their priors and keep the fit with the
    best marginal likelihood.
    """
    from botorch.optim.utils.model_utils import sample_all_priors

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    best = None
    for r in range(max(1, int(restarts))):
        try:
            if r > 0:
                sample_all_priors(model)
            fit_gpytorch_mll(mll)
            model.train()
            with torch.no_grad():
                val = mll(model(*model.train_inputs), model.train_targets).sum().item()
            model.eval()
        except Exception:  # noqa: BLE001 - a failed restart is simply discarded
            continue
        if best is None or val > best[0]:
            best = (val, {k: v.clone() for k, v in model.state_dict().items()})
    if best is None:
        raise RuntimeError("all GP fit attempts failed")
    model.load_state_dict(best[1])
    model.eval()
    return model


def get_fitted_gp_model_from_config(train_x, train_y, train_yvar=None, config=None, **tkwargs):
    """Config-driven counterpart of :func:`get_fitted_gp_model`.

    Used by the DSPy layer when ``gp_config`` is set.
    """
    cfg = dict(GP_CONFIG_DEFAULTS, **(config or {}))
    model = build_gp_from_config(train_x, train_y, train_yvar, cfg)
    return fit_gp_with_restarts(model, cfg["restarts"])
