"""GSM8K GP tuning, step 4: kernel-family comparison with a per-family tuning grid (fair: each family gets its own
lengthscale-prior / noise / ARD grid; the best of each family is compared on CV NLPD and on the learning curve).

Families (inputs are 32-bit exemplar-selection vectors):
  categorical   paper: exponentiated Hamming k = exp(-mean_d 1[x_d != x'_d] / l_d)          (BoTorch CategoricalKernel)
  matern        Matern-5/2 treating bits as continuous (BoTorch default for continuous inputs)
  rbf           squared-exponential on bits
  linear        Bayesian linear regression on bits (additive main effects only)
  poly2         degree-2 polynomial (main effects + pairwise interactions)
  tanimoto      Jaccard/Tanimoto set similarity |x & x'| / |x | x'| (standard kernel for binary fingerprints in molecule BO)
  cat+count     categorical + additive RBF kernel on the number of selected exemplars
  cat+linear    categorical + additive linear kernel (explicit main effects + residual)
Reuses gp_tune's CV machinery (same folds/repeats/metrics) by swapping its model builder.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

import _paths  # noqa: F401  (sets sys.path)
import gp_tune as g
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.kernels import CategoricalKernel
from botorch.models.transforms import Standardize
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import (
    AdditiveKernel,
    Kernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

torch.set_num_threads(1)
NAMES = g.NAMES


class TanimotoKernel(Kernel):
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        inter = x1 @ x2.transpose(-1, -2)
        n1 = x1.sum(-1, keepdim=True)
        n2 = x2.sum(-1, keepdim=True).transpose(-1, -2)
        res = inter / (n1 + n2 - inter).clamp_min(1e-9)
        return torch.diagonal(res, dim1=-1, dim2=-2) if diag else res


class CountKernel(Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        c1 = x1.sum(-1, keepdim=True) / self.lengthscale
        c2 = x2.sum(-1, keepdim=True) / self.lengthscale
        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
        return torch.exp(-0.5 * (c1 - c2.transpose(-1, -2)) ** 2)


KERNELS = ["categorical", "matern", "rbf", "linear", "poly2", "tanimoto", "cat+count", "cat+linear"]
HAS_LS = {"categorical", "matern", "rbf", "cat+count", "cat+linear"}


def build_k(cfg, X, y, yvar):
    d = X.shape[-1]
    tk = dict(dtype=X.dtype, device=X.device)
    kern = cfg["kernel"]
    lo, hi = (float(v) for v in cfg["ls_bounds"].split(","))
    lc = LogTransformedInterval(lo, hi, initial_value=1.0)
    lp = LogNormalPrior(loc=1.4 + math.log(d) * 0.5 + cfg["ls_shift"], scale=cfg["ls_scale"])
    kw = dict(
        ard_num_dims=d if cfg["ard"] else None, lengthscale_prior=lp, lengthscale_constraint=lc
    )

    def scaled(k):
        return ScaleKernel(
            k, outputscale_prior=GammaPrior(2.0, 0.15), outputscale_constraint=GreaterThan(1e-6)
        )

    base = kern.split("+")[0]
    if base in ("categorical", "cat"):
        k = CategoricalKernel(**kw)
    elif base == "matern":
        k = MaternKernel(nu=2.5, **kw)
    elif base == "rbf":
        k = RBFKernel(**kw)
    elif base == "linear":
        k = LinearKernel()
    elif base == "poly2":
        k = PolynomialKernel(power=2)
    elif base == "tanimoto":
        k = TanimotoKernel()
    else:
        raise ValueError(kern)
    covar = scaled(k)
    if kern.endswith("+count"):
        covar = AdditiveKernel(
            covar,
            scaled(
                CountKernel(
                    lengthscale_constraint=LogTransformedInterval(0.5, 64.0, initial_value=8.0)
                )
            ),
        )
    if kern.endswith("+linear"):
        covar = AdditiveKernel(covar, scaled(LinearKernel()))
    nz = cfg["noise"]
    if nz.startswith("fixed:"):
        v = float(nz[6:])
        lik = GaussianLikelihood(noise_constraint=GreaterThan(max(v - 1e-5, 0.0)))
        lik.noise = torch.tensor([v], **tk)
        lik.raw_noise.requires_grad = False
    elif nz == "gamma":
        lik = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **tk), torch.tensor(10.0, **tk)),
            noise_constraint=GreaterThan(1e-4),
        )
    elif nz.startswith("ln"):
        lik = GaussianLikelihood(
            noise_prior=LogNormalPrior(loc=float(nz[2:]), scale=1.0),
            noise_constraint=LogTransformedInterval(1e-4, 1.0),
        )
    else:
        raise ValueError(nz)
    return SingleTaskGP(
        train_X=X,
        train_Y=y.unsqueeze(-1),
        covar_module=covar,
        likelihood=lik,
        outcome_transform=Standardize(m=1),
    )


g.build = build_k  # forked workers inherit the swapped builder
g.FACTORS = {
    "kernel": KERNELS,
    "ard": [True, False],
    "ls_shift": [-3.0, -2.0, -1.0, 0.0],
    "ls_scale": [0.5, 1.0],
    "ls_bounds": ["0.1,100"],
    "noise": ["ln-1", "gamma"],
    "restarts": [1],
}


def canon(c):
    c = dict(c)
    if c["kernel"] not in HAS_LS:
        c["ard"], c["ls_shift"], c["ls_scale"] = True, 0.0, 1.0
    return c


def key(c):
    return "|".join(f"{k}={c[k]}" for k in g.FACTORS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lc-sizes", default="20,40,80,120")
    ap.add_argument("--lc-reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = [torch.load(f, weights_only=False) for f in a.design]
    X = torch.cat([p["x"] for p in parts]).double()
    Y = torch.cat([p["y"] for p in parts]).double()
    n, d = X.shape
    rep_y = torch.cat(
        [p["repeat_y"] for p in parts if "repeat_y" in p and p["repeat_y"].shape[0] > 0]
    ).double()
    noise_sd = rep_y.std(dim=1, unbiased=True).mean(dim=0)
    YV = {j: torch.zeros(n, dtype=torch.double) for j in range(2)}
    grid = []
    for kern in KERNELS:
        if kern in HAS_LS:
            for ard, sh, sc, nz in itertools.product(
                [True, False], [-3.0, -2.0, -1.0, 0.0], [0.5, 1.0], ["ln-1", "gamma"]
            ):
                grid.append(
                    dict(
                        kernel=kern,
                        ard=ard,
                        ls_shift=sh,
                        ls_scale=sc,
                        ls_bounds="0.1,100",
                        noise=nz,
                        restarts=1,
                    )
                )
        else:
            for nz in ["ln-1", "gamma"]:
                grid.append(
                    canon(
                        dict(
                            kernel=kern,
                            ard=True,
                            ls_shift=0.0,
                            ls_scale=1.0,
                            ls_bounds="0.1,100",
                            noise=nz,
                            restarts=1,
                        )
                    )
                )
    rp = f"{a.out}/gp_kernels_results.json"
    results = json.load(open(rp)) if os.path.exists(rp) else {}
    todo = [(c, j) for c in grid for j in range(2) if f"{key(c)}#{j}" not in results]
    print(
        f"design n={n} d={d}; {len(grid)} configs over {len(KERNELS)} kernel families; {len(todo)} tasks",
        flush=True,
    )
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(g.cv_task, c, j, X, Y, YV, a.folds, a.repeats): (c, j) for c, j in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            c, j = futs[fu]
            try:
                results[f"{key(c)}#{j}"] = dict(cfg=c, objective=NAMES[j], **fu.result())
            except Exception as e:
                results[f"{key(c)}#{j}"] = dict(
                    cfg=c, objective=NAMES[j], failed=f"{type(e).__name__}: {str(e)[:100]}"
                )
            if i % 20 == 0 or i == len(todo):
                json.dump(results, open(rp, "w"))
                print(f"  {i}/{len(todo)} done", flush=True)

    def score(c):
        return sum(results.get(f"{key(c)}#{j}", {}).get("nlpd", float("inf")) for j in range(2))

    def row(c):
        parts = []
        for j in range(2):
            r = results[f"{key(c)}#{j}"]
            parts.append(
                "FAILED"
                if "failed" in r
                else f"nlpd {r['nlpd']:6.3f} rmse/c {r['rmse_vs_const']:.3f} rho {r['spearman']:.3f} cov {r['cover95']:.2f} rmse/noise {r['rmse'] / noise_sd[j].item():.2f}"
            )
        return f"{score(c):8.3f}  acc[{parts[0]}]  ece[{parts[1]}]  {key(c)}"

    best = {}
    for kern in KERNELS:
        cs = [c for c in grid if c["kernel"] == kern and score(c) < float("inf")]
        if cs:
            best[kern] = min(cs, key=score)
    lines = [
        f"Kernel-family comparison on the GSM8K design (n={n}, d={d}); {a.folds}-fold CV x {a.repeats}; each family tuned over its own grid (ARD x ls-shift x ls-scale x noise)\n",
        "== best config per kernel family (sorted by summed NLPD; lower is better)",
    ]
    for kern in sorted(best, key=lambda k: score(best[k])):
        lines.append(f"  {kern:<12} " + row(best[kern]))
    lines.append("\n== mean score over each family's whole grid (robustness to hyperparameters)")
    for kern in KERNELS:
        sc = [score(c) for c in grid if c["kernel"] == kern and score(c) < float("inf")]
        nf = sum(1 for c in grid if c["kernel"] == kern and score(c) == float("inf"))
        lines.append(
            f"  {kern:<12} mean {np.mean(sc):8.3f}  best {min(sc):8.3f}  worst {max(sc):8.3f}  (n={len(sc)}, failed={nf})"
            if sc
            else f"  {kern:<12} all failed"
        )
    sizes = [int(s) for s in a.lc_sizes.split(",") if int(s) < n - 10]
    lc = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {
            ex.submit(g.lc_task, c, j, X, Y, YV, m, a.lc_reps): (kern, j, m)
            for kern, c in best.items()
            for j in range(2)
            for m in sizes
        }
        for fu in as_completed(futs):
            kern, j, m = futs[fu]
            try:
                lc[f"{kern}#{j}#{m}"] = fu.result()
            except Exception as e:
                lc[f"{kern}#{j}#{m}"] = dict(failed=str(e)[:80])
    json.dump(lc, open(f"{a.out}/gp_kernels_learning_curve.json", "w"))
    lines.append(
        f"\n== learning curve of each family's best config: held-out NLPD / Spearman at m training points (mean of {a.lc_reps} splits)"
    )
    for kern in sorted(best, key=lambda k: score(best[k])):
        for j in range(2):
            cells = [
                f"m={m}: "
                + (
                    "FAIL"
                    if "failed" in lc[f"{kern}#{j}#{m}"]
                    else f"{lc[f'{kern}#{j}#{m}']['nlpd']:.3f}/{lc[f'{kern}#{j}#{m}']['spearman']:.2f}"
                )
                for m in sizes
            ]
            lines.append(f"  {kern:<12} {NAMES[j]:<18} " + "  ".join(cells))
    open(f"{a.out}/gp_kernels_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, name in enumerate(NAMES):
        for kern in sorted(best, key=lambda k: score(best[k])):
            axes[j].plot(
                sizes,
                [lc[f"{kern}#{j}#{m}"].get("nlpd", np.nan) for m in sizes],
                marker="o",
                label=kern,
                lw=2.5 if kern == "categorical" else 1,
            )
        axes[j].set_xlabel("# training points")
        axes[j].set_ylabel("held-out NLPD")
        axes[j].set_title(name)
        axes[j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{a.out}/gp_kernels_learning_curve.png", dpi=140)
    fig, axes = plt.subplots(2, len(best), figsize=(4.2 * len(best), 8))
    y = Y.numpy()
    for col, kern in enumerate(sorted(best, key=lambda k: score(best[k]))):
        for j, name in enumerate(NAMES):
            ax = axes[j][col]
            r = results[f"{key(best[kern])}#{j}"]
            mu, sd = np.array(r["mu"]), np.array(r["sd"])
            ax.errorbar(
                y[:, j],
                mu,
                yerr=1.96 * sd,
                fmt="o",
                ms=3,
                alpha=0.6,
                ecolor="#9ecae1",
                color="#1f77b4",
                capsize=1.5,
            )
            lo, hi = min(y[:, j].min(), mu.min()), max(y[:, j].max(), mu.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(
                f"{kern}: NLPD {r['nlpd']:.2f} rho {r['spearman']:.2f} cov {r['cover95']:.2f}",
                fontsize=9,
            )
            ax.set_xlabel(f"observed {name}")
            if col == 0:
                ax.set_ylabel("held-out prediction")
    fig.tight_layout()
    fig.savefig(f"{a.out}/gp_kernels_fit.png", dpi=120)
    print(
        f"wrote {a.out}/gp_kernels_report.txt, gp_kernels_fit.png, gp_kernels_learning_curve.png",
        flush=True,
    )


if __name__ == "__main__":
    main()
