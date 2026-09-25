"""Hyperparameter search for COM-BOM's CategoricalKernel GP on a random design (kernel family fixed).

Input: one or more design files from ``dspy_combom.design.collect_design`` (any task; objective names and
the validation size are read from the file). Output in --out: gp_tune_report.txt, gp_config_tuned.json
(pass it to ``COMBOM(gp_config=...)``), held-out fit plots per objective and a learning-curve plot.

Search space (all with SingleTaskGP + Standardize, exactly as combom.gp_modeling builds it):
  ard            per-dimension (ARD) vs one shared lengthscale
  ls_prior       'lognormal' (BoTorch/Hvarfner dim-scaled prior, loc = 1.4 + 0.5 log d + shift, scale),
                 'saas' (BoTorch MAP-SAAS shrinkage prior: tau/l^2 ~ HalfCauchy(1), tau ~ HalfCauchy(0.1)), or 'none' (MLE)
  ls_shift       shift of the log-lengthscale prior location; ls_scale: its spread
  ls_bounds      lengthscale constraint interval
  outputscale    'paper' Gamma(1.1,0.2) | 'gamma2' Gamma(2,0.15) (classic GPyTorch/BoTorch) | 'noprior' | 'fixed1' (no ScaleKernel)
  mean           constant | zero
  noise          fixed:<v> (standardised units) | gamma (paper 'standard') | ln-2 (paper 'high') | ln-4 (BoTorch>=0.12 default)
                 | ln-1 | known (homoscedastic noise floor from repeated evaluations) | known_binom (accuracy: p(1-p)/n_val)
  restarts       1 | 5  (BoTorch wisdom: restart MAP from prior samples, keep the best marginal likelihood)
Scoring: repeated k-fold CV NLPD (primary), RMSE / const-predictor RMSE, Spearman, 95% coverage.
Stages: (1) one-factor-at-a-time around the current setting, (2) random sample of the full space, (3) OFAT around the best.
Then: learning curve (held-out NLPD/Spearman vs #train points) for the top configs vs the paper config, and fit plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.kernels import CategoricalKernel
from botorch.models.map_saas import add_saas_prior
from botorch.models.transforms import Standardize
from botorch.optim.utils.model_utils import sample_all_priors
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from scipy.stats import norm, spearmanr

torch.set_num_threads(1)
NAMES = ["accuracy", "calibration_error"]
BASE = dict(
    ard=True,
    ls_prior="lognormal",
    ls_shift=0.0,
    ls_scale=1.0,
    ls_bounds="0.1,100",
    outputscale="paper",
    mean="constant",
    noise="gamma",
    restarts=1,
)
PAPER = dict(BASE, ls_scale=1.73205, noise="fixed:1e-3")
FACTORS = {
    "ard": [True, False],
    "ls_prior": ["lognormal", "saas", "none"],
    "ls_shift": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0],
    "ls_scale": [0.5, 0.75, 1.0, 1.5, 1.73205, 2.5],
    "ls_bounds": ["0.1,100", "0.01,1000"],
    "outputscale": ["paper", "gamma2", "noprior", "fixed1"],
    "mean": ["constant", "zero"],
    "noise": [
        "fixed:1e-3",
        "fixed:1e-2",
        "fixed:0.1",
        "fixed:0.3",
        "fixed:1.0",
        "gamma",
        "ln-2",
        "ln-4",
        "ln-1",
        "known",
        "known_binom",
    ],
    "restarts": [1, 5],
}


def key(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in FACTORS)


def canon(cfg):
    cfg = dict(cfg)
    if cfg["ls_prior"] != "lognormal":
        cfg["ls_shift"], cfg["ls_scale"] = 0.0, 1.0  # irrelevant factors -> canonical
    return cfg


def build(cfg, X, y, yvar):
    d = X.shape[-1]
    tk = dict(dtype=X.dtype, device=X.device)
    lo, hi = (float(v) for v in cfg["ls_bounds"].split(","))
    lc = LogTransformedInterval(lo, hi, initial_value=1.0)
    if cfg["ls_prior"] == "lognormal":
        lp = LogNormalPrior(loc=1.4 + math.log(d) * 0.5 + cfg["ls_shift"], scale=cfg["ls_scale"])
        base = CategoricalKernel(
            ard_num_dims=d if cfg["ard"] else None, lengthscale_prior=lp, lengthscale_constraint=lc
        )
    elif cfg["ls_prior"] == "saas":
        base = CategoricalKernel(ard_num_dims=d if cfg["ard"] else None, lengthscale_constraint=lc)
        add_saas_prior(base, tau=None, log_scale=True)
    else:
        base = CategoricalKernel(ard_num_dims=d if cfg["ard"] else None, lengthscale_constraint=lc)
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
        raise ValueError(cfg["outputscale"])
    nz = cfg["noise"]
    train_yvar = None
    if nz.startswith("fixed:"):
        v = float(nz[6:])
        lik = GaussianLikelihood(noise_constraint=GreaterThan(v - 1e-5 if v > 1e-5 else 0.0))
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
    elif nz in ("known", "known_binom"):
        lik = None
        train_yvar = yvar.to(X).unsqueeze(-1)
    else:
        raise ValueError(nz)
    m = SingleTaskGP(
        train_X=X,
        train_Y=y.unsqueeze(-1),
        train_Yvar=train_yvar,
        covar_module=covar,
        likelihood=lik,
        outcome_transform=Standardize(m=1),
    )
    if cfg["mean"] == "zero":
        m.mean_module = ZeroMean()
    return m


def fit(m, restarts):
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    best = None
    for r in range(restarts):
        try:
            if r > 0:
                sample_all_priors(m)
            fit_gpytorch_mll(mll)
            m.train()
            with torch.no_grad():
                val = mll(m(*m.train_inputs), m.train_targets).sum().item()
            m.eval()
        except Exception:
            continue
        if best is None or val > best[0]:
            best = (val, {k: v.clone() for k, v in m.state_dict().items()})
    if best is None:
        raise RuntimeError("all fit attempts failed")
    m.load_state_dict(best[1])
    m.eval()
    return m


def predict(cfg, Xtr, ytr, yvtr, Xte):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fit(build(cfg, Xtr, ytr, yvtr), cfg["restarts"])
        with torch.no_grad():
            post = m.posterior(Xte, observation_noise=True)
    return post.mean.flatten(), post.variance.clamp_min(1e-12).sqrt().flatten()


def metrics(y, mu, sd):
    r = y - mu
    return dict(
        rmse=r.pow(2).mean().sqrt().item(),
        nlpd=(0.5 * math.log(2 * math.pi) + sd.log() + 0.5 * (r / sd) ** 2).mean().item(),
        cover95=(r.abs() <= 1.96 * sd).double().mean().item(),
        spearman=float(spearmanr(mu.numpy(), y.numpy()).correlation),
        rmse_vs_const=(r.pow(2).mean().sqrt() / (y - y.mean()).pow(2).mean().sqrt()).item(),
    )


def cv_task(cfg, j, X, Y, YV, folds, repeats):
    """Repeated k-fold CV of one config for objective j. Returns averaged metrics + last-repeat predictions."""
    n = X.shape[0]
    y = Y[:, j]
    yv = YV[j]
    out = []
    t0 = time.time()
    for rep in range(repeats):
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(rep))
        mu, sd = torch.zeros(n, dtype=X.dtype), torch.zeros(n, dtype=X.dtype)
        for f in idx.chunk(folds):
            mask = torch.ones(n, dtype=torch.bool)
            mask[f] = False
            mu[f], sd[f] = predict(cfg, X[mask], y[mask], yv[mask], X[f])
        out.append(metrics(y, mu, sd))
    agg = {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    agg["nlpd_sd"] = float(np.std([o["nlpd"] for o in out]))
    agg["seconds"] = time.time() - t0
    agg["mu"] = mu.tolist()
    agg["sd"] = sd.tolist()
    return agg


def lc_task(cfg, j, X, Y, YV, m, reps):
    n = X.shape[0]
    y = Y[:, j]
    yv = YV[j]
    res = []
    for rep in range(reps):
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(100 + rep))
        tr, te = idx[:m], idx[m:]
        mu, sd = predict(cfg, X[tr], y[tr], yv[tr], X[te])
        res.append(metrics(y[te], mu, sd))
    return {k: float(np.mean([o[k] for o in res])) for k in res[0]}


def export(cfg, noise_sd, n_val):
    """Turn a tuned config into ``COMBOM(gp_config=...)`` form (one dict, or one per objective).

    Known-noise settings carry the variance measured from the design's repeats, and the
    validation size for the binomial variant, so the GP inside the BO loop matches the tuned one.
    """
    per_obj = []
    for j in range(2):
        c = dict(cfg)
        if c["noise"] in ("known", "known_binom"):
            c["known_noise_var"] = float(noise_sd[j]) ** 2
            if c["noise"] == "known_binom":
                if j == 0:
                    c["n_val"] = n_val
                else:
                    c["noise"] = "known"
        per_obj.append(c)
    return per_obj[0] if per_obj[0] == per_obj[1] else per_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument(
        "--n-val", type=int, default=None, help="validation size, if the design file lacks n_val"
    )
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--lc-sizes", default="20,40,80,120")
    ap.add_argument("--lc-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = [torch.load(f, weights_only=False) for f in a.design]
    global NAMES
    NAMES = list(parts[0].get("names", NAMES))
    n_val = parts[0].get("n_val", a.n_val)
    X = torch.cat([p["x"] for p in parts]).double()
    Y = torch.cat([p["y"] for p in parts]).double()
    n, d = X.shape
    rep_y = torch.cat(
        [p["repeat_y"] for p in parts if "repeat_y" in p and p["repeat_y"].shape[0] > 0]
    ).double()
    noise_sd = rep_y.std(dim=1, unbiased=True).mean(
        dim=0
    )  # per objective, from repeated evaluations of the same subsets
    for j, name in enumerate(NAMES):
        ratio = Y[:, j].std() / noise_sd[j].clamp_min(1e-12)
        print(
            f"design n={n} d={d}; {name}: design sd {Y[:, j].std():.4f}, noise sd {noise_sd[j]:.4f}"
            f" (repeats {tuple(rep_y.shape[:2])}), signal/noise {ratio:.2f}",
            flush=True,
        )
    # binomial noise only makes sense for a proportion (objective 0) with a known validation size
    if n_val is None or Y[:, 0].min() < 0 or Y[:, 0].max() > 1:
        FACTORS["noise"] = [v for v in FACTORS["noise"] if v != "known_binom"]
    # per-point known noise variances: homoscedastic floor ('known'); binomial p(1-p)/n_val for accuracy ('known_binom')
    YV = {}
    for j in range(2):
        YV[("known", j)] = torch.full((n,), noise_sd[j].item() ** 2, dtype=torch.double)
        YV[("known_binom", j)] = (
            (Y[:, 0] * (1 - Y[:, 0]) / (n_val or 1)) if j == 0 else YV[("known", j)]
        )

    def yv_for(cfg):
        return {
            j: YV[(cfg["noise"] if cfg["noise"] == "known_binom" else "known", j)] for j in range(2)
        }

    results_path = f"{a.out}/gp_tune_results.json"
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    def run_configs(cfgs, stage):
        todo = [
            (canon(c), j) for c in cfgs for j in range(2) if f"{key(canon(c))}#{j}" not in results
        ]
        print(
            f"[{stage}] {len(cfgs)} configs, {len(todo)} (config, objective) tasks to run",
            flush=True,
        )
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {
                ex.submit(cv_task, c, j, X, Y, yv_for(c), a.folds, a.repeats): (c, j)
                for c, j in todo
            }
            for i, fu in enumerate(as_completed(futs), 1):
                c, j = futs[fu]
                try:
                    results[f"{key(c)}#{j}"] = dict(cfg=c, objective=NAMES[j], **fu.result())
                except Exception as e:
                    results[f"{key(c)}#{j}"] = dict(
                        cfg=c, objective=NAMES[j], failed=f"{type(e).__name__}: {str(e)[:100]}"
                    )
                if i % 20 == 0 or i == len(todo):
                    json.dump(results, open(results_path, "w"))
                    print(f"  [{stage}] {i}/{len(todo)} done", flush=True)
        json.dump(results, open(results_path, "w"))

    def score(c):  # sum of per-objective CV NLPD (lower is better); failures -> inf
        vals = [results.get(f"{key(canon(c))}#{j}", {}).get("nlpd", float("inf")) for j in range(2)]
        return sum(vals)

    def ofat(center):
        out = [dict(center)]
        for f, levels in FACTORS.items():
            for v in levels:
                if v != center[f]:
                    out.append(dict(center, **{f: v}))
        return out

    # stage 1: paper + current + OFAT around current
    run_configs([PAPER, BASE] + ofat(BASE), "stage1-ofat")
    # stage 2: random sample of the full space
    rng = random.Random(a.seed)
    rand = [{f: rng.choice(levels) for f, levels in FACTORS.items()} for _ in range(a.n_random)]
    run_configs(rand, "stage2-random")
    # stage 3: OFAT around the best so far, then again around the new best (coordinate descent, 2 rounds)
    for rnd in range(2):
        allc = {key(results[k]["cfg"]): results[k]["cfg"] for k in results}
        best = min(allc.values(), key=score)
        print(f"[stage3 round {rnd}] centre: {key(best)} score {score(best):.3f}", flush=True)
        run_configs(ofat(best), f"stage3-ofat{rnd}")
    # ---- report
    allc = {key(results[k]["cfg"]): results[k]["cfg"] for k in results}
    ranked = sorted(allc.values(), key=score)
    lines = [
        f"GP tuning on design (n={n}, d={d}); {a.folds}-fold CV x {a.repeats} repeats; {len(allc)} configs; score = summed NLPD over {NAMES}\n",
        f"noise floor sd: {noise_sd.tolist()}; design sd: {Y.std(dim=0).tolist()}\n",
    ]

    def row(c, tag=""):
        parts = []
        for j in range(2):
            r = results[f"{key(c)}#{j}"]
            parts.append(
                "FAILED"
                if "failed" in r
                else f"nlpd {r['nlpd']:6.3f} rmse/c {r['rmse_vs_const']:.3f} rho {r['spearman']:.3f} cov {r['cover95']:.2f} rmse/noise {r['rmse'] / noise_sd[j].item():.2f}"
            )
        return f"{score(c):8.3f}  {NAMES[0]}[{parts[0]}]  {NAMES[1]}[{parts[1]}]  {tag}{key(c)}"

    lines.append("== top 25 by summed NLPD")
    lines += [row(c) for c in ranked[:25]]
    lines.append("\n== reference configs")
    lines.append(row(PAPER, "PAPER   "))
    lines.append(row(BASE, "CURRENT "))
    for j in range(2):
        bj = min(
            allc.values(), key=lambda c: results.get(f"{key(c)}#{j}", {}).get("nlpd", float("inf"))
        )
        lines.append(f"\n== best for {NAMES[j]} alone: " + row(bj))
    lines.append(
        "\n== marginal effect of each factor level (mean score over all evaluated configs having that level; lower is better)"
    )
    for f, levels in FACTORS.items():
        cells = []
        for v in levels:
            sc = [score(c) for c in allc.values() if c[f] == v and score(c) < float("inf")]
            cells.append(f"{v}: {np.mean(sc):.3f} (n={len(sc)})" if sc else f"{v}: -")
        lines.append(f"  {f:<12} " + " | ".join(cells))
    # learning curve for top configs + paper + current
    sizes = [int(s) for s in a.lc_sizes.split(",") if int(s) < n - 10]
    lc_cfgs = [PAPER, BASE] + [c for c in ranked[: a.top] if key(c) not in (key(PAPER), key(BASE))]
    lc = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {
            ex.submit(lc_task, c, j, X, Y, yv_for(c), m, a.lc_reps): (key(c), j, m)
            for c in lc_cfgs
            for j in range(2)
            for m in sizes
        }
        for fu in as_completed(futs):
            k, j, m = futs[fu]
            try:
                lc[f"{k}#{j}#{m}"] = fu.result()
            except Exception as e:
                lc[f"{k}#{j}#{m}"] = dict(failed=str(e)[:80])
    json.dump(lc, open(f"{a.out}/gp_tune_learning_curve.json", "w"))
    lines.append(
        f"\n== learning curve: held-out NLPD / Spearman when trained on m points (mean of {a.lc_reps} random splits), sizes {sizes}"
    )
    for c in lc_cfgs:
        tag = (
            "PAPER"
            if key(c) == key(PAPER)
            else "CURRENT"
            if key(c) == key(BASE)
            else f"rank{ranked.index(c) + 1}"
        )
        for j in range(2):
            cells = [
                f"m={m}: "
                + (
                    "FAIL"
                    if "failed" in lc[f"{key(c)}#{j}#{m}"]
                    else f"{lc[f'{key(c)}#{j}#{m}']['nlpd']:.3f}/{lc[f'{key(c)}#{j}#{m}']['spearman']:.2f}"
                )
                for m in sizes
            ]
            lines.append(f"  {tag:<8} {NAMES[j]:<18} " + "  ".join(cells))
        lines.append(f"           {key(c)}")
    json.dump(export(ranked[0], noise_sd, n_val), open(f"{a.out}/gp_config_tuned.json", "w"))
    lines.append(f"\nwrote {a.out}/gp_config_tuned.json (rank 1); pass it as COMBOM(gp_config=...)")
    open(f"{a.out}/gp_tune_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    # plots: observed vs held-out prediction for paper / current / best, and residual histogram vs noise floor
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = ranked[0]
    for j, name in enumerate(NAMES):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
        y = Y[:, j].numpy()
        for ax, (tag, c) in zip(
            axes[:3], [("paper", PAPER), ("current", BASE), ("tuned best", best)], strict=False
        ):
            r = results[f"{key(c)}#{j}"]
            if "failed" in r:
                ax.set_title(f"{tag}: failed")
                continue
            mu, sd = np.array(r["mu"]), np.array(r["sd"])
            ax.errorbar(
                y,
                mu,
                yerr=1.96 * sd,
                fmt="o",
                ms=4,
                alpha=0.7,
                ecolor="#9ecae1",
                color="#1f77b4",
                capsize=2,
            )
            lo, hi = min(y.min(), mu.min()), max(y.max(), mu.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel(f"observed {name}")
            ax.set_ylabel("held-out GP prediction (95% band)")
            ax.set_title(
                f"{tag}: NLPD {r['nlpd']:.2f} rmse/const {r['rmse_vs_const']:.2f} rho {r['spearman']:.2f} cov {r['cover95']:.2f}\n{key(c)}",
                fontsize=7,
            )
        ax = axes[3]
        r = results[f"{key(best)}#{j}"]
        res = y - np.array(r["mu"])
        ax.hist(res, bins=15, color="#1f77b4", alpha=0.75, label="held-out residuals (tuned best)")
        xs = np.linspace(res.min(), res.max(), 200)
        ax.plot(
            xs,
            len(res) * (xs[1] - xs[0]) * norm.pdf(xs, 0, noise_sd[j].item()),
            "r-",
            label=f"eval noise sd={noise_sd[j]:.4f}",
        )
        ax.set_xlabel(f"residual ({name})")
        ax.legend(fontsize=8)
        ax.set_title("distance of each point from the fitted curve", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{a.out}/gp_tune_fit_{name}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, name in enumerate(NAMES):
        ax = axes[j]
        for c in lc_cfgs:
            tag = (
                "paper"
                if key(c) == key(PAPER)
                else "current"
                if key(c) == key(BASE)
                else f"rank{ranked.index(c) + 1}"
            )
            ys = [lc[f"{key(c)}#{j}#{m}"].get("nlpd", np.nan) for m in sizes]
            ax.plot(
                sizes,
                ys,
                marker="o",
                label=tag,
                lw=2 if tag in ("paper", "current", "rank1") else 1,
            )
        ax.set_xlabel("# training points")
        ax.set_ylabel("held-out NLPD")
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{a.out}/gp_tune_learning_curve.png", dpi=140)
    print(
        f"wrote {a.out}/gp_tune_report.txt, gp_tune_results.json, gp_tune_fit_*.png, gp_tune_learning_curve.png",
        flush=True,
    )


if __name__ == "__main__":
    main()
