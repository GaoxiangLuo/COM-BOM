---
name: combom-gp-tuning
description: Tune and justify the GP surrogate hyperparameters of COM-BOM (categorical-kernel GP over exemplar subsets) on a task. Collects a random design of exemplar subsets, checks signal against evaluation noise, cross-validates GP configurations, and produces the evidence (noise floor, held-out fit, learning curve, comparison with the paper's settings) behind the chosen gp_config. Use when setting or defending COM-BOM's GP hyperparameters for a benchmark, model or exemplar pool.
---

# Tuning and justifying COM-BOM's GP

COM-BOM's sample efficiency depends on its GP surrogate. The paper's settings (dimension-scaled
LogNormal lengthscale prior with scale √3, fixed observation noise 1e-3) were set on MMLU-Pro.
They are hyperparameters, not constants. On a new benchmark, model or exemplar pool they must be
tuned, and the chosen values must be backed by evidence. This skill produces both: a
`gp_config_tuned.json` for `COMBOM(gp_config=...)`, and a report that justifies it.

The kernel stays the paper's `CategoricalKernel`; only its hyperparameters are tuned.
`reference.md` defines every option and gives the GSM8K numbers as a worked example.

Run from `integrations/dspy/` after `uv sync`, with the repository root and `integrations/dspy`
on `PYTHONPATH`, and the model served by vLLM (see `integrations/dspy/README.md`).

## 1. Collect a random design

Score random exemplar subsets exactly as the optimizer will score them, and re-score a few to
measure evaluation noise:

```python
from dspy_combom import COMBOM
from dspy_combom.design import collect_design

opt = COMBOM(metric=metric, fixed_demos=fixed, num_threads=64)   # the settings you will optimise with
collect_design(opt, student, trainset=pool, valset=valset,
               n_iid=60, n_stratified=20, n_repeat=5, reps=3, seed=0, out="design_seed0.pt")
```

- Use the same program, pool, validation set and objectives as the COM-BOM run you are tuning
  for. A design from a different pool or validation size is a different landscape.
- The design mixes half-probability subsets (about half the pool each) with subsets of uniformly
  random size. Without the second kind, the GP never sees how the objectives change with the
  number of demos.
- The cost is `n_iid + n_stratified + n_repeat × (reps − 1)` full validation evaluations (90 in
  the example). Run it as a detached job; it saves after every evaluation and resumes from `out`.
- Aim for 100-150 points in total by collecting several seeds; the tuner merges design files.
- **Never tune or validate on BO trajectories.** They cluster around good subsets and make every
  GP look predictive. On GSM8K, trajectory cross-validation gave Spearman 0.97, against 0.84 on a
  random design.

## 2. Check that there is signal to model

`collect_design` logs, per objective, the spread across subsets, the spread between repeats of
the same subset (the noise floor) and their ratio. If the ratio is around 1 or below, differences
between subsets are mostly evaluation noise. No GP setting can be justified on that data. Raise
the validation size (noise shrinks like 1/√n) and collect again. On GSM8K, 300 validation
questions were not enough and 1000 were.

## 3. Run the search

```bash
uv run python tools/gp_tune.py --design design_seed*.pt --out gp_tune/ --workers 24
```

This runs on CPU only. It scores each configuration by held-out NLPD from repeated k-fold
cross-validation, and also reports RMSE relative to a constant predictor, Spearman, 95% coverage
and RMSE relative to the noise floor. The search always includes the paper's configuration
(`PAPER`) as a reference. It runs in three stages:

1. One factor at a time around a baseline.
2. Random configurations from the full space.
3. Two rounds of refinement around the best configuration.

Outputs in `gp_tune/`:

| file | contents |
| --- | --- |
| `gp_tune_report.txt` | top configurations, the paper and baseline rows, the best per objective, the average effect of each option, and the learning curve |
| `gp_config_tuned.json` | rank 1, ready for `COMBOM(gp_config=...)`; known-noise variants carry the measured variance |
| `gp_tune_fit_<objective>.png` | held-out prediction against observation for the paper, baseline and tuned configurations, and residuals against the noise floor |
| `gp_tune_learning_curve.png` | held-out NLPD against the number of training points |
| `gp_tune_results.json` | every configuration's scores; rerunning resumes from it |

## 4. Accept or reject the winner

Rank 1 is justified only if all of these hold:

- **Calibrated:** 95% held-out coverage is between 0.9 and 0.98.
- **Predictive:** held-out Spearman on accuracy is clearly positive (GSM8K: about 0.7). ECE is
  usually harder to predict (GSM8K: about 0.36); if both are near 0, go back to step 2.
- **At the noise floor:** held-out residual RMSE is about 1× the noise floor. A much higher value
  means the GP misses structure. A much lower one suggests the noise floor is underestimated,
  for example from too few repeats.
- **Better where BO runs:** on the learning curve at 20-40 training points, where BO starts, it
  beats the paper's configuration. Full-data cross-validation often cannot separate reasonable
  configurations; the learning curve can.
- **Not a lucky draw:** its lead over the runner-up exceeds the NLPD spread across CV repeats
  (`nlpd_sd` in `gp_tune_results.json`). If several configurations tie, prefer the one closest
  to the paper's settings and state that they tie.

## 5. Write the justification

Report, for each tuned value:

1. The design: number of points, how they were sampled, validation size, and the noise floor
   against the spread across subsets.
2. The comparison with the paper's configuration: held-out NLPD, coverage and Spearman for both,
   from full cross-validation and at 20-40 training points.
3. Which options mattered: the average-effect-per-option section of the report shows the
   decisive options and the ones that tie. Most choices usually tie; say so rather than claiming
   each value was tuned precisely.
4. The fit plots and the learning curve.
5. Optional kernel check: compare kernel families on the same design
   (`benchmarks/gsm8k/gp_kernels.py`) to show the categorical kernel is not the bottleneck.

On the benchmarks measured so far, the observation-noise model is the one decisive choice. The
paper's fixed 1e-3 noise is overconfident on GSM8K and even on held-out MMLU-Pro points, while any
inferred or measured noise model works. The lengthscale prior shift matters mainly at small
training sizes. `reference.md` gives the numbers.
