# Reference: GP options and the GSM8K worked example

## GP configuration keys (`dspy_combom/gp.py`, `tools/gp_tune.py`)

All configurations are a BoTorch `SingleTaskGP` with a `CategoricalKernel` over the 0/1 subset
vector and a `Standardize` outcome transform. Pass one dict, or a list of two (one per objective),
as `COMBOM(gp_config=...)`.

| key | values | meaning |
| --- | --- | --- |
| `ard` | `true` / `false` | one lengthscale per pool example, or a shared one |
| `ls_prior` | `lognormal` / `saas` / `none` | dimension-scaled LogNormal (BoTorch default, the paper's prior), MAP-SAAS shrinkage, or no prior |
| `ls_shift`, `ls_scale` | floats | LogNormal location `1.4 + 0.5·log d + ls_shift` and scale (paper: shift 0, scale √3) |
| `ls_bounds` | `"lo,hi"` | lengthscale constraint |
| `outputscale` | `paper` / `gamma2` / `noprior` / `fixed1` | Gamma(1.1, 0.2), Gamma(2, 0.15), unconstrained, or no ScaleKernel |
| `mean` | `constant` / `zero` | GP mean |
| `noise` | `fixed:<v>` / `gamma` / `ln<loc>` / `known` / `known_binom` | fixed variance in standardized units (paper `fixed:1e-3`), Gamma prior, LogNormal(loc, 1) prior, variance measured from repeats, or binomial p(1−p)/n for accuracy |
| `restarts` | int | MAP restarts from prior samples; the best marginal likelihood is kept |
| `kernel` | `categorical` / `tanimoto` | Tanimoto (set overlap, no lengthscale) is an alternative |

`gp_tune.py` writes `known_noise_var` and `n_val` into the exported config when a known-noise
option wins, so the GP inside the BO loop matches the tuned one.

Why these knobs: the categorical kernel averages Hamming mismatches over the `d` pool bits. With
the paper's prior, the median lengthscale is about 23 at d = 32, which makes every pair of subsets
about 0.98 correlated. Shifting the prior down lets the GP distinguish subsets.

## GSM8K, Llama-3.1-8B base (the worked example)

**Signal-to-noise**
- With 300 validation questions, the spread across subsets (sd 0.0205 accuracy, 0.0211 ECE) was
  close to the spread between repeats of one subset (0.0141, 0.0182). No GP configuration
  predicted held-out points.
- With 1000 questions, repeat noise fell to 0.0087 (accuracy) and 0.0058 (ECE), and the GP became
  predictive. Most of the apparent spread at 300 questions was noise.
- With the predefined gold-exemplar pool there was about one point of real structure per
  objective. An earlier 32-demo pool, bootstrapped by a custom script that kept any correct
  sample, had roughly six times that.

**Design:** 119 half-probability subsets plus 36 subsets stratified by demo count, and 5 subsets
re-scored 3 times. All were scored on 1000 questions.

**Tuning (367 configurations, 10-fold cross-validation × 2 repeats):**
- The noise model is the only decisive choice. Fixed 1e-3 (the paper's setting) scored held-out
  NLPD −2.33 on accuracy with coverage 0.75-0.83. Any inferred or known noise model reached about
  −3.2.
- SAAS and no-prior (maximum likelihood) were clearly worse than the LogNormal prior.
- The winner was `ls_shift −2`, `ls_scale 0.5`, `noise ln-1`, `outputscale gamma2`,
  `mean constant`, 1 restart (`configs/gsm8k/gp_goldpool.json`).
- Its held-out fit on accuracy: RMSE/constant 0.76, Spearman 0.69, coverage 0.96, residual RMSE
  1.06 × the noise floor. On ECE: Spearman 0.36 and residual RMSE 1.7 × the noise floor, so ECE
  is only weakly predictable.
- The gain concentrates where BO starts. With 20 training points, held-out NLPD was −2.93 / −2.87
  (accuracy / ECE) against −2.35 / −1.70 for the setting in use before the search (the paper's
  lengthscale prior at scale 1.0, with noise inferred under a Gamma prior).

**Kernel families:**
- Categorical beat Matérn, RBF, linear, polynomial and count-additive kernels, and was the most
  robust to its hyperparameters.
- Tanimoto was marginally better on GSM8K (NLPD −3.33 vs −3.24 on accuracy) and on MMLU-Pro.
- On MMLU-Pro trajectories, the paper's fixed noise gave held-out NLPD +73 and coverage 0.66,
  against −3.2 with inferred noise. The noise model matters far more than the kernel.

**Transfer to other pools:**
- On a 100-subset random design over the custom-script 32-demo pool, the gold-tuned configuration
  was adequate (Spearman 0.84 / 0.69, coverage about 0.95).
- A configuration tuned directly on that design gained only about 0.1 NLPD, so the gold-tuned
  configuration was kept.
- Cross-validating on BO trajectories instead inflated Spearman to 0.97.
- The reported GSM8K results use a different pool: 20 demos built by DSPy's `BootstrapFewShot`
  (`COMBOM(bootstrap=True)`, `configs/gsm8k/pool_dspy_bootstrap_llama31_8b.json`). Those runs
  reused `gp_goldpool.json` without re-validating it on a design over that pool. Run steps 1-4 on
  that pool to justify the configuration there.
