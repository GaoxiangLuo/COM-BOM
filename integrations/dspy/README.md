# COM-BOM for DSPy

COM-BOM as a DSPy `Teleprompter`. It searches over exemplar subsets with multi-objective Bayesian
optimization and returns the Pareto front over accuracy and expected calibration error (ECE). Every
optimizer DSPy ships maximizes a single scalar metric and returns one program.

This directory is a separate uv project on current DSPy, BoTorch and GPyTorch. It reuses the
release `combom` package from the repository root unchanged, so the paper's code and environment
are unaffected.

## Setup

```bash
cd integrations/dspy
uv sync
```

Tested with Python 3.12, `dspy==3.3.1`, `botorch==0.18.1`, `gpytorch==1.15.2`, `torch==2.14.0` and
`optuna==5.0.0` (see `uv.lock`). The project is not an installable package, so put the repository
root (for `combom`) and this directory (for `dspy_combom`) on `PYTHONPATH`:

```bash
export PYTHONPATH=$(git rev-parse --show-toplevel):$(git rev-parse --show-toplevel)/integrations/dspy
```

The benchmark scripts under `benchmarks/gsm8k/` set this up themselves.

## Usage

```python
import dspy
from dspy_combom import COMBOM, SelfConsistency

class CoT(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()

# majority vote over n samples; `confidence` = majority fraction, which the ECE objective reads
student = SelfConsistency(CoT, n=16, temperature=0.7, field="answer")

# bootstrap=True: build the exemplar pool from labeled questions with DSPy's BootstrapFewShot
optimizer = COMBOM(metric=exact_match, bootstrap=True, n_initial_pts=20, num_trials=100, num_threads=64)
program = optimizer.compile(student, trainset=questions, valset=valset)

program.pareto_programs  # the accuracy / ECE front, each entry a compiled program
program.combom_history   # every evaluated subset with both objectives
program.combom_pool      # the bootstrapped exemplar pool that was searched
```

With `bootstrap=True`, the pool is built the same way DSPy's few-shot optimizers build demos:
the program, prompted with 16 labeled examples, answers each training question once, and the
answers that pass the metric become candidate demos. Without it, `trainset` is used as the pool
as given. `compile` returns the front point with the largest hypervolume contribution. Pass `gp_config` to
use a GP tuned for your task (see [GP tuning](#gp-tuning)).

| module | contents |
| --- | --- |
| `dspy_combom/teleprompter.py` | `COMBOM`, the DSPy optimizer |
| `dspy_combom/bo_loop.py` | the COM-BOM loop from `train.py`, without the MMLU-Pro task and YAML plumbing |
| `dspy_combom/acquisition.py` | qNEHVI with the paper's Sobol sampler pinned (BoTorch >= 0.16 defaults to IID samples) |
| `dspy_combom/gp.py` | a config-driven `CategoricalKernel` GP for per-benchmark tuning |
| `dspy_combom/pool.py` | `bootstrap_pool`, the exemplar pool built with DSPy's `BootstrapFewShot` |
| `dspy_combom/design.py` | random designs of exemplar subsets for GP tuning |
| `tools/gp_tune.py` | cross-validated GP hyperparameter search on a design |
| `dspy_combom/modules.py` | `SelfConsistency`, which exposes a `confidence` field |
| `dspy_combom/objectives.py` | whole-set accuracy and ECE objectives |
| `dspy_combom/lm.py`, `adapters.py` | a raw-completion LM and adapter for base models |

## GSM8K benchmark

We compare COM-BOM with DSPy's own exemplar optimizers on GSM8K, using `meta-llama/Llama-3.1-8B`
(base model). The comparison is designed to be fair rather than to show COM-BOM at its best: every
method runs with DSPy's defaults, on the same 32 training questions and the same 1000-question
validation set, with the same budget of 100 validation evaluations and three seeds. Every program
is scored with self-consistency over 16 samples at temperature 0.7, and the chosen programs are
scored on the full 1319-question test set.

COM-BOM uses `COMBOM(bootstrap=True)`, which builds its pool with DSPy's `BootstrapFewShot`. The
DSPy optimizers bootstrap their own demos from the same questions. Three settings differ from pure
DSPy defaults, and they apply to every method:

- **Budget.** Each method gets 100 validation evaluations. `BootstrapFewShotWithRandomSearch`
  evaluates 97 candidates plus its 3 built-in programs.
- **Selection score.** The task has two objectives, accuracy and ECE. COM-BOM optimizes both. The
  DSPy optimizers maximize one number, so they select programs by `accuracy - ECE` instead of
  accuracy alone.
- **Demo traces.** Bootstrapped demos record the majority answer that passed the metric (see
  [Notes on DSPy compatibility](#notes-on-dspy-compatibility)).

| method | test accuracy | test ECE | accuracy - ECE | validation hypervolume | demos |
| --- | --- | --- | --- | --- | --- |
| **COM-BOM**, `bootstrap=True` | 63.7 | **19.1** | **44.6** | **134** (13) | 8-11 |
| DSPy `BootstrapFewShotWithRandomSearch` | 63.2 | 20.3 | 42.9 | 101 (7) | 16 |
| DSPy `BootstrapFewShotWithOptuna` | 65.1 | 22.8 | 42.3 | 88 (8) | 1 |
| DSPy `LabeledFewShot` (k=16, no search) | 60.6 | 20.7 | 39.9 | - | 17 |
| no exemplars | 60.0 | 22.3 | 37.8 | - | - |
| COM-BOM on the predefined gold demos | 63.5 | 20.0 | 43.5 | 118 (11) | 13-18 |

Test columns are means over three seeds, except for the two single-program rows. For COM-BOM, the
reported program is the Pareto-front point with the best validation `accuracy - ECE`. That score
favors the DSPy optimizers, which maximize it directly, while COM-BOM optimizes the whole front.
Every COM-BOM seed (44.1-45.0) scores above every seed of either DSPy optimizer (40.8-43.5).
COM-BOM also uses about half as many demos as random search, so its prompts are shorter. Its test
front spans 53-67 accuracy and 15-21 ECE.

Hypervolume is computed over all 100 programs a method evaluated on validation, in squared
percentage points (accuracy points times ECE points). Larger means the programs cover more of the accuracy / ECE plane. The reference point is
accuracy 50 / ECE 25. The value in parentheses uses the no-exemplar program as the reference
(61.8 / 23.9 on validation). Because that reference has higher accuracy than most exemplar
programs, only a few points count and the values vary more across seeds. COM-BOM still ranks first.

The last row runs COM-BOM on the 32 predefined gold demos instead of a bootstrapped pool. These
are the first 100 evaluations of earlier 200-evaluation runs, and they are reported for reference.

![sample efficiency](../../assets/sample_efficiency_gsm8k.png)

The figure plots the best-so-far validation `accuracy - ECE` against the number of evaluations,
as a mean over three seeds with the min-max band. COM-BOM's first 20 evaluations are random, and
during them all methods sit near 42. Once the Bayesian optimization starts, COM-BOM passes 44
at evaluation 26 and 45 at evaluation 47, and ends at 45.6. The DSPy optimizers stay near
43 (random search 43.2, Optuna 42.8).

### Reproducing

Serve the model from the repository root, whose environment pins vLLM:

```bash
uv run vllm serve meta-llama/Llama-3.1-8B --port 8000 --enable-prefix-caching \
    --data-parallel-size 4 --max-model-len 8192
```

Then run the following from `integrations/dspy/benchmarks/gsm8k`. `pilot.py`'s defaults are the
setup above (Llama-3.1-8B base, 1000 validation questions, the full test set, a budget of 100
evaluations, DSPy's defaults for the DSPy optimizers, and the tuned GP for COM-BOM). Outputs go to
`<repo>/runs/dspy-combom-gsm8k/`; set `COMBOM_RUNS_DIR` to write them elsewhere.

```bash
for seed in 0 1 2; do
  uv run python pilot.py --method combom --seed $seed                # DSPy-bootstrapped pool
  uv run python pilot.py --method combom --pool gold --seed $seed    # predefined gold demos
  uv run python pilot.py --method random_search --seed $seed
  uv run python pilot.py --method optuna --seed $seed
done
uv run python pilot.py --method labeled
uv run python pilot.py --method none
uv run python summarize.py
```

Each run scores its chosen program on the test set; COM-BOM runs also score every point of their
Pareto front. `summarize.py` prints the table above. The pool in
`configs/gsm8k/pool_dspy_bootstrap_llama31_8b.json` was built by `bootstrap_pool.py`, which runs
DSPy's `BootstrapFewShot` exactly as `COMBOM(bootstrap=True)` does. Because bootstrapping samples
from the model, a rebuilt pool can differ.

### GP tuning

The GP hyperpriors are tuned per benchmark on a random design, not on optimizer trajectories. The
observation-noise model matters most: the paper's fixed noise of 1e-3 is overconfident on GSM8K.
`dspy_combom.design.collect_design` scores random subsets with a `COMBOM` optimizer's own evaluator,
and `tools/gp_tune.py` cross-validates GP configurations on the result and writes
`gp_config_tuned.json` for `COMBOM(gp_config=...)`. Both work for any task. On GSM8K,
`benchmarks/gsm8k/collect_design.py` collected the design and `benchmarks/gsm8k/gp_kernels.py`
compared kernel families.

For a coding agent, the `combom-gp-tuning` skill in `.claude/skills/` walks through tuning the GP
on a new task and writing up the evidence behind the chosen configuration.

| config | tuned on | used for |
| --- | --- | --- |
| `configs/gsm8k/gp_goldpool.json` | random design over the predefined pool | all reported COM-BOM runs |

## Notes on DSPy compatibility

The optimizer, scoring, pool bootstrapping, majority vote and GSM8K metric use DSPy's own APIs. A few
things DSPy does not support directly are handled as follows.

- **Whole-set objectives.** DSPy metrics score one example at a time, and `dspy.Evaluate` reports
  their mean. ECE is only defined over a whole set, so `COMBOM` runs `dspy.Evaluate` and computes
  both objectives from its per-example results (`dspy_combom/objectives.py`). Accuracy is the same
  number `dspy.Evaluate` reports.
- **Baselines selected by `accuracy - ECE`.** DSPy's optimizers rank candidates by
  `dspy.Evaluate`'s mean score, which has no hook for a whole-set score. The benchmark therefore
  replaces the `Evaluate` class inside those optimizers' modules
  (`benchmarks/gsm8k/pilot.py:install_scalarized_evaluate`). This is benchmark code only; using
  `COMBOM` does not need it.
- **Demos from multi-sample predictors.** `dspy.Predict(n=16)` records all 16 samples in the
  trace, and `BootstrapFewShot` turns a trace into a demo using the first sample, even when the
  majority answer that passed the metric came from other samples. The demo can then carry a
  wrong answer. `SelfConsistency` replaces its trace entry with the majority completion, so
  bootstrapped demos match what the metric accepted. The same issue affects plain DSPy programs
  that combine `dspy.Predict(n=...)` with `dspy.majority`.
- **GSM8K loader.** `benchmarks/gsm8k/data.py` copies `dspy.datasets.gsm8k.GSM8K` with two changes:
  it loads the dataset as `openai/gsm8k`, because current `huggingface_hub` rejects DSPy's
  `gsm8k` id, and it extends the validation split from 300 to 1000 questions. With
  `--n-val 300` or less, the validation set is identical to DSPy's.
- **Base-model LM.** `dspy.LM(model_type="text")` adds `BEGIN RESPONSE:` to the prompt, so the
  benchmark uses `dspy_combom.CompletionLM`, a `dspy.BaseLM` subclass that sends the prompt as the
  adapter wrote it. DSPy 3.4 deprecates this way of subclassing `BaseLM` in favor of LM engines;
  `CompletionLM` will need porting when that release is out.
