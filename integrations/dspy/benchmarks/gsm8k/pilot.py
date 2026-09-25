"""GSM8K benchmark: COM-BOM against DSPy's exemplar optimizers, all under DSPy's defaults.

Every method optimizes the same program (SelfConsistency: 16 samples, majority answer, vote share as
confidence) on the same 32 training questions and validation set, with the same budget of validation
evaluations. The chosen program (and, for COM-BOM, every Pareto-front program) is scored on the test set.

    python pilot.py --method {combom,random_search,optuna,labeled,none} --seed 0

Methods:
  combom         COMBOM over a bootstrapped pool (--pool dspy, what COMBOM(bootstrap=True) builds) or the
                 predefined gold demos (--pool gold)
  random_search  dspy BootstrapFewShotWithRandomSearch with DSPy's defaults, selected by accuracy - ECE
  optuna         dspy BootstrapFewShotWithOptuna with DSPy's defaults, selected by accuracy - ECE
  labeled        dspy LabeledFewShot (k=16), no search
  none           only the fixed format demo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import _paths
import dspy
from data import GSM8K, parse_integer_answer
from dspy.datasets.gsm8k import gsm8k_metric

from dspy_combom import COMBOM, DEFAULT_OBJECTIVES, SelfConsistency
from dspy_combom.adapters import PlainTextAdapter
from dspy_combom.lm import CompletionLM

CONFIGS = Path(_paths.INTEGRATION_ROOT) / "configs" / "gsm8k"
POOLS = {"dspy": CONFIGS / "pool_dspy_bootstrap_llama31_8b.json", "gold": None}
N_SAMPLES, TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS = 16, 0.7, 1.0, -1, 512
MAX_BOOTSTRAPPED, MAX_LABELED = 4, 16  # DSPy's defaults for its few-shot optimizers
MAX_ERRORS = 10**9  # a failed or unparseable item scores 0 instead of aborting the evaluation
metric = gsm8k_metric  # DSPy's GSM8K metric: exact match on the parsed final integer


class GSM8KCoT(dspy.Signature):
    """Solve the math word problem. Think step by step, then give the final numeric answer."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField(desc="the final integer answer only")


def numeric_key(value):
    """Cluster key for majority voting: the parsed integer, or None if no number is present."""
    text = str(value or "")
    if not any(c.isdigit() for c in text):
        return None
    return str(parse_integer_answer(text))


def configure(port, model):
    """Base-model completion endpoint with a plain-text prompt format."""
    lm = CompletionLM(
        model,
        api_base=f"http://localhost:{port}/v1",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stop=["\nQuestion:"],
    )
    labels = {"question": "Question:", "reasoning": "Reasoning:", "answer": "The answer is:"}
    adapter = PlainTextAdapter(
        labels=labels, output_template="{label} {value}\n", query_suffix="{label} "
    )
    dspy.configure(lm=lm, adapter=adapter)
    return lm


def make_student():
    return SelfConsistency(
        GSM8KCoT,
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        field="answer",
        normalize=numeric_key,
    )


def load_data(n_val, n_test):
    """Format demo, the 32 training questions (gold reasoning), validation and test sets."""
    ds = GSM8K()
    mk = lambda e: dspy.Example(
        question=e.question, reasoning=e.gold_reasoning, answer=e.answer
    ).with_inputs("question")
    qa = lambda e: dspy.Example(question=e.question, answer=e.answer).with_inputs("question")
    return (
        mk(ds.train[0]),
        [mk(e) for e in ds.train[1:33]],
        [qa(e) for e in ds.dev[:n_val]],
        [qa(e) for e in ds.test[:n_test]],
    )


def load_pool(name, questions):
    """The exemplar pool COM-BOM searches: a bootstrapped pool file, or the gold questions."""
    if not POOLS[name]:
        return questions
    return [
        dspy.Example(**{k: p[k] for k in ("question", "reasoning", "answer")}).with_inputs(
            "question"
        )
        for p in json.load(open(POOLS[name]))
    ]


def install_scalarized_evaluate(lam: float, log_dir: Path):
    """Make DSPy's optimizers select programs by 100 * (accuracy - lam * ECE) over the whole valset.

    DSPy's optimizers build `Evaluate` from their own module namespace and read `.score`, so the name is
    swapped there. Every evaluation is appended to evaluations.jsonl and the best program so far is saved to
    best_program.json.
    """
    import dspy.teleprompt.random_search as rs_mod
    import dspy.teleprompt.teleprompt_optuna as opt_mod

    acc_obj, ece_obj = DEFAULT_OBJECTIVES
    state = {"n": 0, "best": float("-inf")}
    lock = threading.Lock()

    class ScalarizedEvaluate(dspy.Evaluate):
        def __call__(self, program, *args, **kwargs):
            res = super().__call__(program, *args, **kwargs)
            acc, ece = acc_obj(res.results), ece_obj(res.results)
            score = 100.0 * (acc - lam * ece)
            with lock:
                state["n"] += 1
                rec = dict(evaluation=state["n"], accuracy=acc, calibration_error=ece, score=score)
                with open(log_dir / "evaluations.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if score > state["best"]:
                    state["best"] = score
                    program.save(log_dir / "best_program.json")
            return dspy.evaluate.evaluate.EvaluationResult(
                score=round(score, 2), results=res.results
            )

    for m in (rs_mod, opt_mod):
        m.Evaluate = ScalarizedEvaluate


def score(program, dataset, threads):
    ev = dspy.Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=threads,
        display_progress=False,
        max_errors=MAX_ERRORS,
        failure_score=0.0,
    )
    res = ev(program)
    return {o.name: o(res.results) for o in DEFAULT_OBJECTIVES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method", choices=["combom", "random_search", "optuna", "labeled", "none"], required=True
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=100, help="validation evaluations per search")
    ap.add_argument(
        "--pool", choices=list(POOLS), default="dspy", help="combom: exemplar pool to search"
    )
    ap.add_argument("--n-initial", type=int, default=20, help="combom: random initial evaluations")
    ap.add_argument(
        "--gp-config",
        default=str(CONFIGS / "gp_goldpool.json"),
        help="combom: GP config JSON file ('' for the paper's GP)",
    )
    ap.add_argument("--lam", type=float, default=1.0, help="baselines select by accuracy - lam*ECE")
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1319)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", default=_paths.RUNS_DIR)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for name in ("httpx", "LiteLLM", "dspy.evaluate.evaluate"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # one fixed demo that shows the output format, and the 32 training questions every method draws on
    format_demo, questions, valset, testset = load_data(a.n_val, a.n_test)
    configure(a.port, a.model)
    student = make_student()

    name = a.method + (f"_pool{a.pool}" if a.method == "combom" else "")
    out = Path(a.out) / f"{a.model.split('/')[-1].lower()}_{name}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluations.jsonl").unlink(missing_ok=True)
    print(f"method={a.method} val={len(valset)} test={len(testset)} -> {out}", flush=True)

    t0 = time.time()
    extra = {}
    bootstrap = dict(
        metric=metric,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED,
        max_labeled_demos=MAX_LABELED,
        num_threads=a.threads,
    )
    if a.method == "none":
        program = student.reset_copy()
        program.predict.demos = [format_demo]
    elif a.method == "labeled":
        program = dspy.LabeledFewShot(k=MAX_LABELED).compile(student, trainset=questions)
        program.predict.demos = [format_demo] + list(program.predict.demos)
    elif (
        a.method == "random_search"
    ):  # evaluates budget - 3 candidates plus its 3 built-in programs
        install_scalarized_evaluate(a.lam, out)
        opt = dspy.BootstrapFewShotWithRandomSearch(
            num_candidate_programs=a.budget - 3, max_errors=MAX_ERRORS, **bootstrap
        )
        program = opt.compile(student, trainset=[format_demo] + questions, valset=valset)
    elif a.method == "optuna":  # one trial per evaluation
        install_scalarized_evaluate(a.lam, out)
        opt = dspy.BootstrapFewShotWithOptuna(num_candidate_programs=a.budget, **bootstrap)
        program = opt.compile(
            student,
            trainset=[format_demo] + questions,
            valset=valset,
            max_demos=MAX_BOOTSTRAPPED,
        )
    else:
        pool = load_pool(a.pool, questions)
        gp_config = json.load(open(a.gp_config)) if a.gp_config else None
        opt = COMBOM(
            metric=metric,
            n_initial_pts=a.n_initial,
            num_trials=a.budget,
            num_threads=a.threads,
            seed=a.seed,
            fixed_demos=[format_demo],
            max_errors=MAX_ERRORS,
            gp_config=gp_config,
            ref_point=(0.0, 1.0),
            checkpoint_path=str(out / "checkpoint.pt"),
        )
        program = opt.compile(student, trainset=pool, valset=valset)
        extra["history"] = program.combom_history
        extra["pareto_test"] = [
            dict(
                iteration=p["iteration"],
                val=p["objectives"],
                n_demos=sum(p["x"]),
                test=score(p["program"], testset, a.threads),
            )
            for p in program.pareto_programs
        ]
    search_s = time.time() - t0
    test = score(program, testset, a.threads)
    n_demos = len(program.predict.demos)
    print(
        f"RESULT {a.method} test acc={test['accuracy']:.4f} ece={test['calibration_error']:.4f} "
        f"demos={n_demos} search={search_s / 60:.1f} min",
        flush=True,
    )
    program.save(out / "program.json")
    json.dump(
        {"args": vars(a), "test": test, "n_demos": n_demos, "search_s": search_s, **extra},
        open(out / "result.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
