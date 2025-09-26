"""Train the COM-BOM exemplar selection pipeline from a YAML config.

This script loads a configuration file, validates the requested task,
and runs the trust-region multi-objective Bayesian optimisation loop.
Use the files under `configs/` as templates for your own experiments.
"""

import argparse
import warnings
from pathlib import Path

import torch
import yaml
from botorch.models.model import ModelList
from tqdm.auto import tqdm

from combom import (
    TrustRegionManager,
    get_acquisition_function,
    get_fitted_gp_model,
    optimize_acqf_discrete_local_search,
)
from tasks import BaseTask
from utils import set_random_seed

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "mmlupro.yaml"


def _build_task_kwargs(cfg: dict[str, object]) -> dict[str, object]:
    """Extract keyword arguments for constructing the selected task."""
    task_kwargs: dict[str, object] = {
        "n": int(cfg["n_samples"]),
        "model": cfg["hf_model"],
        "temperature": float(cfg["temperature"]),
        "port": int(cfg["port"]),
        "num_workers": int(cfg["num_workers"]),
    }
    task_value = cfg.get("task")
    if task_value is not None:
        task_kwargs["task"] = task_value
    return task_kwargs


def main(cfg: dict[str, object]) -> None:
    """Run the COM-BOM Bayesian optimisation loop for a given configuration."""
    if cfg["dataset"] not in BaseTask.registry:
        available = ", ".join(sorted(BaseTask.registry)) or "<none>"
        raise ValueError(f"Unknown dataset {cfg['dataset']!r}. Available datasets: {available}")

    task_kwargs = _build_task_kwargs(cfg)
    evaluation_task = BaseTask.create(cfg["dataset"], **task_kwargs)

    model_basename = cfg["hf_model"].split("/")[-1].replace("-", "_")
    evaluation_split = "validation"

    n_initial_pts = int(cfg["n_initial_pts"])
    total_iterations = int(cfg["n_bo_iterations"])
    seed = int(cfg["seed"])
    n_samples = int(cfg["n_samples"])
    temperature = float(cfg["temperature"])

    n_bo_iterations = total_iterations - n_initial_pts

    if n_bo_iterations < 0:
        raise ValueError("n_bo_iterations must be greater than or equal to n_initial_pts")

    tkwargs = {"dtype": torch.float64, "device": "cuda"}

    generator = torch.Generator()
    generator.manual_seed(seed)
    initial_x = torch.randint(
        0,
        2,
        (n_initial_pts, evaluation_task.decision_dim),
        dtype=torch.float64,
        generator=generator,
    )
    metric_weights = evaluation_task.metric_weights(tkwargs=tkwargs)
    n_metrics = len(evaluation_task.metrics)

    initial_completed = 0
    bo_completed = 0

    with tqdm(
        total=total_iterations,
        desc="Initial rounds",
        unit="round",
        dynamic_ncols=True,
    ) as progress:

        def update_postfix(initial_done: int, completed_bo: int) -> None:
            postfix = {
                "initial": f"{initial_done}/{n_initial_pts}" if n_initial_pts else "0/0",
                "bo": f"{completed_bo}/{n_bo_iterations}" if n_bo_iterations else "0/0",
            }
            progress.set_postfix(postfix, refresh=False)

        update_postfix(initial_completed, bo_completed)
        log = progress.write

        initial_y_rows: list[torch.Tensor] = []
        for init_idx in range(n_initial_pts):
            progress.set_description(f"Initial {init_idx + 1}/{n_initial_pts}")
            candidate = initial_x[init_idx : init_idx + 1]
            candidate_y = evaluation_task.evaluate(candidate, split=evaluation_split)
            initial_y_rows.append(candidate_y)
            initial_completed += 1
            update_postfix(initial_completed, bo_completed)
            progress.update()

        if initial_y_rows:
            initial_y = torch.cat(initial_y_rows, dim=0)
        else:
            initial_y = torch.empty((0, n_metrics), dtype=torch.float64)

        train_x = initial_x.to(**tkwargs)
        train_y = initial_y.to(**tkwargs)

        for itr_no in range(n_bo_iterations):
            progress.set_description(f"BO iter {itr_no + 1}/{n_bo_iterations}")
            log(f"--- BO Iteration: {itr_no} ---")
            log("-" * 50)

            transformed_y = metric_weights * train_y
            if itr_no == 0:
                best_value = [
                    transformed_y[:, idx].max().item() for idx in range(transformed_y.shape[1])
                ]
                tr_manager = TrustRegionManager(
                    name="tr_1",
                    n_categorical=evaluation_task.decision_dim,
                    init_radius=min(10, evaluation_task.decision_dim),
                    best_value=best_value,
                )
                log(f"Trust region state: {tr_manager.__dict__}")
                tr_manager.initialize_center(train_x=train_x, train_y=transformed_y, **tkwargs)
            else:
                tr_manager.adjust_radius(
                    new_x=train_x[-1],
                    new_value=metric_weights * train_y[-1],
                )

            log(f"Data loaded - Shape: {train_x.shape}, {train_y.shape}")

            # Log the best value seen so far for each objective
            for metric_idx, metric in enumerate(evaluation_task.metrics):
                column = train_y[:, metric_idx]
                metric_label = metric.name.replace("_", " ").title()
                if metric.goal == "maximize":
                    value, index = column.max(0)
                    log(f"Current {metric_label} best (max): {value.item():.4f} at {index.item()}")
                else:
                    value, index = column.min(0)
                    log(f"Current {metric_label} best (min): {value.item():.4f} at {index.item()}")

            # Fit one GP per objective using the dimscaled categorical kernel
            gp_models = []
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for objective_idx in range(train_y.shape[1]):
                    gp_models.append(
                        get_fitted_gp_model(
                            train_x=train_x,
                            train_y=train_y[:, objective_idx],
                            train_yvar=None,
                            prior_choice="dimscaled",
                            kernel_choice="categorical",
                            noise_type="fixed",
                            fixed_noise_value=0.001,
                            **tkwargs,
                        )
                    )

            model = ModelList(*gp_models)

            # Construct noisy EHVI acquisition that respects the objective weights
            acqf, pareto_points = get_acquisition_function(
                model=model,
                train_x=train_x,
                train_y=train_y,
                weights=metric_weights,
                **tkwargs,
            )

            # Optimise the acquisition over {0,1}^d using local search in the trust region
            discrete_choices = evaluation_task.discrete_choices(tkwargs=tkwargs)
            next_x, _ = optimize_acqf_discrete_local_search(
                acq_function=acqf,
                discrete_choices=discrete_choices,
                q=1,
                num_restarts=20,
                max_batch_size=128,
                raw_samples=4096,
                inequality_constraints=None,
                X_avoid=None,
                batch_initial_conditions=None,
                pareto_points=pareto_points,
                n_local_steps=tr_manager.current_radius,
                init_sample_in_tr=True,
                tr_manager=tr_manager,
            )

            next_x = next_x.to(**tkwargs)
            next_y = evaluation_task.evaluate(next_x, split=evaluation_split).to(**tkwargs)

            train_x = torch.cat([train_x, next_x], dim=0)
            train_y = torch.cat([train_y, next_y], dim=0)

            bo_completed += 1
            update_postfix(initial_completed, bo_completed)
            progress.update()

    task_suffix = f"_{cfg['task']}" if cfg.get("task") else ""
    exp_folder_name = (
        f"{cfg['dataset']}{task_suffix}_{model_basename}"
        f"_temp{temperature}_n{n_samples}_"
        f"initround{n_initial_pts}_boround{total_iterations}_seed{seed}"
    ).lower()

    exp_dir = Path(cfg["save_dir"]) / exp_folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"x": train_x.cpu(), "y": train_y.cpu()},
        exp_dir / "combom.pt",
    )


if __name__ == "__main__":
    """CLI entry point: parse YAML config path and launch training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)

    if not isinstance(raw_cfg, dict):
        raise ValueError("Configuration file must define a mapping of parameters.")

    common_cfg = {k: v for k, v in raw_cfg.items() if k not in {"train", "test"}}
    train_overrides = raw_cfg.get("train", {})
    cfg = common_cfg.copy()
    cfg.update(train_overrides)

    cfg["seed"] = int(cfg["seed"])
    cfg["save_dir"] = str(cfg.get("save_dir", "runs"))
    cfg.setdefault("num_workers", 32)

    set_random_seed(cfg["seed"])
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    main(cfg)
