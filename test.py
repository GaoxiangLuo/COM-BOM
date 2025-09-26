"""Test the COM-BOM exemplar selection results from a YAML config.

This script loads the same configuration file as train.py, finds the corresponding
checkpoint file, and evaluates the Pareto frontier solutions along with baselines.
"""

import argparse
import csv
from pathlib import Path

import torch
import yaml
from botorch.utils.multi_objective.pareto import is_non_dominated

from tasks import BaseTask

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "mmlupro.yaml"


def _build_task_kwargs(cfg: dict[str, object]) -> dict[str, object]:
    """Extract keyword arguments for constructing the selected task."""
    task_kwargs: dict[str, object] = {
        "n": int(cfg["n_samples"]),
        "model": cfg["hf_model"],
        "temperature": float(cfg["temperature"]),
        "port": int(cfg["port"]),
    }
    task_value = cfg.get("task")
    if task_value is not None:
        task_kwargs["task"] = task_value
    text_embedding_model = cfg.get("text_embedding_model")
    if text_embedding_model:
        task_kwargs["text_embedding_model"] = text_embedding_model
    prompt_type = cfg.get("prompt_type")
    if prompt_type:
        task_kwargs["prompt_type"] = prompt_type
    return task_kwargs


def find_checkpoint_path(cfg: dict[str, object]) -> Path:
    """Find the checkpoint file based on config parameters."""
    model_basename = cfg["hf_model"].split("/")[-1].replace("-", "_")
    task_suffix = f"_{cfg['task']}" if cfg.get("task") else ""

    exp_folder_name = (
        f"{cfg['dataset']}{task_suffix}_{model_basename}"
        f"_temp{cfg['temperature']}_n{cfg['n_samples']}_"
        f"initround{cfg['n_initial_pts']}_boround{cfg['n_bo_iterations']}_seed{cfg['seed']}"
    ).lower()

    checkpoint_path = Path(cfg["save_dir"]) / exp_folder_name / "combom.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def main(cfg: dict[str, object]) -> None:
    """Test COM-BOM results and baselines for a given configuration."""
    if cfg["dataset"] not in BaseTask.registry:
        available = ", ".join(sorted(BaseTask.registry)) or "<none>"
        raise ValueError(f"Unknown dataset {cfg['dataset']!r}. Available datasets: {available}")

    # Initialize evaluation task (same as train.py)
    task_kwargs = _build_task_kwargs(cfg)
    evaluation_task = BaseTask.create(cfg["dataset"], **task_kwargs)

    # Setup results file in the same experiment folder as the checkpoint
    model_basename = cfg["hf_model"].split("/")[-1].replace("-", "_")
    task_suffix = f"_{cfg['task']}" if cfg.get("task") else ""

    exp_folder_name = (
        f"{cfg['dataset']}{task_suffix}_{model_basename}"
        f"_temp{cfg['temperature']}_n{cfg['n_samples']}_"
        f"initround{cfg['n_initial_pts']}_boround{cfg['n_bo_iterations']}_seed{cfg['seed']}"
    ).lower()

    exp_dir = Path(cfg["save_dir"]) / exp_folder_name
    results_file = exp_dir / "results.csv"

    # Create results file with header
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy", "calibration_error", "iteration"])

    k_value = int(cfg.get("k", 10))

    # Test baselines
    print("Testing baselines...")

    # No exemplars baseline
    X_none = torch.zeros(1, evaluation_task.decision_dim)
    y_none = evaluation_task.evaluate(X_none, split="test")
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["No Exemplars", y_none[0, 0].item(), y_none[0, 1].item(), 0])
    print(f"No exemplars: Accuracy={y_none[0, 0]:.4f}, Calibration Error={y_none[0, 1]:.4f}")

    # All exemplars baseline
    X_all = torch.ones(1, evaluation_task.decision_dim)
    y_all = evaluation_task.evaluate(X_all, split="test")
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["All Exemplars", y_all[0, 0].item(), y_all[0, 1].item(), 0])
    print(f"All exemplars: Accuracy={y_all[0, 0]:.4f}, Calibration Error={y_all[0, 1]:.4f}")

    if cfg["dataset"] == "mmlupro":
        try:
            nearest_acc, nearest_calibration_error = evaluation_task.nearest_baseline_metrics(
                k_value
            )
            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [f"Nearest (k={k_value})", nearest_acc, nearest_calibration_error, 0]
                )
            print(
                f"Nearest baseline (k={k_value}): Accuracy={nearest_acc:.4f}, "
                f"Calibration Error={nearest_calibration_error:.4f}"
            )
            diversity_acc, diversity_calibration_error = evaluation_task.diversity_baseline_metrics(
                k_value
            )
            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"Diversity (k={k_value})",
                        diversity_acc,
                        diversity_calibration_error,
                        0,
                    ]
                )
            print(
                f"Diversity baseline (k={k_value}): Accuracy={diversity_acc:.4f}, "
                f"Calibration Error={diversity_calibration_error:.4f}"
            )
        except Exception as exc:
            print(f"Skipping nearest/diversity baselines due to error: {exc}")

    # Load and test COM-BOM checkpoint
    checkpoint_path = find_checkpoint_path(cfg)
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path)
    X = ckpt["x"]  # [n_iterations, decision_dim]
    y = ckpt["y"]  # [n_iterations, 2] (accuracy, calibration error)

    # Flip calibration error for Pareto calculation (minimize -> maximize)
    y_pareto = y.clone()
    y_pareto[:, 1] = -y_pareto[:, 1]

    # Find Pareto frontier
    pareto_mask = is_non_dominated(y_pareto)
    pareto_points = y[pareto_mask]
    pareto_indices = torch.where(pareto_mask)[0]

    print(f"Found {len(pareto_points)} Pareto-optimal solutions")

    # Test each Pareto point
    for i, idx in enumerate(pareto_indices):
        X_pareto = X[idx].unsqueeze(0)
        y_test = evaluation_task.evaluate(X_pareto, split="test")

        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["COMBOM Pareto Point", y_test[0, 0].item(), y_test[0, 1].item(), idx.item()]
            )

        print(
            f"Pareto point {i}: Accuracy={y_test[0, 0]:.4f}, Calibration Error={y_test[0, 1]:.4f} (iteration {idx})"
        )

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    """CLI entry point: parse YAML config path and launch testing."""
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
    test_overrides = raw_cfg.get("test", {})
    cfg = common_cfg.copy()
    cfg.update(test_overrides)

    # Ensure required fields are properly typed
    cfg["seed"] = int(cfg["seed"])
    cfg["save_dir"] = str(cfg.get("save_dir", "runs"))

    main(cfg)
