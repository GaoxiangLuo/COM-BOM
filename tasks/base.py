from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True)
class MetricSpec:
    """Definition of a task metric."""

    name: str
    goal: str  # "maximize" or "minimize"

    def __post_init__(self) -> None:
        if self.goal not in {"maximize", "minimize"}:
            raise ValueError("goal must be 'maximize' or 'minimize'")

    @property
    def weight(self) -> float:
        return 1.0 if self.goal == "maximize" else -1.0


class BaseTask(ABC):
    """Abstract base class for exemplar selection tasks."""

    registry: ClassVar[dict[str, type[BaseTask]]] = {}
    task_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        task_name = getattr(cls, "task_name", None)
        if task_name:
            BaseTask.registry[task_name] = cls

    @classmethod
    def create(cls, task_name: str, **kwargs) -> BaseTask:
        try:
            task_cls = cls.registry[task_name]
        except KeyError as exc:
            available = ", ".join(sorted(cls.registry)) or "<none>"
            raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}") from exc
        return task_cls(**kwargs)

    @property
    @abstractmethod
    def decision_dim(self) -> int:
        """Return the number of binary decision variables."""

    @property
    @abstractmethod
    def metrics(self) -> Sequence[MetricSpec]:
        """Return ordered metric specifications."""

    @abstractmethod
    def evaluate(self, X: torch.Tensor, *, split: str = "validation") -> torch.Tensor:
        """Evaluate metrics for the provided candidate matrix."""

    def metric_weights(self, *, tkwargs: dict[str, object]) -> torch.Tensor:
        weights = [metric.weight for metric in self.metrics]
        return torch.tensor(weights, **tkwargs)

    def discrete_choices(self, *, tkwargs: dict[str, object]) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(self.decision_dim, 1, **tkwargs),
                torch.ones(self.decision_dim, 1, **tkwargs),
            ],
            dim=1,
        )


__all__ = ["BaseTask", "MetricSpec"]
