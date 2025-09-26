"""Trust-region utilities tailored for categorical Bayesian optimisation."""

import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated


class TrustRegionManager:
    """Track and adapt the trust-region radius during optimisation."""

    def __init__(
        self,
        name: str,
        n_categorical: int,
        n_objectives: int = 2,
        min_radius: int = 1,
        max_radius: int | None = None,
        init_radius: int | None = None,
        radius_multiplier: float = 1.5,
        succ_tol: int = 3,
        fail_tol: int = 10,
        best_value: list[float] = None,
    ):
        """Initialise a trust-region controller.

        Args:
            name: Identifier used in logging.
            n_categorical: Number of categorical dimensions in the search space.
            n_objectives: Number of objectives the BO loop optimises.
            min_radius: Lower bound on the trust-region radius.
            max_radius: Optional upper bound (defaults to ``n_categorical``).
            init_radius: Optional initial radius (defaults to 20% of ``max_radius``).
            radius_multiplier: Factor used when expanding/contracting the radius.
            succ_tol: Number of consecutive successful iterations before expansion.
            fail_tol: Number of consecutive failures before contraction.
            best_value: Running best objective values (after weight transform).
        """
        self.tr_name = name
        self.n_categorical = n_categorical
        self.max_radius = max_radius if max_radius is not None else n_categorical
        self.min_radius = min_radius
        self.init_radius = init_radius if init_radius is not None else int(0.2 * self.max_radius)
        self.radius_multiplier = radius_multiplier
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol

        self.current_radius = self.init_radius
        self.succ_count = 0
        self.fail_count = 0
        self.center = None
        if best_value is None:
            self.best_value = [-1 * float("inf")] * n_objectives
        else:
            self.best_value = best_value

    def adjust_radius(self, new_x: torch.Tensor, new_value: torch.Tensor) -> None:
        """Update the radius after observing a new candidate.

        The method assumes all objectives are cast as maximisation problems.

        Args:
            new_x: Newly evaluated candidate.
            new_value: Objective values (already multiplied by optimisation weights).
        """
        improved_one_obj = False
        for i in range(len(new_value)):
            if new_value[i] > self.best_value[i]:
                improved_one_obj = True
                self.best_value[i] = new_value[i]
                break
        if improved_one_obj:
            self.center = new_x
            self.succ_count += 1
            self.fail_count = 0
        else:
            self.succ_count = 0
            self.fail_count += 1

        if self.succ_count == self.succ_tol:
            self.current_radius = min(
                int(self.current_radius * self.radius_multiplier), self.max_radius
            )
            self.succ_count = 0
        elif self.fail_count == self.fail_tol:
            self.current_radius = max(
                int(self.current_radius / self.radius_multiplier), self.min_radius
            )
            self.fail_count = 0

        print(f">>> {self.tr_name} | current radius: {self.current_radius}")

    def initialize_center(self, train_x: torch.Tensor, train_y: torch.Tensor, **tkwargs: dict):
        """Select the trust-region centre using hypervolume contribution."""
        # assumes both objectives are to be maximized
        # i.e. train_y is appropriately negated before being passed in
        is_non_dominated_idx = is_non_dominated(train_y)
        pareto_obj = train_y[is_non_dominated_idx].clone()
        pareto_points = train_x[is_non_dominated_idx].clone()
        num_current_pareto = pareto_obj.shape[0]
        ref_point = train_y.min(dim=0)[0] - torch.abs(train_y.min(dim=0)[0]) * 0.1

        masks = torch.eye(
            num_current_pareto,
            dtype=torch.bool,
            device=ref_point.device,
        )  # Shape: (num_current_pareto, num_current_pareto)

        # Create batches of Pareto frontiers, where each batch excludes one point.
        # The result `batch_paretos` has shape:
        # (num_current_pareto, num_current_pareto - 1, num_objectives)
        batch_paretos = torch.cat(
            [
                # Select all points *except* the one indicated by the mask `m`
                pareto_obj[~m, :].unsqueeze(dim=0)  # Unsqueeze to add batch dim
                for m in masks
            ],
            dim=0,  # Concatenate along the new batch dimension
        )

        # works for 2 objectives only
        partitioning_without_point = DominatedPartitioning(Y=batch_paretos, ref_point=ref_point)
        pareto_partitioning = DominatedPartitioning(Y=pareto_obj, ref_point=ref_point)

        # Compute the hypervolume for each Pareto set *excluding* one point
        hv_without_point = partitioning_without_point.compute_hypervolume()

        hv = pareto_partitioning.compute_hypervolume()
        # The contribution of each point is the total HV minus the HV without that point
        hv_contributions = hv - hv_without_point

        # Find the point with the maximum contribution to hypervolume
        best_hv_contribution_idx = torch.argmax(hv_contributions)
        self.center = pareto_points[best_hv_contribution_idx]
