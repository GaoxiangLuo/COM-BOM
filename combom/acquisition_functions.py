"""Utilities for building and optimising multi-objective acquisition functions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models.model import ModelList
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


def get_acquisition_function(
    model: ModelList,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    disable_noisy: bool = False,
    **tkwargs: dict,
) -> Tuple[AcquisitionFunction, torch.Tensor]:
    """Construct a (noisy) EHVI acquisition function for binary search spaces.

    Args:
        model: Joint GP surrogate over all objectives.
        train_x: Previously evaluated points.
        train_y: Objective values corresponding to ``train_x``.
        weights: Optional tensor of objective weights (defaults to [1, -1]).
        disable_noisy: If ``True`` use deterministic qEHVI instead of qNEHVI.
        **tkwargs: Torch keyword arguments (dtype/device).

    Returns:
        Tuple ``(acqf, pareto_points)`` where ``acqf`` is the EHVI module and
        ``pareto_points`` are the current non-dominated points.
    """
    if weights is None:
        weights = torch.tensor([1.0, -1.0], **tkwargs)
    else:
        weights = weights.to(**tkwargs)

    # transform objectives into maximisation form for Pareto/ref point calculations
    transformed_train_y = train_y * weights

    # find pareto optimal points in the transformed (maximization) space
    pareto_points = train_x[is_non_dominated(transformed_train_y)].clone()

    # calculate the reference point in the transformed (maximization) space
    reference_point = (
        transformed_train_y.min(dim=0)[0] - torch.abs(transformed_train_y.min(dim=0)[0]) * 0.1
    )

    # Define the objective function using the original weights
    objective = WeightedMCMultiOutputObjective(weights=weights)

    # Initialize qNEHVI with the transformed reference point and the weighted objective
    if not disable_noisy:
        acqf = qNoisyExpectedHypervolumeImprovement(
            X_baseline=train_x,
            model=model,
            ref_point=reference_point,
            objective=objective,
            cache_root=False,
        )
    else:
        with torch.no_grad():
            pred = model.posterior(train_x).mean
            pred = pred * weights
        partitioning = FastNondominatedPartitioning(
            ref_point=reference_point,
            Y=pred,
        )

        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference_point,
            objective=objective,
            partitioning=partitioning,
        )
    return acqf, pareto_points


### Trust region search


def sample_within_tr(
    x_center: torch.Tensor,
    n_points: int,
    tr_radius: int,
    discrete_choices,
) -> torch.Tensor:
    """Sample points within trust region by perturbing categorical variables.

    Args:
        x_center: Trust region center point (n_dims,)
        n_points: Number of points to sample
        tr_radius: Current trust region radius (max Hamming distance)
        discrete_choices: List specifying number of categories for each dimension

    Returns:
        Sampled points tensor (n_points, n_dims)
    """
    device = x_center.device
    dtype = x_center.dtype
    n_dims = len(discrete_choices)

    # Replicate center point
    x_samples = x_center.clone() * torch.ones((n_points, n_dims), device=device, dtype=dtype)

    # For each point, randomly perturb up to tr_radius dimensions
    n_perturb = torch.randint(low=1, high=tr_radius + 1, size=(n_points,), device=device)

    for i in range(n_points):
        # Randomly choose which dimensions to perturb
        dims_to_perturb = torch.randperm(n_dims, device=device)[: n_perturb[i]]
        for dim in dims_to_perturb:
            # Get current value for this dimension
            current_val = x_samples[i, dim].item()
            # Get all possible values except current
            choices = [v.item() for v in discrete_choices[dim.item()] if v != current_val]
            # Randomly select new value
            if len(choices) > 0:  # Safety check in case of single category
                new_val = np.random.choice(choices)
                x_samples[i, dim] = new_val

    return x_samples


## Methods below are implemented in botorch https://github.com/meta-pytorch/botorch/blob/main/botorch/optim/optimize.py


def _split_batch_eval_acqf(
    acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int
) -> Tensor:
    """Evaluate ``acq_function`` on ``X`` by splitting into smaller batches."""
    return torch.cat([acq_function(X_) for X_ in X.split(max_batch_size)])


def _combine_initial_conditions(
    provided_initial_conditions: Tensor,
    generated_initial_conditions: Tensor,
    dim=0,
) -> Tensor:
    """Concatenate provided and generated initial conditions if available."""
    if provided_initial_conditions is not None and generated_initial_conditions is not None:
        return torch.cat([provided_initial_conditions, generated_initial_conditions], dim=dim)
    if provided_initial_conditions is not None:
        return provided_initial_conditions
    if generated_initial_conditions is not None:
        return generated_initial_conditions
    raise ValueError("Either `batch_initial_conditions` or `raw_samples` must be set.")


def _filter_infeasible(X: Tensor, inequality_constraints: None) -> Tensor:
    """Remove points that violate linear inequality constraints."""
    is_feasible = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for inds, weights, bound in inequality_constraints:
        is_feasible &= (X[..., inds] * weights).sum(dim=-1) >= bound
    return X[is_feasible]


def _filter_invalid(X: Tensor, X_avoid: Tensor) -> Tensor:
    """Remove duplicates present in ``X_avoid`` from ``X``."""
    return X[~(X_avoid.unsqueeze(-2) == X).all(dim=-1).any(dim=-2)]


def _gen_batch_initial_conditions_local_search(
    discrete_choices,
    raw_samples: int,
    X_avoid: Tensor,
    inequality_constraints: None,
    min_points: int,
    max_tries: int = 100,
) -> Tensor:
    """Generate random initial conditions that satisfy constraints."""
    device = discrete_choices[0].device
    dtype = discrete_choices[0].dtype
    dim = len(discrete_choices)
    X = torch.zeros(0, dim, device=device, dtype=dtype)
    for _ in range(max_tries):
        X_new = torch.zeros(raw_samples, dim, device=device, dtype=dtype)
        for i, c in enumerate(discrete_choices):
            X_new[:, i] = c[torch.randint(low=0, high=len(c), size=(raw_samples,), device=c.device)]
        X = torch.unique(torch.cat((X, X_new)), dim=0)
        X = _filter_invalid(X=X, X_avoid=X_avoid)
        X = _filter_infeasible(X=X, inequality_constraints=inequality_constraints)
        if len(X) >= min_points:
            return X
    raise RuntimeError(f"Failed to generate at least {min_points} initial conditions")


def _gen_starting_points_local_search(
    discrete_choices,
    raw_samples: int,
    batch_initial_conditions: Tensor,
    X_avoid: Tensor,
    inequality_constraints: None,
    min_points: int,
    acq_function: AcquisitionFunction,
    max_batch_size: int = 2048,
    max_tries: int = 100,
    pareto_points=None,
    init_sample_in_tr: bool = False,
    tr_manager=None,
) -> Tensor:
    """Select promising starting points for the discrete local search routine."""
    required_min_points = min_points
    provided_X0 = None
    generated_X0 = None

    if batch_initial_conditions is not None:
        provided_X0 = _filter_invalid(X=batch_initial_conditions.squeeze(1), X_avoid=X_avoid)
        provided_X0 = _filter_infeasible(
            X=provided_X0, inequality_constraints=inequality_constraints
        ).unsqueeze(1)
        required_min_points -= batch_initial_conditions.shape[0]

    if required_min_points > 0:
        if init_sample_in_tr:
            generated_X0 = sample_within_tr(
                x_center=tr_manager.center,
                n_points=raw_samples,
                tr_radius=tr_manager.current_radius,
                discrete_choices=discrete_choices,
            )
        else:
            generated_X0 = _gen_batch_initial_conditions_local_search(
                discrete_choices=discrete_choices,
                raw_samples=raw_samples,
                X_avoid=X_avoid,
                inequality_constraints=inequality_constraints,
                min_points=min_points,
                max_tries=max_tries,
            )
        if pareto_points is not None:
            spray_points = _generate_neighbors(
                x=pareto_points,
                discrete_choices=discrete_choices,
                X_avoid=X_avoid,
                inequality_constraints=inequality_constraints,
            )
            generated_X0 = torch.cat([generated_X0, spray_points], dim=0)

        # pick the best starting points
        with torch.no_grad():
            acqvals_init = _split_batch_eval_acqf(
                acq_function=acq_function,
                X=generated_X0.unsqueeze(1),
                max_batch_size=max_batch_size,
            ).unsqueeze(-1)

        generated_X0 = generated_X0[acqvals_init.topk(k=min_points, largest=True, dim=0).indices]
        if init_sample_in_tr:
            distances = (generated_X0 != tr_manager.center).sum(dim=1)
            print(
                f"min distance: {distances.min()}, \
                max distance: {distances.max()}, mean distance: {distances.float().mean()}"
            )

    return _combine_initial_conditions(
        provided_initial_conditions=provided_X0 if provided_X0 is not None else None,
        generated_initial_conditions=generated_X0 if generated_X0 is not None else None,
    )


def _generate_neighbors(
    x: Tensor,
    discrete_choices,
    X_avoid: Tensor,
    inequality_constraints: None,
) -> Tensor:
    """Enumerate 1-flip perturbations of ``x`` that satisfy constraints."""
    # generate all 1D perturbations
    npts = sum([len(c) for c in discrete_choices])
    x_loc = x.repeat(npts, 1)
    j = 0
    for i, c in enumerate(discrete_choices):
        x_loc[j : j + len(c), i] = c
        j += len(c)
    # remove invalid and infeasible points (also remove x)
    x_loc = _filter_invalid(X=x_loc, X_avoid=torch.cat((X_avoid, x)))
    x_loc = _filter_infeasible(X=x_loc, inequality_constraints=inequality_constraints)
    return x_loc


def optimize_acqf_discrete_local_search(
    acq_function: AcquisitionFunction,
    discrete_choices,
    q: int,
    inequality_constraints: None,
    X_avoid: None,
    batch_initial_conditions: None,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    max_batch_size: int = 2048,
    max_tries: int = 100,
    unique: bool = True,
    pareto_points=None,
    n_local_steps: int = 20,
    init_sample_in_tr: bool = False,
    tr_manager=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Optimize acquisition function over a lattice.

    This is useful when d is large and enumeration of the search space
    isn't possible. For q > 1 this function always performs sequential
    greedy optimization (with proper conditioning on generated candidates).

    NOTE: While this method supports arbitrary lattices, it has only been
    thoroughly tested for {0, 1}^d. Consider it to be in alpha stage for
    the more general case.

    Args:
        acq_function: An AcquisitionFunction
        discrete_choices: A list of possible discrete choices for each dimension.
            Each element in the list is expected to be a torch tensor.
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        X_avoid: An `n x d` tensor of candidates that we aren't allowed to pick.
        batch_initial_conditions: A tensor of size `n x 1 x d` to specify the
            initial conditions. Set this if you do not want to use default
            initialization strategy.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        max_tries: Maximum number of iterations to try when generating initial
            conditions.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).
        pareto_points: Optional tensor of Pareto points to seed local search.
        n_local_steps: Maximum number of discrete local search steps.
        init_sample_in_tr: Whether to sample initial points inside the trust region.
        tr_manager: Trust-region manager that provides center and radius when
            ``init_sample_in_tr`` is ``True``.

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if batch_initial_conditions is not None:
        if not (
            len(batch_initial_conditions.shape) == 3 and batch_initial_conditions.shape[-2] == 1
        ):
            raise ValueError(
                "batch_initial_conditions must have shape `n x 1 x d` if "
                f"given (received shape {batch_initial_conditions.shape})."
            )

    candidate_list = []
    base_X_pending = acq_function.X_pending if q > 1 else None
    base_X_avoid = X_avoid
    device = discrete_choices[0].device
    dtype = discrete_choices[0].dtype
    dim = len(discrete_choices)
    if X_avoid is None:
        X_avoid = torch.zeros(0, dim, device=device, dtype=dtype)

    inequality_constraints = inequality_constraints or []
    for _ in range(q):
        # generate some starting points
        X0 = _gen_starting_points_local_search(
            discrete_choices=discrete_choices,
            raw_samples=raw_samples,
            batch_initial_conditions=batch_initial_conditions,
            X_avoid=X_avoid,
            inequality_constraints=inequality_constraints,
            min_points=num_restarts,
            acq_function=acq_function,
            max_batch_size=max_batch_size,
            max_tries=max_tries,
            pareto_points=pareto_points,
            init_sample_in_tr=init_sample_in_tr,
            tr_manager=tr_manager,
        )

        if init_sample_in_tr:
            _distances = (tr_manager.center != X0).sum(dim=1)

        batch_initial_conditions = None

        # optimize from the best starting points
        best_xs = torch.zeros(len(X0), dim, device=device, dtype=dtype)
        best_acqvals = torch.zeros(len(X0), 1, device=device, dtype=dtype)
        for j, x in enumerate(X0):
            curr_x, curr_acqval = x.clone(), acq_function(x.unsqueeze(1))
            n_iters = 0
            starting_distance = 0
            if init_sample_in_tr:
                assert n_local_steps == tr_manager.current_radius, (
                    "n_local_steps must be equal to tr_manager.center"
                )
                starting_distance = (curr_x != tr_manager.center).sum()
            while True and n_iters < (n_local_steps - starting_distance):
                n_iters += 1
                # this generates all feasible neighbors that are one bit away
                X_loc = _generate_neighbors(
                    x=curr_x,
                    discrete_choices=discrete_choices,
                    X_avoid=X_avoid,
                    inequality_constraints=inequality_constraints,
                )
                if init_sample_in_tr:
                    _distances = (X_loc != tr_manager.center).sum(dim=1)

                # there may not be any neighbors
                if len(X_loc) == 0:
                    break
                with torch.no_grad():
                    acqval_loc = _split_batch_eval_acqf(
                        acq_function=acq_function,
                        X=X_loc.unsqueeze(1),
                        max_batch_size=max_batch_size,
                    )
                # break if no neighbor is better than the current point (local optimum)
                if acqval_loc.max() <= curr_acqval:
                    break
                best_ind = acqval_loc.argmax().item()
                curr_x, curr_acqval = X_loc[best_ind].unsqueeze(0), acqval_loc[best_ind]
            best_xs[j, :], best_acqvals[j] = curr_x, curr_acqval

        # pick the best
        best_idx = best_acqvals.argmax()
        candidate_list.append(best_xs[best_idx].unsqueeze(0))

        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )

            # Update points to avoid if unique is True
            if unique:
                X_avoid = (
                    torch.cat([base_X_avoid, candidates], dim=-2)
                    if base_X_avoid is not None
                    else candidates
                )

    # Reset acq_func to original X_pending state
    if q > 1:
        acq_function.set_X_pending(base_X_pending)
    with torch.no_grad():
        acq_value = acq_function(candidates)  # compute joint acquisition value
    return candidates, acq_value
