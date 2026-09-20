"""Native objective and optimizer diagnostics for the PushObj pilot."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def native_objective_breakdown(
    z_obs_pred: dict,
    z_obs_goal: dict,
    step: Optional[int],
    alpha: float,
    base: float,
) -> dict:
    """Reproduce ``planning.objectives.create_objective_fn(..., staged)``.

    The outer ``mean(dim=1)`` after applying normalized coefficients is kept
    intentionally because it is part of the released planner's objective.
    """

    horizon = int(z_obs_pred["visual"].shape[1])
    terminal = step is not None and step < horizon - 1
    if terminal:
        visual = F.mse_loss(
            z_obs_pred["visual"][:, -1:], z_obs_goal["visual"], reduction="none"
        ).mean(dim=tuple(range(1, z_obs_pred["visual"].ndim)))
        proprio = F.mse_loss(
            z_obs_pred["proprio"][:, -1:],
            z_obs_goal["proprio"],
            reduction="none",
        ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
        coefficients = None
        stage = "terminal"
    else:
        coefficients = np.asarray(
            [base**i for i in range(horizon)], dtype=np.float32
        )
        coefficients = torch.as_tensor(
            coefficients / coefficients.sum(), device=z_obs_pred["visual"].device
        )
        visual_by_t = F.mse_loss(
            z_obs_pred["visual"], z_obs_goal["visual"], reduction="none"
        ).mean(dim=tuple(range(2, z_obs_pred["visual"].ndim)))
        proprio_by_t = F.mse_loss(
            z_obs_pred["proprio"], z_obs_goal["proprio"], reduction="none"
        ).mean(dim=tuple(range(2, z_obs_pred["proprio"].ndim)))
        visual = (visual_by_t * coefficients).mean(dim=1)
        proprio = (proprio_by_t * coefficients).mean(dim=1)
        stage = "full_horizon"
    total = visual + float(alpha) * proprio
    return {
        "stage": stage,
        "horizon": horizon,
        "coefficients": None
        if coefficients is None
        else coefficients.detach().cpu().tolist(),
        "visual": visual,
        "proprio": proprio,
        "total": total,
    }


def optimizer_state_summary(optimizer: torch.optim.Optimizer, parameter: torch.Tensor) -> dict:
    state = optimizer.state.get(parameter, {})
    summary = {"optimizer": optimizer.__class__.__name__}
    for key, value in state.items():
        if torch.is_tensor(value):
            detached = value.detach().float()
            summary[key] = {
                "shape": list(value.shape),
                "norm": float(detached.norm().cpu()),
                "mean": float(detached.mean().cpu()),
            }
        elif isinstance(value, (int, float, bool, str)):
            summary[key] = value
        else:
            summary[key] = str(value)
    return summary


def scheduler_state_summary(scheduler) -> Optional[dict]:
    if scheduler is None:
        return None
    state = scheduler.state_dict()
    keep = ("T_max", "eta_min", "base_lrs", "last_epoch", "_step_count", "_last_lr")
    return {key: state[key] for key in keep if key in state}


def signed(value: float, atol: float = 0.0) -> int:
    if value > atol:
        return 1
    if value < -atol:
        return -1
    return 0

