"""Hard-gate checks for PushObj action units and the native staged objective."""

from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from planning.objectives import create_objective_fn


def _f(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def check_action_contract(targets_path: Path, seed: int = 100, frameskip: int = 5) -> dict:
    with targets_path.open("rb") as handle:
        targets = pickle.load(handle)

    chosen = random.Random(seed).sample(targets["segments"], 1)[0]
    stored_pixel_actions = torch.as_tensor(chosen["actions"], dtype=torch.float32)
    env_api_actions = stored_pixel_actions / 100.0
    normalized_actions = (env_api_actions - ACTION_MEAN) / ACTION_STD

    if normalized_actions.shape[0] % frameskip:
        raise AssertionError("Action sequence is not divisible by frameskip")

    model_actions = rearrange(
        normalized_actions, "(t f) d -> 1 t (f d)", f=frameskip
    )
    unpacked_normalized = rearrange(
        model_actions, "b t (f d) -> b (t f) d", f=frameskip
    )[0]
    reconstructed_env_api = unpacked_normalized * ACTION_STD + ACTION_MEAN
    reconstructed_pixel = reconstructed_env_api * 100.0

    normalized_unpack_error = torch.max(
        torch.abs(unpacked_normalized - normalized_actions)
    )
    env_roundtrip_error = torch.max(
        torch.abs(reconstructed_env_api - env_api_actions)
    )
    pixel_roundtrip_error = torch.max(
        torch.abs(reconstructed_pixel - stored_pixel_actions)
    )

    if _f(env_roundtrip_error) > 1e-6:
        raise AssertionError(f"Action round trip failed: {_f(env_roundtrip_error)}")
    if model_actions.shape[-1] != frameskip * 2:
        raise AssertionError(f"Unexpected model action dimension: {model_actions.shape}")

    return {
        "sample_selection": {
            "seed": seed,
            "shape": chosen.get("shape"),
            "stored_sequence_shape": list(stored_pixel_actions.shape),
        },
        "units": {
            "stored_plan_target": "pixel displacement",
            "environment_api": "relative displacement divided by action_scale=100",
            "environment_internal_controller": "environment API action multiplied by 100",
            "model": "environment API action normalized by dataset ACTION_MEAN/ACTION_STD",
        },
        "action_dim": 2,
        "frameskip": frameskip,
        "model_action_dim": int(model_actions.shape[-1]),
        "model_action_shape": list(model_actions.shape),
        "environment_action_shape": [1, *list(env_api_actions.shape)],
        "dtype": str(model_actions.dtype),
        "ranges": {
            "normalized_model_min": _f(normalized_actions.min()),
            "normalized_model_max": _f(normalized_actions.max()),
            "environment_api_min": _f(env_api_actions.min()),
            "environment_api_max": _f(env_api_actions.max()),
            "stored_pixel_min": _f(stored_pixel_actions.min()),
            "stored_pixel_max": _f(stored_pixel_actions.max()),
        },
        "first_action": {
            "normalized_model": normalized_actions[0].tolist(),
            "environment_api": env_api_actions[0].tolist(),
            "stored_pixel_and_internal_controller": stored_pixel_actions[0].tolist(),
        },
        "roundtrip_max_abs_error": {
            "model_pack_unpack": _f(normalized_unpack_error),
            "normalize_denormalize_environment_api": _f(env_roundtrip_error),
            "full_pixel_roundtrip": _f(pixel_roundtrip_error),
        },
        "passed": True,
    }


def check_objective_contract(
    alpha: float = 1.0, base: float = 2.0, action_horizon: int = 5
) -> dict:
    warnings.filterwarnings("ignore", message="Using a target size")
    generator = torch.Generator().manual_seed(271828)
    # VWorldModel.rollout returns the initial observation plus one prediction for
    # every model action, so five model actions produce six objective frames.
    rollout_observation_horizon = action_horizon + 1
    visual = torch.randn(2, rollout_observation_horizon, 3, 4, generator=generator)
    proprio = torch.randn(2, rollout_observation_horizon, 4, generator=generator)
    goal_visual = torch.randn(2, 1, 3, 4, generator=generator)
    goal_proprio = torch.randn(2, 1, 4, generator=generator)
    pred = {"visual": visual, "proprio": proprio}
    goal = {"visual": goal_visual, "proprio": goal_proprio}

    objective = create_objective_fn(alpha=alpha, base=base, mode="staged")
    metric = torch.nn.MSELoss(reduction="none")

    terminal_visual = metric(visual[:, -1:], goal_visual).mean(dim=(1, 2, 3))
    terminal_proprio = metric(proprio[:, -1:], goal_proprio).mean(dim=(1, 2))
    expected_terminal = terminal_visual + alpha * terminal_proprio

    coeffs = torch.tensor([base**i for i in range(visual.shape[1])])
    coeffs = coeffs / coeffs.sum()
    full_visual_by_t = metric(visual, goal_visual).mean(dim=(2, 3))
    full_proprio_by_t = metric(proprio, goal_proprio).mean(dim=2)
    full_visual = (full_visual_by_t * coeffs).mean(dim=1)
    full_proprio = (full_proprio_by_t * coeffs).mean(dim=1)
    expected_full = full_visual + alpha * full_proprio

    observed_early = objective(pred, goal, step=0)
    observed_boundary_minus_one = objective(
        pred, goal, step=rollout_observation_horizon - 2
    )
    observed_boundary = objective(pred, goal, step=rollout_observation_horizon - 1)
    observed_late = objective(pred, goal, step=10)

    errors = {
        "early_terminal": _f(torch.max(torch.abs(observed_early - expected_terminal))),
        "boundary_minus_one_terminal": _f(
            torch.max(torch.abs(observed_boundary_minus_one - expected_terminal))
        ),
        "boundary_full": _f(torch.max(torch.abs(observed_boundary - expected_full))),
        "late_full": _f(torch.max(torch.abs(observed_late - expected_full))),
    }
    if max(errors.values()) > 1e-6:
        raise AssertionError(f"Objective contract failed: {errors}")

    return {
        "mode": "staged",
        "alpha": alpha,
        "base": base,
        "model_action_horizon": action_horizon,
        "rollout_observation_horizon": rollout_observation_horizon,
        "stage_rule": {
            "terminal_steps": list(range(rollout_observation_horizon - 1)),
            "full_horizon_from_step": rollout_observation_horizon - 1,
        },
        "full_horizon_coefficients_before_outer_mean": coeffs.tolist(),
        "component_example": {
            "terminal_visual": terminal_visual.tolist(),
            "terminal_proprio": terminal_proprio.tolist(),
            "full_visual": full_visual.tolist(),
            "full_proprio": full_proprio.tolist(),
        },
        "closure_max_abs_error": errors,
        "passed": True,
    }


def main() -> None:
    targets_path = Path("data/pushobj_eval/val_T/plan_targets.pkl")
    result = {
        "action_contract": check_action_contract(targets_path),
        "objective_contract": check_objective_contract(),
        "all_passed": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
