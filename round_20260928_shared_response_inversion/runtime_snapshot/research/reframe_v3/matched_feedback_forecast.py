"""Round 3 S2/S3: matched-feedback model forecast for the saved C branches.

The module contains a latent-input mirror of the released world-model rollout
and GD planner.  S2 verifies it against the original RGB-input path.  S3 then
predicts the already-defined C intervention without making environment calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

from research.contrast_probe import native_objective_breakdown
from research.reframe_v3.common_continuation import (
    _build_planner,
    _candidate_tensor,
    _rng_equal,
    _tensor_versions,
    continuation_seed,
)
from research.reframe_v3.shadow_selection_audit import (
    _capture_rng_state,
    _jsonable,
    _load_anchor_observations,
    _load_runtime,
    _seed_all,
    _transform_anchor_obs,
)


EPSILON = 1e-6
PARITY_THRESHOLDS = {
    "rollout_atol": 1e-6,
    "rollout_rtol": 1e-6,
    "cost_atol": 1e-8,
    "gradient_atol": 1e-6,
    "gradient_rtol": 1e-6,
    "gd_action_atol": 1e-6,
    "gd_action_rtol": 1e-6,
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _parse_donor(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("donor must use SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if not split or not raw_path:
        raise argparse.ArgumentTypeError("donor must use nonempty SPLIT=PATH")
    return split, Path(raw_path)


def _clone_zobs(z_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in z_obs.items()}


def _cpu_nested(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_nested(item) for item in value)
    return value


def _join_encoded_observation_action(model, z_obs: dict, action: torch.Tensor):
    """Mirror ``VWorldModel.encode`` after observation encoding."""
    visual = z_obs["visual"]
    proprio = z_obs["proprio"]
    if visual.shape[:2] != proprio.shape[:2]:
        raise ValueError("visual/proprio batch-time dimensions do not match")
    if action.shape[:2] != visual.shape[:2]:
        raise ValueError(
            f"action batch-time {tuple(action.shape[:2])} does not match "
            f"observation {tuple(visual.shape[:2])}"
        )
    action_emb = model.encode_act(action)
    if model.concat_dim == 0:
        return torch.cat(
            [visual, proprio.unsqueeze(2), action_emb.unsqueeze(2)], dim=2
        )
    if model.concat_dim == 1:
        tokens = visual.shape[2]
        proprio_tiled = repeat(
            proprio.unsqueeze(2), "b t 1 a -> b t f a", f=tokens
        )
        proprio_repeated = proprio_tiled.repeat(
            1, 1, 1, model.num_proprio_repeat
        )
        action_tiled = repeat(
            action_emb.unsqueeze(2), "b t 1 a -> b t f a", f=tokens
        )
        action_repeated = action_tiled.repeat(1, 1, 1, model.num_action_repeat)
        return torch.cat([visual, proprio_repeated, action_repeated], dim=3)
    raise ValueError(f"Unsupported concat_dim={model.concat_dim}")


def rollout_from_zobs(
    model,
    z_obs_initial: dict[str, torch.Tensor],
    normalized_model_actions: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the native autoregressive model from an already encoded observation."""
    num_obs_init = int(z_obs_initial["visual"].shape[1])
    if normalized_model_actions.shape[1] < num_obs_init:
        raise ValueError(
            "actions must cover every supplied initial observation before prediction"
        )
    act_initial = normalized_model_actions[:, :num_obs_init]
    future_actions = normalized_model_actions[:, num_obs_init:]
    z = _join_encoded_observation_action(model, z_obs_initial, act_initial)
    time_index = 0
    while time_index < future_actions.shape[1]:
        z_pred = model.predict(z[:, -model.num_hist :])
        z_new = z_pred[:, -1:, ...]
        z_new = model.replace_actions_from_z(
            z_new, future_actions[:, time_index : time_index + 1]
        )
        z = torch.cat([z, z_new], dim=1)
        time_index += 1
    z_pred = model.predict(z[:, -model.num_hist :])
    z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
    z_obses, _ = model.separate_emb(z)
    return z_obses, z


def plan_from_zobs(
    planner,
    z_obs_current: dict[str, torch.Tensor],
    z_obs_goal: dict[str, torch.Tensor],
    actions: torch.Tensor | None = None,
    step: int | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Mirror ``GDPlanner.plan`` while accepting encoded observations."""
    actions = planner.init_actions(z_obs_current, actions).to(planner.device)
    actions.requires_grad_(True)
    optimizer = planner.get_action_optimizer(actions)
    scheduler = planner.get_scheduler(optimizer)
    n_evals = int(actions.shape[0])
    for iteration in range(planner.opt_steps):
        optimizer.zero_grad()
        predicted, _ = rollout_from_zobs(planner.wm, z_obs_current, actions)
        loss = planner.objective_fn(predicted, z_obs_goal, step=step)
        total_loss = loss.mean() * n_evals
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            # Keep the original RNG consumption even when action_noise is zero.
            actions += torch.randn_like(actions) * planner.action_noise
        planner.wandb_run.log(
            {f"{planner.logging_prefix}/loss": total_loss.item(), "step": iteration + 1}
        )
    return actions, np.full(n_evals, np.inf)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu().item())


def _zobs_max_abs(left: dict, right: dict) -> dict:
    return {key: _max_abs(left[key], right[key]) for key in sorted(left)}


def _zobs_close(left: dict, right: dict, atol: float, rtol: float) -> bool:
    return all(
        torch.allclose(left[key], right[key], atol=atol, rtol=rtol)
        for key in left
    )


def _terminal_reference_cost(z_obs: dict, z_goal: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = F.mse_loss(z_obs["visual"][:, -1:], z_goal["visual"])
    proprio = F.mse_loss(z_obs["proprio"][:, -1:], z_goal["proprio"])
    return visual + proprio, visual, proprio


def run_parity(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "LATENT_INTERFACE_PARITY.json"
    raw_dir = args.out / "latent_interface_parity_raw"
    if result_path.exists() or (raw_dir.exists() and any(raw_dir.iterdir())):
        raise FileExistsError("latent parity outputs already exist")
    raw_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    model, preprocessor, config = _load_runtime(args.checkpoint_dir, device)
    frameskip = int(config.frameskip)
    planner = _build_planner(model, preprocessor, frameskip)
    versions_before = _tensor_versions(model)
    records = []
    full_gd_iterations = 0

    for mpc_iter in args.mpc_iter:
        current, goal = _load_anchor_observations(
            args.donor_dir, args.sample_id, mpc_iter
        )
        transformed_current, transformed_goal = _transform_anchor_obs(
            preprocessor, current, goal, device
        )
        candidate = _candidate_tensor(
            args.donor_dir, args.sample_id, mpc_iter, "g99_after"
        ).to(device)
        with torch.no_grad():
            z_current = model.encode_obs(transformed_current)
            z_goal = model.encode_obs(transformed_goal)
            direct_rollout, direct_full = model.rollout(
                obs_0=transformed_current, act=candidate
            )
            latent_rollout, latent_full = rollout_from_zobs(
                model, z_current, candidate
            )
            direct_cost = native_objective_breakdown(
                direct_rollout, z_goal, mpc_iter, alpha=1.0, base=2.0
            )["total"]
            latent_cost = native_objective_breakdown(
                latent_rollout, z_goal, mpc_iter, alpha=1.0, base=2.0
            )["total"]

        direct_action = candidate.detach().clone().requires_grad_(True)
        direct_for_grad, _ = model.rollout(
            obs_0=transformed_current, act=direct_action
        )
        direct_grad_cost = native_objective_breakdown(
            direct_for_grad, z_goal, mpc_iter, alpha=1.0, base=2.0
        )["total"].sum()
        direct_gradient = torch.autograd.grad(direct_grad_cost, direct_action)[0]

        latent_action = candidate.detach().clone().requires_grad_(True)
        latent_for_grad, _ = rollout_from_zobs(model, z_current, latent_action)
        latent_grad_cost = native_objective_breakdown(
            latent_for_grad, z_goal, mpc_iter, alpha=1.0, base=2.0
        )["total"].sum()
        latent_gradient = torch.autograd.grad(latent_grad_cost, latent_action)[0]

        parity_seed = int(args.seed + mpc_iter)
        _seed_all(parity_seed)
        original_rng_before = _capture_rng_state()
        original_plan, _ = planner.plan(
            obs_0=current, obs_g=goal, actions=None, step=mpc_iter
        )
        original_rng_after = _capture_rng_state()
        full_gd_iterations += int(planner.opt_steps)

        _seed_all(parity_seed)
        latent_rng_before = _capture_rng_state()
        latent_plan, _ = plan_from_zobs(
            planner, z_current, z_goal, actions=None, step=mpc_iter
        )
        latent_rng_after = _capture_rng_state()
        full_gd_iterations += int(planner.opt_steps)

        rollout_close = _zobs_close(
            direct_rollout,
            latent_rollout,
            PARITY_THRESHOLDS["rollout_atol"],
            PARITY_THRESHOLDS["rollout_rtol"],
        ) and torch.allclose(
            direct_full,
            latent_full,
            atol=PARITY_THRESHOLDS["rollout_atol"],
            rtol=PARITY_THRESHOLDS["rollout_rtol"],
        )
        cost_abs = _max_abs(direct_cost, latent_cost)
        gradient_abs = _max_abs(direct_gradient, latent_gradient)
        gradient_close = torch.allclose(
            direct_gradient,
            latent_gradient,
            atol=PARITY_THRESHOLDS["gradient_atol"],
            rtol=PARITY_THRESHOLDS["gradient_rtol"],
        )
        gd_action_abs = _max_abs(original_plan, latent_plan)
        gd_action_close = torch.allclose(
            original_plan,
            latent_plan,
            atol=PARITY_THRESHOLDS["gd_action_atol"],
            rtol=PARITY_THRESHOLDS["gd_action_rtol"],
        )
        record = {
            "sample_id": args.sample_id,
            "mpc_iter": mpc_iter,
            "seed": parity_seed,
            "objective_stage": native_objective_breakdown(
                direct_rollout, z_goal, mpc_iter, alpha=1.0, base=2.0
            )["stage"],
            "rollout_max_abs": _zobs_max_abs(direct_rollout, latent_rollout),
            "full_latent_max_abs": _max_abs(direct_full, latent_full),
            "rollout_close": rollout_close,
            "cost_abs_difference": cost_abs,
            "cost_close": cost_abs <= PARITY_THRESHOLDS["cost_atol"],
            "action_gradient_max_abs": gradient_abs,
            "action_gradient_close": gradient_close,
            "gd_action_max_abs": gd_action_abs,
            "gd_action_close": gd_action_close,
            "rng_before_exact": _rng_equal(original_rng_before, latent_rng_before),
            "rng_after_exact": _rng_equal(original_rng_after, latent_rng_after),
        }
        record["passed"] = all(
            (
                record["rollout_close"],
                record["cost_close"],
                record["action_gradient_close"],
                record["gd_action_close"],
                record["rng_before_exact"],
                record["rng_after_exact"],
            )
        )
        records.append(record)
        torch.save(
            _cpu_nested(
                {
                    "raw_current": current,
                    "raw_goal": goal,
                    "transformed_current": transformed_current,
                    "transformed_goal": transformed_goal,
                    "encoded_current": z_current,
                    "encoded_goal": z_goal,
                    "input_action": candidate,
                    "direct_rollout": direct_rollout,
                    "latent_rollout": latent_rollout,
                    "direct_full_latent": direct_full,
                    "latent_full_latent": latent_full,
                    "direct_cost": direct_cost,
                    "latent_cost": latent_cost,
                    "direct_action_gradient": direct_gradient,
                    "latent_action_gradient": latent_gradient,
                    "original_gd_action": original_plan,
                    "latent_gd_action": latent_plan,
                }
            ),
            raw_dir / f"sample{args.sample_id}_mpc{mpc_iter}.pt",
        )

    fixed_model_exact = versions_before == _tensor_versions(model)
    result = {
        "stage": "S2_latent_interface_parity",
        "device": str(device),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "donor_dir": str(args.donor_dir.resolve()),
        "thresholds": PARITY_THRESHOLDS,
        "threshold_source": "predeclared FP32 interface-equivalence tolerances; not tuned after results",
        "observations": records,
        "full_gd_iterations": full_gd_iterations,
        "model_parameters_and_buffers_unchanged": fixed_model_exact,
        "passed": bool(
            len(records) == 2
            and all(record["passed"] for record in records)
            and fixed_model_exact
        ),
    }
    _write_json(result_path, result)
    return result


def _manifest_physical_groups(manifest_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in manifest_rows:
        grouped[row["physical_branch_id"]].append(row)
    output = []
    for physical_id, rows in grouped.items():
        first = rows[0]
        output.append(
            {
                "physical_branch_id": physical_id,
                "anchor_id": first["anchor_id"],
                "anchor_ordinal": int(first["anchor_ordinal"]),
                "split": first["split"],
                "shape": first["shape"],
                "sample_id": int(first["sample_id"]),
                "mpc_iter": int(first["mpc_iter"]),
                "labels": ";".join(sorted(row["label"] for row in rows)),
                "candidate_ids": ";".join(
                    sorted(row["candidate_id"] for row in rows)
                ),
                "logical_rows": rows,
            }
        )
    return sorted(output, key=lambda row: row["physical_branch_id"])


def _advance_one_chunk(model, z_current: dict, action: torch.Tensor) -> dict:
    predicted, _ = rollout_from_zobs(model, z_current, action)
    return {key: value[:, -1:].detach() for key, value in predicted.items()}


def _selection_diagnostics(
    rows: list[dict], prediction_field: str, truth_field: str, epsilon: float = EPSILON
) -> dict:
    if not rows:
        raise ValueError("selection diagnostics require candidates")
    predictions = [float(row[prediction_field]) for row in rows]
    truths = [float(row[truth_field]) for row in rows]
    minimum_prediction = min(predictions)
    minimum_truth = min(truths)
    exact_indices = [
        index for index, value in enumerate(predictions) if value == minimum_prediction
    ]
    tie_indices = [
        index
        for index, value in enumerate(predictions)
        if value <= minimum_prediction + epsilon
    ]
    selected_index = exact_indices[0]
    sorted_predictions = sorted(set(predictions))
    margin = (
        sorted_predictions[1] - sorted_predictions[0]
        if len(sorted_predictions) > 1
        else 0.0
    )
    return {
        "candidate_count": len(rows),
        "selected_key": rows[selected_index]["candidate_key"],
        "exact_argmin_keys": ";".join(rows[index]["candidate_key"] for index in exact_indices),
        "epsilon_tie_keys": ";".join(rows[index]["candidate_key"] for index in tie_indices),
        "prediction_margin": margin,
        "selected_truth_regret": truths[selected_index] - minimum_truth,
        "tie_aware_truth_regret_min": min(truths[index] for index in tie_indices)
        - minimum_truth,
        "tie_aware_truth_regret_max": max(truths[index] for index in tie_indices)
        - minimum_truth,
        "true_best_keys": ";".join(
            row["candidate_key"]
            for row in rows
            if float(row[truth_field]) == minimum_truth
        ),
    }


def _forecast_blind(args, manifest_rows: list[dict]):
    device = torch.device(args.device)
    started = time.perf_counter()
    model, preprocessor, config = _load_runtime(args.checkpoint_dir, device)
    cuda_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    ) if torch.cuda.is_available() and device.type == "cuda" else None
    if cuda_index is not None:
        # This runtime requires an initialized CUDA context before resetting stats.
        torch.cuda.reset_peak_memory_stats(cuda_index)
    forecast_model = model
    policy_model = model
    planner = _build_planner(policy_model, preprocessor, int(config.frameskip))
    versions_before = _tensor_versions(model)
    donors = {split: path.resolve() for split, path in args.donor}
    if set(donors) != {"val_T", "val_L"}:
        raise ValueError(f"Expected val_T and val_L donors, got {sorted(donors)}")

    logged_frozen_scores = {
        (row["anchor_id"], row["candidate_id"]): float(row["c_model"])
        for row in _read_csv(args.b_candidate_scores)
        if row["analysis"] == "b1_common_anchor" and row["variant"] == "frozen"
    }
    sidecar_dir = args.out / "matched_feedback_sidecars"
    if sidecar_dir.exists() and any(sidecar_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {sidecar_dir}")
    sidecar_dir.mkdir(exist_ok=True)

    logical_blind = []
    physical_blind = []
    anchor_cache = {}
    reproduction_errors = []
    first_chunk_failures = []
    nonfinite = []
    groups = _manifest_physical_groups(manifest_rows)

    for group in groups:
        anchor_key = (group["split"], group["sample_id"], group["mpc_iter"])
        if anchor_key not in anchor_cache:
            current, goal = _load_anchor_observations(
                donors[group["split"]], group["sample_id"], group["mpc_iter"]
            )
            transformed_current, transformed_goal = _transform_anchor_obs(
                preprocessor, current, goal, device
            )
            with torch.no_grad():
                anchor_cache[anchor_key] = {
                    "z_current": forecast_model.encode_obs(transformed_current),
                    "z_goal": forecast_model.encode_obs(transformed_goal),
                }
        z_initial = anchor_cache[anchor_key]["z_current"]
        z_goal = anchor_cache[anchor_key]["z_goal"]

        full_actions = []
        for logical in group["logical_rows"]:
            full_action = _candidate_tensor(
                donors[group["split"]],
                group["sample_id"],
                group["mpc_iter"],
                logical["candidate_id"],
            ).to(device)
            full_actions.append(full_action)
            with torch.no_grad():
                predicted_open, _ = rollout_from_zobs(
                    forecast_model, z_initial, full_action
                )
                breakdown = native_objective_breakdown(
                    predicted_open,
                    z_goal,
                    group["mpc_iter"],
                    alpha=1.0,
                    base=2.0,
                )
                j_model = float(breakdown["total"].item())
            logged = logged_frozen_scores[(group["anchor_id"], logical["candidate_id"])]
            reproduction_errors.append(abs(j_model - logged))
            logical_blind.append(
                {
                    "anchor_id": group["anchor_id"],
                    "anchor_ordinal": group["anchor_ordinal"],
                    "split": group["split"],
                    "shape": group["shape"],
                    "sample_id": group["sample_id"],
                    "mpc_iter": group["mpc_iter"],
                    "label": logical["label"],
                    "candidate_id": logical["candidate_id"],
                    "physical_branch_id": group["physical_branch_id"],
                    "J_model_open_plan_frozen_forecaster": j_model,
                    "J_model_objective_stage": breakdown["stage"],
                    "logged_frozen_J_model": logged,
                    "J_model_reproduction_abs_error": abs(j_model - logged),
                }
            )
        first_chunks = [action[:, :1] for action in full_actions]
        if not all(torch.equal(first_chunks[0], other) for other in first_chunks[1:]):
            first_chunk_failures.append(group["physical_branch_id"])
        first_chunk = first_chunks[0]

        with torch.no_grad():
            z_current = _advance_one_chunk(
                forecast_model, _clone_zobs(z_initial), first_chunk
            )
        chosen_chunks = [first_chunk.detach().cpu()]
        tail_plans = []
        for tail_index in range(4):
            seed = continuation_seed(
                args.seed, group["anchor_ordinal"], tail_index
            )
            _seed_all(seed)
            plan, _ = plan_from_zobs(
                planner,
                z_current,
                z_goal,
                actions=None,
                step=group["mpc_iter"] + 1 + tail_index,
            )
            plan = plan.detach()
            selected = plan[:, :1].clone()
            tail_plans.append(plan.cpu())
            chosen_chunks.append(selected.cpu())
            with torch.no_grad():
                z_current = _advance_one_chunk(
                    forecast_model, z_current, selected
                )

        with torch.no_grad():
            qhat, qhat_visual, qhat_proprio = _terminal_reference_cost(
                z_current, z_goal
            )
        qhat_value = float(qhat.item())
        if not math.isfinite(qhat_value):
            nonfinite.append(group["physical_branch_id"])
        sidecar = sidecar_dir / f"{group['physical_branch_id']}.pt"
        torch.save(
            _cpu_nested(
                {
                    "initial_z_obs": z_initial,
                    "goal_z_obs": z_goal,
                    "first_chunk": first_chunk,
                    "tail_full_plans": tail_plans,
                    "executed_model_chunks": chosen_chunks,
                    "final_predicted_z_obs": z_current,
                    "Qhat_total": qhat,
                    "Qhat_visual": qhat_visual,
                    "Qhat_proprio": qhat_proprio,
                }
            ),
            sidecar,
        )
        physical_blind.append(
            {
                "physical_branch_id": group["physical_branch_id"],
                "anchor_id": group["anchor_id"],
                "anchor_ordinal": group["anchor_ordinal"],
                "split": group["split"],
                "shape": group["shape"],
                "sample_id": group["sample_id"],
                "mpc_iter": group["mpc_iter"],
                "labels": group["labels"],
                "candidate_ids": group["candidate_ids"],
                "Qhat_common_continuation": qhat_value,
                "Qhat_visual": float(qhat_visual.item()),
                "Qhat_proprio": float(qhat_proprio.item()),
                "prediction_sidecar": str(sidecar.resolve()),
            }
        )

    model_exact = versions_before == _tensor_versions(model)
    wall_clock = time.perf_counter() - started
    max_memory = (
        int(torch.cuda.max_memory_allocated(cuda_index))
        if torch.cuda.is_available() and device.type == "cuda"
        else 0
    )
    validation = {
        "logical_predictions": len(logical_blind),
        "physical_predictions": len(physical_blind),
        "first_chunk_group_failures": first_chunk_failures,
        "nonfinite_predictions": nonfinite,
        "frozen_open_cost_reproduction_max_abs_error": max(
            reproduction_errors, default=math.inf
        ),
        "frozen_open_cost_reproduction_tolerance": 1e-6,
        "model_parameters_and_buffers_unchanged": model_exact,
    }
    validation["passed"] = bool(
        validation["logical_predictions"] == 18
        and validation["physical_predictions"] == 17
        and not first_chunk_failures
        and not nonfinite
        and validation["frozen_open_cost_reproduction_max_abs_error"] <= 1e-6
        and model_exact
    )
    budget = {
        "new_environment_calls": 0,
        "physical_matched_feedback_predictions": len(physical_blind),
        "logical_open_plan_predictions": len(logical_blind),
        "tail_replans": 4 * len(physical_blind),
        "internal_gd_iterations": 400 * len(physical_blind),
        "outer_model_transitions": 5 * len(physical_blind),
        "wall_clock_s": wall_clock,
        "peak_cuda_memory_bytes": max_memory,
    }
    return logical_blind, physical_blind, validation, budget


def _evaluate_blind_predictions(
    logical_blind: list[dict],
    physical_blind: list[dict],
    c_dir: Path,
) -> tuple[list[dict], dict]:
    # This is the first point at which saved C outcomes are read.
    c_logical = {
        (row["anchor_id"], row["label"]): row
        for row in _read_csv(c_dir / "logical_branch_results.csv")
    }
    c_physical = {
        row["physical_branch_id"]: row
        for row in _read_csv(c_dir / "physical_branch_results.csv")
    }
    qhat_by_physical = {
        row["physical_branch_id"]: row for row in physical_blind
    }
    joined = []
    for blind in logical_blind:
        truth_logical = c_logical[(blind["anchor_id"], blind["label"])]
        truth_physical = c_physical[blind["physical_branch_id"]]
        qhat = float(
            qhat_by_physical[blind["physical_branch_id"]][
                "Qhat_common_continuation"
            ]
        )
        j_model = float(blind["J_model_open_plan_frozen_forecaster"])
        j_env = float(truth_logical["recorded_pool_env_cost"])
        q_env = float(truth_physical["final_reference_terminal_cost"])
        e_open = j_env - j_model
        b_pi = q_env - j_env
        total_gap = q_env - j_model
        joined.append(
            {
                **blind,
                "Qhat_common_continuation": qhat,
                "J_env_open_plan": j_env,
                "Q_env_common_continuation": q_env,
                "open_plan_model_error_e": e_open,
                "continuation_target_shift_b_pi": b_pi,
                "Q_env_minus_J_model": total_gap,
                "decomposition_residual": total_gap - (e_open + b_pi),
                "matched_feedback_error_Q_env_minus_Qhat": q_env - qhat,
                "abs_open_plan_model_error": abs(e_open),
                "abs_cross_target_J_model_error": abs(total_gap),
                "abs_matched_feedback_error": abs(q_env - qhat),
                "final_state_dist": float(truth_physical["final_state_dist"]),
                "final_success": truth_physical["final_success"],
                "prediction_sidecar": qhat_by_physical[blind["physical_branch_id"]][
                    "prediction_sidecar"
                ],
            }
        )

    by_anchor = defaultdict(list)
    for row in joined:
        by_anchor[row["anchor_id"]].append(row)
    selection_rows = []
    for anchor_id, rows in sorted(by_anchor.items()):
        frozen = next(row for row in rows if row["label"] == "frozen")
        for row in rows:
            for field in (
                "J_model_open_plan_frozen_forecaster",
                "J_env_open_plan",
                "Qhat_common_continuation",
                "Q_env_common_continuation",
            ):
                row[f"{field}_delta_from_frozen"] = float(row[field]) - float(
                    frozen[field]
                )
            row["matched_relative_error"] = (
                row["Qhat_common_continuation_delta_from_frozen"]
                - row["Q_env_common_continuation_delta_from_frozen"]
            )
            row["open_score_to_common_target_relative_error"] = (
                row["J_model_open_plan_frozen_forecaster_delta_from_frozen"]
                - row["Q_env_common_continuation_delta_from_frozen"]
            )

        all_candidates = [
            {
                **row,
                "candidate_key": row["label"],
            }
            for row in rows
        ]
        legal_candidates = [
            row for row in all_candidates if row["label"] in {"frozen", "adapted"}
        ]
        for scope, candidates in (
            ("all_three_logical_labels", all_candidates),
            ("legal_frozen_adapted", legal_candidates),
        ):
            for predictor, field in (
                ("open_plan_J_model", "J_model_open_plan_frozen_forecaster"),
                ("matched_feedback_Qhat", "Qhat_common_continuation"),
            ):
                selection_rows.append(
                    {
                        "anchor_id": anchor_id,
                        "scope": scope,
                        "predictor": predictor,
                        **_selection_diagnostics(
                            candidates,
                            field,
                            "Q_env_common_continuation",
                        ),
                    }
                )

    decomposition_max = max(abs(float(row["decomposition_residual"])) for row in joined)
    raw_matched = [float(row["abs_matched_feedback_error"]) for row in joined]
    raw_cross = [float(row["abs_cross_target_J_model_error"]) for row in joined]
    rel_matched = [
        abs(float(row["matched_relative_error"]))
        for row in joined
        if row["label"] != "frozen"
    ]
    rel_cross = [
        abs(float(row["open_score_to_common_target_relative_error"]))
        for row in joined
        if row["label"] != "frozen"
    ]
    selection_summary = defaultdict(dict)
    for scope in ("all_three_logical_labels", "legal_frozen_adapted"):
        for predictor in ("open_plan_J_model", "matched_feedback_Qhat"):
            subset = [
                row
                for row in selection_rows
                if row["scope"] == scope and row["predictor"] == predictor
            ]
            selection_summary[scope][predictor] = {
                "anchors": len(subset),
                "exact_selected_truth_regret_mean": statistics.fmean(
                    float(row["selected_truth_regret"]) for row in subset
                ),
                "tie_aware_truth_regret_min_mean": statistics.fmean(
                    float(row["tie_aware_truth_regret_min"]) for row in subset
                ),
                "tie_aware_zero_regret_anchors": sum(
                    float(row["tie_aware_truth_regret_min"]) <= EPSILON
                    for row in subset
                ),
            }
    summary = {
        "logical_rows": len(joined),
        "anchors": len(by_anchor),
        "decomposition_max_abs_residual": decomposition_max,
        "raw_error": {
            "matched_Qhat_to_Qenv_mae": statistics.fmean(raw_matched),
            "cross_target_Jmodel_to_Qenv_mae": statistics.fmean(raw_cross),
        },
        "relative_to_frozen_error": {
            "matched_Qhat_to_Qenv_mae": statistics.fmean(rel_matched),
            "cross_target_Jmodel_to_Qenv_mae": statistics.fmean(rel_cross),
        },
        "selection": dict(selection_summary),
        "selection_rows": selection_rows,
    }
    return joined, summary


def run_forecast(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    target_paths = (
        args.out / "blind_open_plan_predictions.csv",
        args.out / "blind_matched_feedback_predictions.csv",
        args.out / "matched_feedback_predictions.csv",
        args.out / "matched_feedback_selection.csv",
        args.out / "matched_feedback_summary.json",
        args.out / "matched_feedback_validation.json",
        args.out / "budgets.json",
    )
    if any(path.exists() for path in target_paths):
        raise FileExistsError("matched-feedback output already exists")
    manifest_rows = _read_csv(args.manifest)
    if len(manifest_rows) != 18:
        raise ValueError(f"Expected 18 preregistered logical rows, got {len(manifest_rows)}")

    logical_blind, physical_blind, validation, budget = _forecast_blind(
        args, manifest_rows
    )
    _write_csv(args.out / "blind_open_plan_predictions.csv", logical_blind)
    _write_csv(args.out / "blind_matched_feedback_predictions.csv", physical_blind)
    predictions_saved_ns = time.time_ns()

    joined, evaluation = _evaluate_blind_predictions(
        logical_blind, physical_blind, args.c_dir
    )
    _write_csv(args.out / "matched_feedback_predictions.csv", joined)
    _write_csv(
        args.out / "matched_feedback_selection.csv", evaluation.pop("selection_rows")
    )
    validation.update(
        {
            "predictions_saved_before_truth_evaluation": True,
            "predictions_saved_unix_ns": predictions_saved_ns,
            "decomposition_max_abs_residual": evaluation[
                "decomposition_max_abs_residual"
            ],
            "truth_rows_joined": len(joined),
        }
    )
    validation["passed"] = bool(
        validation["passed"]
        and validation["truth_rows_joined"] == 18
        and validation["decomposition_max_abs_residual"] <= 1e-12
    )
    _write_json(args.out / "matched_feedback_summary.json", evaluation)
    _write_json(args.out / "matched_feedback_validation.json", validation)
    _write_json(args.out / "budgets.json", budget)
    return {"evaluation": evaluation, "validation": validation, "budget": budget}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parity = subparsers.add_parser("parity")
    parity.add_argument("--checkpoint-dir", type=Path, required=True)
    parity.add_argument("--donor-dir", type=Path, required=True)
    parity.add_argument("--sample-id", type=int, default=0)
    parity.add_argument("--mpc-iter", type=int, action="append", required=True)
    parity.add_argument("--out", type=Path, required=True)
    parity.add_argument("--device", default="cuda:0")
    parity.add_argument("--seed", type=int, default=290920)

    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--checkpoint-dir", type=Path, required=True)
    forecast.add_argument("--manifest", type=Path, required=True)
    forecast.add_argument("--c-dir", type=Path, required=True)
    forecast.add_argument("--b-candidate-scores", type=Path, required=True)
    forecast.add_argument("--donor", type=_parse_donor, action="append", required=True)
    forecast.add_argument("--out", type=Path, required=True)
    forecast.add_argument("--device", default="cuda:0")
    forecast.add_argument("--seed", type=int, default=270920)

    args = parser.parse_args()
    result = run_parity(args) if args.command == "parity" else run_forecast(args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    passed = result.get("passed", result.get("validation", {}).get("passed", False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
