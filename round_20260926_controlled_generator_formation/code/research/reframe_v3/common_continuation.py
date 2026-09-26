"""Phase C: isolate one pool-selection decision under a common controller.

Each physical branch replays the complete Frozen-donor prefix, executes one
pre-registered candidate chunk, and then uses the unchanged Frozen checkpoint
for four zero-warmstart GD replans with branch-independent RNG seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.reframe_v3.shadow_selection_audit import (
    _capture_rng_state,
    _jsonable,
    _load_runtime,
    _seed_all,
    _write_csv,
)


LOGICAL_LABELS = ("frozen", "adapted", "oracle")


def continuation_seed(base_seed: int, anchor_ordinal: int, tail_index: int) -> int:
    return int(base_seed + anchor_ordinal * 10 + tail_index)


def deduplicate_first_chunks(logical_branches: list[dict]) -> list[dict]:
    physical = []
    for logical in logical_branches:
        match = next(
            (
                branch
                for branch in physical
                if torch.equal(branch["first_chunk"], logical["first_chunk"])
            ),
            None,
        )
        if match is None:
            physical.append(
                {
                    "labels": [logical["label"]],
                    "candidate_ids": {logical["label"]: logical["candidate_id"]},
                    "first_chunk": logical["first_chunk"].detach().cpu().clone(),
                }
            )
        else:
            match["labels"].append(logical["label"])
            match["candidate_ids"][logical["label"]] = logical["candidate_id"]
    return physical


def paired_outcome(before: dict, after: dict, comparison: str) -> dict:
    return {
        "comparison": comparison,
        "before_label": before["label"],
        "after_label": after["label"],
        "delta_first_chunk_state_dist": float(after["first_chunk_state_dist"])
        - float(before["first_chunk_state_dist"]),
        "delta_first_chunk_reference_terminal_cost": float(
            after["first_chunk_reference_terminal_cost"]
        )
        - float(before["first_chunk_reference_terminal_cost"]),
        "delta_final_state_dist": float(after["final_state_dist"])
        - float(before["final_state_dist"]),
        "delta_final_reference_terminal_cost": float(
            after["final_reference_terminal_cost"]
        )
        - float(before["final_reference_terminal_cost"]),
        "delta_final_success": int(bool(after["final_success"]))
        - int(bool(before["final_success"])),
    }


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV input: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_donor(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("donor must use SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if not split or not raw_path:
        raise argparse.ArgumentTypeError("donor must use nonempty SPLIT=PATH")
    return split, Path(raw_path)


def _candidate_tensor(
    donor_dir: Path, sample_id: int, mpc_iter: int, candidate_id: str
) -> torch.Tensor:
    try:
        gd_text, side = candidate_id.split("_", 1)
        gd_iter = int(gd_text.removeprefix("g"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid candidate id: {candidate_id}") from exc
    if side not in ("before", "after"):
        raise ValueError(f"Invalid candidate side: {candidate_id}")
    path = donor_dir / "tensors" / f"s{sample_id}_m{mpc_iter}_g{gd_iter}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate tensor: {path}")
    payload = torch.load(path, map_location="cpu")
    tensor = payload[f"u_{side}"].detach().cpu()
    if tuple(tensor.shape) != (1, 5, 10):
        raise ValueError(f"Unexpected candidate shape in {path}: {tensor.shape}")
    return tensor


def _load_anchor_assets(donor_dir: Path, sample_id: int, mpc_iter: int) -> dict:
    history_path = donor_dir / "real_evidence" / f"s{sample_id}_history_mpc{mpc_iter}.pt"
    initial_path = donor_dir / "real_evidence" / f"sample_{sample_id:03d}_initial.npz"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history asset: {history_path}")
    if not initial_path.exists():
        raise FileNotFoundError(f"Missing initial asset: {initial_path}")
    history = torch.load(history_path, map_location="cpu")
    with np.load(initial_path, allow_pickle=False) as archive:
        initial = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required_initial = {
        "initial_state",
        "goal_state",
        "goal_visual",
        "goal_proprio",
    }
    missing = required_initial - set(initial)
    if missing:
        raise ValueError(f"Initial asset {initial_path} is missing {sorted(missing)}")
    prefix = history.get("executed_prefix")
    if prefix is None or tuple(prefix.shape) != (1, mpc_iter, 10):
        raise ValueError(
            f"Unexpected prefix shape at {history_path}: "
            f"{None if prefix is None else tuple(prefix.shape)}"
        )
    return {
        "prefix": prefix.detach().cpu(),
        "current_observation": {
            key: np.asarray(value).copy()
            for key, value in history["current_real_observation"].items()
        },
        "current_state": np.asarray(history["current_public_state"]).copy(),
        **initial,
    }


def _to_environment_actions(model_actions: torch.Tensor, preprocessor, frameskip: int):
    actions = model_actions.detach().cpu()
    batch, horizon, packed = actions.shape
    if packed % frameskip != 0:
        raise ValueError(f"Packed action dimension {packed} is not divisible by {frameskip}")
    environment_dim = packed // frameskip
    unpacked = actions.reshape(batch, horizon * frameskip, environment_dim)
    return preprocessor.denormalize_actions(unpacked)[0].numpy()


def _as_batch_time(obs: dict) -> dict:
    return {
        key: np.expand_dims(np.expand_dims(np.asarray(value), 0), 0)
        for key, value in obs.items()
    }


def _reference_terminal_cost(model, preprocessor, obs: dict, goal: dict) -> float:
    from utils import move_to_device

    device = next(model.parameters()).device
    transformed_obs = move_to_device(
        preprocessor.transform_obs(_as_batch_time(obs)), device
    )
    transformed_goal = move_to_device(preprocessor.transform_obs(goal), device)
    model.eval()
    with torch.no_grad():
        obs_z = model.encode_obs(transformed_obs)
        goal_z = model.encode_obs(transformed_goal)
        visual = F.mse_loss(obs_z["visual"], goal_z["visual"])
        proprio = F.mse_loss(obs_z["proprio"], goal_z["proprio"])
    return float((visual + proprio).item())


def _rng_equal(left: dict, right: dict) -> bool:
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0]:
        return False
    if not np.array_equal(left["numpy"][1], right["numpy"][1]):
        return False
    if left["numpy"][2:] != right["numpy"][2:]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    if len(left["torch_cuda"]) != len(right["torch_cuda"]):
        return False
    return all(
        torch.equal(first, second)
        for first, second in zip(left["torch_cuda"], right["torch_cuda"])
    )


class _WandbSink:
    def log(self, *args, **kwargs):
        return None


class _EvaluatorStub:
    def __init__(self, frameskip: int):
        self.frameskip = int(frameskip)


def _build_planner(model, preprocessor, frameskip: int):
    import planning.gd as gd_module
    from planning.objectives import create_objective_fn

    gd_module.tqdm = lambda iterable: iterable
    return gd_module.GDPlanner(
        horizon=5,
        action_noise=0.0,
        sample_type="zero",
        lr=0.1,
        opt_steps=100,
        eval_every=-1,
        wm=model,
        action_dim=10,
        objective_fn=create_objective_fn(alpha=1.0, base=2.0, mode="staged"),
        preprocessor=preprocessor,
        evaluator=_EvaluatorStub(frameskip),
        wandb_run=_WandbSink(),
        log_filename=None,
        optimizer="adam",
        use_cosine_scheduler=True,
    )


def _tensor_versions(model) -> tuple[int, ...]:
    tensors = list(model.parameters()) + list(model.buffers())
    return tuple(int(tensor._version) for tensor in tensors)


def _logical_manifest(
    selection_rows: list[dict], donors: dict[str, Path], mpc_iter: int
) -> tuple[list[dict], list[dict]]:
    relevant = [
        row
        for row in selection_rows
        if row["analysis"] == "b1_common_anchor"
        and int(row["mpc_iter"]) == mpc_iter
    ]
    anchors = sorted({row["anchor_id"] for row in relevant})
    logical_manifest = []
    physical_specs = []
    for anchor_ordinal, anchor_id in enumerate(anchors):
        anchor_rows = [row for row in relevant if row["anchor_id"] == anchor_id]
        by_variant = {row["variant"]: row for row in anchor_rows}
        for required in ("frozen", "predictor_official_shadow"):
            if required not in by_variant:
                raise ValueError(f"Anchor {anchor_id} is missing B1 variant {required}")
        frozen = by_variant["frozen"]
        adapted = by_variant["predictor_official_shadow"]
        split = frozen["split"]
        if split not in donors:
            raise ValueError(f"No donor path registered for split {split}")
        donor_dir = donors[split]
        sample_id = int(frozen["sample_id"])
        choices = [
            {
                "label": "frozen",
                "candidate_id": frozen["model_best_candidate_id"],
                "recorded_pool_env_cost": float(frozen["selected_env_cost"]),
            },
            {
                "label": "adapted",
                "candidate_id": adapted["model_best_candidate_id"],
                "recorded_pool_env_cost": float(adapted["selected_env_cost"]),
            },
            {
                "label": "oracle",
                "candidate_id": frozen["env_best_candidate_id"],
                "recorded_pool_env_cost": float(frozen["env_best_cost"]),
            },
        ]
        for choice in choices:
            full_action = _candidate_tensor(
                donor_dir, sample_id, mpc_iter, choice["candidate_id"]
            )
            choice["first_chunk"] = full_action[:, :1].clone()
        physical = deduplicate_first_chunks(choices)
        physical_by_label = {}
        for branch_index, branch in enumerate(physical):
            physical_id = f"{split}_{frozen['shape']}_s{sample_id}_m{mpc_iter}_b{branch_index}"
            branch.update(
                {
                    "physical_branch_id": physical_id,
                    "anchor_id": anchor_id,
                    "anchor_ordinal": anchor_ordinal,
                    "split": split,
                    "shape": frozen["shape"],
                    "sample_id": sample_id,
                    "mpc_iter": mpc_iter,
                    "donor_dir": donor_dir,
                }
            )
            physical_specs.append(branch)
            for label in branch["labels"]:
                physical_by_label[label] = physical_id
        for choice in choices:
            logical_manifest.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_ordinal": anchor_ordinal,
                    "split": split,
                    "shape": frozen["shape"],
                    "sample_id": sample_id,
                    "mpc_iter": mpc_iter,
                    "label": choice["label"],
                    "candidate_id": choice["candidate_id"],
                    "recorded_pool_env_cost": choice["recorded_pool_env_cost"],
                    "physical_branch_id": physical_by_label[choice["label"]],
                    "first_chunk_deduplicated": sum(
                        physical_by_label[other] == physical_by_label[choice["label"]]
                        for other in LOGICAL_LABELS
                    )
                    > 1,
                }
            )
    return logical_manifest, physical_specs


def _run_physical_branch(
    spec: dict,
    assets: dict,
    model,
    preprocessor,
    planner,
    frameskip: int,
    base_seed: int,
    rng_references: dict,
    rng_states: dict,
    output_dir: Path,
) -> tuple[dict, list[str]]:
    from env.pusht.pusht_wrapper import PushTWrapper
    from research.replay_oracle import preserve_global_rng_state

    prefix = assets["prefix"]
    first_chunk = spec["first_chunk"]
    combined = torch.cat([prefix, first_chunk], dim=1)
    environment_actions = _to_environment_actions(combined, preprocessor, frameskip)
    eval_seed = int(100 * spec["sample_id"] + 1)
    goal_state = np.asarray(assets["goal_state"])
    goal_obs = {
        "visual": np.asarray(assets["goal_visual"]),
        "proprio": np.asarray(assets["goal_proprio"]),
    }
    validation_failures = []
    with preserve_global_rng_state():
        env = PushTWrapper(with_velocity=True, with_target=True)
    env.update_env({"shape": spec["shape"]})
    try:
        all_obs, all_states = env.rollout(
            eval_seed,
            np.asarray(assets["initial_state"]),
            environment_actions,
        )
        prefix_environment_steps = int(prefix.shape[1] * frameskip)
        anchor_obs = {
            key: np.asarray(value)[prefix_environment_steps]
            for key, value in all_obs.items()
        }
        anchor_state = np.asarray(all_states)[prefix_environment_steps]
        expected_anchor_obs = {
            key: np.asarray(value)[0, -1]
            for key, value in assets["current_observation"].items()
        }
        expected_anchor_state = np.asarray(assets["current_state"]).reshape(
            -1, anchor_state.shape[-1]
        )[0]
        for key in ("visual", "proprio"):
            if not np.array_equal(anchor_obs[key], expected_anchor_obs[key]):
                validation_failures.append(
                    f"{spec['physical_branch_id']}:anchor_{key}"
                )
        if not np.array_equal(anchor_state, expected_anchor_state):
            validation_failures.append(f"{spec['physical_branch_id']}:anchor_state")

        current_obs = {key: np.asarray(value)[-1] for key, value in all_obs.items()}
        current_state = np.asarray(all_states)[-1]
        first_metrics = env.eval_state(goal_state, current_state)
        first_reference_cost = _reference_terminal_cost(
            model, preprocessor, current_obs, goal_obs
        )
        stride_obs = {
            key: [anchor_obs[key], current_obs[key]] for key in ("visual", "proprio")
        }
        stride_states = [anchor_state, current_state]
        chosen_chunks = [first_chunk]
        tail_full_plans = []
        planner_seeds = []

        for tail_index in range(4):
            seed = continuation_seed(base_seed, spec["anchor_ordinal"], tail_index)
            planner_seeds.append(seed)
            _seed_all(seed)
            rng_before = _capture_rng_state()
            rng_key = f"{spec['anchor_id']}|tail{tail_index}"
            if rng_key not in rng_references:
                rng_references[rng_key] = rng_before
            elif not _rng_equal(rng_references[rng_key], rng_before):
                validation_failures.append(
                    f"{spec['physical_branch_id']}:rng_before_tail{tail_index}"
                )
            obs_batch = _as_batch_time(current_obs)
            full_plan, _ = planner.plan(
                obs_0=obs_batch,
                obs_g=goal_obs,
                actions=None,
                step=spec["mpc_iter"] + 1 + tail_index,
            )
            rng_after = _capture_rng_state()
            rng_states[f"{spec['physical_branch_id']}|tail{tail_index}"] = {
                "seed": seed,
                "before": rng_before,
                "after": rng_after,
            }
            full_plan = full_plan.detach().cpu()
            selected = full_plan[:, :1].clone()
            tail_full_plans.append(full_plan.numpy())
            chosen_chunks.append(selected)
            step_actions = _to_environment_actions(
                selected, preprocessor, frameskip
            )
            step_obs, _, _, step_info = env.step_multiple(step_actions)
            current_obs = {
                key: np.asarray(value)[-1] for key, value in step_obs.items()
            }
            current_state = np.asarray(step_info["state"])[-1]
            for key in stride_obs:
                stride_obs[key].append(current_obs[key])
            stride_states.append(current_state)

        final_metrics = env.eval_state(goal_state, current_state)
        final_reference_cost = _reference_terminal_cost(
            model, preprocessor, current_obs, goal_obs
        )
        sidecar = output_dir / "trajectories" / f"{spec['physical_branch_id']}.npz"
        np.savez_compressed(
            sidecar,
            visual=np.stack(stride_obs["visual"]),
            proprio=np.stack(stride_obs["proprio"]),
            states=np.stack(stride_states),
            prefix_model_actions=prefix.numpy(),
            executed_model_actions=torch.cat(chosen_chunks, dim=1).numpy(),
            tail_full_plans=np.stack(tail_full_plans),
            planner_seeds=np.asarray(planner_seeds, dtype=np.int64),
        )
        result = {
            "physical_branch_id": spec["physical_branch_id"],
            "anchor_id": spec["anchor_id"],
            "anchor_ordinal": spec["anchor_ordinal"],
            "split": spec["split"],
            "shape": spec["shape"],
            "sample_id": spec["sample_id"],
            "mpc_iter": spec["mpc_iter"],
            "labels": ";".join(spec["labels"]),
            "candidate_ids": json.dumps(spec["candidate_ids"], separators=(",", ":")),
            "first_chunk_state_dist": float(first_metrics["state_dist"]),
            "first_chunk_success": bool(first_metrics["success"]),
            "first_chunk_reference_terminal_cost": first_reference_cost,
            "final_state_dist": float(final_metrics["state_dist"]),
            "final_success": bool(final_metrics["success"]),
            "final_reference_terminal_cost": final_reference_cost,
            "prefix_replay_environment_steps": prefix_environment_steps,
            "first_decision_environment_steps": frameskip,
            "tail_environment_steps": 4 * frameskip,
            "total_environment_steps": prefix_environment_steps + 5 * frameskip,
            "tail_replans": 4,
            "tail_gd_steps": 400,
            "trajectory_sidecar": str(sidecar),
        }
        return result, validation_failures
    finally:
        env.close()


def _aggregate_effects(rows: list[dict], tolerance: float = 1e-6) -> dict:
    if not rows:
        return {"n": 0}
    state = [float(row["delta_final_state_dist"]) for row in rows]
    cost = [float(row["delta_final_reference_terminal_cost"]) for row in rows]
    return {
        "n": len(rows),
        "final_state_dist_improved": sum(value < -tolerance for value in state),
        "final_state_dist_worsened": sum(value > tolerance for value in state),
        "final_state_dist_unchanged": sum(abs(value) <= tolerance for value in state),
        "mean_delta_final_state_dist": statistics.fmean(state),
        "median_delta_final_state_dist": statistics.median(state),
        "final_reference_cost_improved": sum(value < -tolerance for value in cost),
        "final_reference_cost_worsened": sum(value > tolerance for value in cost),
        "final_reference_cost_unchanged": sum(abs(value) <= tolerance for value in cost),
        "mean_delta_final_reference_terminal_cost": statistics.fmean(cost),
        "median_delta_final_reference_terminal_cost": statistics.median(cost),
        "net_success_delta": sum(int(row["delta_final_success"]) for row in rows),
    }


def run(args) -> dict:
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trajectories").mkdir()
    started = time.perf_counter()
    donors = {split: path.resolve() for split, path in args.donor}
    if set(donors) != {"val_T", "val_L"}:
        raise ValueError(f"Expected exactly val_T and val_L donors, got {sorted(donors)}")
    selection_rows = _read_csv(args.selection_metrics)
    logical_manifest, physical_specs = _logical_manifest(
        selection_rows, donors, mpc_iter=4
    )
    if len(logical_manifest) != 18:
        raise ValueError(f"Expected 18 logical branches, got {len(logical_manifest)}")
    if len(physical_specs) > 18:
        raise ValueError(f"Physical branch cap exceeded: {len(physical_specs)}")

    # This manifest is materialized before any new environment branch executes.
    _write_csv(args.out / "pre_execution_manifest.csv", logical_manifest)
    with (args.out / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(
                {
                    "selection_metrics": args.selection_metrics.resolve(),
                    "checkpoint_dir": args.checkpoint_dir.resolve(),
                    "donors": donors,
                    "logical_branches": len(logical_manifest),
                    "physical_branches": len(physical_specs),
                    "tail_chunks": 4,
                    "warmstart_rule": "zero_environment_action_normalized_by_preprocessor",
                    "common_continuation_model": "unchanged_frozen_checkpoint",
                    "base_seed": args.seed,
                }
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    device = torch.device(args.device)
    model, preprocessor, model_config = _load_runtime(args.checkpoint_dir, device)
    frameskip = int(model_config.frameskip)
    if frameskip != 5:
        raise ValueError(f"PushObj V3 continuation requires frameskip=5, got {frameskip}")
    planner = _build_planner(model, preprocessor, frameskip)
    model_versions_before = _tensor_versions(model)
    rng_references = {}
    rng_states = {}
    physical_results = []
    validation_failures = []

    assets_cache = {}
    for spec in physical_specs:
        cache_key = (spec["split"], spec["sample_id"], spec["mpc_iter"])
        if cache_key not in assets_cache:
            assets_cache[cache_key] = _load_anchor_assets(
                spec["donor_dir"], spec["sample_id"], spec["mpc_iter"]
            )
        result, failures = _run_physical_branch(
            spec,
            assets_cache[cache_key],
            model,
            preprocessor,
            planner,
            frameskip,
            args.seed,
            rng_references,
            rng_states,
            args.out,
        )
        physical_results.append(result)
        validation_failures.extend(failures)

    fixed_model_exact = model_versions_before == _tensor_versions(model)
    if not fixed_model_exact:
        validation_failures.append("frozen_model_tensor_versions_changed")
    by_physical = {row["physical_branch_id"]: row for row in physical_results}
    logical_results = []
    for manifest in logical_manifest:
        physical = by_physical[manifest["physical_branch_id"]]
        logical_results.append(
            {
                **manifest,
                "first_chunk_state_dist": physical["first_chunk_state_dist"],
                "first_chunk_success": physical["first_chunk_success"],
                "first_chunk_reference_terminal_cost": physical[
                    "first_chunk_reference_terminal_cost"
                ],
                "final_state_dist": physical["final_state_dist"],
                "final_success": physical["final_success"],
                "final_reference_terminal_cost": physical[
                    "final_reference_terminal_cost"
                ],
                "trajectory_sidecar": physical["trajectory_sidecar"],
            }
        )

    paired_rows = []
    for anchor_id in sorted({row["anchor_id"] for row in logical_results}):
        rows = {
            row["label"]: row
            for row in logical_results
            if row["anchor_id"] == anchor_id
        }
        if set(rows) != set(LOGICAL_LABELS):
            validation_failures.append(f"{anchor_id}:missing_logical_label")
            continue
        for label in ("adapted", "oracle"):
            paired_rows.append(
                {
                    "anchor_id": anchor_id,
                    "split": rows[label]["split"],
                    "shape": rows[label]["shape"],
                    "sample_id": rows[label]["sample_id"],
                    "mpc_iter": rows[label]["mpc_iter"],
                    "before_candidate_id": rows["frozen"]["candidate_id"],
                    "after_candidate_id": rows[label]["candidate_id"],
                    "same_physical_intervention": rows["frozen"]["physical_branch_id"]
                    == rows[label]["physical_branch_id"],
                    **paired_outcome(
                        rows["frozen"],
                        rows[label],
                        f"{label}_minus_frozen",
                    ),
                }
            )

    nonfinite = []
    numeric_fields = (
        "first_chunk_state_dist",
        "first_chunk_reference_terminal_cost",
        "final_state_dist",
        "final_reference_terminal_cost",
    )
    for row in logical_results:
        for field in numeric_fields:
            if not math.isfinite(float(row[field])):
                nonfinite.append(f"{row['anchor_id']}:{row['label']}:{field}")
    validation = {
        "logical_branches": len(logical_manifest),
        "physical_branches": len(physical_specs),
        "deduplicated_branches": len(logical_manifest) - len(physical_specs),
        "late_anchors": len({row["anchor_id"] for row in logical_manifest}),
        "prefix_or_rng_failures": validation_failures,
        "nonfinite_metrics": nonfinite,
        "fixed_frozen_model_exact": fixed_model_exact,
        "environment_rollouts": len(physical_specs),
        "prefix_replay_environment_steps": sum(
            int(row["prefix_replay_environment_steps"])
            for row in physical_results
        ),
        "first_decision_environment_steps": sum(
            int(row["first_decision_environment_steps"])
            for row in physical_results
        ),
        "tail_environment_steps": sum(
            int(row["tail_environment_steps"]) for row in physical_results
        ),
        "total_environment_steps": sum(
            int(row["total_environment_steps"]) for row in physical_results
        ),
        "tail_replans": 4 * len(physical_specs),
        "world_model_rollout_and_backward_steps": 400 * len(physical_specs),
    }
    validation["passed"] = bool(
        validation["logical_branches"] == 18
        and validation["physical_branches"] <= 18
        and validation["late_anchors"] == 6
        and not validation_failures
        and not nonfinite
        and fixed_model_exact
    )
    adapted_pairs = [
        row for row in paired_rows if row["comparison"] == "adapted_minus_frozen"
    ]
    oracle_pairs = [
        row for row in paired_rows if row["comparison"] == "oracle_minus_frozen"
    ]
    summary = {
        "stage": "C",
        "adapted_first_decision_vs_frozen": _aggregate_effects(adapted_pairs),
        "pool_oracle_first_decision_vs_frozen": _aggregate_effects(oracle_pairs),
        "validation": validation,
        "wall_clock_s": time.perf_counter() - started,
        "device": str(device),
    }

    _write_csv(args.out / "physical_branch_results.csv", physical_results)
    _write_csv(args.out / "logical_branch_results.csv", logical_results)
    _write_csv(args.out / "paired_effects.csv", paired_rows)
    with (args.out / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(validation), handle, indent=2, sort_keys=True)
    with (args.out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
    torch.save(rng_states, args.out / "rng_states.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-metrics", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--donor", type=_parse_donor, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=270920)
    parser.add_argument("--tail-chunks", type=int, default=4)
    args = parser.parse_args()
    if args.tail_chunks != 4:
        raise ValueError(f"V3 common continuation is frozen at 4 tail chunks, got {args.tail_chunks}")
    summary = run(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    if not summary["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

