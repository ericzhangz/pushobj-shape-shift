"""Common-anchor shadow-adaptation audit for Reframe V3 phase B.

The runner never calls the environment.  It reuses Frozen-donor histories,
candidate tensors, and logged oracle consequences, while recomputing only the
world-model scores before and after factual AdaJEPA updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch


GD_STEPS = (0, 24, 49, 74, 99)
SIDES = ("before", "after")
DEFAULT_EPSILON = 1e-6


def official_prefix_schedule(anchor_iter: int) -> list[tuple[int, ...]]:
    """Official replay-buffer update sequence before one MPC anchor."""
    if anchor_iter < 1:
        return []
    return [tuple(range(end + 1)) for end in range(anchor_iter)]


def experience_reuse_schedules() -> dict[str, list[tuple[int, ...]]]:
    """Pre-registered time-separated E1/E2 controls for MPC index 4."""
    e1 = (0, 1)
    e2 = (2, 3)
    return {
        "frozen": [],
        "e1_once": [e1],
        "e2_once": [e2],
        "e1_then_e2": [e1, e2],
        "e1_then_e1": [e1, e1],
        "e2_then_e2": [e2, e2],
    }


def processed_transition_counts(
    schedule: list[tuple[int, ...]],
) -> tuple[int, int]:
    processed = sum(len(batch) for batch in schedule)
    unique = len({index for batch in schedule for index in batch})
    return processed, unique


def selection_summary(rows: list[dict], epsilon: float) -> dict:
    if not rows:
        raise ValueError("candidate pool is empty")
    ids = [str(row["candidate_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate pool contains duplicate candidate ids")
    for row in rows:
        if not (
            math.isfinite(float(row["c_model"]))
            and math.isfinite(float(row["c_env"]))
        ):
            raise ValueError("candidate pool contains non-finite costs")

    model_best = min(rows, key=lambda row: float(row["c_model"]))
    env_best = min(rows, key=lambda row: float(row["c_env"]))
    model_min = float(model_best["c_model"])
    env_min = float(env_best["c_env"])
    tied = [
        row
        for row in rows
        if float(row["c_model"]) <= model_min + float(epsilon)
    ]
    tied_regrets = [float(row["c_env"]) - env_min for row in tied]
    env_costs = [float(row["c_env"]) for row in rows]
    sorted_model_costs = sorted(float(row["c_model"]) for row in rows)
    return {
        "pool_size": len(rows),
        "model_best_candidate_id": model_best["candidate_id"],
        "env_best_candidate_id": env_best["candidate_id"],
        "model_best_cost": model_min,
        "env_best_cost": env_min,
        "selected_env_cost": float(model_best["c_env"]),
        "r_selected": float(model_best["c_env"]) - env_min,
        "numerical_tie_epsilon": float(epsilon),
        "model_tie_count_epsilon": len(tied),
        "r_selected_tie_min": min(tied_regrets),
        "r_selected_tie_max": max(tied_regrets),
        "model_best_margin": (
            sorted_model_costs[1] - sorted_model_costs[0]
            if len(sorted_model_costs) > 1
            else math.inf
        ),
        "environment_pool_span": max(env_costs) - min(env_costs),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_runtime(checkpoint_dir: Path, device: torch.device):
    import hydra
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import (
        ACTION_MEAN,
        ACTION_STD,
        PROPRIO_MEAN,
        PROPRIO_STD,
        STATE_MEAN,
        STATE_STD,
    )
    from omegaconf import OmegaConf
    from plan import load_model
    from preprocessor import Preprocessor

    config_path = checkpoint_dir / "hydra.yaml"
    checkpoint_path = checkpoint_dir / "checkpoints" / "model_latest.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    config = OmegaConf.load(config_path)
    model = load_model(
        checkpoint_path,
        config,
        config.num_action_repeat,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    transform_config = config.env.dataset.get("transform", None)
    transform = (
        hydra.utils.instantiate(transform_config)
        if transform_config is not None
        else default_transform()
    )
    preprocessor = Preprocessor(
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        state_mean=STATE_MEAN[:5],
        state_std=STATE_STD[:5],
        proprio_mean=PROPRIO_MEAN[:4],
        proprio_std=PROPRIO_STD[:4],
        transform=transform,
    )
    return model, preprocessor, config


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL asset: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _load_candidate_pool(
    donor_dir: Path,
    sample_id: int,
    mpc_iter: int,
    branch_rows: list[dict],
) -> list[dict]:
    indexed = {}
    for row in branch_rows:
        if int(row["sample_id"]) != sample_id or int(row["mpc_iter"]) != mpc_iter:
            continue
        candidate_id = f"g{int(row['gd_iter'])}_{str(row['candidate']).removeprefix('u_')}"
        if candidate_id in indexed:
            raise ValueError(
                f"Duplicate branch record for sample={sample_id}, mpc={mpc_iter}, "
                f"candidate={candidate_id}"
            )
        indexed[candidate_id] = row

    candidates = []
    for gd_iter in GD_STEPS:
        tensor_path = donor_dir / "tensors" / f"s{sample_id}_m{mpc_iter}_g{gd_iter}.pt"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing candidate tensor sidecar: {tensor_path}")
        tensors = torch.load(tensor_path, map_location="cpu")
        for side in SIDES:
            candidate_id = f"g{gd_iter}_{side}"
            if candidate_id not in indexed:
                raise ValueError(
                    f"Missing branch record for sample={sample_id}, mpc={mpc_iter}, "
                    f"candidate={candidate_id}"
                )
            branch = indexed[candidate_id]
            tensor_key = f"u_{side}"
            if tensor_key not in tensors:
                raise ValueError(f"{tensor_path} is missing {tensor_key}")
            action = tensors[tensor_key].detach().cpu()
            if tuple(action.shape) != (1, 5, 10):
                raise ValueError(
                    f"Unexpected action shape {tuple(action.shape)} in {tensor_path}:{tensor_key}"
                )
            c_env_ref = float(branch["c_env_ref"])
            c_env_samever = float(branch["c_env_samever"])
            if abs(c_env_ref - c_env_samever) > 1e-9:
                raise ValueError(
                    "Frozen donor has different reference/current environment costs: "
                    f"{donor_dir}, sample={sample_id}, mpc={mpc_iter}, {candidate_id}"
                )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "gd_iter": gd_iter,
                    "side": side,
                    "action": action,
                    "c_model_logged": float(branch["c_hat_native"]),
                    "c_env": c_env_ref,
                    "objective_stage_logged": str(branch["objective_stage"]),
                }
            )
    return candidates


def _load_anchor_observations(
    donor_dir: Path, sample_id: int, mpc_iter: int
) -> tuple[dict, dict]:
    history_path = donor_dir / "real_evidence" / f"s{sample_id}_history_mpc{mpc_iter}.pt"
    initial_path = donor_dir / "real_evidence" / f"sample_{sample_id:03d}_initial.npz"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing anchor history: {history_path}")
    if not initial_path.exists():
        raise FileNotFoundError(f"Missing goal evidence: {initial_path}")
    history = torch.load(history_path, map_location="cpu")
    if "current_real_observation" not in history:
        raise ValueError(f"Anchor history is missing current_real_observation: {history_path}")
    with np.load(initial_path, allow_pickle=False) as archive:
        for key in ("goal_visual", "goal_proprio"):
            if key not in archive.files:
                raise ValueError(f"Goal evidence is missing {key}: {initial_path}")
        goal = {
            "visual": np.asarray(archive["goal_visual"]),
            "proprio": np.asarray(archive["goal_proprio"]),
        }
    current = {
        key: np.asarray(value)
        for key, value in history["current_real_observation"].items()
    }
    return current, goal


def _load_segments(
    donor_dir: Path,
    sample_id: int,
    count: int,
    frameskip: int,
    preprocessor,
    *,
    full_resolution: bool = False,
) -> list[tuple[dict, torch.Tensor]]:
    segments = []
    for segment_index in range(count):
        path = (
            donor_dir
            / "real_evidence"
            / f"s{sample_id}_executed_mpc{segment_index}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing factual segment: {path}")
        with np.load(path, allow_pickle=False) as archive:
            for key in ("visual", "proprio", "normalized_model_actions"):
                if key not in archive.files:
                    raise ValueError(f"Factual segment is missing {key}: {path}")
            visual = np.asarray(archive["visual"])
            proprio = np.asarray(archive["proprio"])
            actions = np.asarray(archive["normalized_model_actions"])
        if len(visual) != frameskip + 1 or len(proprio) != frameskip + 1:
            raise ValueError(
                f"Expected one complete {frameskip}-substep chunk in {path}, "
                f"got visual={len(visual)}, proprio={len(proprio)}"
            )
        stride = 1 if full_resolution else frameskip
        model_obs = {
            "visual": visual[np.newaxis, ::stride],
            "proprio": proprio[np.newaxis, ::stride],
        }
        transformed = preprocessor.transform_obs(model_obs)
        action_tensor = torch.as_tensor(actions).detach().cpu()
        if tuple(action_tensor.shape) != (1, 1, 10):
            raise ValueError(f"Unexpected factual action shape in {path}: {action_tensor.shape}")
        segments.append((transformed, action_tensor))
    return segments


def _move_obs(obs: dict, device: torch.device) -> dict:
    from utils import move_to_device

    return move_to_device({key: value.clone() for key, value in obs.items()}, device)


def _transform_anchor_obs(preprocessor, current: dict, goal: dict, device: torch.device):
    return (
        _move_obs(preprocessor.transform_obs(current), device),
        _move_obs(preprocessor.transform_obs(goal), device),
    )


def _score_pool(
    model,
    transformed_current: dict,
    transformed_goal: dict,
    candidates: list[dict],
    mpc_iter: int,
) -> tuple[list[dict], str]:
    from research.contrast_probe import native_objective_breakdown

    model.eval()
    device = next(model.parameters()).device
    scored = []
    stages = set()
    with torch.no_grad():
        goal_z = model.encode_obs(transformed_goal)
        for candidate in candidates:
            action = candidate["action"].to(device)
            predicted_z, _ = model.rollout(obs_0=transformed_current, act=action)
            breakdown = native_objective_breakdown(
                predicted_z,
                goal_z,
                mpc_iter,
                alpha=1.0,
                base=2.0,
            )
            stages.add(breakdown["stage"])
            scored.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "gd_iter": candidate["gd_iter"],
                    "side": candidate["side"],
                    "c_model": float(breakdown["total"].item()),
                    "c_env": float(candidate["c_env"]),
                }
            )
    if len(stages) != 1:
        raise AssertionError(f"Mixed objective stages while scoring one pool: {stages}")
    return scored, next(iter(stages))


def _support_loss(trainer, segments, indices: tuple[int, ...]) -> float:
    if not indices:
        return math.nan
    obs_seqs = [segments[index][0] for index in indices]
    act_seqs = [segments[index][1] for index in indices]
    if len(obs_seqs) > 1:
        obs_seqs, act_seqs = trainer._merge_segments(obs_seqs, act_seqs)
    prepared = [
        trainer._prepare_segment(obs, action)
        for obs, action in zip(obs_seqs, act_seqs)
    ]
    predictor_training = trainer.wm.predictor.training
    encoder_training = trainer.wm.encoder.training
    trainer.wm.predictor.eval()
    trainer.wm.encoder.eval()
    try:
        with torch.no_grad():
            losses = [
                trainer._prediction_loss(trainer.wm.encode(obs, action))
                for obs, action in prepared
            ]
        return float(torch.stack(losses).mean().item())
    finally:
        trainer.wm.predictor.train(predictor_training)
        trainer.wm.encoder.train(encoder_training)


def _apply_schedule(trainer, segments, schedule) -> list[float]:
    losses = []
    for batch in schedule:
        obs_seqs = [segments[index][0] for index in batch]
        act_seqs = [segments[index][1] for index in batch]
        losses.extend(trainer.finetune(obs_seqs, act_seqs, merge=True))
    return [float(value) for value in losses]


def _snapshot_exact(trainer) -> bool:
    return all(torch.equal(tensor.detach(), saved) for tensor, saved in trainer._snapshot)


def _reset_to_base(model, trainers) -> bool:
    for trainer in trainers:
        trainer.reset()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return all(_snapshot_exact(trainer) for trainer in trainers)


def _encoder_versions(model) -> tuple[int, ...]:
    tensors = list(model.encoder.parameters()) + list(model.encoder.buffers())
    return tuple(int(tensor._version) for tensor in tensors)


def _decorate_candidate_rows(
    scored: list[dict],
    summary: dict,
    metadata: dict,
) -> list[dict]:
    initial = next(row for row in scored if row["candidate_id"] == "g0_before")
    rows = []
    for row in scored:
        delta_model = float(row["c_model"]) - float(initial["c_model"])
        delta_env = float(row["c_env"]) - float(initial["c_env"])
        rows.append(
            {
                **metadata,
                "candidate_id": row["candidate_id"],
                "gd_iter": row["gd_iter"],
                "side": row["side"],
                "c_model": row["c_model"],
                "c_env_ref": row["c_env"],
                "delta_model_from_g0_before": delta_model,
                "delta_env_from_g0_before": delta_env,
                "contrast_error_from_g0_before": delta_model - delta_env,
                "is_exact_model_best": row["candidate_id"]
                == summary["model_best_candidate_id"],
                "is_exact_env_best": row["candidate_id"]
                == summary["env_best_candidate_id"],
            }
        )
    return rows


def _mean_abs_contrast(scored: list[dict]) -> float:
    initial = next(row for row in scored if row["candidate_id"] == "g0_before")
    errors = []
    for row in scored:
        delta_model = float(row["c_model"]) - float(initial["c_model"])
        delta_env = float(row["c_env"]) - float(initial["c_env"])
        errors.append(abs(delta_model - delta_env))
    return statistics.fmean(errors)


def _selection_row(
    scored: list[dict],
    epsilon: float,
    metadata: dict,
    support: dict,
    update: dict,
) -> tuple[dict, dict]:
    summary = selection_summary(scored, epsilon)
    row = {
        **metadata,
        **summary,
        "mean_abs_contrast_error": _mean_abs_contrast(scored),
        **support,
        **update,
    }
    return row, summary


def _paired_row(before: dict, after: dict, comparison: str) -> dict:
    return {
        "analysis": after["analysis"],
        "comparison": comparison,
        "anchor_id": after["anchor_id"],
        "split": after["split"],
        "shape": after["shape"],
        "sample_id": after["sample_id"],
        "mpc_iter": after["mpc_iter"],
        "before_variant": before["variant"],
        "after_variant": after["variant"],
        "before_model_best_candidate_id": before["model_best_candidate_id"],
        "after_model_best_candidate_id": after["model_best_candidate_id"],
        "selection_changed": before["model_best_candidate_id"]
        != after["model_best_candidate_id"],
        "r_selected_before": before["r_selected"],
        "r_selected_after": after["r_selected"],
        "delta_r_selected": after["r_selected"] - before["r_selected"],
        "r_selected_tie_min_before": before["r_selected_tie_min"],
        "r_selected_tie_min_after": after["r_selected_tie_min"],
        "delta_r_selected_tie_min": after["r_selected_tie_min"]
        - before["r_selected_tie_min"],
        "selected_env_cost_before": before["selected_env_cost"],
        "selected_env_cost_after": after["selected_env_cost"],
        "delta_selected_env_cost": after["selected_env_cost"]
        - before["selected_env_cost"],
        "mean_abs_contrast_error_before": before["mean_abs_contrast_error"],
        "mean_abs_contrast_error_after": after["mean_abs_contrast_error"],
        "delta_mean_abs_contrast_error": after["mean_abs_contrast_error"]
        - before["mean_abs_contrast_error"],
        "common_support_loss_before": after["common_support_loss_before"],
        "common_support_loss_after": after["common_support_loss_after"],
        "common_support_loss_delta": after["common_support_loss_delta"],
        "before_variant_common_support_loss_delta": before[
            "common_support_loss_delta"
        ],
        "after_variant_common_support_loss_delta": after[
            "common_support_loss_delta"
        ],
        "delta_common_support_loss_delta": after["common_support_loss_delta"]
        - before["common_support_loss_delta"],
        "update_steps": after["update_steps"],
        "processed_transitions": after["processed_transitions"],
        "unique_factual_transitions": after["unique_factual_transitions"],
        "before_variant_processed_transitions": before["processed_transitions"],
        "after_variant_processed_transitions": after["processed_transitions"],
        "before_variant_unique_factual_transitions": before[
            "unique_factual_transitions"
        ],
        "after_variant_unique_factual_transitions": after[
            "unique_factual_transitions"
        ],
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def _aggregate_pairs(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    deltas = [float(row["delta_r_selected_tie_min"]) for row in rows]
    support = [float(row["delta_common_support_loss_delta"]) for row in rows]
    return {
        "n": len(rows),
        "selection_changed": sum(bool(row["selection_changed"]) for row in rows),
        "tie_aware_regret_improved": sum(value < -DEFAULT_EPSILON for value in deltas),
        "tie_aware_regret_worsened": sum(value > DEFAULT_EPSILON for value in deltas),
        "tie_aware_regret_unchanged": sum(abs(value) <= DEFAULT_EPSILON for value in deltas),
        "mean_delta_r_selected_tie_min": statistics.fmean(deltas),
        "median_delta_r_selected_tie_min": statistics.median(deltas),
        "mean_support_loss_delta": statistics.fmean(support),
        "support_delta_vs_regret_delta_pearson": _pearson(support, deltas),
    }


def _parse_donor(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("donor must use SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if not split or not raw_path:
        raise argparse.ArgumentTypeError("donor must use nonempty SPLIT=PATH")
    return split, Path(raw_path)


def run(args) -> dict:
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(args.device)
    model, preprocessor, model_config = _load_runtime(args.checkpoint_dir, device)

    from planning.adajepa import AdaJEPATrainer

    predictor_trainer = AdaJEPATrainer(
        wm=model,
        lr=args.predictor_lr,
        steps=1,
        optimizer_name="adam",
        finetune_encoder=False,
        last_layer_only=True,
    )
    full_trainer = AdaJEPATrainer(
        wm=model,
        lr=args.predictor_lr,
        steps=1,
        optimizer_name="adam",
        finetune_encoder=True,
        last_layer_only=True,
        encoder_lr=args.encoder_lr,
        encoder_last_layer_only=True,
    )
    trainers = (predictor_trainer, full_trainer)
    if not _reset_to_base(model, trainers):
        raise AssertionError("Could not establish the base checkpoint state")

    anchors = []
    candidate_rows = []
    main_selection_rows = []
    main_pair_rows = []
    reuse_selection_rows = []
    reuse_pair_rows = []
    reuse_control_rows = []
    full_pair_rows = []
    rng_states = {}
    reproduction_errors = []
    stage_mismatches = []
    reset_failures = []
    fixed_encoder_failures = []
    candidate_rollouts = 0
    adaptation_steps = 0
    desired_anchor_count = len(args.donor) * 3 * len(args.anchors)
    anchor_ordinal = 0

    for split, donor_dir in args.donor:
        donor_dir = donor_dir.resolve()
        metadata_path = donor_dir / "run_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing donor metadata: {metadata_path}")
        donor_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if donor_metadata.get("arm") != "FROZEN":
            raise ValueError(f"Donor is not FROZEN: {donor_dir}")
        branch_rows = _read_jsonl(donor_dir / "oracle" / "branch_records.jsonl")
        shape = str(donor_metadata["shapes"][0])
        frameskip = int(donor_metadata["frameskip"])
        for sample_id in range(3):
            for mpc_iter in args.anchors:
                anchor_id = f"{split}|{shape}|s{sample_id}|m{mpc_iter}"
                anchor_seed = int(args.seed + anchor_ordinal)
                anchor_ordinal += 1
                anchor_record = {
                    "anchor_id": anchor_id,
                    "split": split,
                    "shape": shape,
                    "sample_id": sample_id,
                    "mpc_iter": mpc_iter,
                    "donor_dir": str(donor_dir),
                    "rng_seed": anchor_seed,
                    "available": False,
                    "missing_reason": "",
                }
                try:
                    candidates = _load_candidate_pool(
                        donor_dir, sample_id, mpc_iter, branch_rows
                    )
                    current, goal = _load_anchor_observations(
                        donor_dir, sample_id, mpc_iter
                    )
                    segments = _load_segments(
                        donor_dir,
                        sample_id,
                        mpc_iter,
                        frameskip,
                        preprocessor,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    anchor_record["missing_reason"] = str(exc)
                    anchors.append(anchor_record)
                    continue
                anchor_record["available"] = True
                anchors.append(anchor_record)

                transformed_current, transformed_goal = _transform_anchor_obs(
                    preprocessor, current, goal, device
                )
                if not _reset_to_base(model, trainers):
                    reset_failures.append(f"{anchor_id}:before_base")
                base_scored, base_stage = _score_pool(
                    model,
                    transformed_current,
                    transformed_goal,
                    candidates,
                    mpc_iter,
                )
                candidate_rollouts += len(candidates)
                for scored, source in zip(base_scored, candidates):
                    reproduction_errors.append(
                        abs(float(scored["c_model"]) - float(source["c_model_logged"]))
                    )
                    if base_stage != source["objective_stage_logged"]:
                        stage_mismatches.append(
                            f"{anchor_id}:{source['candidate_id']}:"
                            f"{base_stage}!={source['objective_stage_logged']}"
                        )
                all_indices = tuple(range(mpc_iter))
                base_support = _support_loss(
                    predictor_trainer, segments, all_indices
                )
                frozen_meta = {
                    "analysis": "b1_common_anchor",
                    "variant": "frozen",
                    "trainable": "none",
                    "anchor_id": anchor_id,
                    "split": split,
                    "shape": shape,
                    "sample_id": sample_id,
                    "mpc_iter": mpc_iter,
                    "objective_stage": base_stage,
                    "rng_seed": anchor_seed,
                }
                frozen_support = {
                    "common_support_loss_before": base_support,
                    "common_support_loss_after": base_support,
                    "common_support_loss_delta": 0.0,
                    "training_support_loss_before": base_support,
                    "training_support_loss_after": base_support,
                    "training_support_loss_delta": 0.0,
                }
                frozen_update = {
                    "update_steps": 0,
                    "processed_transitions": 0,
                    "unique_factual_transitions": 0,
                    "step_losses": "[]",
                    "fixed_encoder_unchanged": True,
                    "base_reset_exact": True,
                    "candidate_rollouts": len(candidates),
                }
                frozen_row, frozen_summary = _selection_row(
                    base_scored,
                    args.epsilon,
                    frozen_meta,
                    frozen_support,
                    frozen_update,
                )
                main_selection_rows.append(frozen_row)
                candidate_rows.extend(
                    _decorate_candidate_rows(base_scored, frozen_summary, frozen_meta)
                )

                def evaluate_variant(
                    analysis: str,
                    variant: str,
                    trainable: str,
                    trainer,
                    schedule: list[tuple[int, ...]],
                ):
                    nonlocal candidate_rollouts, adaptation_steps
                    reset_exact = _reset_to_base(model, trainers)
                    if not reset_exact:
                        reset_failures.append(f"{anchor_id}:{analysis}:{variant}:before")
                    _seed_all(anchor_seed)
                    rng_before = _capture_rng_state()
                    common_before = _support_loss(trainer, segments, all_indices)
                    used_indices = tuple(
                        sorted({index for batch in schedule for index in batch})
                    )
                    training_before = _support_loss(trainer, segments, used_indices)
                    encoder_before = _encoder_versions(model)
                    step_losses = _apply_schedule(trainer, segments, schedule)
                    encoder_after = _encoder_versions(model)
                    common_after = _support_loss(trainer, segments, all_indices)
                    training_after = _support_loss(trainer, segments, used_indices)
                    scored, stage = _score_pool(
                        model,
                        transformed_current,
                        transformed_goal,
                        candidates,
                        mpc_iter,
                    )
                    rng_after = _capture_rng_state()
                    rng_states[f"{anchor_id}|{analysis}|{variant}"] = {
                        "seed": anchor_seed,
                        "before": rng_before,
                        "after": rng_after,
                    }
                    _restore_rng_state(rng_before)
                    processed, unique = processed_transition_counts(schedule)
                    fixed_encoder = encoder_before == encoder_after
                    if trainable == "predictor" and not fixed_encoder:
                        fixed_encoder_failures.append(f"{anchor_id}:{analysis}:{variant}")
                    adaptation_steps += len(step_losses)
                    candidate_rollouts += len(candidates)
                    metadata = {
                        "analysis": analysis,
                        "variant": variant,
                        "trainable": trainable,
                        "anchor_id": anchor_id,
                        "split": split,
                        "shape": shape,
                        "sample_id": sample_id,
                        "mpc_iter": mpc_iter,
                        "objective_stage": stage,
                        "rng_seed": anchor_seed,
                    }
                    support = {
                        "common_support_loss_before": common_before,
                        "common_support_loss_after": common_after,
                        "common_support_loss_delta": common_after - common_before,
                        "training_support_loss_before": training_before,
                        "training_support_loss_after": training_after,
                        "training_support_loss_delta": training_after - training_before,
                    }
                    update = {
                        "update_steps": len(step_losses),
                        "processed_transitions": processed,
                        "unique_factual_transitions": unique,
                        "step_losses": json.dumps(step_losses, separators=(",", ":")),
                        "fixed_encoder_unchanged": fixed_encoder,
                        "base_reset_exact": reset_exact,
                        "candidate_rollouts": len(candidates),
                    }
                    selection, summary = _selection_row(
                        scored, args.epsilon, metadata, support, update
                    )
                    decorated = _decorate_candidate_rows(scored, summary, metadata)
                    if not _reset_to_base(model, trainers):
                        reset_failures.append(f"{anchor_id}:{analysis}:{variant}:after")
                    return selection, decorated

                predictor_row, predictor_candidates = evaluate_variant(
                    "b1_common_anchor",
                    "predictor_official_shadow",
                    "predictor",
                    predictor_trainer,
                    official_prefix_schedule(mpc_iter),
                )
                main_selection_rows.append(predictor_row)
                candidate_rows.extend(predictor_candidates)
                main_pair_rows.append(
                    _paired_row(
                        frozen_row,
                        predictor_row,
                        "predictor_official_shadow_minus_frozen",
                    )
                )

                if args.include_full_adaptation and mpc_iter == 4:
                    full_row, full_candidates = evaluate_variant(
                        "b1_common_anchor",
                        "official_full_shadow",
                        "predictor+encoder",
                        full_trainer,
                        official_prefix_schedule(mpc_iter),
                    )
                    main_selection_rows.append(full_row)
                    candidate_rows.extend(full_candidates)
                    full_pair_rows.append(
                        _paired_row(
                            frozen_row,
                            full_row,
                            "official_full_shadow_minus_frozen",
                        )
                    )

                if mpc_iter == 4:
                    reuse_frozen_meta = {
                        **frozen_meta,
                        "analysis": "b3_time_separated_experience",
                    }
                    reuse_frozen = {
                        **frozen_row,
                        "analysis": "b3_time_separated_experience",
                    }
                    reuse_selection_rows.append(reuse_frozen)
                    reuse_variant_rows = {"frozen": reuse_frozen}
                    candidate_rows.extend(
                        _decorate_candidate_rows(
                            base_scored, frozen_summary, reuse_frozen_meta
                        )
                    )
                    for variant, schedule in experience_reuse_schedules().items():
                        if variant == "frozen":
                            continue
                        row, decorated = evaluate_variant(
                            "b3_time_separated_experience",
                            variant,
                            "predictor",
                            predictor_trainer,
                            schedule,
                        )
                        reuse_selection_rows.append(row)
                        reuse_variant_rows[variant] = row
                        candidate_rows.extend(decorated)
                        reuse_pair_rows.append(
                            _paired_row(
                                reuse_frozen,
                                row,
                                f"{variant}_minus_frozen",
                            )
                        )
                    for control in ("e1_then_e1", "e2_then_e2"):
                        reuse_control_rows.append(
                            _paired_row(
                                reuse_variant_rows[control],
                                reuse_variant_rows["e1_then_e2"],
                                f"e1_then_e2_minus_{control}",
                            )
                        )

    available_anchors = sum(bool(row["available"]) for row in anchors)
    validation = {
        "desired_anchors": desired_anchor_count,
        "available_anchors": available_anchors,
        "missing_anchors": desired_anchor_count - available_anchors,
        "candidate_pools_scored": len(main_selection_rows)
        + len(reuse_selection_rows),
        "base_cost_reproduction_max_abs_error": max(reproduction_errors, default=None),
        "base_cost_reproduction_tolerance": args.reproduction_tolerance,
        "base_cost_reproduction_failures": sum(
            error > args.reproduction_tolerance for error in reproduction_errors
        ),
        "objective_stage_mismatches": stage_mismatches,
        "fixed_encoder_failures": fixed_encoder_failures,
        "base_reset_failures": reset_failures,
        "candidate_rollouts": candidate_rollouts,
        "adaptation_steps": adaptation_steps,
        "new_environment_calls": 0,
    }
    validation["passed"] = bool(
        validation["missing_anchors"] == 0
        and validation["base_cost_reproduction_failures"] == 0
        and not stage_mismatches
        and not fixed_encoder_failures
        and not reset_failures
    )

    main_predictor_pairs = [
        row
        for row in main_pair_rows
        if row["comparison"] == "predictor_official_shadow_minus_frozen"
    ]
    reuse_by_variant = {}
    for variant in experience_reuse_schedules():
        if variant == "frozen":
            continue
        subset = [
            row
            for row in reuse_pair_rows
            if row["after_variant"] == variant
        ]
        reuse_by_variant[variant] = _aggregate_pairs(subset)
    summary = {
        "stage": "B",
        "main_predictor_shadow": _aggregate_pairs(main_predictor_pairs),
        "official_full_shadow_late": _aggregate_pairs(full_pair_rows),
        "time_separated_experience": reuse_by_variant,
        "new_data_vs_repetition_controls": {
            control: _aggregate_pairs(
                [
                    row
                    for row in reuse_control_rows
                    if row["comparison"] == f"e1_then_e2_minus_{control}"
                ]
            )
            for control in ("e1_then_e1", "e2_then_e2")
        },
        "validation": validation,
        "wall_clock_s": time.perf_counter() - started,
        "device": str(device),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "epsilon": args.epsilon,
    }

    _write_csv(args.out / "anchors.csv", anchors)
    _write_csv(args.out / "candidate_scores.csv", candidate_rows)
    _write_csv(args.out / "selection_metrics.csv", main_selection_rows)
    _write_csv(args.out / "paired_effects.csv", main_pair_rows + full_pair_rows)
    _write_csv(args.out / "experience_reuse_metrics.csv", reuse_selection_rows)
    _write_csv(args.out / "experience_reuse_paired_effects.csv", reuse_pair_rows)
    _write_csv(
        args.out / "experience_reuse_control_comparisons.csv",
        reuse_control_rows,
    )
    with (args.out / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(validation), handle, indent=2, sort_keys=True)
    with (args.out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
    torch.save(rng_states, args.out / "rng_states.pt")
    with (args.out / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(
                {
                    "checkpoint_dir": args.checkpoint_dir,
                    "donors": args.donor,
                    "anchors": args.anchors,
                    "seed": args.seed,
                    "epsilon": args.epsilon,
                    "predictor_lr": args.predictor_lr,
                    "encoder_lr": args.encoder_lr,
                    "include_full_adaptation": args.include_full_adaptation,
                    "frameskip": int(model_config.frameskip),
                    "model_action_horizon": 5,
                    "model_action_dim": 10,
                }
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--donor",
        type=_parse_donor,
        action="append",
        required=True,
        help="Repeat as SPLIT=PATH; expected val_T and val_L Frozen donor runs",
    )
    parser.add_argument("--anchors", default="2,4")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=260920)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-6)
    parser.add_argument("--predictor-lr", type=float, default=5e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--include-full-adaptation", action="store_true")
    args = parser.parse_args()
    args.anchors = tuple(
        int(value.strip()) for value in str(args.anchors).split(",") if value.strip()
    )
    if args.anchors != (2, 4):
        raise ValueError(f"V3 anchors are frozen at (2, 4), got {args.anchors}")
    summary = run(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    if not summary["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
