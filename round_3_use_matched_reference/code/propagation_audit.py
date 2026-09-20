"""Small post-S3 diagnostic: locate matched-feedback consequence compression.

This audit uses only saved C observations/actions and the frozen checkpoint.  It
does not call an environment or re-run GD.  Actual future actions are used only
in the explicitly labelled teacher-forced diagnostic branch.
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

from research.reframe_v3.matched_feedback_forecast import rollout_from_zobs
from research.reframe_v3.shadow_selection_audit import (
    _jsonable,
    _load_anchor_observations,
    _load_runtime,
    _transform_anchor_obs,
)


EPSILON = 1e-6
LATENT_REPRO_TOLERANCE = 2e-6


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
    return split, Path(raw_path)


def _range(values: list[float]) -> float:
    if not values:
        raise ValueError("range requires values")
    return max(values) - min(values)


def _cost(z_obs: dict, z_goal: dict, time_index: int) -> float:
    visual = F.mse_loss(
        z_obs["visual"][:, time_index : time_index + 1], z_goal["visual"]
    )
    proprio = F.mse_loss(
        z_obs["proprio"][:, time_index : time_index + 1], z_goal["proprio"]
    )
    return float((visual + proprio).item())


def _latent_error(predicted: dict, real: dict, time_index: int) -> tuple[float, float, float]:
    visual = F.mse_loss(
        predicted["visual"][:, time_index : time_index + 1],
        real["visual"][:, time_index : time_index + 1],
    )
    proprio = F.mse_loss(
        predicted["proprio"][:, time_index : time_index + 1],
        real["proprio"][:, time_index : time_index + 1],
    )
    return float((visual + proprio).item()), float(visual.item()), float(proprio.item())


def _sequential_rollout(model, initial_z: dict, actions: torch.Tensor) -> dict:
    """Mirror S3/C replanning semantics, retaining one latent frame per chunk.

    S3 advances from the latest single-frame latent after every executed chunk.
    That is a different model input history from one call over the concatenated
    action sequence, so this diagnostic must repeat the same chunk boundary.
    """
    current = {key: value[:, :1] for key, value in initial_z.items()}
    frames = {key: [value] for key, value in current.items()}
    for action_index in range(actions.shape[1]):
        predicted, _ = rollout_from_zobs(
            model, current, actions[:, action_index : action_index + 1]
        )
        current = {key: value[:, -1:].detach() for key, value in predicted.items()}
        for key, value in current.items():
            frames[key].append(value)
    return {key: torch.cat(values, dim=1) for key, values in frames.items()}


def _encode_each_frame(model, transformed_obs: dict) -> dict:
    """Encode each stride frame with C's original one-frame scoring batch shape."""
    time_steps = int(next(iter(transformed_obs.values())).shape[1])
    encoded_frames = []
    for time_index in range(time_steps):
        frame = {
            key: value[:, time_index : time_index + 1]
            for key, value in transformed_obs.items()
        }
        encoded_frames.append(model.encode_obs(frame))
    return {
        key: torch.cat([encoded[key] for encoded in encoded_frames], dim=1)
        for key in encoded_frames[0]
    }


def _real_state_one_step_rollout(
    model, real_z: dict, actions: torch.Tensor
) -> dict:
    """Predict each next frame from its saved real latent and executed action."""
    frames = {key: [value[:, :1]] for key, value in real_z.items()}
    for action_index in range(actions.shape[1]):
        current = {
            key: value[:, action_index : action_index + 1]
            for key, value in real_z.items()
        }
        predicted, _ = rollout_from_zobs(
            model, current, actions[:, action_index : action_index + 1]
        )
        for key, value in predicted.items():
            frames[key].append(value[:, -1:].detach())
    return {key: torch.cat(values, dim=1) for key, values in frames.items()}


def _unique_physical_prediction_rows(rows: list[dict]) -> list[dict]:
    by_physical = {}
    for row in rows:
        physical_id = row["physical_branch_id"]
        if physical_id in by_physical:
            prior = by_physical[physical_id]
            for field in (
                "Qhat_common_continuation",
                "Q_env_common_continuation",
                "prediction_sidecar",
            ):
                if row[field] != prior[field]:
                    raise ValueError(f"Duplicate physical prediction differs: {physical_id}:{field}")
        else:
            by_physical[physical_id] = row
    return sorted(by_physical.values(), key=lambda row: row["physical_branch_id"])


def _aggregate_ranges(depth_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in depth_rows:
        grouped[(row["anchor_id"], int(row["depth"]))].append(row)
    output = []
    for (anchor_id, depth), rows in sorted(grouped.items()):
        real_range = _range([float(row["real_goal_cost"]) for row in rows])
        free_range = _range([float(row["free_model_goal_cost"]) for row in rows])
        teacher_range = _range(
            [float(row["teacher_forced_model_goal_cost"]) for row in rows]
        )
        reset_range = _range(
            [float(row["real_state_one_step_model_goal_cost"]) for row in rows]
        )
        output.append(
            {
                "anchor_id": anchor_id,
                "depth": depth,
                "physical_branches": len(rows),
                "real_goal_cost_range": real_range,
                "free_model_goal_cost_range": free_range,
                "teacher_forced_model_goal_cost_range": teacher_range,
                "real_state_one_step_model_goal_cost_range": reset_range,
                "free_to_real_range_ratio": (
                    free_range / real_range if real_range > EPSILON else math.nan
                ),
                "teacher_to_real_range_ratio": (
                    teacher_range / real_range if real_range > EPSILON else math.nan
                ),
                "real_state_one_step_to_real_range_ratio": (
                    reset_range / real_range if real_range > EPSILON else math.nan
                ),
            }
        )
    return output


def run(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    targets = (
        args.out / "propagation_depth_metrics.csv",
        args.out / "feedback_contraction_by_depth.csv",
        args.out / "propagation_audit_summary.json",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("propagation audit output already exists")
    started = time.perf_counter()
    donors = {split: path.resolve() for split, path in args.donor}
    if set(donors) != {"val_T", "val_L"}:
        raise ValueError(f"Expected val_T and val_L donors, got {sorted(donors)}")
    predictions = _unique_physical_prediction_rows(_read_csv(args.predictions))
    physical_truth = {
        row["physical_branch_id"]: row for row in _read_csv(args.c_physical)
    }

    device = torch.device(args.device)
    model, preprocessor, _ = _load_runtime(args.checkpoint_dir, device)
    versions_before = tuple(int(tensor._version) for tensor in list(model.parameters()) + list(model.buffers()))
    depth_rows = []
    validation_failures = []
    final_free_errors = []
    final_teacher_errors = []
    final_reset_errors = []
    final_qenv_crosscheck_errors = []
    final_qhat_crosscheck_errors = []
    sidecar_reconstruction_errors = []
    replayed_initial_alignment_errors = []
    goal_reencoding_alignment_errors = []

    for prediction in predictions:
        physical_id = prediction["physical_branch_id"]
        truth = physical_truth[physical_id]
        split = prediction["split"]
        sample_id = int(prediction["sample_id"])
        mpc_iter = int(prediction["mpc_iter"])
        forecast_sidecar = torch.load(
            Path(prediction["prediction_sidecar"]), map_location=device
        )
        initial_z = {
            key: value.to(device) for key, value in forecast_sidecar["initial_z_obs"].items()
        }
        goal_z = {
            key: value.to(device) for key, value in forecast_sidecar["goal_z_obs"].items()
        }
        _, goal = _load_anchor_observations(
            donors[split], sample_id, mpc_iter
        )
        with np.load(Path(truth["trajectory_sidecar"]), allow_pickle=False) as archive:
            visual = np.asarray(archive["visual"])
            proprio = np.asarray(archive["proprio"])
            actual_actions_np = np.asarray(archive["executed_model_actions"])
        if visual.shape[0] != 6 or proprio.shape[0] != 6:
            raise ValueError(f"Expected six stride observations in {physical_id}")
        actual_actions = torch.as_tensor(actual_actions_np, device=device)
        real_obs = {
            "visual": visual[np.newaxis],
            "proprio": proprio[np.newaxis],
        }
        transformed_real, transformed_goal = _transform_anchor_obs(
            preprocessor, real_obs, goal, device
        )
        with torch.no_grad():
            real_z = _encode_each_frame(model, transformed_real)
            reencoded_goal_z = model.encode_obs(transformed_goal)
            teacher_z = _sequential_rollout(model, initial_z, actual_actions)
            reset_z = _real_state_one_step_rollout(model, real_z, actual_actions)
        replayed_initial_alignment_errors.append(
            max(
                float((initial_z[key] - real_z[key][:, :1]).abs().max().item())
                for key in initial_z
            )
        )
        goal_reencoding_alignment_errors.append(
            max(
                float((goal_z[key] - reencoded_goal_z[key]).abs().max().item())
                for key in goal_z
            )
        )

        predicted_chunks = [
            torch.as_tensor(chunk, device=device)
            for chunk in forecast_sidecar["executed_model_chunks"]
        ]
        predicted_actions = torch.cat(predicted_chunks, dim=1)
        if not torch.equal(predicted_actions[:, :1], actual_actions[:, :1]):
            validation_failures.append(f"{physical_id}:first_chunk")
        with torch.no_grad():
            free_z = _sequential_rollout(model, initial_z, predicted_actions)

        saved_final = forecast_sidecar["final_predicted_z_obs"]
        reconstruction_error = max(
            float(
                (
                    free_z[key][:, -1:].detach().cpu()
                    - saved_final[key].detach().cpu()
                )
                .abs()
                .max()
                .item()
            )
            for key in free_z
        )
        sidecar_reconstruction_errors.append(reconstruction_error)
        if reconstruction_error > LATENT_REPRO_TOLERANCE:
            validation_failures.append(f"{physical_id}:forecast_sidecar")

        for depth in range(6):
            free_total, free_visual, free_proprio = _latent_error(
                free_z, real_z, depth
            )
            teacher_total, teacher_visual, teacher_proprio = _latent_error(
                teacher_z, real_z, depth
            )
            reset_total, reset_visual, reset_proprio = _latent_error(
                reset_z, real_z, depth
            )
            if depth == 0:
                action_delta_l2 = 0.0
                action_delta_max = 0.0
            else:
                action_delta = (
                    predicted_actions[:, depth - 1 : depth]
                    - actual_actions[:, depth - 1 : depth]
                )
                action_delta_l2 = float(torch.linalg.vector_norm(action_delta).item())
                action_delta_max = float(action_delta.abs().max().item())
            depth_rows.append(
                {
                    "physical_branch_id": physical_id,
                    "anchor_id": prediction["anchor_id"],
                    "labels": truth["labels"],
                    "depth": depth,
                    "real_goal_cost": _cost(real_z, goal_z, depth),
                    "free_model_goal_cost": _cost(free_z, goal_z, depth),
                    "teacher_forced_model_goal_cost": _cost(teacher_z, goal_z, depth),
                    "real_state_one_step_model_goal_cost": _cost(
                        reset_z, goal_z, depth
                    ),
                    "free_latent_mse": free_total,
                    "free_visual_mse": free_visual,
                    "free_proprio_mse": free_proprio,
                    "teacher_forced_latent_mse": teacher_total,
                    "teacher_forced_visual_mse": teacher_visual,
                    "teacher_forced_proprio_mse": teacher_proprio,
                    "real_state_one_step_latent_mse": reset_total,
                    "real_state_one_step_visual_mse": reset_visual,
                    "real_state_one_step_proprio_mse": reset_proprio,
                    "preceding_action_delta_l2": action_delta_l2,
                    "preceding_action_delta_max_abs": action_delta_max,
                }
            )

        real_final = _cost(real_z, goal_z, 5)
        free_final = _cost(free_z, goal_z, 5)
        teacher_final = _cost(teacher_z, goal_z, 5)
        reset_final = _cost(reset_z, goal_z, 5)
        recorded_qenv = float(truth["final_reference_terminal_cost"])
        recorded_qhat = float(prediction["Qhat_common_continuation"])
        final_free_errors.append(abs(free_final - real_final))
        final_teacher_errors.append(abs(teacher_final - real_final))
        final_reset_errors.append(abs(reset_final - real_final))
        final_qenv_crosscheck_errors.append(abs(real_final - recorded_qenv))
        final_qhat_crosscheck_errors.append(abs(free_final - recorded_qhat))

    ranges = _aggregate_ranges(depth_rows)
    physical_raw_matched_mae = statistics.fmean(final_free_errors)
    teacher_final_mae = statistics.fmean(final_teacher_errors)
    reset_final_mae = statistics.fmean(final_reset_errors)
    depth_summary = {}
    for depth in range(6):
        rows = [row for row in depth_rows if int(row["depth"]) == depth]
        depth_summary[str(depth)] = {
            "free_goal_cost_mae": statistics.fmean(
                abs(float(row["free_model_goal_cost"]) - float(row["real_goal_cost"]))
                for row in rows
            ),
            "teacher_goal_cost_mae": statistics.fmean(
                abs(
                    float(row["teacher_forced_model_goal_cost"])
                    - float(row["real_goal_cost"])
                )
                for row in rows
            ),
            "free_latent_mse_mean": statistics.fmean(
                float(row["free_latent_mse"]) for row in rows
            ),
            "teacher_latent_mse_mean": statistics.fmean(
                float(row["teacher_forced_latent_mse"]) for row in rows
            ),
            "real_state_one_step_goal_cost_mae": statistics.fmean(
                abs(
                    float(row["real_state_one_step_model_goal_cost"])
                    - float(row["real_goal_cost"])
                )
                for row in rows
            ),
            "real_state_one_step_latent_mse_mean": statistics.fmean(
                float(row["real_state_one_step_latent_mse"]) for row in rows
            ),
            "preceding_action_delta_l2_mean": statistics.fmean(
                float(row["preceding_action_delta_l2"]) for row in rows
            ),
        }

    meaningful_range_rows = [
        row for row in ranges if int(row["depth"]) == 5 and float(row["real_goal_cost_range"]) > EPSILON
    ]
    first_step_range_rows = [
        row for row in ranges if int(row["depth"]) == 1 and float(row["real_goal_cost_range"]) > EPSILON
    ]
    versions_after = tuple(int(tensor._version) for tensor in list(model.parameters()) + list(model.buffers()))
    summary = {
        "stage": "post_S3_propagation_audit",
        "physical_branches": len(predictions),
        "new_environment_calls": 0,
        "new_GD_iterations": 0,
        "sequential_rollout_paths": 2 * len(predictions),
        "real_state_one_step_rollouts": 5 * len(predictions),
        "model_transitions": 15 * len(predictions),
        "wall_clock_s": time.perf_counter() - started,
        "physical_primary_matched_final_mae": physical_raw_matched_mae,
        "teacher_forced_actual_action_final_mae": teacher_final_mae,
        "real_state_one_step_actual_action_final_mae": reset_final_mae,
        "teacher_forcing_reduces_final_mae": teacher_final_mae < physical_raw_matched_mae,
        "depth": depth_summary,
        "first_step_mean_free_to_real_range_ratio": statistics.fmean(
            float(row["free_to_real_range_ratio"]) for row in first_step_range_rows
        ) if first_step_range_rows else math.nan,
        "final_mean_free_to_real_range_ratio": statistics.fmean(
            float(row["free_to_real_range_ratio"]) for row in meaningful_range_rows
        ) if meaningful_range_rows else math.nan,
        "final_mean_teacher_to_real_range_ratio": statistics.fmean(
            float(row["teacher_to_real_range_ratio"]) for row in meaningful_range_rows
        ) if meaningful_range_rows else math.nan,
        "final_mean_real_state_one_step_to_real_range_ratio": statistics.fmean(
            float(row["real_state_one_step_to_real_range_ratio"])
            for row in meaningful_range_rows
        ) if meaningful_range_rows else math.nan,
        "validation": {
            "failures": validation_failures,
            "sidecar_reconstruction_max_abs_error": max(
                sidecar_reconstruction_errors, default=math.inf
            ),
            "sidecar_reconstruction_tolerance": LATENT_REPRO_TOLERANCE,
            "recorded_final_cost_crosscheck_max_abs_error": max(
                final_qenv_crosscheck_errors + final_qhat_crosscheck_errors,
                default=math.inf,
            ),
            "recorded_Qenv_crosscheck_max_abs_error": max(
                final_qenv_crosscheck_errors, default=math.inf
            ),
            "recorded_Qhat_crosscheck_max_abs_error": max(
                final_qhat_crosscheck_errors, default=math.inf
            ),
            "blind_anchor_vs_replayed_initial_max_abs_error": max(
                replayed_initial_alignment_errors, default=math.inf
            ),
            "blind_goal_vs_reencoded_goal_max_abs_error": max(
                goal_reencoding_alignment_errors, default=math.inf
            ),
            "model_parameters_and_buffers_unchanged": versions_before == versions_after,
        },
    }
    summary["validation"]["passed"] = bool(
        len(predictions) == 17
        and not validation_failures
        and summary["validation"]["sidecar_reconstruction_max_abs_error"]
        <= LATENT_REPRO_TOLERANCE
        and summary["validation"]["recorded_final_cost_crosscheck_max_abs_error"] <= 1e-6
        and summary["validation"]["model_parameters_and_buffers_unchanged"]
    )
    _write_csv(args.out / "propagation_depth_metrics.csv", depth_rows)
    _write_csv(args.out / "feedback_contraction_by_depth.csv", ranges)
    _write_json(args.out / "propagation_audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--c-physical", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--donor", type=_parse_donor, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    if not result["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
