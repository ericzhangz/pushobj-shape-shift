"""Completed-RGB representation closure and a time-held mechanics reference.

No environment, candidate or privileged state loader. Registration sees both
observed images: representation closure is deliberately NOT a forecast metric.
CSV signed_visual_task_error means predicted-minus-actual readout, i.e. -E;
it must not be inserted unchanged into the frozen E=true-minus-pred identity.
"""

import argparse
import csv
import json
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

from research.reframe_v3.completed_phase_evidence import phase_chains
from research.reframe_v3.contact_kinematic_reference import (
    contact_geometry, fit_sticking_scale, sticking_completion,
)
from research.reframe_v3.finite_task_reference import successor_from_history
from research.reframe_v3.rgb_rigid_observer import observe_rgb, register_pair, repaint_pair
from research.reframe_v3.round6_reference import _local_hub_loader
from research.reframe_v3.shadow_selection_audit import _load_runtime


def write_csv(path, rows):
    if not rows:
        raise ValueError("no measurements")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_completed(donor, sample, count=4):
    """Read precisely count completed chunks, excluding every other NPZ field."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("positive completed chunk count required")
    images, proprio, actions = [], [], []
    for chunk in range(count):
        with np.load(donor / "real_evidence" / f"s{sample}_executed_mpc{chunk}.npz",
                     allow_pickle=False) as archive:
            image = archive["visual"].copy()
            prop = archive["proprio"].copy()
            action = archive["normalized_model_actions"].copy()
        if image.shape != (6, 224, 224, 3) or prop.shape != (6, 4) or action.shape != (1, 1, 10):
            raise ValueError("unexpected completed evidence layout")
        if chunk and (not np.array_equal(images[-1][-1], image[0])
                      or not np.array_equal(proprio[-1][-1], prop[0])):
            raise ValueError("noncontinuous completed evidence")
        images.append(image if not chunk else image[1:])
        proprio.append(prop if not chunk else prop[1:])
        actions.append(action.reshape(5, 2))
    return np.concatenate(images), np.concatenate(proprio), np.concatenate(actions)


@torch.no_grad()
def encode(model, preprocessor, images, proprio, device):
    result = []
    for start in range(0, len(images), 16):
        observed = preprocessor.transform_obs({"visual": images[None, start:start+16],
                                               "proprio": proprio[None, start:start+16]})
        observed = {key: value.to(device) for key, value in observed.items()}
        result.append(model.encode_obs(observed))
    return {key: torch.cat([item[key] for item in result], dim=1) for key in result[0]}


def movement_rows(images, split, sample):
    observed = [observe_rgb(frame) for frame in images]
    reference = np.asarray(observed[0]["object_centroid"], dtype=float)
    rows = []
    for step in range(20):
        before, after = observed[step:step+2]
        fit = register_pair(images[step], images[step+1])
        affine = np.asarray(fit["affine"])
        geometry = contact_geometry(before["object_mask"], before["pusher_center"],
                                    before["pusher_radius"], reference)
        next_reference = affine[:, :2] @ reference + affine[:, 2]
        dp = np.asarray(after["pusher_center"]) - before["pusher_center"]
        point = geometry["point"]
        point_delta = affine[:, :2] @ point + affine[:, 2] - point
        velocity = next_reference - reference
        theta = float(fit["theta_radians"])
        r = geometry["r"]
        linear_point_delta = velocity + theta*np.array([-r[1], r[0]])
        next_gap = contact_geometry(after["object_mask"], after["pusher_center"],
                                    after["pusher_radius"], next_reference)["gap"]
        rows.append({"split": split, "sample": sample, "step": step,
                     "is_prefix": step < 10, "gap_px": geometry["gap"],
                     "next_gap_px": next_gap, "rx": float(r[0]), "ry": float(r[1]),
                     "dpx": float(dp[0]), "dpy": float(dp[1]), "theta": theta,
                     "vx": float(velocity[0]), "vy": float(velocity[1]),
                     "pusher_displacement_px": float(np.linalg.norm(dp)),
                     "object_reference_displacement_px": float(np.linalg.norm(velocity)),
                     "finite_sticking_residual_px": float(np.linalg.norm(point_delta-dp)),
                     "linear_sticking_residual_px": float(np.linalg.norm(linear_point_delta-dp)),
                     "iou": float(fit["silhouette_iou"]),
                     "chamfer_rmse_px": float(fit["chamfer_rmse_px"]),
                     "trimmed_chamfer_rmse_px": float(fit["trimmed_chamfer_rmse_px"]),
                     "source_boundary_fraction": float(fit["source_boundary_fraction"]),
                     "target_boundary_fraction": float(fit["target_boundary_fraction"]),
                     "optimizer_converged": fit["optimizer_converged"],
                     "alternative_start_objective_gap": float(fit["alternative_start_objective_gap"])})
        reference = next_reference
    return rows


def held_mechanics(rows):
    """Conditional response given REAL held pusher displacement, not action rollout.

    Each interval also receives its real starting scene geometry. Only the
    scalar completion parameter is held fixed from the earlier prefix.
    """
    output, fits = [], []
    for threshold in (1., 2., 3.):
        prefix = [row for row in rows if row["is_prefix"] and row["gap_px"] <= threshold]
        fit = fit_sticking_scale(prefix)
        fits.append({"gap_threshold_px": threshold, **fit})
        if fit["status"] != "fit":
            continue
        for row in rows:
            if row["is_prefix"] or row["gap_px"] > threshold:
                continue
            velocity, omega = sticking_completion([row["rx"], row["ry"]],
                                                  [row["dpx"], row["dpy"]],
                                                  fit["ell_squared"])
            dv = velocity - [row["vx"], row["vy"]]
            radius = np.hypot(row["rx"], row["ry"])
            output.append({**row, "gap_threshold_px": threshold,
                           "ell_squared": fit["ell_squared"],
                           "translation_error_px": float(np.linalg.norm(dv)),
                           "rotation_error_rad": abs(omega-row["theta"]),
                           "rotation_lever_error_px": radius*abs(omega-row["theta"]),
                           "predicted_theta": omega})
    return output, fits


@torch.no_grad()
def closure_rows(model, preprocessor, images, proprio, actions, goal_z, device,
                 split, sample, out):
    encoded = encode(model, preprocessor, images, proprio, device)
    frozen = {}
    for chain in phase_chains(encoded, torch.as_tensor(actions[None], device=device)):
        for index, start in enumerate(chain["lowlevel_starts"]):
            frozen[start] = successor_from_history(model, chain["observed"],
                                                   chain["actions"], index)
    rows = []
    for start in range(16):
        end = start + 5
        fit = register_pair(images[start], images[end])
        stationary = {**fit, "affine": np.array([[1., 0., 0.], [0., 1., 0.]]),
                      "theta_radians": 0.}
        repaints = {
            "RIGID-SOURCE-BACKGROUND": repaint_pair(images[start], images[end], fit),
            "RIGID-RETROSPECTIVE-BACKGROUND": repaint_pair(
                images[start], images[end], fit, background="retrospective_target"),
            "STATIC-OBJECT-MOVED-PUSHER": repaint_pair(images[start], images[end], stationary),
        }
        rgb = np.stack(list(repaints.values()))
        repaint_z = encode(model, preprocessor, rgb, np.repeat(proprio[end:end+1], 3, 0), device)
        actual = encoded["visual"][:, end:end+1]
        actual_cost = float((actual-goal_z["visual"]).square().mean())
        source_change = float((encoded["visual"][:, start:start+1]-actual).square().mean())
        for index, arm in enumerate([*repaints, "FROZEN-NATIVE-REAL-HISTORY"]):
            visual = (repaint_z["visual"][:, index:index+1] if arm in repaints
                      else frozen[start]["visual"])
            cost = float((visual-goal_z["visual"]).square().mean())
            rows.append({"split": split, "sample": sample, "start": start, "end": end,
                         "arm": arm, "registration_iou": float(fit["silhouette_iou"]),
                         "registration_chamfer_rmse_px": float(fit["chamfer_rmse_px"]),
                         "registration_trimmed_chamfer_rmse_px": float(fit["trimmed_chamfer_rmse_px"]),
                         "registration_source_boundary_fraction": float(fit["source_boundary_fraction"]),
                         "registration_target_boundary_fraction": float(fit["target_boundary_fraction"]),
                         "registration_optimizer_converged": fit["optimizer_converged"],
                         "registration_alternative_start_objective_gap": float(fit["alternative_start_objective_gap"]),
                         "visual_mse": float((visual-actual).square().mean()),
                         "source_to_target_visual_mse": source_change,
                         "actual_visual_task_cost": actual_cost, "visual_task_cost": cost,
                         "signed_visual_task_error": cost-actual_cost,
                         "absolute_visual_task_error": abs(cost-actual_cost),
                         "rgb_mae_255": (float(np.abs(repaints[arm].astype(float)-images[end]).mean())
                                         if arm in repaints else ""),
                         "uses_target_registration": arm in repaints and not arm.startswith("STATIC"),
                         "uses_target_pusher_pixels": arm in repaints})
        if start in (0, 15):
            montage = np.concatenate([images[start], images[end], *repaints.values()], axis=1)
            cv2.imwrite(str(out / f"{split}{sample}_s{start}_source_true_warp_upper_static.png"),
                        cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return rows


def summary(closure, movement, mechanics):
    result = {"cases": 4, "completed_transitions": len(movement),
              "new_environment_calls": 0, "new_candidate_predictions": 0,
              "closure_not_a_forecast": True, "closure": {}, "mechanics": {}}
    for arm in dict.fromkeys(row["arm"] for row in closure):
        selected = [row for row in closure if row["arm"] == arm]
        case_values = []
        for split in ("T", "L"):
            for sample in (0, 1):
                case = [row for row in selected if row["split"] == split and row["sample"] == sample]
                case_values.append({"case": f"{split}{sample}", "windows": len(case),
                                    "visual_mse": float(np.mean([r["visual_mse"] for r in case])),
                                    "task_mae": float(np.mean([r["absolute_visual_task_error"] for r in case]))})
        result["closure"][arm] = {
            "case_values": case_values,
            "mean_visual_mse": float(np.mean([r["visual_mse"] for r in selected])),
            "mean_visual_task_mae": float(np.mean([r["absolute_visual_task_error"] for r in selected])),
            "sum_mse_over_sum_observed_change_mse": float(sum(r["visual_mse"] for r in selected)
                / sum(r["source_to_target_visual_mse"] for r in selected))}
    for threshold in (1., 2., 3.):
        contacts = [row for row in movement if row["gap_px"] <= threshold]
        held = [row for row in mechanics if row["gap_threshold_px"] == threshold]
        result["mechanics"][str(threshold)] = {
            "all_geometry_selected_intervals": len(contacts), "held_predictions": len(held),
            "finite_sticking_residual_mean_px": (float(np.mean([r["finite_sticking_residual_px"] for r in contacts]))
                                                 if contacts else None),
            "held_translation_mae_px": (float(np.mean([r["translation_error_px"] for r in held])) if held else None),
            "held_rotation_lever_mae_px": (float(np.mean([r["rotation_lever_error_px"] for r in held])) if held else None)}
    result["registration"] = {
        "micro_iou_min": min(r["iou"] for r in movement),
        "micro_iou_mean": float(np.mean([r["iou"] for r in movement])),
        "micro_chamfer_max_px": max(r["chamfer_rmse_px"] for r in movement)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    with patch.object(torch.hub, "load", _local_hub_loader()):
        model, preprocessor, config = _load_runtime(args.checkpoint_dir, torch.device(args.device))
    if int(config.frameskip) != 5:
        raise ValueError("requires native stride five")
    json_dump(args.out / "RUN_CONTRACT.json", {
        "cases": ["T0", "T1", "L0", "L1"], "completed_chunks": [0, 1, 2, 3],
        "dynamics_fit_lowlevel_steps": [0, 9], "dynamics_held_steps": [10, 19],
        "registration_uses_both_observed_endpoints": True,
        "representation_uses_observed_target_pusher_pixels": True,
        "mechanics_is_conditional_on_real_next_pusher_displacement": True,
        "mechanics_uses_real_current_scene_each_interval": True,
        "neither_probe_is_new_action_forecasting": True,
        "goal_is_evaluation_only": True, "state_and_candidate_loaders": False,
        "new_environment_calls": 0})
    movement, closure, mechanics, fits = [], [], [], []
    for split in ("T", "L"):
        donor = args.donor_root / f"donor_{split}_seed101_n12_capture"
        for sample in (0, 1):
            images, proprio, actions = load_completed(donor, sample)
            current_movement = movement_rows(images, split, sample)
            held, current_fits = held_mechanics(current_movement)
            movement.extend(current_movement)
            mechanics.extend(held)
            fits.append({"case": f"{split}{sample}", "fits": current_fits})
            with np.load(donor / "real_evidence" / f"sample_{sample:03d}_initial.npz",
                         allow_pickle=False) as archive:
                goal_z = encode(model, preprocessor, archive["goal_visual"][0],
                                archive["goal_proprio"][0], args.device)
            closure.extend(closure_rows(model, preprocessor, images, proprio, actions,
                                        goal_z, args.device, split, sample, args.out))
            print(f"completed {split}{sample}: 20 observed intervals, 16 representation windows", flush=True)
    write_csv(args.out / "movement.csv", movement)
    write_csv(args.out / "representation.csv", closure)
    if mechanics:
        write_csv(args.out / "held_mechanics.csv", mechanics)
    json_dump(args.out / "prefix_fits.json", fits)
    result = summary(closure, movement, mechanics)
    result["wall_seconds"] = time.perf_counter()-started
    json_dump(args.out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
