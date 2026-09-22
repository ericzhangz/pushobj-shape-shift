"""Round 4 analysis of saved C vectors; no model or environment calls.

The W coordinate is a block-scaled flattening of visual and proprio latents.
Thus native terminal cost equals 0.5 * squared Euclidean norm in this coordinate.
Saved real futures and reset/persistence modes are evaluation-only diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


PARITY_TOL = 1e-6
CLOSURE_TOL = 1e-10
NORM_MASK = 1e-10
MODES = ("free", "teacher", "reset", "persistence")
SCORE_FIELDS = {
    "real": ("real_goal_cost", None),
    "free": ("free_model_goal_cost", "free_latent_mse"),
    "teacher": ("teacher_forced_model_goal_cost", "teacher_forced_latent_mse"),
    "reset": ("real_state_one_step_model_goal_cost", "real_state_one_step_latent_mse"),
}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict]) -> None:
    if path.exists() or not rows:
        raise ValueError(f"output exists or table is empty: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def w_vector(latent: dict, depth: int) -> np.ndarray:
    """Flatten only observation latents, with native MSE block weights."""
    visual = latent["visual"][0, depth].detach().cpu().numpy().astype(np.float64).ravel()
    proprio = latent["proprio"][0, depth].detach().cpu().numpy().astype(np.float64).ravel()
    if not visual.size or not proprio.size:
        raise ValueError("empty observation block")
    value = np.concatenate((visual * math.sqrt(2.0 / visual.size),
                            proprio * math.sqrt(2.0 / proprio.size)))
    if not np.isfinite(value).all():
        raise ValueError("nonfinite latent")
    return value


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def norm(a: np.ndarray) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def cost(z: np.ndarray, goal: np.ndarray) -> float:
    difference = z - goal
    return 0.5 * dot(difference, difference)


def pair_metrics(zi: np.ndarray, zj: np.ndarray, pi: np.ndarray,
                 pj: np.ndarray, goal: np.ndarray) -> dict:
    """Signed contrast and vector-error identities, with no norm division."""
    delta, predicted_delta = zi - zj, pi - pj
    radius = (zi + zj) / 2 - goal
    predicted_radius = (pi + pj) / 2 - goal
    real_contrast = cost(zi, goal) - cost(zj, goal)
    predicted_contrast = cost(pi, goal) - cost(pj, goal)
    midpoint = dot(predicted_radius - radius, (predicted_delta + delta) / 2)
    effect = dot((predicted_radius + radius) / 2, predicted_delta - delta)
    d, predicted_d = norm(delta), norm(predicted_delta)
    radial = (predicted_d - d) ** 2
    angular = 2 * (predicted_d * d - dot(predicted_delta, delta))
    vector_error_sq = dot(predicted_delta - delta, predicted_delta - delta)
    return {
        "real_contrast": real_contrast,
        "predicted_contrast": predicted_contrast,
        "contrast_error": predicted_contrast - real_contrast,
        "midpoint_contribution": midpoint,
        "effect_contribution": effect,
        "signed_closure": predicted_contrast - real_contrast - midpoint - effect,
        "real_effect_norm": d,
        "predicted_effect_norm": predicted_d,
        "effect_vector_error_sq": vector_error_sq,
        "radial_error_sq": radial,
        "direction_error_sq": angular,
        "vector_closure": vector_error_sq - radial - angular,
        "effect_cosine": dot(predicted_delta, delta) / (predicted_d * d)
        if min(predicted_d, d) > NORM_MASK else math.nan,
        "real_task_projection": dot(radius, delta),
        "predicted_task_projection": dot(predicted_radius, predicted_delta),
        "real_task_cosine": dot(radius, delta) / (norm(radius) * d)
        if min(norm(radius), d) > NORM_MASK else math.nan,
        "predicted_task_cosine": dot(predicted_radius, predicted_delta)
        / (norm(predicted_radius) * predicted_d)
        if min(norm(predicted_radius), predicted_d) > NORM_MASK else math.nan,
    }


def _pair_classes(left: dict, right: dict) -> tuple[bool, bool]:
    labels = (set(left["labels"].split(";")), set(right["labels"].split(";")))
    frozen_adapted = ("frozen" in labels[0] and "adapted" in labels[1]) or (
        "adapted" in labels[0] and "frozen" in labels[1])
    frozen_nonfrozen = ("frozen" in labels[0]) != ("frozen" in labels[1])
    return frozen_nonfrozen, frozen_adapted


def _finite_gain(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], row["mode"])][row["depth"]] = row
    output = []
    for (pair_id, mode), by_depth in sorted(grouped.items()):
        if mode not in ("free", "teacher"):
            continue
        reference = by_depth[1]
        baseline_valid = min(reference["real_effect_norm"],
                             reference["predicted_effect_norm"]) > NORM_MASK
        previous = None
        chain_valid = baseline_valid
        telescoped = 0.0
        for depth in range(1, 6):
            row = by_depth[depth]
            endpoint_valid = baseline_valid and min(row["real_effect_norm"],
                                                    row["predicted_effect_norm"]) > NORM_MASK
            adjacent_valid = endpoint_valid and previous is not None
            if endpoint_valid:
                value = math.log(row["predicted_effect_norm"] / reference["predicted_effect_norm"])
                value -= math.log(row["real_effect_norm"] / reference["real_effect_norm"])
                if adjacent_valid:
                    step = math.log(row["predicted_effect_norm"] / previous["predicted_effect_norm"])
                    step -= math.log(row["real_effect_norm"] / previous["real_effect_norm"])
                    if chain_valid:
                        telescoped += step
                        gap = value - telescoped
                    else:
                        gap = math.nan
                else:
                    step = 0.0 if depth == 1 else math.nan
                    gap = 0.0 if depth == 1 else math.nan
                previous = row
            else:
                value = step = gap = math.nan
                previous = None
                chain_valid = False
            output.append({
                "pair_id": pair_id, "anchor_id": row["anchor_id"], "mode": mode,
                "depth": depth, "frozen_nonfrozen": row["frozen_nonfrozen"],
                "frozen_adapted": row["frozen_adapted"],
                "real_norm": row["real_effect_norm"],
                "predicted_norm": row["predicted_effect_norm"],
                "endpoint_gain_defined": endpoint_valid,
                "adjacent_gain_defined": adjacent_valid or (depth == 1 and endpoint_valid),
                "telescoping_defined": math.isfinite(gap),
                "undefined_reason": "" if endpoint_valid else "depth1_or_current_norm_at_or_below_mask",
                "lambda_from_depth1": value, "adjacent_log_gain_difference": step,
                "telescoping_gap": gap,
            })
    return output


def run(args) -> dict:
    staging = args.out.with_name(args.out.name + "_staging")
    if args.out.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.out}")
    propagation = json.loads(args.propagation_summary.read_text(encoding="utf-8"))
    if (not propagation.get("validation", {}).get("passed")
            or propagation.get("physical_branches") != 17
            or propagation.get("vector_sidecars") != 17
            or propagation.get("model_transitions") != 255
            or propagation.get("new_environment_calls") != 0
            or propagation.get("new_GD_iterations") != 0
            or Path(propagation.get("vector_output_path", "")).resolve() != args.vectors.resolve()):
        raise ValueError("propagation/vector source contract not passed")
    scalar = {(row["physical_branch_id"], int(row["depth"])): row
              for row in _rows(args.scalar_depth)}
    if len(scalar) != 17 * 6:
        raise ValueError(f"expected 102 unique scalar rows, found {len(scalar)}")
    paths = sorted(args.vectors.glob("*.pt"))
    if len(paths) != 17:
        raise ValueError(f"expected 17 vector sidecars, found {len(paths)}")
    branches = defaultdict(list)
    parity_errors = []
    vector_dims = set()
    for path in paths:
        item = torch.load(path, map_location="cpu")
        physical_id = item["physical_branch_id"]
        if path.stem != physical_id:
            raise ValueError(f"sidecar/name mismatch: {path}")
        for mode in ("real", "free", "teacher", "reset"):
            latent = item[f"{mode}_z"]
            if latent["visual"].shape[1] != 6 or latent["proprio"].shape[1] != 6:
                raise ValueError(f"latent time shape: {physical_id}:{mode}")
        if item["actual_actions"].shape != (1, 5, 10) or item["feedback_actions"].shape != (1, 5, 10):
            raise ValueError(f"action shape: {physical_id}")
        goal = w_vector(item["goal_z"], 0)
        branch = {"id": physical_id, "anchor_id": item["anchor_id"], "labels": item["labels"],
                  "goal": goal, "vectors": {mode: [] for mode in ("real",) + MODES}}
        for depth in range(6):
            for mode in ("real", "free", "teacher", "reset"):
                branch["vectors"][mode].append(w_vector(item[f"{mode}_z"], depth))
            branch["vectors"]["persistence"].append(branch["vectors"]["real"][max(depth - 1, 0)])
            real = branch["vectors"]["real"][depth]
            entry = scalar[(physical_id, depth)]
            if entry["anchor_id"] != item["anchor_id"] or entry["labels"] != item["labels"]:
                raise ValueError(f"scalar/vector identity mismatch: {physical_id}:{depth}")
            for mode, (score_field, mse_field) in SCORE_FIELDS.items():
                value = branch["vectors"][mode][depth]
                recorded_score = float(entry[score_field])
                if not math.isfinite(recorded_score):
                    raise ValueError(f"nonfinite saved scalar: {physical_id}:{depth}:{score_field}")
                parity_error = abs(cost(value, goal) - recorded_score)
                if not math.isfinite(parity_error):
                    raise ValueError(f"nonfinite scalar parity: {physical_id}:{depth}:{score_field}")
                parity_errors.append(parity_error)
                if mse_field:
                    recorded_mse = float(entry[mse_field])
                    if not math.isfinite(recorded_mse):
                        raise ValueError(f"nonfinite saved scalar: {physical_id}:{depth}:{mse_field}")
                    parity_error = abs(cost(value, real) - recorded_mse)
                    if not math.isfinite(parity_error):
                        raise ValueError(f"nonfinite MSE parity: {physical_id}:{depth}:{mse_field}")
                    parity_errors.append(parity_error)
                    for block in ("visual", "proprio"):
                        latent_value = item[f"{mode}_z"][block][0, depth].double()
                        real_value = item["real_z"][block][0, depth].double()
                        block_mse = float(torch.mean((latent_value - real_value) ** 2))
                        field = f"{mode if mode != 'teacher' else 'teacher_forced'}_{block}_mse"
                        if mode == "reset":
                            field = f"real_state_one_step_{block}_mse"
                        saved = float(entry[field])
                        if not math.isfinite(saved) or not math.isfinite(block_mse):
                            raise ValueError(f"nonfinite block MSE: {physical_id}:{depth}:{field}")
                        parity_errors.append(abs(block_mse - saved))
            vector_dims.add(real.size)
        branches[item["anchor_id"]].append(branch)
    if len(branches) != 6 or any(len(group) not in (2, 3) for group in branches.values()):
        raise ValueError("unexpected anchor/physical grouping")
    if max(parity_errors) > PARITY_TOL:
        raise ValueError(f"scalar/block parity failed: {max(parity_errors)}")
    pair_rows, contrast_rows, forced_rows, forced_pair_rows = [], [], [], []
    pair_count = 0
    for anchor_id, group in sorted(branches.items()):
        group.sort(key=lambda entry: entry["id"])
        reference_goal = group[0]["goal"]
        if any(np.max(np.abs(branch["goal"] - reference_goal)) > PARITY_TOL for branch in group[1:]):
            raise ValueError(f"goal mismatch within anchor: {anchor_id}")
        for branch in group:
            for depth in range(1, 6):
                z, teacher, reset = (branch["vectors"][mode][depth]
                                     for mode in ("real", "teacher", "reset"))
                p = teacher - reset
                eta = reset - z
                forced_rows.append({
                    "anchor_id": anchor_id, "physical_branch_id": branch["id"], "depth": depth,
                    "input_recursion_norm_sq": dot(p, p), "real_input_residual_norm_sq": dot(eta, eta),
                    "cross_inner_product": dot(p, eta),
                    "teacher_error_norm_sq": dot(teacher - z, teacher - z),
                    "closure": dot(teacher - z, teacher - z)
                    - dot(p, p) - dot(eta, eta) - 2 * dot(p, eta),
                })
        for left, right in combinations(group, 2):
            pair_count += 1
            pair_id = f"{left['id']}__{right['id']}"
            nonfrozen, legal = _pair_classes(left, right)
            for depth in range(1, 6):
                real_i, real_j = (branch["vectors"]["real"][depth] for branch in (left, right))
                for mode in MODES:
                    pred_i, pred_j = (branch["vectors"][mode][depth] for branch in (left, right))
                    metrics = pair_metrics(real_i, real_j, pred_i, pred_j, reference_goal)
                    common = {"pair_id": pair_id, "anchor_id": anchor_id,
                              "left_id": left["id"], "right_id": right["id"],
                              "left_labels": left["labels"], "right_labels": right["labels"],
                              "frozen_nonfrozen": nonfrozen, "frozen_adapted": legal,
                              "depth": depth, "mode": mode}
                    pair_rows.append({**common, **metrics})
                    contrast_rows.append({**common, **{key: metrics[key] for key in (
                        "real_contrast", "predicted_contrast", "contrast_error",
                        "midpoint_contribution", "effect_contribution", "signed_closure")}})
                p_i = left["vectors"]["teacher"][depth] - left["vectors"]["reset"][depth]
                p_j = right["vectors"]["teacher"][depth] - right["vectors"]["reset"][depth]
                e_i = left["vectors"]["reset"][depth] - real_i
                e_j = right["vectors"]["reset"][depth] - real_j
                dp, de = p_i - p_j, e_i - e_j
                forced_pair_rows.append({
                    "pair_id": pair_id, "anchor_id": anchor_id, "depth": depth,
                    "frozen_nonfrozen": nonfrozen, "frozen_adapted": legal,
                    "recursion_effect_norm_sq": dot(dp, dp), "residual_effect_norm_sq": dot(de, de),
                    "cross_inner_product": dot(dp, de),
                    "teacher_effect_error_norm_sq": dot(dp + de, dp + de),
                    "closure": dot(dp + de, dp + de) - dot(dp, dp) - dot(de, de) - 2 * dot(dp, de),
                })
    if pair_count != 16 or len(pair_rows) != 16 * 5 * 4:
        raise ValueError(f"pair cardinality: {pair_count}, {len(pair_rows)} rows")
    if sum(row["frozen_nonfrozen"] for row in pair_rows if row["depth"] == 5 and row["mode"] == "free") != 11:
        raise ValueError("nonfrozen pair cardinality")
    if sum(row["frozen_adapted"] for row in pair_rows if row["depth"] == 5 and row["mode"] == "free") != 6:
        raise ValueError("legal pair cardinality")
    gains = _finite_gain(pair_rows)
    snapshot_rows = [{"pair_id": row["pair_id"], "anchor_id": row["anchor_id"],
                      "mode": row["mode"], "depth": row["depth"],
                      "frozen_nonfrozen": row["frozen_nonfrozen"],
                      "frozen_adapted": row["frozen_adapted"],
                      "real_norm": row["real_effect_norm"],
                      "predicted_norm": row["predicted_effect_norm"],
                      "pointwise_ratio": row["predicted_effect_norm"] / row["real_effect_norm"]
                      if row["real_effect_norm"] > NORM_MASK else math.nan,
                      "ratio_defined": row["real_effect_norm"] > NORM_MASK}
                     for row in pair_rows if row["mode"] in ("reset", "persistence")]
    mask_rows = []
    for subset, select in (("all", lambda row: True),
                           ("frozen_nonfrozen", lambda row: row["frozen_nonfrozen"]),
                           ("frozen_adapted", lambda row: row["frozen_adapted"])):
        for mode in MODES:
            for depth in range(1, 6):
                rows = [row for row in pair_rows if select(row) and row["mode"] == mode and row["depth"] == depth]
                mask_rows.append({"subset": subset, "mode": mode, "depth": depth,
                                  "pairs": len(rows),
                                  "both_norms_above_mask": sum(min(row["real_effect_norm"],
                                      row["predicted_effect_norm"]) > NORM_MASK for row in rows),
                                  "real_norm_above_mask": sum(row["real_effect_norm"] > NORM_MASK for row in rows),
                                  "effect_cosine_defined": sum(math.isfinite(row["effect_cosine"]) for row in rows),
                                  "real_task_cosine_defined": sum(math.isfinite(row["real_task_cosine"]) for row in rows),
                                  "predicted_task_cosine_defined": sum(math.isfinite(row["predicted_task_cosine"]) for row in rows)})
    all_closures = [abs(row[key]) for row in pair_rows for key in ("signed_closure", "vector_closure")]
    all_closures += [abs(row["closure"]) for row in forced_rows + forced_pair_rows]
    all_closures += [abs(row["telescoping_gap"]) for row in gains if math.isfinite(row["telescoping_gap"])]
    summary = {
        "stage": "G1_Round4_saved_vector_geometry", "anchors": len(branches),
        "physical_branches": sum(map(len, branches.values())), "physical_pairs": pair_count,
        "pair_depth_mode_rows": len(pair_rows), "frozen_nonfrozen_pairs": 11,
        "legal_frozen_adapted_pairs": 6, "latent_flattened_dimensions": sorted(vector_dims),
        "W": "visual 2/n_visual; proprio 2/n_proprio; C=0.5||z-g||_W^2",
        "norm_mask": NORM_MASK, "scalar_parity_max_abs": max(parity_errors),
        "scalar_parity_tolerance": PARITY_TOL, "closure_max_abs": max(all_closures),
        "closure_tolerance": CLOSURE_TOL,
        "new_model_transitions": 0, "new_environment_calls": 0,
        "new_GD_iterations": 0, "new_training_updates": 0,
        "truth_role": "evaluation_only; C future observations/actions never used for update",
        "passed": max(all_closures) <= CLOSURE_TOL,
    }
    staging.mkdir(parents=True)
    _write_rows(staging / "PAIR_GEOMETRY.csv", pair_rows)
    _write_rows(staging / "PAIR_CONTRAST_DECOMPOSITION.csv", contrast_rows)
    _write_rows(staging / "FINITE_PAIR_GAIN.csv", gains)
    _write_rows(staging / "POINTWISE_SNAPSHOTS.csv", snapshot_rows)
    _write_rows(staging / "MASK_COVERAGE.csv", mask_rows)
    _write_rows(staging / "OBSERVED_FORCED_ERROR.csv", forced_rows)
    _write_rows(staging / "FORCED_PAIR_ERROR.csv", forced_pair_rows)
    _write_json(staging / "GEOMETRY_CLOSURE.json", summary)
    if summary["passed"]:
        staging.rename(args.out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--scalar-depth", type=Path, required=True)
    parser.add_argument("--propagation-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
