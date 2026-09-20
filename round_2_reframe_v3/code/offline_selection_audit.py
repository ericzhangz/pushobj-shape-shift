"""Audit logged optimizer endpoints without loading a model or environment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_GD_STEPS = (0, 24, 49, 74, 99)
NUMERICAL_TIE_EPSILON = 1e-6
GROUP_FIELDS = ("run_id", "sample_id", "mpc_iter")
IDENTITY_FIELDS = (
    "phase",
    "split",
    "run_path",
    "action_unit_version",
    "adaptation_buffer_ids",
    "arm",
    "encoder_version",
    "model_version",
    "objective_version",
    "predictor_version",
    "real_history_id",
    "run_id",
    "sample_id",
    "seed",
    "shape",
    "mpc_iter",
)
FLOAT_FIELDS = (
    "c_env_ref_after",
    "c_env_ref_before",
    "c_env_samever_after",
    "c_env_samever_before",
    "c_hat_native_after",
    "c_hat_native_before",
    "delta_env_ref",
    "delta_env_samever",
    "delta_pred",
)
DELTA_ENDPOINT_FIELDS = (
    ("delta_pred", "c_hat_native_before", "c_hat_native_after"),
    ("delta_env_samever", "c_env_samever_before", "c_env_samever_after"),
    ("delta_env_ref", "c_env_ref_before", "c_env_ref_after"),
)
VERSION_FIELDS = (
    "action_unit_version",
    "encoder_version",
    "model_version",
    "objective_version",
    "predictor_version",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(IDENTITY_FIELDS) | set(FLOAT_FIELDS) | {"gd_iter"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")
    for row_number, row in enumerate(rows, start=2):
        row["gd_iter"] = int(row["gd_iter"])
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(
                    f"Row {row_number} has non-finite {field}: {row[field]}"
                )
        for delta_field, before_field, after_field in DELTA_ENDPOINT_FIELDS:
            expected = row[after_field] - row[before_field]
            if not math.isclose(
                row[delta_field], expected, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Row {row_number} {delta_field} does not match endpoints: "
                    f"stored={row[delta_field]} expected={expected}"
                )
    return rows


def endpoint(row: dict, side: str) -> dict:
    suffix = "before" if side == "before" else "after"
    return {
        "candidate_id": f"g{row['gd_iter']}_{side}",
        "gd_iter": row["gd_iter"],
        "side": side,
        "c_model": row[f"c_hat_native_{suffix}"],
        "c_env": row[f"c_env_samever_{suffix}"],
        "c_env_ref": row[f"c_env_ref_{suffix}"],
    }


def base_fields(rows: list[dict]) -> dict:
    first = rows[0]
    result = {field: first[field] for field in IDENTITY_FIELDS}
    result["objective_stage"] = "terminal" if int(first["mpc_iter"]) < 5 else "full_horizon"
    return result


def analyze_group(rows: list[dict]) -> tuple[list[dict], dict]:
    ordered = sorted(rows, key=lambda row: row["gd_iter"])
    key = tuple(ordered[0][field] for field in GROUP_FIELDS)
    actual_steps = tuple(row["gd_iter"] for row in ordered)
    if actual_steps != EXPECTED_GD_STEPS:
        raise ValueError(
            f"Replan {key} expected GD snapshots {EXPECTED_GD_STEPS}, got {actual_steps}"
        )
    for field in VERSION_FIELDS:
        values = {row[field] for row in ordered}
        if len(values) != 1:
            raise ValueError(f"Replan {key} has mixed {field}: {sorted(values)}")
    base = base_fields(ordered)
    initial = endpoint(ordered[0], "before")
    seen = [initial]
    path_rows = []
    for row in ordered:
        before = endpoint(row, "before")
        after = endpoint(row, "after")
        if before["candidate_id"] != initial["candidate_id"]:
            seen.append(before)
        seen.append(after)
        model_best = min(seen, key=lambda candidate: candidate["c_model"])
        env_best = min(seen, key=lambda candidate: candidate["c_env"])
        g_model = initial["c_model"] - after["c_model"]
        g_env = initial["c_env"] - after["c_env"]
        omega = (after["c_env"] - after["c_model"]) - (
            initial["c_env"] - initial["c_model"]
        )
        path_rows.append(
            {
                **base,
                "gd_iter": row["gd_iter"],
                "endpoint_candidate_id": after["candidate_id"],
                "c_model_initial": initial["c_model"],
                "c_env_initial": initial["c_env"],
                "c_model_endpoint": after["c_model"],
                "c_env_endpoint": after["c_env"],
                "g_model": g_model,
                "g_env": g_env,
                "omega": omega,
                "best_so_far_model_candidate_id": model_best["candidate_id"],
                "best_so_far_env_candidate_id": env_best["candidate_id"],
                "best_so_far_r_selected": model_best["c_env"] - env_best["c_env"],
            }
        )

    pool = []
    for row in ordered:
        pool.extend((endpoint(row, "before"), endpoint(row, "after")))
    model_best = min(pool, key=lambda candidate: candidate["c_model"])
    env_best = min(pool, key=lambda candidate: candidate["c_env"])
    final_record = endpoint(ordered[-1], "after")
    errors = [candidate["c_env"] - candidate["c_model"] for candidate in pool]
    r_selected = model_best["c_env"] - env_best["c_env"]
    r_final = final_record["c_env"] - env_best["c_env"]
    eta = final_record["c_model"] - model_best["c_model"]
    osc = max(errors) - min(errors)
    model_costs = sorted(candidate["c_model"] for candidate in pool)
    model_ties = [
        candidate
        for candidate in pool
        if candidate["c_model"] <= model_best["c_model"] + NUMERICAL_TIE_EPSILON
    ]
    tie_regrets = [candidate["c_env"] - env_best["c_env"] for candidate in model_ties]
    selection = {
        **base,
        "pool_size": len(pool),
        "model_best_candidate_id": model_best["candidate_id"],
        "env_best_candidate_id": env_best["candidate_id"],
        "final_record_candidate_id": final_record["candidate_id"],
        "r_selected": r_selected,
        "r_final_record": r_final,
        "eta_model_final_record": eta,
        "error_oscillation": osc,
        "numerical_tie_epsilon": NUMERICAL_TIE_EPSILON,
        "model_tie_count_epsilon": len(model_ties),
        "r_selected_tie_min": min(tie_regrets),
        "r_selected_tie_max": max(tie_regrets),
        "model_best_margin": model_costs[1] - model_costs[0],
        "selected_bound_slack": osc - r_selected,
        "final_bound_slack": osc + eta - r_final,
        "g_model_final": initial["c_model"] - final_record["c_model"],
        "g_env_final": initial["c_env"] - final_record["c_env"],
        "omega_final": (final_record["c_env"] - final_record["c_model"])
        - (initial["c_env"] - initial["c_model"]),
    }
    return path_rows, selection


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(values: list[float], draws: int = 5000) -> tuple[float, float]:
    rng = random.Random(260920)
    estimates = []
    for _ in range(draws):
        estimates.append(mean([rng.choice(values) for _ in values]))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def build_case_metrics(selection_rows: list[dict]) -> list[dict]:
    metric_fields = (
        "r_selected",
        "r_final_record",
        "eta_model_final_record",
        "error_oscillation",
        "g_model_final",
        "g_env_final",
        "omega_final",
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in selection_rows:
        key = (
            row["phase"],
            row["split"],
            row["shape"],
            row["arm"],
            row["objective_stage"],
            row["sample_id"],
        )
        grouped[key].append(row)
    result = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        phase, split, shape, arm, objective_stage, sample_id = key
        rows = grouped[key]
        record = {
            "phase": phase,
            "split": split,
            "shape": shape,
            "arm": arm,
            "objective_stage": objective_stage,
            "sample_id": sample_id,
            "case_id": f"{phase}|{split}|{shape}|{sample_id}",
            "independent_unit": "matched_initial_goal_case",
            "n_replans": len(rows),
        }
        for field in metric_fields:
            values = [float(row[field]) for row in rows]
            record[f"mean_{field}"] = mean(values)
            record[f"median_{field}"] = statistics.median(values)
        result.append(record)
    return result


def build_strata_summary(case_rows: list[dict]) -> list[dict]:
    metric_fields = (
        "r_selected",
        "r_final_record",
        "eta_model_final_record",
        "error_oscillation",
        "g_model_final",
        "g_env_final",
        "omega_final",
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in case_rows:
        key = (
            row["phase"],
            row["split"],
            row["shape"],
            row["arm"],
            row["objective_stage"],
        )
        grouped[key].append(row)
    result = []
    for key in sorted(grouped):
        phase, split, shape, arm, objective_stage = key
        rows = grouped[key]
        record = {
            "phase": phase,
            "split": split,
            "shape": shape,
            "arm": arm,
            "objective_stage": objective_stage,
            "independent_unit": "matched_initial_goal_case",
            "n_cases": len(rows),
            "n_replans": sum(int(row["n_replans"]) for row in rows),
            "numerical_tie_epsilon": NUMERICAL_TIE_EPSILON,
        }
        for field in metric_fields:
            values = [float(row[f"mean_{field}"]) for row in rows]
            low, high = bootstrap_mean_ci(values)
            record[f"mean_case_{field}"] = mean(values)
            record[f"median_case_{field}"] = statistics.median(values)
            record[f"bootstrap95_low_case_{field}"] = low
            record[f"bootstrap95_high_case_{field}"] = high
        result.append(record)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(input_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"refusing to overwrite nonempty output directory: {output_dir}"
        )
    rows = read_rows(input_path)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)

    all_path_rows = []
    selection_rows = []
    for key in sorted(grouped):
        path_rows, selection = analyze_group(grouped[key])
        all_path_rows.extend(path_rows)
        selection_rows.append(selection)

    gain_residuals = [
        abs(row["omega"] - (row["g_model"] - row["g_env"]))
        for row in all_path_rows
    ]
    selected_min_slack = min(row["selected_bound_slack"] for row in selection_rows)
    final_min_slack = min(row["final_bound_slack"] for row in selection_rows)
    tolerance = 1e-10
    validation = {
        "input_rows": len(rows),
        "replans": len(selection_rows),
        "expected_gd_steps": list(EXPECTED_GD_STEPS),
        "algebra": {
            "gain_identity_max_abs_residual": max(gain_residuals),
            "selected_bound_min_slack": selected_min_slack,
            "final_bound_min_slack": final_min_slack,
            "tolerance": tolerance,
            "passed": bool(
                max(gain_residuals) <= tolerance
                and selected_min_slack >= -tolerance
                and final_min_slack >= -tolerance
            ),
        },
    }
    if not validation["algebra"]["passed"]:
        raise ValueError(f"Algebra validation failed: {validation['algebra']}")

    case_rows = build_case_metrics(selection_rows)
    strata_rows = build_strata_summary(case_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "path_metrics.csv", all_path_rows)
    write_csv(output_dir / "selection_metrics.csv", selection_rows)
    write_csv(output_dir / "case_metrics.csv", case_rows)
    write_csv(output_dir / "strata_summary.csv", strata_rows)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.out)


if __name__ == "__main__":
    main()
