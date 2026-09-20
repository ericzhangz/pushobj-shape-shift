"""Aggregate contrast, replay, environment, and adaptation records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def metric_row(key: tuple, records: list[dict], epsilon: float) -> dict:
    arm, shape = key
    pred = np.asarray([row["delta_pred"] for row in records], dtype=np.float64)
    env = np.asarray(
        [row["delta_env_samever"] for row in records], dtype=np.float64
    )
    ref = np.asarray([row["delta_env_ref"] for row in records], dtype=np.float64)
    error = pred - env
    predicted_improvement = pred < 0.0
    false_improvement = predicted_improvement & (env > 0.0)
    pred_sign_thresholded = np.where(
        pred > epsilon, 1, np.where(pred < -epsilon, -1, 0)
    )
    env_sign_thresholded = np.where(
        env > epsilon, 1, np.where(env < -epsilon, -1, 0)
    )
    predicted_improvement_thresholded = pred < -epsilon
    false_improvement_thresholded = predicted_improvement_thresholded & (
        env > epsilon
    )
    rho = float(spearmanr(pred, env).statistic) if len(records) > 1 else math.nan
    return {
        "arm": arm,
        "shape": shape,
        "n_contrasts": len(records),
        "n_samples": len({row["sample_id"] for row in records}),
        "aggregation_unit": "contrast (repeated within episode; not independent samples)",
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": rho,
        "sign_accuracy": float(
            np.mean(np.sign(pred).astype(int) == np.sign(env).astype(int))
        ),
        "false_improvement_count": int(false_improvement.sum()),
        "false_improvement_rate_all": float(false_improvement.mean()),
        "predicted_improvement_count": int(predicted_improvement.sum()),
        "false_improvement_rate_given_predicted_improvement": float(
            false_improvement.sum() / predicted_improvement.sum()
        )
        if predicted_improvement.any()
        else math.nan,
        "numerical_tie_epsilon": epsilon,
        "thresholded_sign_accuracy_all": float(
            np.mean(pred_sign_thresholded == env_sign_thresholded)
        ),
        "thresholded_false_improvement_count": int(
            false_improvement_thresholded.sum()
        ),
        "thresholded_false_improvement_rate_all": float(
            false_improvement_thresholded.mean()
        ),
        "thresholded_predicted_improvement_count": int(
            predicted_improvement_thresholded.sum()
        ),
        "thresholded_false_improvement_rate_given_predicted_improvement": float(
            false_improvement_thresholded.sum()
            / predicted_improvement_thresholded.sum()
        )
        if predicted_improvement_thresholded.any()
        else math.nan,
        "mean_abs_delta_pred": float(np.mean(np.abs(pred))),
        "mean_abs_delta_env_samever": float(np.mean(np.abs(env))),
        "max_env_worsening_under_predicted_improvement": float(
            np.max(env[false_improvement])
        )
        if false_improvement.any()
        else 0.0,
        "encoder_coordinate_delta_mae": float(np.mean(np.abs(env - ref))),
    }


def load_completed_runs(runs: list[Path]) -> list[dict]:
    """Reject partial/duplicated runs before calculating any denominators."""
    loaded = []
    seen_paths = set()
    seen_ids = set()
    paired_initials = {}
    paired_samples = {}
    arm_splits = set()
    for run in runs:
        run = run.resolve()
        if run in seen_paths:
            raise ValueError(f"Duplicate run path: {run}")
        seen_paths.add(run)
        with (run / "run_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        budget_path = run / "oracle" / "run_budget_final.json"
        if not budget_path.exists():
            raise ValueError(f"Run is incomplete: {budget_path} is missing")
        with budget_path.open(encoding="utf-8") as handle:
            budget = json.load(handle)
        run_id = metadata["run_id"]
        if run_id in seen_ids:
            raise ValueError(f"Duplicate run_id: {run_id}")
        seen_ids.add(run_id)
        split = Path(metadata["eval_data_path"]).parent.name
        arm_split = (metadata["arm"], split)
        if arm_split in arm_splits:
            raise ValueError(f"Duplicate arm/split would duplicate episodes: {arm_split}")
        arm_splits.add(arm_split)
        data = {"metadata": metadata, "budget": budget, "split": split}
        for name, relative in {
            "contrasts": "oracle/contrast_pairs.jsonl",
            "branches": "oracle/branch_records.jsonl",
            "replay": "metrics/replay_errors.jsonl",
            "adaptations": "traces/adaptation_steps.jsonl",
            "samples": "oracle/query_budget.jsonl",
        }.items():
            rows = read_jsonl(run / relative)
            if not rows:
                raise ValueError(f"Missing records: {run / relative}")
            if any(row["run_id"] != run_id or row["arm"] != metadata["arm"] for row in rows):
                raise ValueError(f"Mixed run identity: {run / relative}")
            data[name] = rows
        samples = data["samples"]
        sample_ids = [row["sample_id"] for row in samples]
        if sorted(sample_ids) != list(range(metadata["n_evals"])):
            raise ValueError(f"Missing or duplicated completed samples: {run}")
        if budget["run_id"] != run_id or budget["n_samples"] != len(samples):
            raise ValueError(f"Final budget does not match samples: {run}")
        identities = [(row["sample_id"], row["shape"], row["seed"]) for row in samples]
        if split in paired_samples and identities != paired_samples[split]:
            raise ValueError(f"Unpaired sample identities across arms: {split}")
        paired_samples[split] = identities
        for sample in samples:
            sample_id = sample["sample_id"]
            with np.load(run / "real_evidence" / f"sample_{sample_id:03d}_initial.npz") as arrays:
                initial = {key: arrays[key].copy() for key in arrays.files}
            pair_key = (split, sample_id)
            if pair_key in paired_initials:
                previous = paired_initials[pair_key]
                if initial.keys() != previous.keys() or any(
                    not np.array_equal(initial[key], previous[key]) for key in initial
                ):
                    raise ValueError(f"Initial/goal evidence differs across arms: {pair_key}")
            paired_initials[pair_key] = initial
        anchor_keys = [(row["sample_id"], row["mpc_iter"]) for row in data["replay"]]
        if len(set(anchor_keys)) != len(anchor_keys) or not all(row["passed"] for row in data["replay"]):
            raise ValueError(f"Duplicated or failed replay anchors: {run}")
        expected = {(s, m, g) for s, m in anchor_keys for g in metadata["oracle_gd_steps"]}
        actual = [(row["sample_id"], row["mpc_iter"], row["gd_iter"]) for row in data["contrasts"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"Missing or duplicated fixed-step contrasts: {run}")
        branch_keys = [(row["sample_id"], row["mpc_iter"], row["gd_iter"], row["candidate"]) for row in data["branches"]]
        expected_branches = {(*key, candidate) for key in expected for candidate in ("u_before", "u_after")}
        if len(branch_keys) != len(expected_branches) or set(branch_keys) != expected_branches:
            raise ValueError(f"Missing or duplicated candidate branches: {run}")
        if any(not math.isfinite(row[key]) for row in data["contrasts"] for key in ("delta_pred", "delta_env_samever", "delta_env_ref")):
            raise ValueError(f"Nonfinite contrast value: {run}")
        loaded.append(data)
    return loaded


def adaptation_summary(rows: list[dict]) -> dict:
    updates = [row for row in rows if row["official_pre_update_step_losses"]]
    evaluated = [row for row in updates if row.get("support_loss_eval_before") is not None and row.get("support_loss_eval_after") is not None]
    improved = sum(row["support_loss_eval_after"] < row["support_loss_eval_before"] for row in evaluated)
    return {
        "finetune_calls": len(rows),
        "zero_update_calls": len(rows) - len(updates),
        "adaptation_events": len(updates),
        "optimizer_update_steps": sum(len(row["official_pre_update_step_losses"]) for row in updates),
        "support_loss_evaluated_update_events": len(evaluated),
        "support_loss_improved_events": improved,
        "support_loss_improved_rate": improved / len(evaluated) if evaluated else None,
    }


def corrected_budget(data: dict) -> dict:
    """The official final evaluator executes padded actions before masking metrics."""
    budget = dict(data["budget"])
    actual_steps = len(data["samples"]) * max(row["returned_model_actions"] for row in data["samples"]) * data["metadata"]["frameskip"]
    budget["recorded_official_final_eval_environment_steps"] = budget["official_final_eval_environment_steps"]
    budget["official_final_eval_environment_steps"] = actual_steps
    budget["official_total_evaluation_environment_steps"] = budget["official_mpc_cumulative_replay_environment_steps"] + actual_steps
    budget["peak_vram_bytes"] = max(budget["peak_vram_bytes"], *(row["peak_vram_bytes"] for row in data["samples"]))
    budget["model_forward_count_scope"] = "VWorldModel predict and encode_obs API calls; excludes direct reference-module calls"
    budget["wall_clock_scope"] = "after workspace construction; excludes model loading and target preparation"
    return budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    loaded = load_completed_runs(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)

    contrasts = []
    branches = []
    replay = []
    adaptations = []
    for data in loaded:
        contrasts.extend(data["contrasts"])
        branches.extend(data["branches"])
        replay.extend(data["replay"])
        adaptations.extend(data["adaptations"])

    grouped = defaultdict(list)
    for record in contrasts:
        grouped[(record["arm"], record["shape"])].append(record)
    metric_rows = [
        metric_row(key, rows, args.epsilon) for key, rows in sorted(grouped.items())
    ]
    write_csv(args.output / "contrast_metrics.csv", metric_rows)
    write_csv(args.output / "contrast_pairs.csv", contrasts)
    write_csv(args.output / "replay_errors.csv", replay)
    write_csv(args.output / "environment_metrics.csv", branches)
    write_csv(args.output / "adaptation_steps.csv", adaptations)
    budgets = [corrected_budget(data) for data in loaded]
    write_csv(args.output / "run_budgets.csv", budgets)
    episode_rows = []
    deployed_rows = []
    for data in loaded:
        for sample in data["samples"]:
            rows = [row for row in data["contrasts"] if row["sample_id"] == sample["sample_id"]]
            episode_rows.append({**metric_row((sample["arm"], sample["shape"]), rows, args.epsilon), "sample_id": sample["sample_id"], "seed": sample["seed"]})
        n_success = sum(math.isfinite(row["action_len"][0]) for row in data["samples"])
        deployed_rows.append({"run_id": data["metadata"]["run_id"], "arm": data["metadata"]["arm"], "split": data["split"], "n_episodes": len(data["samples"]), "n_success": n_success, "success_rate": n_success / len(data["samples"]), "success_source": "official MPC first-success action_len (finite means success)"})
    write_csv(args.output / "episode_contrast_metrics.csv", episode_rows)
    write_csv(args.output / "deployed_environment_metrics.csv", deployed_rows)

    if contrasts:
        fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
        for key, rows in sorted(grouped.items()):
            pred = np.asarray([row["delta_pred"] for row in rows])
            env = np.asarray([row["delta_env_samever"] for row in rows])
            ax.scatter(pred, env, s=22, alpha=0.75, label=f"{key[0]} / {key[1]}")
        all_values = np.asarray(
            [
                value
                for row in contrasts
                for value in (row["delta_pred"], row["delta_env_samever"])
            ]
        )
        lower, upper = float(all_values.min()), float(all_values.max())
        padding = max((upper - lower) * 0.05, 1e-8)
        ax.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            linewidth=1,
            color="0.45",
            label="ideal contrast",
        )
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.axvline(0.0, color="0.75", linewidth=0.8)
        ax.set_xlabel("Predicted delta cost")
        ax.set_ylabel("Environment same-version delta cost")
        ax.set_title("Native GD update contrast")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output / "predicted_vs_environment_contrast.png")
        plt.close(fig)

    replay_passed = sum(bool(row.get("passed")) for row in replay)
    summary = {
        "runs": [str(run.resolve()) for run in args.runs],
        "n_contrasts": len(contrasts),
        "n_oracle_branches": len(branches),
        "zero_replay_passed": replay_passed,
        "zero_replay_total": len(replay),
        **adaptation_summary(adaptations),
        "adaptation_by_run": [{"run_id": data["metadata"]["run_id"], "arm": data["metadata"]["arm"], **adaptation_summary(data["adaptations"])} for data in loaded],
        "deployed_environment_metrics": deployed_rows,
        "environment_metrics_csv_scope": "diagnostic oracle branches, not deployed episode success",
        "episode_contrast_metrics": episode_rows,
        "run_budgets": budgets,
        "metrics": metric_rows,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
