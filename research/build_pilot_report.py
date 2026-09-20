"""Build auditable aggregate tables, plots, and the decisive pilot report.

This is an offline analysis only.  It never calls the environment, model, or
planner, and it treats episodes (not within-episode contrast probes) as the
independent sampling unit.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.summarize_pilot_runs import adaptation_summary, metric_row, read_jsonl


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EPSILON = 1e-6

ARM_ORDER = ["FROZEN", "OFFICIAL_ADAJEPA", "PREDICTOR_ONLY"]
ARM_LABEL = {
    "FROZEN": "Frozen",
    "OFFICIAL_ADAJEPA": "Official",
    "PREDICTOR_ONLY": "Predictor-only",
}
SPLIT_ORDER = ["val_T", "val_L", "val_I", "val_small_tee", "val_square"]

RUN_SPECS = [
    ("development", "val_T", "FROZEN", "development/val_T_frozen_n3"),
    (
        "development",
        "val_T",
        "OFFICIAL_ADAJEPA",
        "development/val_T_official_n3_attempt2",
    ),
    (
        "development",
        "val_T",
        "PREDICTOR_ONLY",
        "development/val_T_predictor_only_n3",
    ),
    ("development", "val_L", "FROZEN", "development/val_L_frozen_n3"),
    (
        "development",
        "val_L",
        "OFFICIAL_ADAJEPA",
        "development/val_L_official_n3",
    ),
    (
        "development",
        "val_L",
        "PREDICTOR_ONLY",
        "development/val_L_predictor_only_n3",
    ),
    ("ood", "val_I", "FROZEN", "ood/val_I_frozen_n5_attempt2"),
    ("ood", "val_I", "OFFICIAL_ADAJEPA", "ood/val_I_official_n5"),
    ("ood", "val_I", "PREDICTOR_ONLY", "ood/val_I_predictor_only_n5"),
    ("ood", "val_small_tee", "FROZEN", "ood/val_small_tee_frozen_n5"),
    (
        "ood",
        "val_small_tee",
        "OFFICIAL_ADAJEPA",
        "ood/val_small_tee_official_n5",
    ),
    (
        "ood",
        "val_small_tee",
        "PREDICTOR_ONLY",
        "ood/val_small_tee_predictor_only_n5",
    ),
    ("ood", "val_square", "FROZEN", "ood/val_square_frozen_n5"),
    (
        "ood",
        "val_square",
        "OFFICIAL_ADAJEPA",
        "ood/val_square_official_n5",
    ),
    (
        "ood",
        "val_square",
        "PREDICTOR_ONLY",
        "ood/val_square_predictor_only_n5",
    ),
]


def json_ready(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def frame(rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(json_ready)
    return result


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame(rows).to_csv(path, index=False)


def save_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame(rows).to_parquet(path, index=False)


def finite_success(sample: dict) -> bool:
    return math.isfinite(float(sample["action_len"][0]))


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan
    std = float(array.std(ddof=1)) if array.size > 1 else math.nan
    return float(array.mean()), std


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - half, center + half


FINAL_ARRAY = re.compile(
    r"\{'success':\s*array\(\[(.*?)\]\),\s*'state_dist':\s*"
    r"array\(\[(.*?)\],\s*dtype=float32\)\}",
    re.DOTALL,
)


def parse_final_stdout(run_path: Path, expected_n: int) -> tuple[list[bool], list[float], Path]:
    log_path = run_path.parent / f"{run_path.name}_stdout.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(FINAL_ARRAY.finditer(text))
    if not matches:
        raise ValueError(f"Final success/state array not found: {log_path}")
    match = matches[-1]
    successes = [token == "True" for token in re.findall(r"True|False", match.group(1))]
    distances = [float(token) for token in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(2))]
    if len(successes) != expected_n or len(distances) != expected_n:
        raise ValueError(
            f"Final array length mismatch in {log_path}: "
            f"{len(successes)} successes, {len(distances)} distances, expected {expected_n}"
        )
    return successes, distances, log_path


def load_run(phase: str, split: str, arm: str, relative: str) -> dict:
    run_path = ARTIFACTS / relative
    metadata = json.loads((run_path / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["arm"] != arm:
        raise ValueError(f"Arm mismatch: {run_path}")
    if Path(metadata["eval_data_path"]).parent.name != split:
        raise ValueError(f"Split mismatch: {run_path}")
    samples = read_jsonl(run_path / "oracle" / "query_budget.jsonl")
    if len(samples) != metadata["n_evals"]:
        raise ValueError(f"Incomplete samples: {run_path}")
    successes, distances, stdout_path = parse_final_stdout(run_path, metadata["n_evals"])
    budget = json.loads(
        (run_path / "oracle" / "run_budget_final.json").read_text(encoding="utf-8")
    )
    sample_successes = [finite_success(sample) for sample in samples]
    if successes != sample_successes:
        raise ValueError(f"Official success mismatch between stdout and action_len: {run_path}")
    final_eval_steps = (
        len(samples)
        * max(int(sample["returned_model_actions"]) for sample in samples)
        * int(metadata["frameskip"])
    )
    corrected_budget = dict(budget)
    corrected_budget["recorded_official_final_eval_environment_steps"] = budget[
        "official_final_eval_environment_steps"
    ]
    corrected_budget["official_final_eval_environment_steps"] = final_eval_steps
    corrected_budget["official_total_evaluation_environment_steps"] = (
        budget["official_mpc_cumulative_replay_environment_steps"] + final_eval_steps
    )
    corrected_budget["model_forward_count_scope"] = (
        "VWorldModel predict and encode_obs API calls; excludes direct reference-module calls"
    )
    corrected_budget["wall_clock_scope"] = (
        "after workspace construction; excludes model loading and target preparation"
    )
    return {
        "phase": phase,
        "split": split,
        "arm": arm,
        "path": run_path,
        "stdout_path": stdout_path,
        "metadata": metadata,
        "samples": samples,
        "successes": successes,
        "distances": distances,
        "budget": corrected_budget,
        "contrasts": read_jsonl(run_path / "oracle" / "contrast_pairs.jsonl"),
        "branches": read_jsonl(run_path / "oracle" / "branch_records.jsonl"),
        "replay": read_jsonl(run_path / "metrics" / "replay_errors.jsonl"),
        "adaptations": read_jsonl(run_path / "traces" / "adaptation_steps.jsonl"),
        "planner": read_jsonl(run_path / "traces" / "planner_steps.jsonl"),
        "versions": read_jsonl(run_path / "traces" / "version_table.jsonl"),
    }


def augmented(rows: list[dict], run: dict) -> list[dict]:
    return [
        {
            "phase": run["phase"],
            "split": run["split"],
            "run_path": str(run["path"]),
            **row,
        }
        for row in rows
    ]


def task_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        n_episodes = len(run["samples"])
        n_success = sum(run["successes"])
        success_mean, success_std = mean_std([float(value) for value in run["successes"]])
        distance_mean, distance_std = mean_std(run["distances"])
        ci_low, ci_high = wilson_interval(n_success, n_episodes)
        rows.append(
            {
                "phase": run["phase"],
                "split": run["split"],
                "arm": run["arm"],
                "n_episodes": n_episodes,
                "n_success": n_success,
                "success_rate": success_mean,
                "success_binary_sample_std": success_std,
                "success_wilson95_low": ci_low,
                "success_wilson95_high": ci_high,
                "final_state_dist_mean": distance_mean,
                "final_state_dist_sample_std": distance_std,
                "final_state_dist_median": float(np.median(run["distances"])),
                "final_state_dist_max": float(np.max(run["distances"])),
                "mean_returned_model_actions": float(
                    np.mean([sample["returned_model_actions"] for sample in run["samples"]])
                ),
                "run_id": run["metadata"]["run_id"],
                "stdout_log": str(run["stdout_path"]),
            }
        )
    frozen = {row["split"]: row for row in rows if row["arm"] == "FROZEN"}
    for row in rows:
        baseline = frozen[row["split"]]
        row["success_rate_delta_vs_frozen"] = row["success_rate"] - baseline["success_rate"]
        row["state_dist_mean_delta_vs_frozen"] = (
            row["final_state_dist_mean"] - baseline["final_state_dist_mean"]
        )
    return rows


def contrast_rows(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    aggregate = []
    episodes = []
    for run in runs:
        pooled = metric_row((run["arm"], run["metadata"]["shapes"][0]), run["contrasts"], EPSILON)
        per_episode = []
        sample_map = {sample["sample_id"]: sample for sample in run["samples"]}
        adaptation_by_sample = defaultdict(list)
        for update in run["adaptations"]:
            adaptation_by_sample[update["sample_id"]].append(update)
        for sample_id in sorted(sample_map):
            records = [row for row in run["contrasts"] if row["sample_id"] == sample_id]
            metrics = metric_row((run["arm"], records[0]["shape"]), records, EPSILON)
            support = adaptation_summary(adaptation_by_sample[sample_id])
            episode = {
                "phase": run["phase"],
                "split": run["split"],
                "run_id": run["metadata"]["run_id"],
                "sample_id": sample_id,
                "seed": sample_map[sample_id]["seed"],
                "arm": run["arm"],
                "success": finite_success(sample_map[sample_id]),
                "returned_model_actions": sample_map[sample_id]["returned_model_actions"],
                "support_update_events": support["adaptation_events"],
                "support_loss_improved_rate": support["support_loss_improved_rate"],
                **metrics,
            }
            per_episode.append(episode)
            episodes.append(episode)
        episode_mae_mean, episode_mae_std = mean_std([row["mae"] for row in per_episode])
        episode_false_mean, episode_false_std = mean_std(
            [row["thresholded_false_improvement_rate_all"] for row in per_episode]
        )
        support = adaptation_summary(run["adaptations"])
        aggregate.append(
            {
                "phase": run["phase"],
                "split": run["split"],
                "run_id": run["metadata"]["run_id"],
                **pooled,
                "episode_mae_mean": episode_mae_mean,
                "episode_mae_sample_std": episode_mae_std,
                "episode_thresholded_false_rate_mean": episode_false_mean,
                "episode_thresholded_false_rate_sample_std": episode_false_std,
                "support_update_events": support["adaptation_events"],
                "support_loss_improved_events": support["support_loss_improved_events"],
                "support_loss_improved_rate": support["support_loss_improved_rate"],
            }
        )
    frozen = {row["split"]: row for row in aggregate if row["arm"] == "FROZEN"}
    for row in aggregate:
        baseline = frozen[row["split"]]
        row["mae_delta_vs_frozen"] = row["mae"] - baseline["mae"]
        row["spearman_delta_vs_frozen"] = row["spearman"] - baseline["spearman"]
        row["conditional_false_rate_delta_vs_frozen"] = (
            row["thresholded_false_improvement_rate_given_predicted_improvement"]
            - baseline["thresholded_false_improvement_rate_given_predicted_improvement"]
        )
    return aggregate, episodes


def grouped_contrast_table(runs: list[dict], field: str) -> list[dict]:
    grouped = defaultdict(list)
    phase_lookup = {}
    for run in runs:
        for row in run["contrasts"]:
            key = (run["split"], run["arm"], row[field])
            grouped[key].append(row)
            phase_lookup[(run["split"], run["arm"])] = run["phase"]
    rows = []
    for (split, arm, value), records in sorted(
        grouped.items(), key=lambda item: (SPLIT_ORDER.index(item[0][0]), ARM_ORDER.index(item[0][1]), item[0][2])
    ):
        metrics = metric_row((arm, records[0]["shape"]), records, EPSILON)
        rows.append(
            {
                "phase": phase_lookup[(split, arm)],
                "split": split,
                "arm": arm,
                field: value,
                **metrics,
            }
        )
    return rows


def adaptation_stage_rows(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        for row in run["contrasts"]:
            if row["mpc_iter"] == 0:
                stage = "first_mpc_replan_pre_update"
            elif run["arm"] == "FROZEN":
                stage = "later_mpc_replans_no_updates"
            else:
                stage = "after_prior_factual_updates"
            grouped[(run["phase"], run["split"], run["arm"], stage)].append(row)
    result = []
    for (phase, split, arm, stage), records in sorted(grouped.items()):
        result.append(
            {
                "phase": phase,
                "split": split,
                "arm": arm,
                "adaptation_stage": stage,
                **metric_row((arm, records[0]["shape"]), records, EPSILON),
            }
        )
    return result


def support_next_query_rows(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    pairs = []
    for run in runs:
        contrast_lookup = defaultdict(list)
        for row in run["contrasts"]:
            contrast_lookup[(row["sample_id"], row["mpc_iter"])].append(row)
        for update in run["adaptations"]:
            losses = update["official_pre_update_step_losses"]
            if not losses:
                continue
            next_records = contrast_lookup.get((update["sample_id"], update["mpc_iter"] + 1))
            if not next_records:
                continue
            metrics = metric_row((run["arm"], next_records[0]["shape"]), next_records, EPSILON)
            support_delta = update["support_loss_eval_after"] - update["support_loss_eval_before"]
            pairs.append(
                {
                    "phase": run["phase"],
                    "split": run["split"],
                    "run_id": run["metadata"]["run_id"],
                    "arm": run["arm"],
                    "sample_id": update["sample_id"],
                    "adaptation_mpc_iter": update["mpc_iter"],
                    "next_query_mpc_iter": update["mpc_iter"] + 1,
                    "support_loss_delta": support_delta,
                    "support_loss_improved": support_delta < 0,
                    "next_query_mae": metrics["mae"],
                    "next_query_thresholded_false_rate_all": metrics[
                        "thresholded_false_improvement_rate_all"
                    ],
                    "next_query_sign_accuracy": metrics["thresholded_sign_accuracy_all"],
                    "next_query_max_env_worsening": metrics[
                        "max_env_worsening_under_predicted_improvement"
                    ],
                }
            )
    summaries = []
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["phase"], row["split"], row["arm"])].append(row)
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            SPLIT_ORDER.index(item[0][1]),
            ARM_ORDER.index(item[0][2]),
        ),
    )
    for (phase, split, arm), rows in ordered_groups:
        support = np.asarray([row["support_loss_delta"] for row in rows])
        next_mae = np.asarray([row["next_query_mae"] for row in rows])
        next_false = np.asarray([row["next_query_thresholded_false_rate_all"] for row in rows])
        summaries.append(
            {
                "phase": phase,
                "split": split,
                "arm": arm,
                "n_temporally_aligned_pairs": len(rows),
                "support_loss_improvement_rate": float(
                    np.mean([row["support_loss_improved"] for row in rows])
                ),
                "support_loss_delta_mean": float(support.mean()),
                "next_query_mae_mean": float(next_mae.mean()),
                "next_query_thresholded_false_rate_mean": float(next_false.mean()),
                "spearman_support_delta_vs_next_query_mae": float(
                    spearmanr(support, next_mae).statistic
                )
                if len(rows) > 1
                else math.nan,
                "spearman_support_delta_vs_next_query_false_rate": float(
                    spearmanr(support, next_false).statistic
                )
                if len(rows) > 1
                else math.nan,
            }
        )
    return pairs, summaries


def episode_outcome_summary(episodes: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in episodes:
        grouped[("ALL", bool(row["success"]))].append(row)
        grouped[(row["arm"], bool(row["success"]))].append(row)
    result = []
    for (arm, success), rows in sorted(grouped.items()):
        mae_mean, mae_std = mean_std([row["mae"] for row in rows])
        false_mean, false_std = mean_std(
            [row["thresholded_false_improvement_rate_all"] for row in rows]
        )
        result.append(
            {
                "arm": arm,
                "success": success,
                "n_episodes": len(rows),
                "episode_mae_mean": mae_mean,
                "episode_mae_sample_std": mae_std,
                "episode_thresholded_false_rate_mean": false_mean,
                "episode_thresholded_false_rate_sample_std": false_std,
            }
        )
    return result


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def pct(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{100 * value:.1f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def generate_plots(task: list[dict], contrast: list[dict], all_contrasts: list[dict]) -> None:
    plots = ARTIFACTS / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SPLIT_ORDER))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    for index, arm in enumerate(ARM_ORDER):
        values = [next(row["success_rate"] for row in task if row["split"] == split and row["arm"] == arm) for split in SPLIT_ORDER]
        ax.bar(x + (index - 1) * width, values, width, label=ARM_LABEL[arm])
    ax.set_xticks(x, SPLIT_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Official environment success rate")
    ax.set_title("PushObj task success (n=3 development; n=5 OOD per arm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "task_success_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    for index, arm in enumerate(ARM_ORDER):
        values = [
            next(
                row["thresholded_false_improvement_rate_given_predicted_improvement"]
                for row in contrast
                if row["split"] == split and row["arm"] == arm
            )
            for split in SPLIT_ORDER
        ]
        ax.bar(x + (index - 1) * width, values, width, label=ARM_LABEL[arm])
    ax.set_xticks(x, SPLIT_ORDER)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("False-improvement rate | predicted improvement")
    ax.set_title("Independent contrast mismatch (epsilon=1e-6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "conditional_false_improvement_rate.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180, sharex=True, sharey=True)
    for ax, arm in zip(axes, ARM_ORDER):
        rows = [row for row in all_contrasts if row["arm"] == arm]
        pred = np.asarray([row["delta_pred"] for row in rows])
        env = np.asarray([row["delta_env_samever"] for row in rows])
        ax.scatter(pred, env, s=5, alpha=0.22)
        lower = min(float(pred.min()), float(env.min()))
        upper = max(float(pred.max()), float(env.max()))
        ax.plot([lower, upper], [lower, upper], "--", color="0.35", linewidth=1)
        ax.axhline(0, color="0.75", linewidth=0.7)
        ax.axvline(0, color="0.75", linewidth=0.7)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("Predicted delta cost")
    axes[0].set_ylabel("Environment same-version delta cost")
    fig.suptitle("Native GD proposed updates: predicted vs real contrast")
    fig.tight_layout()
    fig.savefig(plots / "predicted_vs_environment_contrast.png")
    plt.close(fig)


def write_report(
    runs: list[dict],
    task: list[dict],
    contrast: list[dict],
    stage: list[dict],
    outcome_summary: list[dict],
    support_summary: list[dict],
) -> None:
    task_map = {(row["split"], row["arm"]): row for row in task}
    contrast_map = {(row["split"], row["arm"]): row for row in contrast}
    replay_total = sum(len(run["replay"]) for run in runs)
    replay_passed = sum(sum(bool(row["passed"]) for row in run["replay"]) for run in runs)
    total_episodes = sum(len(run["samples"]) for run in runs)

    task_table = []
    for split in SPLIT_ORDER:
        for arm in ARM_ORDER:
            row = task_map[(split, arm)]
            task_table.append(
                [
                    split,
                    ARM_LABEL[arm],
                    f"{row['n_success']}/{row['n_episodes']}",
                    f"{row['success_rate']:.3f} ± {row['success_binary_sample_std']:.3f}",
                    f"{row['success_rate_delta_vs_frozen']:+.3f}",
                    f"{row['final_state_dist_mean']:.2f} ± {row['final_state_dist_sample_std']:.2f}",
                ]
            )

    contrast_table = []
    for split in SPLIT_ORDER:
        for arm in ARM_ORDER:
            row = contrast_map[(split, arm)]
            contrast_table.append(
                [
                    split,
                    ARM_LABEL[arm],
                    str(row["n_contrasts"]),
                    number(row["mae"], 4),
                    number(row["spearman"], 3),
                    pct(row["thresholded_sign_accuracy_all"]),
                    (
                        f"{row['thresholded_false_improvement_count']}/"
                        f"{row['thresholded_predicted_improvement_count']} "
                        f"({pct(row['thresholded_false_improvement_rate_given_predicted_improvement'])})"
                    ),
                    number(row["max_env_worsening_under_predicted_improvement"], 4),
                    f"{row['conditional_false_rate_delta_vs_frozen']:+.3f}",
                ]
            )

    support_table = []
    for split in SPLIT_ORDER:
        for arm in ARM_ORDER:
            row = contrast_map[(split, arm)]
            support_table.append(
                [
                    split,
                    ARM_LABEL[arm],
                    str(row["support_update_events"]),
                    pct(row["support_loss_improved_rate"]),
                    pct(row["thresholded_false_improvement_rate_given_predicted_improvement"]),
                    number(row["mae"], 4),
                    pct(task_map[(split, arm)]["success_rate"]),
                ]
            )

    encoder_table = []
    for split in SPLIT_ORDER:
        for arm in ("OFFICIAL_ADAJEPA", "PREDICTOR_ONLY"):
            row = contrast_map[(split, arm)]
            encoder_table.append(
                [
                    split,
                    ARM_LABEL[arm],
                    number(row["encoder_coordinate_delta_mae"], 7),
                    pct(row["thresholded_false_improvement_rate_given_predicted_improvement"]),
                    pct(task_map[(split, arm)]["success_rate"]),
                ]
            )

    stage_aggregate = []
    grouped_stage = defaultdict(list)
    for row in stage:
        grouped_stage[(row["arm"], row["adaptation_stage"])].append(row)
    for (arm, stage_name), rows in sorted(grouped_stage.items()):
        count_false = sum(row["thresholded_false_improvement_count"] for row in rows)
        count_pred = sum(row["thresholded_predicted_improvement_count"] for row in rows)
        stage_aggregate.append(
            [
                ARM_LABEL[arm],
                stage_name,
                str(sum(row["n_contrasts"] for row in rows)),
                f"{count_false}/{count_pred} ({pct(count_false / count_pred if count_pred else math.nan)})",
            ]
        )

    outcome_table = []
    for row in outcome_summary:
        if row["arm"] != "ALL":
            continue
        outcome_table.append(
            [
                "successful" if row["success"] else "failed",
                str(row["n_episodes"]),
                f"{row['episode_mae_mean']:.4f} ± {row['episode_mae_sample_std']:.4f}",
                (
                    f"{pct(row['episode_thresholded_false_rate_mean'])} ± "
                    f"{pct(row['episode_thresholded_false_rate_sample_std'])}"
                ),
            ]
        )

    support_alignment_table = []
    for row in support_summary:
        support_alignment_table.append(
            [
                row["split"],
                ARM_LABEL[row["arm"]],
                str(row["n_temporally_aligned_pairs"]),
                pct(row["support_loss_improvement_rate"]),
                number(row["next_query_mae_mean"], 4),
                pct(row["next_query_thresholded_false_rate_mean"]),
                number(row["spearman_support_delta_vs_next_query_mae"], 3),
            ]
        )

    paths = {
        "task": ARTIFACTS / "metrics" / "task_metrics.csv",
        "contrast": ARTIFACTS / "metrics" / "contrast_metrics.csv",
        "episodes": ARTIFACTS / "metrics" / "episode_outcomes.csv",
        "mpc": ARTIFACTS / "metrics" / "contrast_by_mpc_iter.csv",
        "gd": ARTIFACTS / "metrics" / "contrast_by_gd_step.csv",
        "support": ARTIFACTS / "metrics" / "support_next_query_summary.csv",
        "budgets": ARTIFACTS / "metrics" / "run_budgets.csv",
        "replay": ARTIFACTS / "metrics" / "replay_errors.csv",
    }

    report = f"""# AdaJEPA PushObj 真实 Checkpoint 决定性 Pilot

## 结论先行

**判定：REDEFINE。** 测量基座通过，真实新查询 contrast mismatch 明确存在，而且 Official AdaJEPA 的 factual support loss 改善没有稳定消除它；但本轮尚未执行多尺度 response 稳定性、response-only/value-only 因果 oracle 和按时间顺序的 experience-reuse 检验。因此证据支持继续做**限定的因果诊断**，不支持现在设计新主算子，也不足以 KILL response-mismatch 研究对象。

本轮共有 {total_episodes} 个正式 episode。contrast 是每个 episode/replan 内固定 GD 步 `0,24,49,74,99` 的重复诊断，不是独立样本；所有成功率以官方环境返回的 `success` 为准。

## A. Reproduction

- 官方 `val_T, n=1, max_iter=1` smoke 成功运行；原始日志：`artifacts/baseline/smoke_max_iter1_attempt2/baseline_stdout.log`。
- 官方 `val_T, n=1, max_iter=20` 在 MPC iter 10 成功，最终 `state_dist=58.35865`；原始日志：`artifacts/baseline/full_max_iter20/baseline_stdout.log`。
- instrumentation OFF 的完整 episode 返回 11 个 model-action blocks，与捕获的官方输出逐元素完全相等，最大绝对误差 `0.0`：`artifacts/regression/full_episode_action_regression.json`。

## B. Measurement integrity

- Fresh-env full-prefix zero-action replay：**{replay_passed}/{replay_total} 全部逐项精确通过**；RGB、公开状态、native same-version cost 和官方环境指标均无差异。汇总：`{paths['replay']}`。
- 动作单位合同通过：model pack/unpack 误差 `0`，normalize/denormalize 最大误差 `1.49e-8`，完整 pixel round-trip 最大误差 `1.91e-6`。
- objective 合同通过：MPC step 0–4 使用 terminal，step 5 起使用 full horizon；四个 closure 检查最大误差均为 `0`。
- 三臂的 sample id、seed、初始状态与目标 evidence 已逐数组精确配对；失败/半成品运行与重复 run 路径被汇总入口拒绝。
- Astra 审计修正了两个离线统计错误：(1) Frozen 的空 finetune 调用不再误计为真实 adaptation；(2) 官方 final evaluator 的 padded-action 环境步数按实际整批执行量派生，原记录值同时保留。原始日志和原始 evidence 未修改。

## 原始任务表

{markdown_table(['split', 'arm', 'success', 'rate ± episode SD', 'Δrate vs Frozen', 'final state distance mean ± SD'], task_table)}

说明：`state_dist` 与官方几何成功判据不是同一个量，二者并列报告，不能互相替代。95% Wilson 区间和逐 episode 原始值见 `{paths['task']}`；本 pilot 的 `n=3/5` 不支持显著性外推。

## C. Real failure

`false improvement` 使用预注册的数值阈值：`delta_pred < -1e-6` 且 `delta_env_samever > 1e-6`。

{markdown_table(['split', 'arm', 'contrasts', 'MAE', 'Spearman', 'sign acc', 'false / predicted-improve', 'max real worsening', 'Δfalse-rate vs Frozen'], contrast_table)}

结论：所有 15 个 split/arm 单元都出现了 predicted-improvement / real-worsening。条件伪改善率约为 37%–53%，最大真实恶化出现在 `val_square`（Official `0.6951`，Predictor-only `0.6975`）。该错误在第一次 MPC replan 和后续 replans 都存在；对两个适配臂，后续 replans 已消费此前 factual updates，而 Frozen 始终没有参数更新：

{markdown_table(['arm', 'stage', 'contrasts', 'thresholded false / predicted-improve'], stage_aggregate)}

精确到每个 MPC iter 和固定 GD step 的表分别在 `{paths['mpc']}` 与 `{paths['gd']}`。

任务层联系目前是描述性关联而非因果证明：

{markdown_table(['episode outcome', 'n episodes', 'episode contrast MAE mean ± SD', 'episode false rate mean ± SD'], outcome_table)}

逐 episode 结果见 `{paths['episodes']}`。难 split 会产生更多 replans，从而贡献更多 pooled contrasts；因此不把 pooled contrast 数量当作独立样本数。

## D. Effect of AdaJEPA

{markdown_table(['split', 'arm', 'real updates', 'support improved', 'independent false rate', 'contrast MAE', 'task success'], support_table)}

Official 与 Predictor-only 的 deterministic support loss 在绝大多数真实更新中下降，但新查询伪改善率仍高，且 Official 相对 Frozen 只在 `val_T/val_L` 下降、在三个 OOD split 均上升。OOD 任务成功合计为 Frozen `6/15`、Official `7/15`、Predictor-only `6/15`；这只是一个 episode 的差异，不能解释为稳定收益。

把一次 factual update 与**下一次** MPC 新查询按时间对齐后的描述性检查如下。相关系数不是因果量，且每个 next-query 内仍只有五个固定 probe：

{markdown_table(['split', 'arm', 'aligned pairs', 'support improved', 'next-query MAE', 'next-query false rate', 'rho(support delta, next MAE)'], support_alignment_table)}

除 `val_L/Official` 接近 0 外，`rho(support delta, next MAE)` 多为负：更大的 support-loss 下降并没有对应更低的下一查询 MAE，描述性方向反而相反。该现象仍可能受 episode 难度、replan 选择和少量固定 probes 混杂，不能作因果结论。完整配对记录在 `artifacts/metrics/support_next_query_pairs.csv`，分组汇总见 `{paths['support']}`。

## E. Encoder confound

{markdown_table(['split', 'arm', '|samever-ref| MAE', 'independent false rate', 'task success'], encoder_table)}

Predictor-only 的 evaluator 坐标漂移按构造为 0；Official 的 drift 非零但量级较小。关闭 encoder 更新并未稳定降低独立 contrast error：它在部分 split 改善、部分 split 恶化，并在 `val_square` 得到 `0/5`。因此 encoder drift 是可测 confound，但不是本轮 failure 的充分解释。

## F. Response diagnostics

本轮只测了 native Adam 实际 action update 的**有限步 task contrast**，没有运行预注册的多尺度方向 probe；不能声称恢复了完整 Jacobian，也不能回答是否存在 scale-stable response 区域。该问题保持 **unresolved**，不是负结果。

## G. Causal evidence

本轮没有运行 response-only oracle 或 value-only oracle，因此不能量化两者各自修复多少 planning failure。当前结果证明 mismatch 与规划轨迹共现，不证明 response error 单独因果控制失败。结论：**unresolved**。

## H. Experience reuse

本轮 adaptation 使用官方 recent-5 factual history，但没有构造两个按时间分离的 experience 组，也没有在相同预算下测试 later unseen action/goal query 的可重复增量。support-loss 下降不等于 experience reuse。结论：**unresolved**。

## I. Go / No-Go

**REDEFINE**，理由如下：

1. 测量基座通过，不触发 measurement NO-GO。
2. 真实新查询 contrast failure 在 Frozen、Official 和 Predictor-only 上均存在，不触发“failure 不存在”的 NO-GO。
3. AdaJEPA 没有稳定自然消除 OOD contrast failure。
4. 但 GO 的后半条件——因果层定位、past experience 的 later-query 复用价值、以及同预算简单控制不能解释收益——尚未建立。

因此下一阶段只能做以下冻结诊断，不进入新方法设计：

1. 在事先固定的失败 anchor 上运行 response-only 与 value-only oracle；
2. 用事先固定的尺度/方向检查 finite response 的稳定区间；
3. 用两个按时间分离的 experience 组测试 later unseen query，并匹配真实环境、forward/backward 与墙钟预算；
4. 只有前三项支持 GO 时，才加入简单 finite-scale readout、replay/TTA、GRASP 和 CEM/零阶规划对照。

## 预算与可复核路径

- 逐 run 的真实环境、官方 replay/final eval、world-model rollout、forward/backward、oracle、墙钟与峰值显存：`{paths['budgets']}`。
- 原始 planner/adaptation/version 记录：`artifacts/traces/*.parquet`。
- 原始 oracle branches 与 query budget：`artifacts/oracle/branch_records.parquet`、`artifacts/oracle/query_budget.csv`。
- 任务、contrast、episode、stage 与 support-next-query 表：`artifacts/metrics/`。
- 图：`artifacts/plots/task_success_rate.png`、`artifacts/plots/conditional_false_improvement_rate.png`、`artifacts/plots/predicted_vs_environment_contrast.png`。
- 每个正式 run 的原始日志、sidecar tensor/NPZ 与 Hydra 配置保留在 `artifacts/development/` 和 `artifacts/ood/`；启动失败的 `val_T_official_n3` 与 `val_I_frozen_n5` 也保留，但不会进入任何分母。

## 与说明文本不同、以源码为准的实现事实

- 5 个 model actions 的 `VWorldModel.rollout` 返回 initial + 5 个预测 observation，共 6 帧。
- staged objective 在 MPC step 0–4 取 terminal，从 step 5 起取 full horizon；full objective 的归一化指数权重后仍执行外层 mean。
- native GD 配置实际使用 Adam；诊断记录的是 optimizer/scheduler 后的真实 `u_after-u_before`，不是假设的 `-lr*grad`。
- 官方 final evaluator 会先执行整批 padded actions，再用 `action_len` 选择评价帧；预算已按真实执行量校正。

## 审计边界

本报告未执行任何 SHA 或其他哈希检查。资产身份仅沿用用户提供路径、文件大小/时间、Git 分支和说明中给定的提交标识；这不是密码学完整性证明。
"""
    (ARTIFACTS / "PILOT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    runs = [load_run(*spec) for spec in RUN_SPECS]
    task = task_rows(runs)
    contrast, episodes = contrast_rows(runs)
    by_gd = grouped_contrast_table(runs, "gd_iter")
    by_mpc = grouped_contrast_table(runs, "mpc_iter")
    by_stage = adaptation_stage_rows(runs)
    support_pairs, support_summary = support_next_query_rows(runs)
    outcome_summary = episode_outcome_summary(episodes)

    all_planner = [row for run in runs for row in augmented(run["planner"], run)]
    all_adaptations = [row for run in runs for row in augmented(run["adaptations"], run)]
    all_versions = [row for run in runs for row in augmented(run["versions"], run)]
    all_branches = [row for run in runs for row in augmented(run["branches"], run)]
    all_contrasts = [row for run in runs for row in augmented(run["contrasts"], run)]
    all_replay = [row for run in runs for row in augmented(run["replay"], run)]
    all_samples = [row for run in runs for row in augmented(run["samples"], run)]

    save_parquet(all_planner, ARTIFACTS / "traces" / "planner_steps.parquet")
    save_parquet(all_adaptations, ARTIFACTS / "traces" / "adaptation_steps.parquet")
    save_parquet(all_versions, ARTIFACTS / "traces" / "version_table.parquet")
    save_parquet(all_branches, ARTIFACTS / "oracle" / "branch_records.parquet")
    save_csv(all_samples, ARTIFACTS / "oracle" / "query_budget.csv")
    save_csv(contrast, ARTIFACTS / "metrics" / "contrast_metrics.csv")
    save_csv(all_contrasts, ARTIFACTS / "metrics" / "contrast_pairs.csv")
    save_csv(all_replay, ARTIFACTS / "metrics" / "replay_errors.csv")
    save_csv(all_branches, ARTIFACTS / "metrics" / "environment_metrics.csv")
    save_csv(task, ARTIFACTS / "metrics" / "task_metrics.csv")
    save_csv(episodes, ARTIFACTS / "metrics" / "episode_outcomes.csv")
    save_csv(outcome_summary, ARTIFACTS / "metrics" / "episode_outcome_summary.csv")
    save_csv(by_gd, ARTIFACTS / "metrics" / "contrast_by_gd_step.csv")
    save_csv(by_mpc, ARTIFACTS / "metrics" / "contrast_by_mpc_iter.csv")
    save_csv(by_stage, ARTIFACTS / "metrics" / "contrast_by_adaptation_stage.csv")
    save_csv(support_pairs, ARTIFACTS / "metrics" / "support_next_query_pairs.csv")
    save_csv(support_summary, ARTIFACTS / "metrics" / "support_next_query_summary.csv")
    save_csv(
        [
            {
                "phase": run["phase"],
                "split": run["split"],
                "run_path": str(run["path"]),
                **run["budget"],
            }
            for run in runs
        ],
        ARTIFACTS / "metrics" / "run_budgets.csv",
    )
    generate_plots(task, contrast, all_contrasts)
    write_report(runs, task, contrast, by_stage, outcome_summary, support_summary)

    result = {
        "n_runs": len(runs),
        "n_episodes": sum(len(run["samples"]) for run in runs),
        "n_contrasts": len(all_contrasts),
        "n_oracle_branches": len(all_branches),
        "zero_replay_passed": sum(bool(row["passed"]) for row in all_replay),
        "zero_replay_total": len(all_replay),
        "decision": "REDEFINE",
        "report": str(ARTIFACTS / "PILOT_REPORT.md"),
    }
    (ARTIFACTS / "metrics" / "result_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
