"""Fail-fast consistency checks for the completed PushObj pilot artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    result = json.loads(
        (ARTIFACTS / "metrics" / "result_summary.json").read_text(encoding="utf-8")
    )
    require(result["n_runs"] == 15, "Expected 15 formal runs")
    require(result["n_episodes"] == 63, "Expected 63 formal episodes")
    require(result["n_contrasts"] == 4000, "Expected 4000 fixed-step contrasts")
    require(result["n_oracle_branches"] == 8000, "Expected paired oracle branches")
    require(
        result["zero_replay_passed"] == result["zero_replay_total"] == 800,
        "Every zero-action full-prefix replay must pass",
    )
    require(result["decision"] == "REDEFINE", "Decision drifted from the report")

    expected_counts = {
        "traces/adaptation_steps.parquet": 800,
        "traces/version_table.parquet": 863,
        "oracle/branch_records.parquet": 8000,
        "oracle/query_budget.csv": 63,
        "metrics/contrast_metrics.csv": 15,
        "metrics/contrast_pairs.csv": 4000,
        "metrics/replay_errors.csv": 800,
        "metrics/environment_metrics.csv": 8000,
        "metrics/task_metrics.csv": 15,
        "metrics/episode_outcomes.csv": 63,
        "metrics/run_budgets.csv": 15,
    }
    loaded = {}
    for relative, expected in expected_counts.items():
        path = ARTIFACTS / relative
        require(path.exists() and path.stat().st_size > 0, f"Missing output: {path}")
        data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        require(len(data) == expected, f"Unexpected row count for {path}: {len(data)}")
        loaded[relative] = data

    replay = loaded["metrics/replay_errors.csv"]
    require(replay["passed"].astype(bool).all(), "A replay gate is not exact")
    require(
        replay[["run_id", "sample_id", "mpc_iter"]].duplicated().sum() == 0,
        "Duplicated replay anchor",
    )

    versions = loaded["traces/version_table.parquet"]
    require(
        (versions["event"] == "sample_start").sum() == 63,
        "Expected exactly one initial version snapshot per episode",
    )
    require(
        (versions["event"] == "adaptation").sum() == 800,
        "Expected exactly one post-replan version record per replay anchor",
    )

    task = loaded["metrics/task_metrics.csv"]
    require(task["n_episodes"].sum() == 63, "Task episode denominator mismatch")
    require(task["n_success"].sum() == 35, "Task success count mismatch")
    require(
        task.groupby("split")["arm"].nunique().eq(3).all(),
        "A split is missing an arm",
    )

    contrast = loaded["metrics/contrast_metrics.csv"]
    frozen = contrast[contrast["arm"] == "FROZEN"]
    require((frozen["support_update_events"] == 0).all(), "Frozen updates were miscounted")
    require(
        (contrast["thresholded_false_improvement_count"] > 0).all(),
        "A split/arm unexpectedly has no real false-improvement event",
    )

    budgets = loaded["metrics/run_budgets.csv"]
    require(budgets["run_id"].is_unique, "Duplicate formal run in budgets")
    require(
        (
            budgets["official_total_evaluation_environment_steps"]
            == budgets["official_mpc_cumulative_replay_environment_steps"]
            + budgets["official_final_eval_environment_steps"]
        ).all(),
        "Official environment budget does not close",
    )
    require(
        (
            budgets["official_final_eval_environment_steps"]
            >= budgets["recorded_official_final_eval_environment_steps"]
        ).all(),
        "Corrected padded final evaluation is smaller than the original record",
    )

    planner = pd.read_parquet(ARTIFACTS / "traces" / "planner_steps.parquet")
    require(len(planner) == 80000, f"Expected 80000 planner iterations, got {len(planner)}")

    report = (ARTIFACTS / "PILOT_REPORT.md").read_text(encoding="utf-8")
    for heading in (
        "## A. Reproduction",
        "## B. Measurement integrity",
        "## C. Real failure",
        "## D. Effect of AdaJEPA",
        "## E. Encoder confound",
        "## F. Response diagnostics",
        "## G. Causal evidence",
        "## H. Experience reuse",
        "## I. Go / No-Go",
    ):
        require(heading in report, f"Missing report section: {heading}")

    for name in (
        "task_success_rate.png",
        "conditional_false_improvement_rate.png",
        "predicted_vs_environment_contrast.png",
    ):
        path = ARTIFACTS / "plots" / name
        require(path.exists() and path.stat().st_size > 0, f"Missing plot: {path}")

    print(
        json.dumps(
            {
                "status": "passed",
                "formal_runs": 15,
                "episodes": 63,
                "planner_iterations": len(planner),
                "contrasts": 4000,
                "oracle_branches": 8000,
                "zero_replays": "800/800",
                "task_successes": "35/63",
                "decision": "REDEFINE",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
