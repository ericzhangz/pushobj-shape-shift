"""Round 3 S1: reconcile existing A/B/C results on matched targets.

This module performs no model or environment calls.  It recomputes the small
tables directly from the recorded CSV files and candidate action tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from research.reframe_v3.common_continuation import _candidate_tensor


EPSILON = 1e-6


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


def _parse_donor(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("donor must use SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if not split or not raw_path:
        raise argparse.ArgumentTypeError("donor must use nonempty SPLIT=PATH")
    return split, Path(raw_path)


def _f(value) -> float:
    return float(value)


def _direction_counts(values: list[float], epsilon: float = EPSILON) -> dict:
    return {
        "improved": sum(value < -epsilon for value in values),
        "unchanged": sum(abs(value) <= epsilon for value in values),
        "worsened": sum(value > epsilon for value in values),
    }


def _action_metrics(action: torch.Tensor, frozen_action: torch.Tensor) -> dict:
    action = action.detach().cpu()
    frozen_action = frozen_action.detach().cpu()
    delta = action - frozen_action
    first_delta = delta[:, :1]
    return {
        "action_l2": float(torch.linalg.vector_norm(action).item()),
        "first_chunk_l2": float(torch.linalg.vector_norm(action[:, :1]).item()),
        "full_action_delta_from_frozen_l2": float(
            torch.linalg.vector_norm(delta).item()
        ),
        "full_action_delta_from_frozen_max_abs": float(delta.abs().max().item()),
        "first_chunk_delta_from_frozen_l2": float(
            torch.linalg.vector_norm(first_delta).item()
        ),
        "first_chunk_delta_from_frozen_max_abs": float(
            first_delta.abs().max().item()
        ),
        "full_action_exactly_frozen": bool(torch.equal(action, frozen_action)),
        "first_chunk_exactly_frozen": bool(
            torch.equal(action[:, :1], frozen_action[:, :1])
        ),
    }


def recompute_b_selection(
    selection_rows: list[dict],
    candidate_rows: list[dict],
    donors: dict[str, Path],
    epsilon: float = EPSILON,
) -> list[dict]:
    relevant = [
        row
        for row in selection_rows
        if row["analysis"] == "b1_common_anchor"
        and row["variant"] in {"frozen", "predictor_official_shadow"}
    ]
    scores_by_key = defaultdict(list)
    for row in candidate_rows:
        if row["analysis"] != "b1_common_anchor":
            continue
        scores_by_key[(row["anchor_id"], row["variant"])].append(row)

    frozen_by_anchor = {
        row["anchor_id"]: row for row in relevant if row["variant"] == "frozen"
    }
    output = []
    for row in relevant:
        key = (row["anchor_id"], row["variant"])
        pool = scores_by_key[key]
        if len(pool) != 10:
            raise ValueError(f"Expected ten B candidates for {key}, got {len(pool)}")
        model_costs = [_f(candidate["c_model"]) for candidate in pool]
        minimum = min(model_costs)
        exact_ids = [
            candidate["candidate_id"]
            for candidate in pool
            if _f(candidate["c_model"]) == minimum
        ]
        tie_rows = [
            candidate
            for candidate in pool
            if _f(candidate["c_model"]) <= minimum + epsilon
        ]
        env_best = min(_f(candidate["c_env_ref"]) for candidate in pool)
        tie_env_costs = [_f(candidate["c_env_ref"]) for candidate in tie_rows]
        selected_id = row["model_best_candidate_id"]
        if selected_id not in exact_ids:
            raise AssertionError(f"Recorded argmin is not exact for {key}: {selected_id}")

        split = row["split"]
        sample_id = int(row["sample_id"])
        mpc_iter = int(row["mpc_iter"])
        donor = donors[split]
        frozen_id = frozen_by_anchor[row["anchor_id"]]["model_best_candidate_id"]
        action = _candidate_tensor(donor, sample_id, mpc_iter, selected_id)
        frozen_action = _candidate_tensor(donor, sample_id, mpc_iter, frozen_id)
        metrics = _action_metrics(action, frozen_action)
        recomputed_tie_min = min(tie_env_costs) - env_best
        recomputed_tie_max = max(tie_env_costs) - env_best
        if not math.isclose(
            recomputed_tie_min,
            _f(row["r_selected_tie_min"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"tie-min mismatch for {key}")
        if not math.isclose(
            recomputed_tie_max,
            _f(row["r_selected_tie_max"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"tie-max mismatch for {key}")
        output.append(
            {
                "anchor_id": row["anchor_id"],
                "split": split,
                "shape": row["shape"],
                "sample_id": sample_id,
                "mpc_iter": mpc_iter,
                "variant": row["variant"],
                "exact_argmin_candidate_ids": ";".join(exact_ids),
                "epsilon_tie_candidate_ids": ";".join(
                    candidate["candidate_id"] for candidate in tie_rows
                ),
                "epsilon_tie_count": len(tie_rows),
                "selected_candidate_id": selected_id,
                "selected_environment_cost": _f(row["selected_env_cost"]),
                "environment_pool_best": env_best,
                "r_selected": _f(row["r_selected"]),
                "r_selected_tie_min": recomputed_tie_min,
                "r_selected_tie_max": recomputed_tie_max,
                "model_best_margin": _f(row["model_best_margin"]),
                "common_support_loss_before": _f(row["common_support_loss_before"]),
                "common_support_loss_after": _f(row["common_support_loss_after"]),
                "common_support_loss_delta": _f(row["common_support_loss_delta"]),
                **metrics,
            }
        )
    if len(output) != 24:
        raise ValueError(f"Expected 24 B selection rows, got {len(output)}")
    return sorted(output, key=lambda row: (row["anchor_id"], row["variant"]))


def build_matched_target_table(
    logical_rows: list[dict],
    physical_rows: list[dict],
    b_candidate_rows: list[dict],
    donors: dict[str, Path],
) -> list[dict]:
    physical_by_id = {row["physical_branch_id"]: row for row in physical_rows}
    b_scores = {
        (row["anchor_id"], row["variant"], row["candidate_id"]): row
        for row in b_candidate_rows
        if row["analysis"] == "b1_common_anchor"
        and row["variant"] in {"frozen", "predictor_official_shadow"}
    }
    frozen_ids = {
        row["anchor_id"]: row["candidate_id"]
        for row in logical_rows
        if row["label"] == "frozen"
    }
    output = []
    for row in logical_rows:
        physical = physical_by_id[row["physical_branch_id"]]
        split = row["split"]
        sample_id = int(row["sample_id"])
        mpc_iter = int(row["mpc_iter"])
        action = _candidate_tensor(
            donors[split], sample_id, mpc_iter, row["candidate_id"]
        )
        frozen_action = _candidate_tensor(
            donors[split], sample_id, mpc_iter, frozen_ids[row["anchor_id"]]
        )
        variant = (
            "predictor_official_shadow" if row["label"] == "adapted" else "frozen"
        )
        score = b_scores[(row["anchor_id"], variant, row["candidate_id"])]
        j_env = _f(row["recorded_pool_env_cost"])
        q_env = _f(physical["final_reference_terminal_cost"])
        output.append(
            {
                "anchor_id": row["anchor_id"],
                "anchor_ordinal": int(row["anchor_ordinal"]),
                "split": split,
                "shape": row["shape"],
                "sample_id": sample_id,
                "mpc_iter": mpc_iter,
                "label": row["label"],
                "candidate_id": row["candidate_id"],
                "selection_model_variant": variant,
                "selection_model_cost": _f(score["c_model"]),
                "J_env_open_plan": j_env,
                "Q_env_common_continuation": q_env,
                "continuation_target_shift_b_pi": q_env - j_env,
                "first_chunk_reference_terminal_cost": _f(
                    physical["first_chunk_reference_terminal_cost"]
                ),
                "final_reference_terminal_cost": q_env,
                "first_chunk_state_dist": _f(physical["first_chunk_state_dist"]),
                "final_state_dist": _f(physical["final_state_dist"]),
                "first_chunk_success": physical["first_chunk_success"],
                "final_success": physical["final_success"],
                "physical_branch_id": row["physical_branch_id"],
                "first_chunk_deduplicated": row["first_chunk_deduplicated"],
                **_action_metrics(action, frozen_action),
            }
        )
    if len(output) != 18:
        raise ValueError(f"Expected 18 logical C rows, got {len(output)}")
    return sorted(output, key=lambda row: (row["anchor_id"], row["label"]))


def _paired_deltas(rows: list[dict], label: str, field: str) -> list[float]:
    by_anchor = defaultdict(dict)
    for row in rows:
        by_anchor[row["anchor_id"]][row["label"]] = row
    return [
        _f(labels[label][field]) - _f(labels["frozen"][field])
        for labels in by_anchor.values()
    ]


def _render_reports(
    a_rows: list[dict],
    b_pair_rows: list[dict],
    b_rows: list[dict],
    c_rows: list[dict],
) -> tuple[str, str, dict]:
    robust_a = sum(_f(row["r_selected_tie_min"]) > EPSILON for row in a_rows)
    eta_a = sum(_f(row["eta_model_final_record"]) > EPSILON for row in a_rows)
    improve_a = sum(_f(row["g_env_final"]) > EPSILON for row in a_rows)
    worsen_a = sum(_f(row["g_env_final"]) < -EPSILON for row in a_rows)
    main_pairs = [
        row
        for row in b_pair_rows
        if row["comparison"] == "predictor_official_shadow_minus_frozen"
    ]
    b_regret = [_f(row["delta_r_selected_tie_min"]) for row in main_pairs]
    b_support = [_f(row["common_support_loss_delta"]) for row in main_pairs]
    b_counts = _direction_counts(b_regret)
    id_changes = sum(row["selection_changed"].lower() == "true" for row in main_pairs)
    g99_side_only = sum(
        {row["before_model_best_candidate_id"], row["after_model_best_candidate_id"]}
        == {"g99_before", "g99_after"}
        for row in main_pairs
    )

    c_effects = {}
    for label in ("adapted", "oracle"):
        cost = _paired_deltas(c_rows, label, "final_reference_terminal_cost")
        state = _paired_deltas(c_rows, label, "final_state_dist")
        c_effects[label] = {
            "cost": {**_direction_counts(cost), "mean": statistics.fmean(cost), "median": statistics.median(cost)},
            "state": {**_direction_counts(state), "mean": statistics.fmean(state), "median": statistics.median(state)},
        }

    success_by_label = {
        label: sum(str(row["final_success"]).lower() == "true" for row in c_rows if row["label"] == label)
        for label in ("frozen", "adapted", "oracle")
    }
    summary = {
        "A": {
            "replans": len(a_rows),
            "tie_robust_selection_regret": robust_a,
            "recorded_pool_optimizer_gap": eta_a,
            "environment_outcome_improved": improve_a,
            "environment_outcome_worsened": worsen_a,
        },
        "B": {
            "anchors": len(main_pairs),
            "support_loss_decreased": sum(value < 0 for value in b_support),
            "selection_ids_changed": id_changes,
            "g99_before_after_only_changes": g99_side_only,
            "tie_aware_regret": b_counts,
            "mean_delta_tie_aware_regret": statistics.fmean(b_regret),
        },
        "C": {
            "anchors": len({row["anchor_id"] for row in c_rows}),
            "effects": c_effects,
            "final_successes_by_label": success_by_label,
        },
    }

    existing = f"""# Round 3 — Existing-data reanalysis

This is a zero-model-call, zero-environment-call reconciliation from the recorded CSVs.

## A: finite recorded-pool selection

- {len(a_rows)} replans; tie-aware regret exceeds `1e-6` in {robust_a}.
- Recorded-pool optimizer gap exceeds `1e-6` in {eta_a}.
- Full optimization improves recorded environment outcome in {improve_a} and worsens it in {worsen_a}.
- These are nested replans, not {len(a_rows)} independent tasks.

## B: same experience, same query pool

- Support loss decreases in {sum(value < 0 for value in b_support)}/{len(main_pairs)} anchors.
- Candidate ID changes in {id_changes}/{len(main_pairs)}; {g99_side_only} are only `g99_before`/`g99_after` side changes.
- Tie-aware regret: {b_counts['improved']} improved, {b_counts['worsened']} worsened, {b_counts['unchanged']} unchanged; mean delta `{statistics.fmean(b_regret):.12g}`.
- `b_selection_existing_results.csv` records exact argmin sets, epsilon-tie sets, and full/first-chunk action magnitudes without adding a post-hoc behavioral threshold.

## C: open-plan target versus common-continuation target

- `matched_target_existing_results.csv` places `J_env_open_plan` and `Q_env_common_continuation` on the same logical row.
- The difference `b_pi = Q_env - J_env` is an intervention-target shift, not model error.
- Final successes are Frozen {success_by_label['frozen']}/6, adapted {success_by_label['adapted']}/6, oracle {success_by_label['oracle']}/6.
- The three labels within an anchor are interventions on one starting case, not independent episodes.

No new threshold, candidate selection, model query, or environment branch was introduced.
"""
    effect = f"""# C effect-size report

All deltas are label minus Frozen at the same anchor; `epsilon=1e-6` is inherited.

| label | final cost improve / tie / worsen | mean delta | median delta | state improve / tie / worsen | mean state delta | median state delta |
|---|---:|---:|---:|---:|---:|---:|
| adapted | {c_effects['adapted']['cost']['improved']} / {c_effects['adapted']['cost']['unchanged']} / {c_effects['adapted']['cost']['worsened']} | {c_effects['adapted']['cost']['mean']:.12g} | {c_effects['adapted']['cost']['median']:.12g} | {c_effects['adapted']['state']['improved']} / {c_effects['adapted']['state']['unchanged']} / {c_effects['adapted']['state']['worsened']} | {c_effects['adapted']['state']['mean']:.12g} | {c_effects['adapted']['state']['median']:.12g} |
| oracle | {c_effects['oracle']['cost']['improved']} / {c_effects['oracle']['cost']['unchanged']} / {c_effects['oracle']['cost']['worsened']} | {c_effects['oracle']['cost']['mean']:.12g} | {c_effects['oracle']['cost']['median']:.12g} | {c_effects['oracle']['state']['improved']} / {c_effects['oracle']['state']['unchanged']} / {c_effects['oracle']['state']['worsened']} | {c_effects['oracle']['state']['mean']:.12g} | {c_effects['oracle']['state']['median']:.12g} |

The oracle label was selected with full-plan environment outcomes and is diagnostic only.  It is not a deployable candidate generator.  In the legal Frozen/adapted subset, adapted has no anchor with a final fixed-reference cost improvement beyond epsilon.
"""
    return existing, effect, summary


def run(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    donors = {split: path.resolve() for split, path in args.donor}
    if set(donors) != {"val_T", "val_L"}:
        raise ValueError(f"Expected val_T and val_L donors, got {sorted(donors)}")

    a_rows = _read_csv(args.a_selection)
    b_selection = _read_csv(args.b_dir / "selection_metrics.csv")
    b_candidates = _read_csv(args.b_dir / "candidate_scores.csv")
    b_pairs = _read_csv(args.b_dir / "paired_effects.csv")
    c_logical = _read_csv(args.c_dir / "logical_branch_results.csv")
    c_physical = _read_csv(args.c_dir / "physical_branch_results.csv")

    b_output = recompute_b_selection(b_selection, b_candidates, donors)
    c_output = build_matched_target_table(
        c_logical, c_physical, b_candidates, donors
    )
    existing_report, effect_report, summary = _render_reports(
        a_rows, b_pairs, b_output, c_output
    )

    _write_csv(args.out / "b_selection_existing_results.csv", b_output)
    _write_csv(args.out / "matched_target_existing_results.csv", c_output)
    for name, content in (
        ("EXISTING_DATA_REANALYSIS.md", existing_report),
        ("C_effect_size_report.md", effect_report),
    ):
        path = args.out / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(content, encoding="utf-8")
    summary_path = args.out / "existing_data_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-selection", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--donor", type=_parse_donor, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
