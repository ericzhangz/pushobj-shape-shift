"""Validate raw case mappings and tensor identity for the V3 offline audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from offline_selection_audit import GROUP_FIELDS, endpoint, read_rows


INITIAL_KEYS = (
    "visual",
    "proprio",
    "initial_state",
    "goal_visual",
    "goal_proprio",
    "goal_state",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_initial(run_path: Path, sample_id: int) -> dict[str, np.ndarray]:
    path = run_path / "real_evidence" / f"sample_{sample_id:03d}_initial.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing initial/goal evidence: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in INITIAL_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"Initial/goal evidence {path} is missing keys: {missing}")
        return {key: np.asarray(archive[key]) for key in INITIAL_KEYS}


def validate_case_mappings(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[tuple, Path]] = defaultdict(dict)
    for row in rows:
        case_key = (
            row["phase"],
            row["split"],
            row["shape"],
            int(row["sample_id"]),
        )
        source_key = (row["arm"], row["run_id"])
        grouped[case_key][source_key] = Path(row["run_path"])

    results = []
    for case_key in sorted(grouped):
        phase, split, shape, sample_id = case_key
        sources = sorted(grouped[case_key].items())
        loaded = [
            (arm, run_id, load_initial(run_path, sample_id))
            for (arm, run_id), run_path in sources
        ]
        reference = loaded[0][2]
        mismatches = []
        for arm, run_id, arrays in loaded[1:]:
            for key in INITIAL_KEYS:
                if not np.array_equal(reference[key], arrays[key]):
                    mismatches.append(f"{arm}:{run_id}:{key}")
        results.append(
            {
                "phase": phase,
                "split": split,
                "shape": shape,
                "sample_id": sample_id,
                "case_id": f"{phase}|{split}|{shape}|{sample_id}",
                "arms": ";".join(arm for arm, _, _ in loaded),
                "run_ids": ";".join(run_id for _, run_id, _ in loaded),
                "source_count": len(loaded),
                "all_arrays_equal": not mismatches,
                "mismatches": ";".join(mismatches),
            }
        )
    return results


def load_sidecar(run_path: Path, sample_id: int, mpc_iter: int, gd_iter: int) -> dict:
    path = run_path / "tensors" / f"s{sample_id}_m{mpc_iter}_g{gd_iter}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing optimizer tensor sidecar: {path}")
    record = torch.load(path, map_location="cpu")
    for key in ("u_before", "u_after"):
        if key not in record:
            raise ValueError(f"Tensor sidecar {path} is missing {key}")
    return record


def validate_returned_prefixes(grouped: dict[tuple, list[dict]]) -> list[dict]:
    baseline_cache = {}
    results = []
    for group_key in sorted(grouped):
        rows = sorted(grouped[group_key], key=lambda row: row["gd_iter"])
        first = rows[0]
        run_path = Path(first["run_path"])
        if run_path not in baseline_cache:
            path = run_path / "baseline_actions.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing returned actions: {path}")
            baseline_cache[run_path] = torch.load(path, map_location="cpu")
        baseline = baseline_cache[run_path]
        sample_id = int(first["sample_id"])
        mpc_iter = int(first["mpc_iter"])
        sidecar = load_sidecar(run_path, sample_id, mpc_iter, 99)
        recorded = sidecar["u_after"][0, 0].detach().cpu()
        returned = baseline["actions"][sample_id, mpc_iter].detach().cpu()
        same_shape = tuple(recorded.shape) == tuple(returned.shape)
        exact = bool(same_shape and torch.equal(recorded, returned))
        max_abs_error = (
            float(torch.max(torch.abs(recorded - returned))) if same_shape else None
        )
        results.append(
            {
                "run_id": first["run_id"],
                "arm": first["arm"],
                "phase": first["phase"],
                "split": first["split"],
                "shape": first["shape"],
                "sample_id": sample_id,
                "mpc_iter": mpc_iter,
                "same_shape": same_shape,
                "exact_returned_prefix": exact,
                "max_abs_error": max_abs_error,
            }
        )
    return results


def validate_first_chunk_continuations(grouped: dict[tuple, list[dict]]) -> list[dict]:
    """Match the submitted g99 action and its logged first-chunk consequence.

    Oracle outcomes are stored at model stride, whereas real evidence stores all
    environment substeps.  Therefore this validates the common start and the
    state/observation after the complete first model chunk; it intentionally
    does not claim that the four intermediate substeps were archived by the
    oracle.
    """
    results = []
    for group_key in sorted(grouped):
        rows = sorted(grouped[group_key], key=lambda row: row["gd_iter"])
        first = rows[0]
        run_path = Path(first["run_path"])
        sample_id = int(first["sample_id"])
        mpc_iter = int(first["mpc_iter"])
        real_path = (
            run_path
            / "real_evidence"
            / f"s{sample_id}_executed_mpc{mpc_iter}.npz"
        )
        oracle_path = (
            run_path
            / "oracle"
            / f"sample_{sample_id:03d}"
            / f"s{sample_id}_m{mpc_iter}_g99_after_outcome.npz"
        )
        if not real_path.exists():
            raise FileNotFoundError(f"Missing executed continuation evidence: {real_path}")
        if not oracle_path.exists():
            raise FileNotFoundError(f"Missing oracle continuation evidence: {oracle_path}")
        with np.load(real_path, allow_pickle=False) as archive:
            real = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(oracle_path, allow_pickle=False) as archive:
            oracle = {key: np.asarray(archive[key]) for key in archive.files}
        for key in ("visual", "proprio", "states"):
            if key not in real or key not in oracle:
                raise ValueError(
                    f"Continuation evidence is missing {key}: {real_path} / {oracle_path}"
                )
            if len(real[key]) < 2 or len(oracle[key]) < 2:
                raise ValueError(
                    f"Continuation evidence has fewer than two frames for {key}: "
                    f"{real_path} / {oracle_path}"
                )

        comparisons = {}
        max_errors = {}
        for key in ("visual", "proprio", "states"):
            for label, real_index, oracle_index in (
                ("start", 0, 0),
                ("first_chunk_end", -1, 1),
            ):
                name = f"{key}_{label}"
                left = real[key][real_index]
                right = oracle[key][oracle_index]
                same_shape = left.shape == right.shape
                comparisons[name] = bool(same_shape and np.array_equal(left, right))
                max_errors[name] = (
                    float(
                        np.max(
                            np.abs(
                                left.astype(np.float64) - right.astype(np.float64)
                            )
                        )
                    )
                    if same_shape
                    else None
                )

        sidecar = load_sidecar(run_path, sample_id, mpc_iter, 99)
        submitted = sidecar["u_after"][0, 0].detach().cpu().numpy()
        executed = np.asarray(real.get("normalized_model_actions"))
        action_shape_ok = executed.shape[:2] == (1, 1) and executed.shape[2:] == submitted.shape
        action_exact = bool(
            action_shape_ok and np.array_equal(executed[0, 0], submitted)
        )
        results.append(
            {
                "run_id": first["run_id"],
                "arm": first["arm"],
                "phase": first["phase"],
                "split": first["split"],
                "shape": first["shape"],
                "sample_id": sample_id,
                "mpc_iter": mpc_iter,
                "executed_action_exact": action_exact,
                **{f"{key}_exact": value for key, value in comparisons.items()},
                **{f"{key}_max_abs_error": value for key, value in max_errors.items()},
                "all_exact": action_exact and all(comparisons.values()),
            }
        )
    return results


def deduplicate_pool(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: row["gd_iter"])
    first = ordered[0]
    run_path = Path(first["run_path"])
    sample_id = int(first["sample_id"])
    mpc_iter = int(first["mpc_iter"])
    candidates = []
    for row in ordered:
        sidecar = load_sidecar(run_path, sample_id, mpc_iter, int(row["gd_iter"]))
        for side in ("before", "after"):
            costs = endpoint(row, side)
            candidates.append({**costs, "tensor": sidecar[f"u_{side}"].detach().cpu()})

    unique = []
    duplicate_groups = []
    cost_consistent = True
    for candidate in candidates:
        match = next(
            (item for item in unique if torch.equal(item["tensor"], candidate["tensor"])),
            None,
        )
        if match is None:
            unique.append({**candidate, "members": [candidate["candidate_id"]]})
            continue
        match["members"].append(candidate["candidate_id"])
        if not (
            abs(match["c_model"] - candidate["c_model"]) <= 1e-9
            and abs(match["c_env"] - candidate["c_env"]) <= 1e-9
        ):
            cost_consistent = False
    duplicate_groups = [item["members"] for item in unique if len(item["members"]) > 1]
    model_best = min(unique, key=lambda candidate: candidate["c_model"])
    env_best = min(unique, key=lambda candidate: candidate["c_env"])
    final_record = candidates[-1]
    return {
        "run_id": first["run_id"],
        "arm": first["arm"],
        "phase": first["phase"],
        "split": first["split"],
        "shape": first["shape"],
        "sample_id": sample_id,
        "mpc_iter": mpc_iter,
        "objective_stage": "terminal" if mpc_iter < 5 else "full_horizon",
        "raw_pool_size": len(candidates),
        "unique_pool_size": len(unique),
        "duplicate_groups": json.dumps(duplicate_groups, separators=(",", ":")),
        "duplicate_costs_consistent": cost_consistent,
        "model_best_candidate_id": model_best["candidate_id"],
        "env_best_candidate_id": env_best["candidate_id"],
        "r_selected_deduplicated": model_best["c_env"] - env_best["c_env"],
        "r_final_record_deduplicated": final_record["c_env"] - env_best["c_env"],
        "eta_model_final_record_deduplicated": final_record["c_model"]
        - model_best["c_model"],
    }


def verify(input_path: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"refusing to overwrite nonempty output directory: {output_dir}"
        )
    rows = read_rows(input_path)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)

    case_rows = validate_case_mappings(rows)
    returned_rows = validate_returned_prefixes(grouped)
    continuation_rows = validate_first_chunk_continuations(grouped)
    dedup_rows = [deduplicate_pool(grouped[key]) for key in sorted(grouped)]
    validation = {
        "case_mappings_checked": len(case_rows),
        "returned_prefixes_checked": len(returned_rows),
        "continuations_checked": len(continuation_rows),
        "replan_pools_checked": len(dedup_rows),
        "case_mapping_failures": sum(
            1 for row in case_rows if not row["all_arrays_equal"]
        ),
        "returned_prefix_failures": sum(
            1 for row in returned_rows if not row["exact_returned_prefix"]
        ),
        "continuation_failures": sum(
            1 for row in continuation_rows if not row["all_exact"]
        ),
        "duplicate_cost_consistency_failures": sum(
            1 for row in dedup_rows if not row["duplicate_costs_consistent"]
        ),
    }
    validation["passed"] = not any(
        validation[key]
        for key in (
            "case_mapping_failures",
            "returned_prefix_failures",
            "continuation_failures",
            "duplicate_cost_consistency_failures",
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "case_mapping.csv", case_rows)
    write_csv(output_dir / "returned_prefix_validation.csv", returned_rows)
    write_csv(output_dir / "continuation_validation.csv", continuation_rows)
    write_csv(output_dir / "deduplicated_selection_metrics.csv", dedup_rows)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validation = verify(args.input, args.out)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
