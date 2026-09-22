"""Fail-fast source and temporal contract for the Round 4 development data.

This checks archived evidence only. It never loads the world-model checkpoint,
updates parameters, or queries the environment. Oracle outcomes are opened to
validate evaluation assets, never exposed as training inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from research.reframe_v3.shadow_selection_audit import (
    GD_STEPS,
    SIDES,
    official_prefix_schedule,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_outcome(path: Path, current: dict) -> None:
    require(path.is_file(), f"Missing oracle outcome: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(
            {"visual", "proprio", "states", "final_state"} <= set(archive.files),
            f"Incomplete oracle outcome: {path}",
        )
        require(archive["visual"].shape == (6, 224, 224, 3), f"Visual shape: {path}")
        require(archive["proprio"].shape == (6, 4), f"Proprio shape: {path}")
        require(archive["states"].shape[0] == 6, f"State length: {path}")
        require(
            np.array_equal(archive["visual"][0], current["visual"][0, 0])
            and np.array_equal(archive["proprio"][0], current["proprio"][0, 0]),
            f"Outcome start differs from archived anchor observation: {path}",
        )
        require(
            np.array_equal(archive["states"][-1], archive["final_state"]),
            f"Final state mismatch: {path}",
        )


def check_factual(donor: Path, sample_id: int, mpc_iter: int) -> dict:
    history_path = donor / "real_evidence" / f"s{sample_id}_history_mpc{mpc_iter}.pt"
    require(history_path.is_file(), f"Missing anchor history: {history_path}")
    history = torch.load(history_path, map_location="cpu")
    require("current_real_observation" in history, f"Missing current obs: {history_path}")
    current = history["current_real_observation"]
    require(
        tuple(current["visual"].shape) == (1, 1, 224, 224, 3)
        and tuple(current["proprio"].shape) == (1, 1, 4),
        f"Anchor observation shape: {history_path}",
    )
    initial = donor / "real_evidence" / f"sample_{sample_id:03d}_initial.npz"
    require(initial.is_file(), f"Missing goal evidence: {initial}")
    with np.load(initial, allow_pickle=False) as archive:
        require(
            {"goal_visual", "goal_proprio"} <= set(archive.files),
            f"Incomplete goal evidence: {initial}",
        )
    factual_actions = []
    previous_terminal = None
    for index in range(mpc_iter):
        path = donor / "real_evidence" / f"s{sample_id}_executed_mpc{index}.npz"
        require(path.is_file(), f"Missing completed factual segment: {path}")
        with np.load(path, allow_pickle=False) as archive:
            require(
                {"visual", "proprio", "states", "normalized_model_actions"} <= set(archive.files),
                f"Incomplete factual segment: {path}",
            )
            require(archive["visual"].shape == (6, 224, 224, 3), f"Visual shape: {path}")
            require(archive["proprio"].shape == (6, 4), f"Proprio shape: {path}")
            require(
                archive["normalized_model_actions"].shape == (1, 1, 10),
                f"Action shape: {path}",
            )
            if previous_terminal is not None:
                require(
                    np.array_equal(previous_terminal["visual"], archive["visual"][0])
                    and np.array_equal(previous_terminal["proprio"], archive["proprio"][0]),
                    f"Factual segments are not contiguous: {path}",
                )
            previous_terminal = {
                "visual": np.asarray(archive["visual"][-1]),
                "proprio": np.asarray(archive["proprio"][-1]),
                "states": np.asarray(archive["states"][-1]),
            }
            factual_actions.append(np.asarray(archive["normalized_model_actions"]))
    require(
        np.array_equal(
            history["executed_prefix"].detach().cpu().numpy(),
            np.concatenate(factual_actions, axis=1),
        ),
        f"History prefix differs from completed actions: {history_path}",
    )
    require(
        np.array_equal(current["visual"][0, 0], previous_terminal["visual"])
        and np.array_equal(current["proprio"][0, 0], previous_terminal["proprio"])
        and np.array_equal(history["current_public_state"][0], previous_terminal["states"]),
        f"History endpoint differs from last completed factual segment: {history_path}",
    )
    return current


def audit_b(repo: Path) -> tuple[list[dict], list[dict]]:
    base = repo / "artifacts" / "reframe_v3" / "common_anchors_v2"
    config = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
    require(config["anchors"] == [2, 4], "Unexpected B anchor times")
    require(config["frameskip"] == 5, "Unexpected frameskip")
    require(config["model_action_dim"] == 10, "Unexpected action dimension")
    require(config["model_action_horizon"] == 5, "Unexpected action horizon")
    checkpoint = (repo / config["checkpoint_dir"]).resolve()
    require((checkpoint / "hydra.yaml").is_file(), f"Missing model config: {checkpoint}")
    require(
        (checkpoint / "checkpoints" / "model_latest.pth").is_file(),
        f"Missing checkpoint: {checkpoint}",
    )
    from omegaconf import OmegaConf

    model_config = OmegaConf.load(checkpoint / "hydra.yaml")
    require(
        int(model_config.frameskip) == 5
        and int(model_config.num_hist) == 3
        and int(model_config.num_pred) == 1
        and int(model_config.num_action_repeat) == 1,
        f"Model history/prediction configuration mismatch: {checkpoint}",
    )
    anchors = read_csv(base / "anchors.csv")
    scores = read_csv(base / "candidate_scores.csv")
    frozen_rows = [
        row for row in scores
        if row["analysis"] == "b1_common_anchor" and row["variant"] == "frozen"
    ]
    frozen_scores = {
        (row["anchor_id"], row["candidate_id"]): row
        for row in frozen_rows
    }
    require(len(anchors) == 12, f"Expected 12 B anchors, found {len(anchors)}")
    require(len(frozen_rows) == 120, f"Expected 120 B frozen rows, found {len(frozen_rows)}")
    require(len(frozen_scores) == 120, f"Expected 120 B frozen scores, found {len(frozen_scores)}")
    require(len({row["anchor_id"] for row in anchors}) == 12, "Duplicate B anchor")

    allowed_donors = {split: (repo / path).resolve() for split, path in config["donors"]}
    require(len(allowed_donors) == len(config["donors"]) == 2, "Duplicate B donor split")
    branch_cache: dict[Path, dict] = {}
    query_rows, candidate_rows = [], []
    for ordinal, anchor in enumerate(anchors):
        anchor_id = anchor["anchor_id"]
        require(anchor["available"] == "True", f"Unavailable anchor: {anchor_id}")
        donor = Path(anchor["donor_dir"]).resolve()
        require(donor == allowed_donors.get(anchor["split"]), f"Donor/split mismatch: {anchor_id}")
        meta = json.loads((donor / "run_metadata.json").read_text(encoding="utf-8"))
        require(meta["arm"] == "FROZEN", f"Not a frozen donor: {donor}")
        require(meta["frameskip"] == 5, f"Donor frameskip mismatch: {donor}")
        sample_id, mpc_iter = int(anchor["sample_id"]), int(anchor["mpc_iter"])
        require(0 <= sample_id < 3, f"Unexpected sample index: {anchor_id}")
        require(mpc_iter in (2, 4), f"Unexpected query time: {anchor_id}")
        require(
            anchor["shape"] == meta["shapes"][sample_id]
            and anchor_id == f"{anchor['split']}|{anchor['shape']}|s{sample_id}|m{mpc_iter}",
            f"Anchor identity/shape mismatch: {anchor_id}",
        )
        require(
            int(anchor["rng_seed"]) == int(config["seed"]) + ordinal,
            f"Anchor RNG mismatch: {anchor_id}",
        )
        require(
            meta["objective"] == {"mode": "staged", "alpha": 1.0, "base": 2.0}
            and meta["model_action_dim"] == 10
            and meta["action_units"]["model"] == "normalized environment API action packed by frameskip",
            f"Donor objective/action convention mismatch: {donor}",
        )
        schedule = official_prefix_schedule(mpc_iter)
        require(
            len(schedule) == mpc_iter
            and all(0 <= index < mpc_iter for batch in schedule for index in batch),
            f"Factual schedule crosses query cutoff: {anchor_id}",
        )
        current = check_factual(donor, sample_id, mpc_iter)
        if donor not in branch_cache:
            rows = read_jsonl(donor / "oracle" / "branch_records.jsonl")
            branch_cache[donor] = {
                (
                    int(row["sample_id"]),
                    int(row["mpc_iter"]),
                    int(row["gd_iter"]),
                    str(row["candidate"]),
                ): row
                for row in rows
            }
            require(len(branch_cache[donor]) == len(rows), f"Duplicate donor branch: {donor}")
        query_rows.append(
            {
                "anchor_id": anchor_id,
                "case_id": f"{anchor['split']}|{anchor['shape']}|s{sample_id}",
                "query_mpc_iter": mpc_iter,
                "latest_allowed_factual_mpc_iter": mpc_iter - 1,
                "factual_segments": ";".join(str(i) for i in range(mpc_iter)),
                "update_batches": json.dumps(schedule, separators=(",", ":")),
                "rng_seed": int(anchor["rng_seed"]),
                "donor_dir": str(donor),
                "oracle_for_training": False,
            }
        )
        for gd_iter in GD_STEPS:
            tensor_path = donor / "tensors" / f"s{sample_id}_m{mpc_iter}_g{gd_iter}.pt"
            require(tensor_path.is_file(), f"Missing candidate actions: {tensor_path}")
            tensors = torch.load(tensor_path, map_location="cpu")
            for side in SIDES:
                candidate_id = f"g{gd_iter}_{side}"
                require(
                    f"u_{side}" in tensors
                    and tuple(tensors[f"u_{side}"].shape) == (1, 5, 10),
                    f"Candidate action shape: {tensor_path}:{side}",
                )
                branch_key = (sample_id, mpc_iter, gd_iter, f"u_{side}")
                require(branch_key in branch_cache[donor], f"Missing branch: {anchor_id}:{candidate_id}")
                branch = branch_cache[donor][branch_key]
                require(
                    branch["arm"] == "FROZEN"
                    and branch["real_history_id"] == f"s{sample_id}_history_mpc{mpc_iter}"
                    and branch["adaptation_buffer_ids"]
                    == [f"s{sample_id}_executed_mpc{i}" for i in range(mpc_iter)]
                    and branch["action_unit_version"] == "pushobj_norm_envapi_v1"
                    and all(int(branch[key]) == 0 for key in
                            ("model_version", "encoder_version", "predictor_version", "objective_version")),
                    f"Frozen branch identity/version mismatch: {anchor_id}:{candidate_id}",
                )
                score_key = (anchor_id, candidate_id)
                require(score_key in frozen_scores, f"Missing score: {score_key}")
                score = frozen_scores[score_key]
                require(
                    abs(float(branch["c_env_ref"]) - float(score["c_env_ref"])) < 1e-9,
                    f"B score/branch truth mismatch: {score_key}",
                )
                require(
                    branch["objective_stage"] == score["objective_stage"],
                    f"Objective stage mismatch: {score_key}",
                )
                expected_outcome = (
                    donor
                    / "oracle"
                    / f"sample_{sample_id:03d}"
                    / f"s{sample_id}_m{mpc_iter}_g{gd_iter}_{side}_outcome.npz"
                ).resolve()
                outcome = Path(branch["sidecar"]).resolve()
                require(
                    outcome == expected_outcome,
                    f"Outcome path does not match candidate identity: {score_key}: {outcome}",
                )
                check_outcome(outcome, current)
                candidate_rows.append(
                    {
                        "anchor_id": anchor_id,
                        "candidate_id": candidate_id,
                        "action_tensor": str(tensor_path),
                        "outcome_sidecar": str(outcome),
                        "c_env_ref": float(score["c_env_ref"]),
                        "c_env_ref_visual": float(branch["c_env_ref_visual"]),
                        "c_env_ref_proprio": float(branch["c_env_ref_proprio"]),
                        "objective_stage": branch["objective_stage"],
                        "reference_encoder_version": int(branch["encoder_version"]),
                    }
                )
    require(len(candidate_rows) == 120, "Incomplete B candidate outcome pool")
    case_counts = Counter(row["case_id"] for row in query_rows)
    require(
        len(case_counts) == 6 and set(case_counts.values()) == {2}
        and all({row["query_mpc_iter"] for row in query_rows if row["case_id"] == case_id} == {2, 4}
                for case_id in case_counts),
        "B case/time grouping mismatch",
    )
    return query_rows, candidate_rows


def audit_c(repo: Path, queries: list[dict]) -> tuple[dict, list[dict]]:
    from research.reframe_v3.common_continuation import (
        _logical_manifest,
        continuation_seed,
    )

    base = repo / "artifacts" / "reframe_v3" / "common_continuation"
    config = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
    b_config = json.loads(
        (repo / "artifacts" / "reframe_v3" / "common_anchors_v2" / "run_config.json")
        .read_text(encoding="utf-8")
    )
    require(
        Path(config["checkpoint_dir"]).resolve()
        == (repo / b_config["checkpoint_dir"]).resolve(),
        "C checkpoint differs from B frozen checkpoint",
    )
    require(
        config["logical_branches"] == 18
        and config["physical_branches"] == 17
        and config["tail_chunks"] == 4
        and config["warmstart_rule"] == "zero_environment_action_normalized_by_preprocessor",
        "C continuation configuration mismatch",
    )
    donors = {split: Path(path).resolve() for split, path in config["donors"].items()}
    require(
        donors == {split: (repo / path).resolve() for split, path in b_config["donors"]},
        "C donors differ from B frozen donors",
    )
    selection = read_csv(repo / "artifacts" / "reframe_v3" / "common_anchors_v2" / "selection_metrics.csv")
    expected_logical, expected_physical = _logical_manifest(selection, donors, 4)
    require(len(expected_logical) == 18 and len(expected_physical) == 17, "C regenerated branch count")
    b_m4_anchors = {row["anchor_id"] for row in queries if row["query_mpc_iter"] == 4}
    require(
        {row["anchor_id"] for row in expected_logical} == b_m4_anchors,
        "C anchors differ from B mpc4 query set",
    )
    pre_rows = read_csv(base / "pre_execution_manifest.csv")
    logical_rows = read_csv(base / "logical_branch_results.csv")
    require(len(pre_rows) == len(logical_rows) == 18, "C logical manifest/result count")
    expected_by_key = {(row["anchor_id"], row["label"]): row for row in expected_logical}
    for source, name in ((pre_rows, "pre-execution"), (logical_rows, "logical result")):
        require(len({(row["anchor_id"], row["label"]) for row in source}) == 18,
                f"Duplicate C {name} row")
        for row in source:
            key = (row["anchor_id"], row["label"])
            require(key in expected_by_key, f"Unknown C {name} label: {key}")
            expected = expected_by_key[key]
            require(
                row["physical_branch_id"] == expected["physical_branch_id"]
                and row["candidate_id"] == expected["candidate_id"]
                and abs(float(row["recorded_pool_env_cost"])
                        - float(expected["recorded_pool_env_cost"])) < 1e-9,
                f"C {name} differs from frozen B selection: {key}",
            )

    rows = read_csv(base / "physical_branch_results.csv")
    require(len(rows) == 17, f"Expected 17 C physical branches, found {len(rows)}")
    by_id = {row["physical_branch_id"]: row for row in rows}
    require(len(by_id) == 17, "Duplicate C physical branch")
    require(set(by_id) == {spec["physical_branch_id"] for spec in expected_physical},
            "C physical IDs differ from regenerated manifest")
    physical_contract = []
    for spec in expected_physical:
        physical_id = spec["physical_branch_id"]
        row = by_id[physical_id]
        require(
            row["anchor_id"] == spec["anchor_id"]
            and row["labels"].split(";") == spec["labels"]
            and json.loads(row["candidate_ids"]) == spec["candidate_ids"]
            and int(row["mpc_iter"]) == 4,
            f"C physical label/candidate mapping mismatch: {physical_id}",
        )
        require(
            int(row["prefix_replay_environment_steps"]) == 20
            and int(row["first_decision_environment_steps"]) == 5
            and int(row["tail_environment_steps"]) == 20
            and int(row["total_environment_steps"]) == 45
            and int(row["tail_replans"]) == 4
            and int(row["tail_gd_steps"]) == 400,
            f"C branch execution contract mismatch: {physical_id}",
        )
        expected_path = (base / "trajectories" / f"{physical_id}.npz").resolve()
        path = (repo / row["trajectory_sidecar"]).resolve()
        require(path == expected_path, f"C trajectory path mismatch: {physical_id}")
        donor = donors[spec["split"]]
        history = torch.load(
            donor / "real_evidence" / f"s{spec['sample_id']}_history_mpc4.pt",
            map_location="cpu",
        )
        check_outcome_c(
            path,
            history["current_real_observation"],
            history["executed_prefix"],
            spec["first_chunk"],
            [continuation_seed(config["base_seed"], spec["anchor_ordinal"], i)
             for i in range(4)],
        )
        physical_contract.append(
            {
                "physical_branch_id": physical_id,
                "anchor_id": spec["anchor_id"],
                "labels": row["labels"],
                "candidate_ids": row["candidate_ids"],
                "trajectory_sidecar": str(path),
                "oracle_for_training": False,
            }
        )
    return {"physical_branches": 17, "logical_branches": 18, "anchors": 6,
            "oracle_use": "diagnosis_only"}, physical_contract


def check_outcome_c(
    path: Path, current: dict, prefix: torch.Tensor, first_chunk: torch.Tensor,
    planner_seeds: list[int],
) -> None:
    require(path.is_file(), f"Missing C trajectory: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(archive["visual"].shape == (6, 224, 224, 3), f"C visual shape: {path}")
        require(archive["proprio"].shape == (6, 4), f"C proprio shape: {path}")
        require(archive["executed_model_actions"].shape == (1, 5, 10), f"C action shape: {path}")
        require(
            np.array_equal(archive["visual"][0], current["visual"][0, 0])
            and np.array_equal(archive["proprio"][0], current["proprio"][0, 0])
            and np.array_equal(archive["prefix_model_actions"], prefix)
            and np.array_equal(archive["executed_model_actions"][:, :1], first_chunk)
            and np.array_equal(archive["planner_seeds"], planner_seeds),
            f"C trajectory/prefix/action/seed binding mismatch: {path}",
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "RUN_CONFIG.json").write_text(
        json.dumps({"repo": str(repo), "mode": "source_and_time_contract", "environment_calls": 0,
                    "model_calls": 0, "training_steps": 0, "hash_checks": 0}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        queries, candidates = audit_b(repo)
        c_contract, c_rows = audit_c(repo, queries)
        result = {
            "passed": False,
            "status": "PARTIAL_SOURCE_AND_TIME_CONTRACT",
            "source_and_time_contract_passed": True,
            "b_anchors": len(queries),
            "b_cases": len({row["case_id"] for row in queries}),
            "b_frozen_candidate_outcomes": len(candidates),
            "c": c_contract,
            "b_truth_encoding": "archived_six_frame_batch_not_yet_recomputed",
            "b_truth_encoding_passed": False,
            "r3_truth_encoding": "distinct_one_frame_contract",
            "action_outcome_binding": "archived_branch_record_and_exact_filename_only; outcome sidecars contain no action trace",
            "source_tests": "geometry_contract_and_check_processing_not_found_locally",
            "training_use_of_oracle": False,
            "new_environment_calls": 0,
            "model_calls": 0,
            "training_steps": 0,
        }
        write_csv(out / "QUERY_CUTOFFS.csv", queries)
        write_csv(
            out / "CANDIDATE_ACTIONS.csv",
            [{key: row[key] for key in ("anchor_id", "candidate_id", "action_tensor")}
             for row in candidates],
        )
        evaluation_only = out / "evaluation_only"
        evaluation_only.mkdir()
        write_csv(
            evaluation_only / "B_TRUTH.csv",
            [{key: row[key] for key in
              ("anchor_id", "candidate_id", "outcome_sidecar", "c_env_ref",
               "c_env_ref_visual", "c_env_ref_proprio", "objective_stage",
               "reference_encoder_version")}
             for row in candidates],
        )
        write_csv(evaluation_only / "C_PHYSICAL_CONTRACT.csv", c_rows)
        (out / "INPUT_CONTRACT.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        (out / "FAILURE.json").write_text(
            json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
