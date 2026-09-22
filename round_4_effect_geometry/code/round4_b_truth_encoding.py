"""Re-encode archived B outcomes with the original six-frame oracle semantics.

This is evaluation-only. No oracle outcome is passed to an adaptation routine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from research.contrast_probe import native_objective_breakdown
from research.reframe_v3.shadow_selection_audit import (
    GD_STEPS,
    SIDES,
    _load_anchor_observations,
    _load_runtime,
    _move_obs,
    _transform_anchor_obs,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    repo, contract_dir, out = args.repo.resolve(), args.contract_dir.resolve(), args.out.resolve()
    if not 1 <= args.limit <= 120:
        raise ValueError("--limit must be between 1 and 120")
    contract = json.loads((contract_dir / "INPUT_CONTRACT.json").read_text(encoding="utf-8"))
    if not contract["source_and_time_contract_passed"]:
        raise ValueError("Source/time contract has not passed")
    queries = {row["anchor_id"]: row for row in read_csv(contract_dir / "QUERY_CUTOFFS.csv")}
    truth_rows = read_csv(contract_dir / "evaluation_only" / "B_TRUTH.csv")
    action_rows = read_csv(contract_dir / "CANDIDATE_ACTIONS.csv")
    truth_keys = [(row["anchor_id"], row["candidate_id"]) for row in truth_rows]
    action_keys = [(row["anchor_id"], row["candidate_id"]) for row in action_rows]
    if (len(queries) != 12 or len(truth_rows) != 120
            or len(set(truth_keys)) != 120 or set(truth_keys) != set(action_keys)
            or len(action_keys) != 120):
        raise ValueError("Unexpected B query or truth pool size")
    expected_ids = {f"g{gd_iter}_{side}" for gd_iter in GD_STEPS for side in SIDES}
    for anchor_id in queries:
        if {candidate for anchor, candidate in truth_keys if anchor == anchor_id} != expected_ids:
            raise ValueError(f"Incomplete candidate IDs: {anchor_id}")
    for truth in truth_rows:
        query = queries[truth["anchor_id"]]
        if int(truth["reference_encoder_version"]) != 0:
            raise ValueError(f"B truth is not frozen-encoder reference: {truth['anchor_id']}")
        sample_id = int(truth["anchor_id"].split("|")[2].removeprefix("s"))
        mpc_iter = int(query["query_mpc_iter"])
        candidate_id = truth["candidate_id"]
        gd_iter, side = candidate_id.split("_", 1)
        expected = (
            Path(query["donor_dir"]) / "oracle" / f"sample_{sample_id:03d}"
            / f"s{sample_id}_m{mpc_iter}_{gd_iter}_{side}_outcome.npz"
        ).resolve()
        if Path(truth["outcome_sidecar"]).resolve() != expected:
            raise ValueError(f"B truth sidecar identity mismatch: {truth['anchor_id']}:{candidate_id}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA requested but unavailable: {device}")
    b_config = json.loads(
        (repo / "artifacts" / "reframe_v3" / "common_anchors_v2" / "run_config.json")
        .read_text(encoding="utf-8")
    )
    checkpoint_dir = (repo / b_config["checkpoint_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "repo": str(repo),
                "contract_dir": str(contract_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "device": str(device),
                "limit": args.limit,
                "semantics": "single_batch_six_model_frames_per_candidate",
                "tolerance": 1e-6,
                "environment_calls": 0,
                "training_steps": 0,
                "hash_checks": 0,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        model, preprocessor, config = _load_runtime(checkpoint_dir, device)
        if int(config.frameskip) != 5:
            raise ValueError("Checkpoint frameskip differs from archive")
        versions_before = tuple(tensor._version for tensor in
                                list(model.parameters()) + list(model.buffers()))
        goal_cache = {}
        goal_provenance = []
        results = []
        evaluation_only = out / "evaluation_only"
        latent_dir = evaluation_only / "B_TRUTH_LATENTS"
        latent_dir.mkdir(parents=True)
        goal_dir = evaluation_only / "B_GOAL_LATENTS"
        goal_dir.mkdir()
        with torch.no_grad():
            for truth in truth_rows[: args.limit]:
                anchor_id = truth["anchor_id"]
                query = queries[anchor_id]
                if anchor_id not in goal_cache:
                    donor = Path(query["donor_dir"])
                    sample_id = int(anchor_id.split("|")[2].removeprefix("s"))
                    current, goal = _load_anchor_observations(
                        donor, sample_id,
                        int(query["query_mpc_iter"]),
                    )
                    _, transformed_goal = _transform_anchor_obs(
                        preprocessor, current, goal, device
                    )
                    goal_cache[anchor_id] = model.encode_obs(transformed_goal)
                    if not all(torch.isfinite(block).all().item()
                               for block in goal_cache[anchor_id].values()):
                        raise ValueError(f"Non-finite goal latent: {anchor_id}")
                    goal_path = goal_dir / f"{anchor_id.replace('|', '__')}.npz"
                    np.savez_compressed(
                        goal_path,
                        visual=goal_cache[anchor_id]["visual"].detach().cpu().numpy(),
                        proprio=goal_cache[anchor_id]["proprio"].detach().cpu().numpy(),
                    )
                    goal_provenance.append(
                        {
                            "anchor_id": anchor_id,
                            "goal_evidence": str(donor / "real_evidence" / f"sample_{sample_id:03d}_initial.npz"),
                            "checkpoint_dir": str(checkpoint_dir),
                            "reference_encoder_version": 0,
                            "goal_latent": str(goal_path),
                        }
                    )
                outcome = Path(truth["outcome_sidecar"])
                with np.load(outcome, allow_pickle=False) as archive:
                    observed = {
                        "visual": np.asarray(archive["visual"])[np.newaxis],
                        "proprio": np.asarray(archive["proprio"])[np.newaxis],
                    }
                transformed = _move_obs(preprocessor.transform_obs(observed), device)
                encoded = model.encode_obs(transformed)
                if not all(torch.isfinite(block).all().item() for block in encoded.values()):
                    raise ValueError(f"Non-finite outcome latent: {anchor_id}:{truth['candidate_id']}")
                breakdown = native_objective_breakdown(
                    encoded, goal_cache[anchor_id], int(query["query_mpc_iter"]),
                    alpha=1.0, base=2.0,
                )
                computed = float(breakdown["total"].item())
                logged = float(truth["c_env_ref"])
                visual = float(breakdown["visual"].item())
                proprio = float(breakdown["proprio"].item())
                logged_visual = float(truth["c_env_ref_visual"])
                logged_proprio = float(truth["c_env_ref_proprio"])
                if not all(math.isfinite(value) for value in
                           (computed, logged, visual, proprio, logged_visual, logged_proprio)):
                    raise ValueError(f"Non-finite oracle cost: {anchor_id}:{truth['candidate_id']}")
                if breakdown["stage"] != truth["objective_stage"]:
                    raise ValueError(f"Oracle objective stage mismatch: {anchor_id}:{truth['candidate_id']}")
                relative = Path(anchor_id.replace("|", "__")) / f"{truth['candidate_id']}.npz"
                path = latent_dir / relative
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    path,
                    visual=encoded["visual"].detach().cpu().numpy(),
                    proprio=encoded["proprio"].detach().cpu().numpy(),
                )
                results.append(
                    {
                        "anchor_id": anchor_id,
                        "candidate_id": truth["candidate_id"],
                        "logged_c_env_ref": logged,
                        "reencoded_cost": computed,
                        "total_abs_error": abs(computed - logged),
                        "visual_abs_error": abs(visual - logged_visual),
                        "proprio_abs_error": abs(proprio - logged_proprio),
                        "objective_stage": breakdown["stage"],
                        "latent_sidecar": str(path),
                    }
                )
        versions_after = tuple(tensor._version for tensor in
                               list(model.parameters()) + list(model.buffers()))
        if versions_after != versions_before:
            raise AssertionError("Frozen model parameters or buffers changed")
        write_csv(evaluation_only / "REENCODED_COSTS.csv", results)
        write_csv(evaluation_only / "GOAL_PROVENANCE.csv", goal_provenance)
        max_error = max(
            row[key] for row in results
            for key in ("total_abs_error", "visual_abs_error", "proprio_abs_error")
        )
        complete = len(results) == 120
        numeric_pass = max_error <= 1e-6
        passed = complete and numeric_pass
        validation = {
            "passed": passed,
            "status": "FAIL" if not numeric_pass else ("PASS" if complete else "SANITY_PARTIAL"),
            "candidates_checked": len(results),
            "expected_candidates": 120,
            "max_abs_cost_error": max_error,
            "tolerance": 1e-6,
            "model_unchanged": True,
            "reference_encoder": "frozen_checkpoint_version_0",
            "goal_encoding": "same_frozen_checkpoint_from_archived_initial_goal_observation",
            "model_encode_calls": len(results) + len(goal_cache),
            "new_environment_calls": 0,
            "training_steps": 0,
            "wall_s": time.perf_counter() - started,
            "gpu_peak_bytes": (torch.cuda.max_memory_allocated(device)
                               if device.type == "cuda" else None),
        }
        (out / "TRUTH_ENCODING_VALIDATION.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
        if not numeric_pass:
            raise ValueError(f"B truth cost reproduction failed: max error {max_error}")
    except Exception as exc:
        (out / "FAILURE.json").write_text(
            json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
