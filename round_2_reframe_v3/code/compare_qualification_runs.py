"""Compare aligned instrumentation-OFF and instrumentation-ON qualification runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare_values(left, right, path: str, differences: list[str]) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            differences.append(f"{path}: tensor metadata differs")
            return False
        if not torch.equal(left, right):
            differences.append(f"{path}: tensor values differ")
            return False
        return True
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            differences.append(f"{path}: array metadata differs")
            return False
        if not np.array_equal(left, right):
            differences.append(f"{path}: array values differ")
            return False
        return True
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append(f"{path}: dictionary keys differ")
            return False
        return all(
            compare_values(left[key], right[key], f"{path}.{key}", differences)
            for key in sorted(left)
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            differences.append(f"{path}: sequence metadata differs")
            return False
        return all(
            compare_values(a, b, f"{path}[{index}]", differences)
            for index, (a, b) in enumerate(zip(left, right))
        )
    if type(left) is not type(right) or left != right:
        differences.append(f"{path}: {left!r} != {right!r}")
        return False
    return True


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> dict:
    baseline_actions = torch.load(
        baseline_dir / "baseline_actions.pt", map_location="cpu"
    )
    candidate_actions = torch.load(
        candidate_dir / "baseline_actions.pt", map_location="cpu"
    )
    action_differences: list[str] = []
    actions_exact = compare_values(
        baseline_actions["actions"],
        candidate_actions["actions"],
        "actions",
        action_differences,
    )
    action_lengths_exact = compare_values(
        np.asarray(baseline_actions["action_len"]),
        np.asarray(candidate_actions["action_len"]),
        "action_len",
        action_differences,
    )

    baseline_snapshot_dir = baseline_dir / "qualification_snapshots"
    candidate_snapshot_dir = candidate_dir / "qualification_snapshots"
    baseline_names = sorted(path.name for path in baseline_snapshot_dir.glob("*.pt"))
    candidate_names = sorted(path.name for path in candidate_snapshot_dir.glob("*.pt"))
    snapshot_names_exact = baseline_names == candidate_names and bool(baseline_names)
    snapshot_differences = []
    if not snapshot_names_exact:
        snapshot_differences.append(
            f"snapshot names differ: {baseline_names} != {candidate_names}"
        )

    parameters_and_buffers_exact = snapshot_names_exact
    experience_buffers_exact = snapshot_names_exact
    rng_exact = snapshot_names_exact
    snapshot_metadata_exact = snapshot_names_exact
    for name in sorted(set(baseline_names) & set(candidate_names)):
        baseline = torch.load(baseline_snapshot_dir / name, map_location="cpu")
        candidate = torch.load(candidate_snapshot_dir / name, map_location="cpu")
        parameters_and_buffers_exact &= compare_values(
            {
                key: baseline[key]
                for key in (
                    "adapted_predictor_params",
                    "adapted_encoder_params",
                    "predictor_buffers",
                    "encoder_buffers",
                )
            },
            {
                key: candidate[key]
                for key in (
                    "adapted_predictor_params",
                    "adapted_encoder_params",
                    "predictor_buffers",
                    "encoder_buffers",
                )
            },
            f"{name}.parameters_and_buffers",
            snapshot_differences,
        )
        experience_buffers_exact &= compare_values(
            {
                key: baseline[key]
                for key in ("obs_buffer", "act_buffer", "segment_scores")
            },
            {
                key: candidate[key]
                for key in ("obs_buffer", "act_buffer", "segment_scores")
            },
            f"{name}.experience_buffers",
            snapshot_differences,
        )
        rng_exact &= compare_values(
            baseline["rng"], candidate["rng"], f"{name}.rng", snapshot_differences
        )
        snapshot_metadata_exact &= compare_values(
            (baseline["sample_idx"], baseline["mpc_iter"]),
            (candidate["sample_idx"], candidate["mpc_iter"]),
            f"{name}.metadata",
            snapshot_differences,
        )

    report = {
        "baseline": str(baseline_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "snapshots_compared": len(baseline_names)
        if snapshot_names_exact
        else len(set(baseline_names) & set(candidate_names)),
        "snapshot_names_exact": snapshot_names_exact,
        "actions_exact": bool(actions_exact),
        "action_lengths_exact": bool(action_lengths_exact),
        "parameters_and_buffers_exact": bool(parameters_and_buffers_exact),
        "experience_buffers_exact": bool(experience_buffers_exact),
        "rng_exact": bool(rng_exact),
        "snapshot_metadata_exact": bool(snapshot_metadata_exact),
        "differences": (action_differences + snapshot_differences)[:100],
    }
    report["passed"] = all(
        report[key]
        for key in (
            "snapshot_names_exact",
            "actions_exact",
            "action_lengths_exact",
            "parameters_and_buffers_exact",
            "experience_buffers_exact",
            "rng_exact",
            "snapshot_metadata_exact",
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_runs(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
