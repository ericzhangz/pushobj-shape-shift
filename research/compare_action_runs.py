"""Strict tensor comparison for baseline/instrumentation regression runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = torch.load(args.baseline, map_location="cpu")
    candidate = torch.load(args.candidate, map_location="cpu")
    actions_a = baseline["actions"]
    actions_b = candidate["actions"]
    same_shape = tuple(actions_a.shape) == tuple(actions_b.shape)
    exact = same_shape and torch.equal(actions_a, actions_b)
    max_abs_error = (
        float(torch.max(torch.abs(actions_a - actions_b))) if same_shape else None
    )
    lengths_equal = np.array_equal(
        np.asarray(baseline["action_len"]), np.asarray(candidate["action_len"])
    )
    result = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "baseline_shape": list(actions_a.shape),
        "candidate_shape": list(actions_b.shape),
        "same_shape": same_shape,
        "actions_exactly_equal": exact,
        "action_max_abs_error": max_abs_error,
        "action_lengths_equal": lengths_equal,
        "passed": bool(exact and lengths_equal),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

