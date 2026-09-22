"""Optional bounded G1 hybrid: distinguish free-input and feedback-action terms.

Uses frozen F on saved real C latents with saved model-feedback actions. These
oracle-side vectors are evaluation-only and are not a deployment update buffer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from research.reframe_v3.matched_feedback_forecast import rollout_from_zobs
from research.reframe_v3.round4_effect_geometry import dot, w_vector
from research.reframe_v3.shadow_selection_audit import _load_runtime
from research.reframe_v3.common_continuation import _tensor_versions


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists() or not rows:
        raise ValueError(f"output exists or empty: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_terms(common: dict, input_term: np.ndarray, action_term: np.ndarray,
                 residual: np.ndarray, observed: np.ndarray) -> dict:
    terms = (input_term, action_term, residual)
    if not all(np.isfinite(term).all() for term in (*terms, observed)):
        raise ValueError(f"nonfinite hybrid term: {common}")
    reconstructed = input_term + action_term + residual
    result = {
        **common,
        "input_term_norm_sq": dot(input_term, input_term),
        "action_term_norm_sq": dot(action_term, action_term),
        "residual_term_norm_sq": dot(residual, residual),
        "input_action_inner": dot(input_term, action_term),
        "input_residual_inner": dot(input_term, residual),
        "action_residual_inner": dot(action_term, residual),
        "observed_error_norm_sq": dot(observed, observed),
        "closure_max_abs": float(np.max(np.abs(observed - reconstructed))),
        "squared_norm_closure": dot(observed, observed)
        - (dot(input_term, input_term) + dot(action_term, action_term)
           + dot(residual, residual) + 2 * (dot(input_term, action_term)
           + dot(input_term, residual) + dot(action_term, residual))),
    }
    if not all(math.isfinite(value) for key, value in result.items() if key not in common):
        raise ValueError(f"nonfinite hybrid derived metric: {common}")
    return result


def run(args) -> dict:
    staging = args.out.with_name(args.out.name + "_staging")
    if args.out.exists() or staging.exists():
        raise FileExistsError(args.out)
    propagation = json.loads(args.propagation_summary.read_text(encoding="utf-8"))
    b_config = json.loads(args.b_config.read_text(encoding="utf-8"))
    binding = json.loads(args.source_binding.read_text(encoding="utf-8"))
    repo = args.b_config.resolve().parents[3]
    loaded_checkpoint_line = (
        "Resuming from epoch 3: "
        + str(args.checkpoint_dir.resolve() / "checkpoints" / "model_latest.pth")
    )
    execution_log = Path(binding["execution_log"]).read_text(encoding="utf-8")
    vector_marker = '"vector_output_path": ' + json.dumps(str(args.vectors.resolve()))
    if (not propagation.get("validation", {}).get("passed")
            or propagation.get("vector_sidecars") != 17
            or Path(propagation["vector_output_path"]).resolve() != args.vectors.resolve()
            or (repo / b_config["checkpoint_dir"]).resolve() != args.checkpoint_dir.resolve()
            or Path(binding["propagation_summary"]).resolve() != args.propagation_summary.resolve()
            or Path(binding["vector_directory"]).resolve() != args.vectors.resolve()
            or Path(binding["loaded_checkpoint_directory"]).resolve() != args.checkpoint_dir.resolve()
            or loaded_checkpoint_line not in execution_log
            or vector_marker not in execution_log
            or '"model_transitions": 255' not in execution_log):
        raise ValueError("G1 vector/checkpoint source contract is not valid")
    with args.c_physical.open(encoding="utf-8", newline="") as handle:
        c_rows = list(csv.DictReader(handle))
    c_identity = {row["physical_branch_id"]: (row["anchor_id"], row["labels"])
                  for row in c_rows}
    if len(c_rows) != 17 or len(c_identity) != 17:
        raise ValueError("C physical identity manifest must have 17 unique rows")
    paths = sorted(args.vectors.glob("*.pt"))
    if len(paths) != 17:
        raise ValueError(f"expected 17 physical vectors, got {len(paths)}")

    started = time.perf_counter()
    device = torch.device(args.device)
    model, _, _ = _load_runtime(args.checkpoint_dir, device)
    versions_before = _tensor_versions(model)
    predictor_calls = [0]

    def count_call(_module, inputs, _output):
        if inputs[0].shape[0] != 1:
            raise ValueError("hybrid batch expansion")
        predictor_calls[0] += 1

    hook = model.predictor.register_forward_hook(count_call)
    groups = defaultdict(list)
    branch_rows = []
    for path in paths:
        item = torch.load(path, map_location="cpu")
        if (item["physical_branch_id"] != path.stem
                or c_identity.get(path.stem) != (item["anchor_id"], item["labels"])):
            raise ValueError(f"hybrid source identity mismatch: {path}")
        real, free, reset = (item[f"{name}_z"] for name in ("real", "free", "reset"))
        feedback, actual = item["feedback_actions"], item["actual_actions"]
        if (tuple(feedback.shape) != (1, 5, 10)
                or tuple(actual.shape) != (1, 5, 10)
                or not torch.equal(feedback[:, :1], actual[:, :1])):
            raise ValueError(f"hybrid action contract failed: {path.stem}")
        terms = {}
        for depth in range(2, 6):
            time_index = depth - 1
            real_input = {key: value[:, time_index : time_index + 1].to(device)
                          for key, value in real.items()}
            action = feedback[:, time_index : time_index + 1].to(device)
            with torch.no_grad():
                predicted, _ = rollout_from_zobs(model, real_input, action)
            hybrid = {key: value[:, -1:].detach().cpu() for key, value in predicted.items()}
            hybrid_w = w_vector(hybrid, 0)
            free_next = w_vector(free, depth)
            real_next = w_vector(real, depth)
            reset_next = w_vector(reset, depth)
            input_term = free_next - hybrid_w
            action_term = hybrid_w - reset_next
            residual = reset_next - real_next
            common = {"physical_branch_id": path.stem, "anchor_id": item["anchor_id"],
                      "labels": item["labels"], "depth": depth}
            branch_rows.append(record_terms(common, input_term, action_term,
                                            residual, free_next - real_next))
            terms[depth] = (input_term, action_term, residual, free_next - real_next)
        groups[item["anchor_id"]].append({"id": path.stem, "labels": item["labels"], "terms": terms})
    hook.remove()

    pair_rows = []
    for anchor_id, group in sorted(groups.items()):
        group.sort(key=lambda entry: entry["id"])
        for left, right in combinations(group, 2):
            labels = (set(left["labels"].split(";")), set(right["labels"].split(";")))
            legal = ("frozen" in labels[0] and "adapted" in labels[1]) or (
                "adapted" in labels[0] and "frozen" in labels[1])
            nonfrozen = ("frozen" in labels[0]) != ("frozen" in labels[1])
            for depth in range(2, 6):
                left_terms, right_terms = left["terms"][depth], right["terms"][depth]
                differences = tuple(a - b for a, b in zip(left_terms, right_terms))
                common = {"pair_id": f"{left['id']}__{right['id']}", "anchor_id": anchor_id,
                          "depth": depth, "frozen_nonfrozen": nonfrozen,
                          "frozen_adapted": legal}
                pair_rows.append(record_terms(common, *differences))
    if (len(groups) != 6 or len(branch_rows) != 17 * 4 or len(pair_rows) != 16 * 4
            or predictor_calls[0] != 68):
        raise ValueError("hybrid cardinality or model transition budget failed")
    final_pairs = [row for row in pair_rows if row["depth"] == 5]
    if (sum(row["frozen_nonfrozen"] for row in final_pairs) != 11
            or sum(row["frozen_adapted"] for row in final_pairs) != 6):
        raise ValueError("hybrid pair class cardinality failed")
    closures = [abs(float(row[key])) for row in branch_rows + pair_rows
                for key in ("closure_max_abs", "squared_norm_closure")]
    summary = {
        "stage": "G1_optional_ordered_hybrid_feedback_split",
        "interpretation": "ordered algebraic decomposition, not unique causal attribution",
        "anchors": len(groups), "physical_branches": len(paths), "physical_pairs": 16,
        "branch_depth_rows": len(branch_rows), "pair_depth_rows": len(pair_rows),
        "model_transitions": predictor_calls[0], "model_samples": predictor_calls[0],
        "new_environment_calls": 0, "new_GD_iterations": 0, "new_training_updates": 0,
        "model_parameters_and_buffers_unchanged": versions_before == _tensor_versions(model),
        "closure_max_abs": max(closures), "closure_tolerance": 1e-10,
        "wall_clock_s": time.perf_counter() - started,
    }
    summary["passed"] = bool(summary["model_parameters_and_buffers_unchanged"]
                             and summary["closure_max_abs"] <= summary["closure_tolerance"])
    staging.mkdir(parents=True)
    write_csv(staging / "FEEDBACK_SPLIT_BRANCH.csv", branch_rows)
    write_csv(staging / "FEEDBACK_SPLIT_PAIR.csv", pair_rows)
    (staging / "FEEDBACK_SPLIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary["passed"]:
        staging.rename(args.out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--propagation-summary", type=Path, required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--c-physical", type=Path, required=True)
    parser.add_argument("--b-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
