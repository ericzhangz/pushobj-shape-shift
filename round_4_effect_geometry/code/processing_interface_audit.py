"""G2 fixed-action identity/nonidentity processing interface audit.

This is an equivalence test, not a learned method or a prediction-quality test.
No planner iteration, parameter update, or environment call is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from research.contrast_probe import native_objective_breakdown
from research.reframe_v3.common_continuation import _candidate_tensor, _rng_equal, _tensor_versions
from research.reframe_v3.matched_feedback_forecast import (
    PARITY_THRESHOLDS,
    _join_encoded_observation_action,
    rollout_from_zobs,
)
from research.reframe_v3.shadow_selection_audit import (
    _capture_rng_state,
    _jsonable,
    _load_anchor_observations,
    _load_runtime,
    _seed_all,
    _transform_anchor_obs,
)


SHEAR_COEFFICIENT = 0.01
INVERSE_TOL = 1e-7
NONIDENTITY_MARGIN = 10 * PARITY_THRESHOLDS["rollout_atol"]


def checked_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"interface tensor shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("nonfinite interface tensor")
    value = float((left.detach() - right.detach()).abs().max().item())
    if not math.isfinite(value):
        raise ValueError("nonfinite interface error")
    return value


def checked_zobs_max_abs(left: dict, right: dict) -> dict:
    if set(left) != set(right):
        raise ValueError("observation blocks differ")
    return {key: checked_max_abs(left[key], right[key]) for key in sorted(left)}


class ObservationProcessor:
    def __init__(self, coefficient: float):
        self.coefficient = coefficient

    def forward(self, z: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_shift = self.coefficient * z["proprio"][..., :1].unsqueeze(2)
        return {"visual": z["visual"] + visual_shift, "proprio": z["proprio"]}

    def inverse(self, z: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_shift = self.coefficient * z["proprio"][..., :1].unsqueeze(2)
        return {"visual": z["visual"] - visual_shift, "proprio": z["proprio"]}


def boundary_rollout(model, initial_z: dict, actions: torch.Tensor,
                     processor: ObservationProcessor) -> tuple[dict, torch.Tensor]:
    internal, full = rollout_from_zobs(model, processor.forward(initial_z), actions)
    return processor.inverse(internal), full


def stepwise_rollout(model, initial_z: dict, actions: torch.Tensor,
                     processor: ObservationProcessor) -> tuple[dict, torch.Tensor, float]:
    """Materialize P^-1 F_a P at each step while preserving native token cache."""
    if initial_z["visual"].shape[1] != 1 or actions.shape[1] != 5:
        raise ValueError("G2 audit expects one initial frame and five action chunks")
    internal_initial = processor.forward(initial_z)
    cache = _join_encoded_observation_action(model, internal_initial, actions[:, :1])
    external_frames = {key: [value] for key, value in initial_z.items()}
    replacement_max_abs = 0.0
    for action_index in range(5):
        predicted_token = model.predict(cache[:, -model.num_hist :])[:, -1:]
        predicted_obs, _ = model.separate_emb(predicted_token)
        external = processor.inverse(predicted_obs)
        relifted = processor.forward(external)
        for key in external_frames:
            external_frames[key].append(external[key])
        if action_index < 4:
            next_action = actions[:, action_index + 1 : action_index + 2]
            native_replacement = model.replace_actions_from_z(predicted_token.clone(), next_action)
            rebuilt = _join_encoded_observation_action(model, relifted, next_action)
            replacement_max_abs = max(replacement_max_abs, checked_max_abs(native_replacement, rebuilt))
            cache = torch.cat((cache, rebuilt), dim=1)
        else:
            cache = torch.cat((cache, predicted_token), dim=1)
    external_sequence = {key: torch.cat(frames, dim=1) for key, frames in external_frames.items()}
    return external_sequence, cache, replacement_max_abs


def external_cost(predicted: dict, goal: dict, mpc_iter: int) -> torch.Tensor:
    return native_objective_breakdown(predicted, goal, mpc_iter, alpha=1.0, base=2.0)["total"].sum()


def run(args) -> dict:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    repo = args.b_config.resolve().parents[3]
    g0 = json.loads(args.g0_final.read_text(encoding="utf-8"))
    source = json.loads((repo / g0["source_contract"]).read_text(encoding="utf-8"))
    truth = json.loads((repo / g0["truth_encoding_validation"]).read_text(encoding="utf-8"))
    b_config = json.loads(args.b_config.read_text(encoding="utf-8"))
    donor_paths = {split: (repo / path).resolve() for split, path in b_config["donors"]}
    donor_meta = json.loads((args.donor_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if (g0.get("status") != "PASS_DATA_AND_TRUTH_CONTRACT"
            or not source.get("source_and_time_contract_passed") or not truth.get("passed")
            or (repo / b_config["checkpoint_dir"]).resolve() != args.checkpoint_dir.resolve()
            or donor_paths.get("val_T") != args.donor_dir.resolve()
            or donor_meta.get("arm") != "FROZEN"
            or donor_meta.get("frameskip") != 5
            or donor_meta.get("model_action_dim") != 10
            or donor_meta.get("objective") != {"mode": "staged", "alpha": 1.0, "base": 2.0}):
        raise ValueError("G2 checkpoint/donor source differs from frozen G0 contract")
    started = time.perf_counter()
    device = torch.device(args.device)
    model, preprocessor, _ = _load_runtime(args.checkpoint_dir, device)
    versions_before = _tensor_versions(model)
    predictor_calls = [0]
    predictor_samples = [0]

    def count_predictor_call(_module, _inputs, _output):
        batch = int(_inputs[0].shape[0])
        if batch != 1:
            raise ValueError(f"G2 predictor batch expansion: {batch}")
        predictor_calls[0] += 1
        predictor_samples[0] += batch

    hook = model.predictor.register_forward_hook(count_predictor_call)
    identity = ObservationProcessor(0.0)
    shear = ObservationProcessor(SHEAR_COEFFICIENT)
    records = []

    for mpc_iter in (4, 5):
        current, goal = _load_anchor_observations(args.donor_dir, 0, mpc_iter)
        transformed_current, transformed_goal = _transform_anchor_obs(preprocessor, current, goal, device)
        candidate = _candidate_tensor(args.donor_dir, 0, mpc_iter, "g99_after").to(device)
        if tuple(candidate.shape) != (1, 5, 10):
            raise ValueError(f"fixed candidate shape at mpc{mpc_iter}")
        with torch.no_grad():
            z_current = model.encode_obs(transformed_current)
            z_goal = model.encode_obs(transformed_goal)
            inverse_error = max(checked_zobs_max_abs(shear.inverse(shear.forward(z_current)), z_current).values())
            nontrivial_change = checked_max_abs(shear.forward(z_current)["visual"], z_current["visual"])
        if inverse_error > INVERSE_TOL or nontrivial_change <= NONIDENTITY_MARGIN:
            raise ValueError(f"processor inverse/nontrivial check failed at mpc{mpc_iter}")

        fixed_seed = args.seed + mpc_iter
        _seed_all(fixed_seed)
        rng_native_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        with torch.no_grad():
            native, native_full = model.rollout(obs_0=transformed_current, act=candidate)
        native_forward_calls = predictor_calls[0] - previous_calls
        rng_native_after = _capture_rng_state()
        _seed_all(fixed_seed)
        rng_identity_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        with torch.no_grad():
            id_external, id_full = boundary_rollout(model, z_current, candidate, identity)
        identity_forward_calls = predictor_calls[0] - previous_calls
        rng_identity_after = _capture_rng_state()

        _seed_all(fixed_seed)
        rng_boundary_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        with torch.no_grad():
            boundary_external, boundary_full = boundary_rollout(model, z_current, candidate, shear)
        boundary_forward_calls = predictor_calls[0] - previous_calls
        rng_boundary_after = _capture_rng_state()
        _seed_all(fixed_seed)
        rng_stepwise_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        with torch.no_grad():
            stepwise_external, stepwise_full, action_cache_error = stepwise_rollout(
                model, z_current, candidate, shear)
        stepwise_forward_calls = predictor_calls[0] - previous_calls
        rng_stepwise_after = _capture_rng_state()

        actions_for_grad = [candidate.detach().clone().requires_grad_(True) for _ in range(4)]
        _seed_all(fixed_seed + 100)
        rng_native_grad_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        native_grad_out, _ = model.rollout(obs_0=transformed_current, act=actions_for_grad[0])
        native_grad_calls = predictor_calls[0] - previous_calls
        native_gradient = torch.autograd.grad(external_cost(native_grad_out, z_goal, mpc_iter),
                                              actions_for_grad[0])[0]
        rng_native_grad_after = _capture_rng_state()
        _seed_all(fixed_seed + 100)
        rng_identity_grad_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        identity_grad_out, _ = boundary_rollout(model, z_current, actions_for_grad[1], identity)
        identity_grad_calls = predictor_calls[0] - previous_calls
        identity_gradient = torch.autograd.grad(external_cost(identity_grad_out, z_goal, mpc_iter),
                                                actions_for_grad[1])[0]
        rng_identity_grad_after = _capture_rng_state()
        _seed_all(fixed_seed + 100)
        rng_boundary_grad_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        boundary_grad_out, _ = boundary_rollout(model, z_current, actions_for_grad[2], shear)
        boundary_grad_calls = predictor_calls[0] - previous_calls
        boundary_gradient = torch.autograd.grad(external_cost(boundary_grad_out, z_goal, mpc_iter),
                                                actions_for_grad[2])[0]
        rng_boundary_grad_after = _capture_rng_state()
        _seed_all(fixed_seed + 100)
        rng_stepwise_grad_before = _capture_rng_state()
        previous_calls = predictor_calls[0]
        stepwise_grad_out, _, _ = stepwise_rollout(model, z_current, actions_for_grad[3], shear)
        stepwise_grad_calls = predictor_calls[0] - previous_calls
        stepwise_gradient = torch.autograd.grad(external_cost(stepwise_grad_out, z_goal, mpc_iter),
                                                actions_for_grad[3])[0]
        rng_stepwise_grad_after = _capture_rng_state()

        id_output_error = max(checked_zobs_max_abs(native, id_external).values())
        id_full_error = checked_max_abs(native_full, id_full)
        id_cost_error = checked_max_abs(external_cost(native, z_goal, mpc_iter),
                                 external_cost(id_external, z_goal, mpc_iter))
        id_gradient_error = checked_max_abs(native_gradient, identity_gradient)
        step_output_error = max(checked_zobs_max_abs(boundary_external, stepwise_external).values())
        step_full_error = checked_max_abs(boundary_full, stepwise_full)
        step_cost_error = checked_max_abs(external_cost(boundary_external, z_goal, mpc_iter),
                                   external_cost(stepwise_external, z_goal, mpc_iter))
        step_gradient_error = checked_max_abs(boundary_gradient, stepwise_gradient)
        processed_internal_change = checked_max_abs(boundary_full, native_full)
        internal_boundary_action = model.separate_emb(boundary_full)[1]
        internal_stepwise_action = model.separate_emb(stepwise_full)[1]
        action_channel_error = checked_max_abs(internal_boundary_action, internal_stepwise_action)
        record = {
            "sample_id": 0, "mpc_iter": mpc_iter, "candidate_id": "g99_after",
            "seed": fixed_seed, "objective_stage": native_objective_breakdown(
                native, z_goal, mpc_iter, alpha=1.0, base=2.0)["stage"],
            "processor": f"visual += {SHEAR_COEFFICIENT} * proprio_first_coordinate; proprio unchanged",
            "processor_input_change_max_abs": nontrivial_change,
            "processed_internal_change_max_abs": processed_internal_change,
            "processor_inverse_max_abs": inverse_error,
            "identity_output_max_abs": id_output_error,
            "identity_full_cache_max_abs": id_full_error,
            "identity_cost_abs": id_cost_error,
            "identity_action_gradient_max_abs": id_gradient_error,
            "nonidentity_stepwise_output_max_abs": step_output_error,
            "nonidentity_stepwise_full_cache_max_abs": step_full_error,
            "nonidentity_cost_abs": step_cost_error,
            "nonidentity_action_gradient_max_abs": step_gradient_error,
            "nonidentity_action_replacement_max_abs": action_cache_error,
            "nonidentity_action_channel_max_abs": action_channel_error,
            "identity_rng_before_exact": _rng_equal(rng_native_before, rng_identity_before),
            "identity_rng_after_exact": _rng_equal(rng_native_after, rng_identity_after),
            "nonidentity_rng_before_exact": _rng_equal(rng_boundary_before, rng_stepwise_before),
            "nonidentity_rng_after_exact": _rng_equal(rng_boundary_after, rng_stepwise_after),
            "identity_gradient_rng_before_exact": _rng_equal(rng_native_grad_before, rng_identity_grad_before),
            "identity_gradient_rng_after_exact": _rng_equal(rng_native_grad_after, rng_identity_grad_after),
            "nonidentity_gradient_rng_before_exact": _rng_equal(rng_boundary_grad_before, rng_stepwise_grad_before),
            "nonidentity_gradient_rng_after_exact": _rng_equal(rng_boundary_grad_after, rng_stepwise_grad_after),
            "predictor_calls": {
                "native_forward": native_forward_calls, "identity_forward": identity_forward_calls,
                "nonidentity_boundary_forward": boundary_forward_calls,
                "nonidentity_stepwise_forward": stepwise_forward_calls,
                "native_gradient": native_grad_calls, "identity_gradient": identity_grad_calls,
                "nonidentity_boundary_gradient": boundary_grad_calls,
                "nonidentity_stepwise_gradient": stepwise_grad_calls,
            },
        }
        record["passed"] = bool(
            id_output_error <= PARITY_THRESHOLDS["rollout_atol"]
            and id_full_error <= PARITY_THRESHOLDS["rollout_atol"]
            and id_cost_error <= PARITY_THRESHOLDS["cost_atol"]
            and id_gradient_error <= PARITY_THRESHOLDS["gradient_atol"]
            and max(step_output_error, step_full_error, action_cache_error,
                    action_channel_error) <= PARITY_THRESHOLDS["rollout_atol"]
            and step_cost_error <= PARITY_THRESHOLDS["cost_atol"]
            and step_gradient_error <= PARITY_THRESHOLDS["gradient_atol"]
            and processed_internal_change > NONIDENTITY_MARGIN
            and all(record[key] for key in (
                "identity_rng_before_exact", "identity_rng_after_exact",
                "nonidentity_rng_before_exact", "nonidentity_rng_after_exact",
                "identity_gradient_rng_before_exact", "identity_gradient_rng_after_exact",
                "nonidentity_gradient_rng_before_exact", "nonidentity_gradient_rng_after_exact"))
            and all(calls == 5 for calls in record["predictor_calls"].values())
        )
        records.append(record)

    hook.remove()
    result = {
        "stage": "G2_fixed_action_processing_interface", "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "donor_dir": str(args.donor_dir.resolve()), "device": str(device),
        "source_contract": str(args.g0_final.resolve()),
        "thresholds": PARITY_THRESHOLDS, "inverse_tolerance": INVERSE_TOL,
        "nonidentity_margin": NONIDENTITY_MARGIN,
        "threshold_source": "Round3 predeclared FP32 interface tolerances; not retuned",
        "observations": records, "full_100_step_GD_action_parity": "NOT_RUN",
        "new_GD_iterations": 0, "new_environment_calls": 0, "new_training_updates": 0,
        "native_predict_calls_for_two_fixed_action_forward_and_grad": 20,
        "boundary_processed_predict_calls_for_two_fixed_action_forward_and_grad": 20,
        "observed_predictor_calls_all_eight_paths_two_observations": predictor_calls[0],
        "observed_predictor_samples_all_eight_paths_two_observations": predictor_samples[0],
        "model_parameters_and_buffers_unchanged": versions_before == _tensor_versions(model),
        "wall_clock_s": time.perf_counter() - started,
    }
    result["passed"] = bool(len(records) == 2 and all(row["passed"] for row in records)
                            and predictor_calls[0] == 80
                            and predictor_samples[0] == 80
                            and result["model_parameters_and_buffers_unchanged"])
    output = args.out / "PROCESSING_INTERFACE_REPORT.json"
    output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--donor-dir", type=Path, required=True)
    parser.add_argument("--g0-final", type=Path, required=True)
    parser.add_argument("--b-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=290920)
    result = run(parser.parse_args())
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
