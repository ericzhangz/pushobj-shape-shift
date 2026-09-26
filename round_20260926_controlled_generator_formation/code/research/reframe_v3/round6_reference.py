"""Round-6 development reference: shared native-predictor revisions only.

Reads completed real evidence and captured candidate tensors; never reads
counterfactual environment costs.  All output scores are sealed before a
separate evaluator may inspect candidate consequences.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from planning.adajepa import AdaJEPATrainer
from research.contrast_probe import native_objective_breakdown
from research.reframe_v3.common_continuation import _build_planner
from research.reframe_v3.matched_feedback_forecast import (
    _join_encoded_observation_action, plan_from_zobs,
)
from research.reframe_v3.finite_task_reference import (
    completed_path_loss, make_functionals, task_reference_loss,
)
from research.reframe_v3.shadow_selection_audit import (
    _load_anchor_observations, _load_runtime, _load_segments,
    _transform_anchor_obs, _seed_all,
)


ARMS = ("FROZEN", "OFFICIAL-PRED", "FACT-MATCHED", "TASK-LINEAR", "TASK-FINITE")
COMPOSITION_ARMS = ("FROZEN", "FACT-MATCHED", "FACT-SUBPATH-TEACHER",
                    "FACT-SUBPATH-COMPOSED", "TASK-SUBPATH-TEACHER",
                    "TASK-SUBPATH-COMPOSED")
INCREMENT_KINDS = {"INCREMENT-MEAN": "mean_residual",
                   "INCREMENT-SCALAR": "scalar_gain",
                   "INCREMENT-SECANT": "multisecant"}
INPUT_KINDS = {f"INPUT-{label}-{steps}": (kind, steps)
               for steps in (20, 100)
               for label, kind in (("BIAS", "constant"), ("FIBER", "fiber"))}
FROZEN_NODES = ((0, "before"), (24, "after"), (49, "after"), (99, "after"))


def _write_csv(path: Path, rows: list[dict]):
    if not rows or path.exists():
        raise FileExistsError(f"missing rows or output already exists: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _native_candidates(donor: Path, sample: int, mpc: int):
    result = []
    for gd_step, side in FROZEN_NODES:
        path = donor / "tensors" / f"s{sample}_m{mpc}_g{gd_step}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        tensor = torch.load(path, map_location="cpu")[f"u_{side}"].detach().clone()
        if tuple(tensor.shape) != (1, 5, 10) or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid complete action chain: {path}:{side}")
        result.append((f"g{gd_step}_{side}", tensor, str(path), f"u_{side}"))
    return result


def _sealed_candidates(source: Path, args):
    """Reuse only sealed proposals, without opening their environment outcomes."""
    contract = json.loads((source / "RUN_CONTRACT.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    if (summary["status"] != "PREDICTIONS_SEALED_NO_ORACLE_READ"
            or summary["environment_branches"] != 0 or contract["oracle_truth_read"]):
        raise ValueError("candidate source was not sealed independently of truth")
    for key in ("split", "sample", "mpc"):
        if contract[key] != getattr(args, key):
            raise ValueError(f"candidate source changed {key}")
    if Path(contract["donor"]).resolve() != args.donor.resolve():
        raise ValueError("candidate source changed donor")
    with (source / "PREDICTIONS.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["arm"] == "FROZEN"]
    if len(rows) != 8 or len({row["candidate_id"] for row in rows}) != 8:
        raise ValueError("expected eight unique sealed candidates")
    result = []
    for row in rows:
        path, key = Path(row["candidate_tensor"]), row["tensor_key"]
        action = torch.load(path, map_location="cpu")[key].detach().clone()
        if tuple(action.shape) != (1, 5, 10) or not torch.isfinite(action).all():
            raise ValueError(f"invalid sealed action chain: {path}")
        result.append((row["candidate_id"], action, str(path), key))
    return result


@torch.no_grad()
def _factual_support_loss(model, trainer, merged_observations, merged_actions):
    padded = torch.cat((merged_actions, torch.zeros_like(merged_actions[:, :1])), dim=1)
    encoded = model.encode(merged_observations, padded)
    return float(trainer._prediction_loss(encoded).detach())


def _encoded_factual_loss(model, trainer, encoded_observations, actions):
    padded = torch.cat((actions, torch.zeros_like(actions[:, :1])), dim=1)
    joined = _join_encoded_observation_action(model, encoded_observations, padded)
    return trainer._prediction_loss(joined)


def _score_pool(model, current, goal, candidates, mpc, path_sink=None):
    model.eval()
    with torch.no_grad():
        goal_z = model.encode_obs(goal)
        scores = []
        for candidate_id, cpu_action, path, tensor_key in candidates:
            action = cpu_action.to(next(model.parameters()).device)
            path_z, _ = model.rollout(obs_0=current, act=action)
            if path_sink is not None:
                path_sink[candidate_id] = {key: value.detach().cpu().clone()
                                           for key, value in path_z.items()}
            result = native_objective_breakdown(path_z, goal_z, mpc, alpha=1., base=2.)
            if result["stage"] != "terminal":
                raise AssertionError("MPC2/MPC4 must use the native terminal stage")
            if any(not torch.isfinite(result[key]).all() for key in ("total", "visual", "proprio")):
                raise FloatingPointError(f"nonfinite candidate cost before prediction seal: {candidate_id}")
            scores.append({"candidate_id": candidate_id,
                           "candidate_tensor": path, "tensor_key": tensor_key,
                           "c_model": float(result["total"].item()),
                           "visual": float(result["visual"].item()),
                           "proprio": float(result["proprio"].item())})
        return scores


def _adapted_return_plan(model, preprocessor, frameskip, current, goal,
                         warm_start, mpc, seed):
    """Native GD mirror, 100 steps, same fixed Frozen warm-start for both facts."""
    planner = _build_planner(model, preprocessor, frameskip)
    model.eval()
    with torch.no_grad():
        encoded_current = model.encode_obs(current)
        encoded_goal = model.encode_obs(goal)
    _seed_all(seed)
    actions, _ = plan_from_zobs(planner, encoded_current, encoded_goal,
                                actions=warm_start.detach().clone(), step=mpc)
    result = actions.detach().cpu().clone()
    if tuple(result.shape) != (1, 5, 10) or not torch.isfinite(result).all():
        raise ValueError("adapted proposal did not return a finite H5 native action chain")
    return result


def _local_hub_loader():
    cache = Path("D:/EV-TTT/adajepa_runtime/torch/hub/facebookresearch_dinov2_main")
    weights = cache.parent / "checkpoints/dinov2_vits14_pretrain.pth"
    if not (cache / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError("local DINO code and weights required; network download disabled")
    original = torch.hub.load

    def load(repo, name, *args, **kwargs):
        if (repo, name) != ("facebookresearch/dinov2", "dinov2_vits14"):
            raise ValueError("unexpected remote hub dependency")
        return original(str(cache), name, *args, source="local", **kwargs)
    return load


def run(args):
    if args.arm_set in ("actuator", "innovation"):
        from research.reframe_v3.actuator_controls import run_actuator_controls
        return run_actuator_controls(args)
    arms = {"round6": ARMS, "composition": COMPOSITION_ARMS,
            "increment": ("FROZEN", *INCREMENT_KINDS),
            "input": ("FROZEN", "INCREMENT-MEAN", *INPUT_KINDS)}[args.arm_set]
    if args.arm_set != "round6" and args.pool_source is None:
        raise ValueError("new controls require the already sealed common pool")
    if (args.mpc not in (2, 4) or args.steps_per_event < 1
            or args.matched_lr <= 0 or args.task_weight_multiplier <= 0):
        raise ValueError("query must be MPC2/MPC4 and matched updates must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite Round-6 output: {args.out}")
    for required in (args.donor / "real_evidence", args.donor / "tensors"):
        if not required.is_dir():
            raise FileNotFoundError(required)
    _seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    started = time.perf_counter()
    with patch.object(torch.hub, "load", _local_hub_loader()):
        student, preprocessor, config = _load_runtime(args.checkpoint_dir, device)
    reference = copy.deepcopy(student)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainer = AdaJEPATrainer(student, lr=5e-4, steps=1,
                            optimizer_name="adam", finetune_encoder=False,
                            last_layer_only=True)
    selected = trainer._ada_predictor_params
    initial = [value.detach().clone() for value in selected]
    frameskip = int(config.frameskip)
    segments = _load_segments(args.donor, args.sample, args.mpc, frameskip, preprocessor)
    for left, right in zip(segments, segments[1:]):
        if not all(torch.equal(left[0][key][:, -1:], right[0][key][:, :1])
                   for key in left[0]):
            raise AssertionError("real chunk boundary observations differ")
    full_obs, full_actions = AdaJEPATrainer._merge_segments(
        [segment[0] for segment in segments], [segment[1] for segment in segments])
    full_obs = {key: value.to(device) for key, value in full_obs[0].items()}
    full_actions = full_actions[0].to(device)
    with torch.no_grad():
        encoded = student.encode_obs(full_obs)
    current_raw, goal_raw = _load_anchor_observations(args.donor, args.sample, args.mpc)
    current, goal = _transform_anchor_obs(preprocessor, current_raw, goal_raw, device)
    with torch.no_grad():
        goal_z = student.encode_obs(goal)
    candidates = (_native_candidates(args.donor, args.sample, args.mpc)
                  if args.pool_source is None else _sealed_candidates(args.pool_source, args))

    # Scale both task arms using only the initial checkpoint and completed D.
    # Each arm retains precisely the same native factual base term.
    event_tests = {}
    for event in range(1, args.mpc + 1):
        prefix = {key: value[:, :event+1] for key, value in encoded.items()}
        prefix_actions = full_actions[:, :event]
        if args.arm_set != "round6":
            event_tests[event] = {"prefix": prefix, "actions": prefix_actions}
            continue
        finite_tests = make_functionals(reference, prefix, prefix_actions,
                                        goal_z, linear=False)
        linear_tests = make_functionals(reference, prefix, prefix_actions,
                                        goal_z, linear=True)
        with torch.no_grad():
            base_fact = float(_encoded_factual_loss(reference, trainer,
                                                    prefix, prefix_actions))
            base_fin = float(task_reference_loss(reference, reference, prefix,
                                                 prefix_actions, goal_z,
                                                 finite_tests, linear=False))
            base_lin = float(task_reference_loss(reference, reference, prefix,
                                                 prefix_actions, goal_z,
                                                 linear_tests, linear=True))
        denominator = (base_fin + base_lin) / 2
        if denominator <= 0 or not all(math.isfinite(v)
                                       for v in (base_fact, base_fin, base_lin)):
            raise ValueError("cannot scale finite/linear tasks from completed data")
        event_tests[event] = {
            "prefix": prefix, "actions": prefix_actions,
            "finite": finite_tests, "linear": linear_tests,
            "weight": base_fact / denominator,
            "reference_fact": base_fact, "reference_finite": base_fin,
            "reference_linear": base_lin,
        }

    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "RUN_CONTRACT.json", {
        "status": "FROZEN_BEFORE_ANY_CANDIDATE_TRUTH_READ",
        "split": args.split, "sample": args.sample, "mpc": args.mpc,
        "data_cutoff": f"completed chunks 0..{args.mpc-1}",
        "arms": arms, "arm_set": args.arm_set,
        "pool_source": str(args.pool_source) if args.pool_source else None,
        "matched_steps_per_event": args.steps_per_event,
        "official_steps_per_event": 1, "lr": 5e-4,
        "matched_lr": args.matched_lr,
        "task_weight_multiplier": args.task_weight_multiplier,
        "carrier": ("native encode_act ten-coordinate bias or raw-action fiber lift plus profiled observation mean; frozen all original parameters"
                    if args.arm_set == "input" else "native predictor last transformer layer and norm; fixed encoder"),
        "mode_contract": "official arm uses released train-mode dropout; all matched arms use deterministic eval mode on the same cached encoded evidence",
        "functional_family": ({"round6": "one native goal x visual/proprio x suffix 0/2/4-forward/4-reverse",
                               "composition": "total native task readout on all completed subpath endpoints",
                               "increment": "none; finite increment response from completed transitions",
                               "input": "none; checkpoint-defined input boundary reference"}[args.arm_set]),
        "suffix_actions": ("cyclic actions drawn exclusively from completed chunks"
                           if args.arm_set == "round6" else "none; only actually completed contiguous subpaths"),
        "task_arms_loss": ({"round6": "native factual MSE + lambda_event x task-functional squared residual",
                            "composition": "see completed_path_control; task arms use no auxiliary state term",
                            "increment": "not applicable; no optimizer or task loss",
                            "input": "native equal-coordinate factual MSE with shared output bias analytically eliminated"}[args.arm_set]),
        "task_weight_rule": ("lambda_event = F0 factual MSE / mean(F0 finite, F0 linear); identical for both task arms"
                             if args.arm_set == "round6" else "not applicable; no lambda"),
        "task_reference_scales": {
            str(event): {key: value for key, value in cache.items()
                         if key in ("weight", "reference_fact", "reference_finite", "reference_linear")}
            for event, cache in event_tests.items()
        },
        "completed_path_control": ({
            "pairs": "all completed start/end pairs, uniformly weighted; single real frame at each start",
            "state_loss": "native observation-token MSE, excludes action coordinates",
            "task_loss": "square of signed difference between total terminal visual+proprio costs; no per-channel squaring and no auxiliary state loss",
            "teacher_vs_composed": "same starts, endpoints, context lengths, targets and weights; real versus recursively predicted cache",
            "native_support_baseline": "FACT-MATCHED unchanged",
            "novelty": "standard controls, not a claimed new mechanism",
            "extrapolation": "completed maximum path length MPC2/MPC4; queried full plans H5",
            "budget": "equal selected parameters and optimizer steps, not equal FLOPs",
        } if args.arm_set == "composition" else None),
        "increment_response": ({
            "hypothesis": "a common linear action maps frozen-model observation increments to actual increments",
            "data": "full native real histories up to num_hist; only completed transitions",
            "arms": "mean residual; one scalar gain; minimum-change multisecant response",
            "metric": "fixed native task channel scales 1/sqrt(visual_dim), 1/sqrt(proprio_dim)",
            "solver": "FP64 least squares/pseudoinverse at machine-precision rank; no damping or clipping",
            "implementation": "temporary correction of native model.predict observations; unchanged action slots and native rollout",
            "held_forward": "fit first half of available completed transitions, test remaining actual transitions and continuation",
            "novelty": "classical multisecant reference; shared response law is an unverified hypothesis",
        } if args.arm_set == "increment" else None),
        "input_revision": ({
            "input": "e0(a)+b*phi(a); phi=1 or n^T(a-mean_fit_D_action)",
            "n": "unit W^-1 ones from frozen action encoder; no task outcomes",
            "output": "c(b)=mean_D(actual-predicted_with_input_b)",
            "optimization": "b=0; Adam lr=.01; fixed20 and fixed100 steps; no oracle selection between budgets",
            "loss": "mean((r_i(b)-mean_i r_i(b))^2), native394 observation coordinates equally weighted",
            "history": "native real teacher history up to num_hist; no new rollout or query history",
            "held": "first half only for both fit and center; D1 centered fit has zero information about b",
            "novelty": "input-calibration causal reference, not claimed original mechanism",
            "small_fiber_probe": "also score previously checkpoint-defined5-plan fibers at T/L sample0/1 MPC2; no branch truth read",
        } if args.arm_set == "input" else None),
        "candidate_ids": [candidate[0] for candidate in candidates] + ([] if args.pool_source else [
            "official_pred_return", "fact_matched_return",
            "zero_environment_action", "repeat_last_completed_action"]),
        "candidate_generator": "four Frozen GD nodes; predictor-only official and matched factual 100-step native GD from identical g0_before warm-start; normalized zero environment action; repeat last completed packed action",
        "adapted_proposal_seed": args.seed + 1000 + args.sample * 10 + args.mpc,
        "old_development_pool_only": bool(args.old_development),
        "oracle_truth_read": False, "counterfactual_environment_calls": 0,
        "checkpoint_dir": str(args.checkpoint_dir), "donor": str(args.donor),
    })

    training_rows, prediction_rows, budget_rows = [], [], []
    response_states, forward_rows = {}, []
    evidence_elapsed, probe_elapsed = 0., 0.
    fiber_prediction_rows, fiber_candidates = [], []
    if args.arm_set == "input":
        from research.reframe_v3.action_fiber_probe import action_fiber
        from research.reframe_v3.action_input_revision import (
            action_input_revision_context, fit_action_input_revision,
        )
        input_direction, _ = action_fiber(student.action_encoder)
        input_center = full_actions.mean(dim=(0, 1)).detach()
        if args.sample in (0, 1) and args.mpc == 2:
            fiber_source = (args.out.parent / f"fiber_{args.split}_s{args.sample}_m2")
            fiber_summary = json.loads((fiber_source / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
            if fiber_summary["status"] != "PREDICTIONS_SEALED_NO_ORACLE_READ":
                raise ValueError("unsealed fiber probe")
            with (fiber_source / "PREDICTIONS.csv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    path, key = row["candidate_tensor"], row["tensor_key"]
                    action = torch.load(path, map_location="cpu")[key].detach().clone()
                    fiber_candidates.append((row["candidate_id"], action, path, key))
            if len(fiber_candidates) != 5:
                raise ValueError("expected five fixed fiber plans")
    if args.arm_set in ("increment", "input"):
        from research.reframe_v3.increment_response import (
            collect_increment_evidence, fit_increment_response, increment_response_context,
        )
        from research.reframe_v3.increment_response_evaluation import held_forward_probe
        evidence_started = time.perf_counter()
        with torch.no_grad():
            response_evidence = collect_increment_evidence(student, encoded, full_actions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evidence_elapsed = time.perf_counter() - evidence_started
        probe_started = time.perf_counter()
        forward_rows = held_forward_probe(student, encoded, full_actions, goal_z,
                                          args.split, args.sample, args.mpc, arm_set=args.arm_set)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        probe_elapsed = time.perf_counter() - probe_started
    arm_snapshots, proposed_actions = {}, {}
    for arm in arms:
        _seed_all(args.seed)
        with torch.no_grad():
            for parameter, saved in zip(selected, initial):
                parameter.copy_(saved)
                parameter.grad = None
        student.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        arm_started = time.perf_counter()
        updates = 0
        final_primary_loss = None
        final_task_before = None
        final_task_after = None
        step_objectives = []
        for event in range(1, args.mpc + 1):
            if arm == "FROZEN" or args.arm_set in ("increment", "input"):
                break
            if arm == "OFFICIAL-PRED":
                trainer.steps = 1
                step_losses = trainer.finetune(
                    [segment[0] for segment in segments[:event]],
                    [segment[1] for segment in segments[:event]], merge=True)
                updates += len(step_losses)
                final_primary_loss = step_losses[-1]
                continue

            linear = arm == "TASK-LINEAR"
            cache = event_tests[event]
            prefix, prefix_actions = cache["prefix"], cache["actions"]
            tests = cache["linear" if linear else "finite"] if arm in ("TASK-LINEAR", "TASK-FINITE") else None
            if tests is not None and len(tests) != 8 * event:
                raise AssertionError("task-functional family is incomplete")
            if tests is not None:
                with torch.no_grad():
                    final_task_before = float(task_reference_loss(
                        student, reference, prefix, prefix_actions,
                        goal_z, tests, linear=linear))
            for parameter in selected:
                parameter.requires_grad_(True)
            student.predictor.eval()
            optimizer = torch.optim.Adam(selected, lr=args.matched_lr,
                                         betas=(.9, .999), eps=1e-8)
            for _ in range(args.steps_per_event):
                optimizer.zero_grad(set_to_none=True)
                if "SUBPATH" in arm:
                    loss = completed_path_loss(
                        student, prefix, prefix_actions,
                        teacher_forced=arm.endswith("TEACHER"),
                        task_goal=goal_z if arm.startswith("TASK-") else None)
                elif tests is None:
                    fact_loss = _encoded_factual_loss(student, trainer, prefix, prefix_actions)
                    loss = fact_loss
                else:
                    fact_loss = _encoded_factual_loss(student, trainer, prefix, prefix_actions)
                    task_loss = task_reference_loss(student, reference, prefix,
                                                    prefix_actions, goal_z, tests,
                                                    linear=linear)
                    loss = fact_loss + (args.task_weight_multiplier
                                        * cache["weight"] * task_loss)
                loss.backward()
                if not all(parameter.grad is not None and
                           torch.isfinite(parameter.grad).all() for parameter in selected):
                    raise FloatingPointError("missing or nonfinite predictor task gradient")
                optimizer.step()
                updates += 1
                final_primary_loss = float(loss.detach())
                step_objectives.append({"event": event, "loss_before_step": final_primary_loss,
                                        "gradient_l2": float(torch.sqrt(sum(
                                            p.grad.detach().square().sum() for p in selected)))})
            student.predictor.eval()
            if tests is not None:
                with torch.no_grad():
                    final_task_after = float(task_reference_loss(
                        student, reference, prefix, prefix_actions,
                        goal_z, tests, linear=linear))
            for parameter in selected:
                parameter.requires_grad_(False)

        student.eval()
        response_stack = ExitStack()
        if arm in INCREMENT_KINDS:
            response = fit_increment_response(*response_evidence, kind=INCREMENT_KINDS[arm])
            response_states[arm] = response
            response_stack.enter_context(increment_response_context(student, response))
        if arm in INPUT_KINDS:
            kind, fit_steps = INPUT_KINDS[arm]
            response = fit_action_input_revision(
                student, encoded, full_actions, kind=kind,
                n=input_direction, center=input_center, steps=fit_steps)
            response_states[arm] = response
            response_stack.enter_context(action_input_revision_context(
                student, response["b"], kind=kind, n=response["n"], center=response["center"]))
            response_stack.enter_context(increment_response_context(student, response["output_response"]))
            updates = fit_steps
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_elapsed = time.perf_counter() - arm_started
        support = _factual_support_loss(student, trainer, full_obs, full_actions)
        scores = _score_pool(student, current, goal, candidates, args.mpc)
        if fiber_candidates:
            fiber_prediction_rows += [{"split": args.split, "sample": args.sample,
                                       "mpc": args.mpc, "arm": arm, **row}
                                      for row in _score_pool(student, current, goal, fiber_candidates, args.mpc)]
        if args.pool_source is None and arm in ("OFFICIAL-PRED", "FACT-MATCHED"):
            name = "official_pred_return" if arm == "OFFICIAL-PRED" else "fact_matched_return"
            proposed_actions[name] = _adapted_return_plan(
                student, preprocessor, frameskip, current, goal,
                candidates[0][1].to(device), args.mpc,
                args.seed + 1000 + args.sample * 10 + args.mpc)
        arm_snapshots[arm] = [parameter.detach().clone() for parameter in selected]
        for row in scores:
            prediction_rows.append({"split": args.split, "sample": args.sample,
                                    "mpc": args.mpc, "arm": arm, **row})
        training_rows.append({"split": args.split, "sample": args.sample,
                              "mpc": args.mpc, "arm": arm, "updates": updates,
                              "factual_support_loss": support,
                              "last_primary_loss": "" if final_primary_loss is None else final_primary_loss,
                              "last_task_loss_before": "" if final_task_before is None else final_task_before,
                              "last_task_loss_after": "" if final_task_after is None else final_task_after,
                              "task_weight": "" if final_task_after is None else (
                                  args.task_weight_multiplier * event_tests[args.mpc]["weight"]),
                              "step_objectives": json.dumps(step_objectives)})
        if args.arm_set in ("composition", "increment", "input"):
            with torch.no_grad():
                for teacher in (True, False):
                    for is_task in (False, True):
                        label = ("teacher" if teacher else "composed") + ("_task" if is_task else "_state")
                        training_rows[-1]["completed_" + label + "_loss"] = float(completed_path_loss(
                            student, encoded, full_actions, teacher_forced=teacher,
                            task_goal=goal_z if is_task else None))
        budget_rows.append({"split": args.split, "sample": args.sample,
                            "mpc": args.mpc, "arm": arm,
                            "fit_wall_seconds": fit_elapsed,
                            "fit_score_and_proposal_wall_seconds": time.perf_counter() - arm_started,
                            "updated_parameters": (10 if arm in INPUT_KINDS else
                                                   sum(p.numel() for p in selected) if updates else 0),
                            "updates": updates,
                            "peak_vram_bytes": torch.cuda.max_memory_allocated()
                            if torch.cuda.is_available() else 0})
        print(arm, "support", support, "score_min", min(row["c_model"] for row in scores),
              "updates", updates, "seconds", round(time.perf_counter() - arm_started, 2), flush=True)
        response_stack.close()

    if args.pool_source is None:
        _generate_and_score_extras(args, reference, student, preprocessor, frameskip,
                                   current, goal, full_actions, proposed_actions,
                                   selected, arm_snapshots, arms, budget_rows,
                                   prediction_rows, candidates)

    if args.arm_set == "composition":
        selected_ids = {id(p) for p in selected}
        names = [name for name, p in student.named_parameters() if id(p) in selected_ids]
        if len(names) != len(selected):
            raise AssertionError("selected parameter names incomplete")
        name_by_id = {id(p): name for name, p in student.named_parameters()}
        names = [name_by_id[id(p)] for p in selected]
        torch.save({arm: {name: value.detach().cpu() for name, value in zip(names, snapshot)}
                    for arm, snapshot in arm_snapshots.items()}, args.out / "predictor_updates.pt")
    if args.arm_set in ("increment", "input"):
        torch.save(response_states, args.out / f"{args.arm_set}_responses.pt")
        _write_csv(args.out / "HELD_FORWARD.csv", forward_rows)
    if fiber_prediction_rows:
        _write_csv(args.out / "FIBER_PREDICTIONS.csv", fiber_prediction_rows)

    if len(prediction_rows) != len(arms) * len(candidates):
        raise AssertionError("incomplete prediction manifest")
    if not all(math.isfinite(row["c_model"]) for row in prediction_rows):
        raise FloatingPointError("nonfinite sealed prediction")
    _write_csv(args.out / "PREDICTIONS.csv", prediction_rows)
    _write_csv(args.out / "TRAINING.csv", training_rows)
    _write_csv(args.out / "BUDGET.csv", budget_rows)
    _write_json(args.out / "RUN_SUMMARY.json", {
        "status": "PREDICTIONS_SEALED_NO_ORACLE_READ",
        "candidate_count": len(candidates), "arms": len(arms),
        "prediction_rows": len(prediction_rows),
        "wall_seconds": time.perf_counter() - started,
        "environment_branches": 0,
        "reference_model_parameters_require_grad": any(p.requires_grad for p in reference.parameters()),
        "actual_chunk_count": len(segments),
        "shared_increment_evidence_wall_seconds": evidence_elapsed,
        "held_forward_diagnostic_wall_seconds": probe_elapsed,
    })


def _generate_and_score_extras(args, reference, student, preprocessor, frameskip,
                               current, goal, full_actions, proposed_actions,
                               selected, arm_snapshots, arms, budget_rows,
                               prediction_rows, candidates):
    # Fixed score-independent actions contain no model or oracle selection.
    frozen_planner = _build_planner(reference, preprocessor, frameskip)
    with torch.no_grad():
        encoded_current = reference.encode_obs(current)
        proposed_actions["zero_environment_action"] = frozen_planner.init_actions(
            encoded_current, None).detach().cpu().clone()
    proposed_actions["repeat_last_completed_action"] = (
        full_actions[:, -1:].repeat(1, 5, 1).detach().cpu().clone())
    extra_dir = args.out / "candidate_actions"
    extra_dir.mkdir(exist_ok=False)
    extra_candidates = []
    for candidate_id in ("official_pred_return", "fact_matched_return",
                         "zero_environment_action", "repeat_last_completed_action"):
        action = proposed_actions[candidate_id]
        if tuple(action.shape) != (1, 5, 10) or not torch.isfinite(action).all():
            raise ValueError(f"invalid generated candidate {candidate_id}")
        path = extra_dir / f"{candidate_id}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save({"u_after": action}, path)
        extra_candidates.append((candidate_id, action, str(path), "u_after"))
    for arm, budget in zip(arms, budget_rows):
        with torch.no_grad():
            for parameter, saved in zip(selected, arm_snapshots[arm]):
                parameter.copy_(saved)
        student.eval()
        scored_at = time.perf_counter()
        for row in _score_pool(student, current, goal, extra_candidates, args.mpc):
            prediction_rows.append({"split": args.split, "sample": args.sample,
                                    "mpc": args.mpc, "arm": arm, **row})
        budget["additional_candidate_scoring_wall_seconds"] = time.perf_counter() - scored_at
    candidates += extra_candidates



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("T", "L"), required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--mpc", type=int, choices=(2, 4), required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("D:/EV-TTT/pushobj_shape_shift"))
    parser.add_argument("--steps-per-event", type=int, default=4)
    parser.add_argument("--matched-lr", type=float, default=1e-4)
    parser.add_argument("--task-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260924)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--old-development", action="store_true")
    parser.add_argument("--arm-set", choices=("round6", "composition", "increment", "input", "actuator", "innovation"), default="round6")
    parser.add_argument("--pool-source", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
