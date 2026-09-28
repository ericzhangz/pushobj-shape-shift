"""D-only observed-subsystem experiment on the existing native rollout.

This is a gray-box reference, not an originality claim. No environment or
counterfactual-outcome loader is imported. The main Round-6 runner dispatches
here; scoring and the model operator remain the existing primary path.
"""

from contextlib import contextmanager
from functools import partial
import csv
import json
import time
from unittest.mock import patch

import torch

from planning.adajepa import AdaJEPATrainer
from research.reframe_v3.action_input_revision import _residuals
from research.reframe_v3.actuator_response import fit_actuator_response, make_actuator_transition
from research.reframe_v3.completed_phase_evidence import merge_completed_lowlevel, phase_chains
from research.reframe_v3.endpoint_matched_actions import endpoint_matched_actions
from research.reframe_v3.visual_innovation import (
    collect_innovation_evidence, fit_visual_innovation, make_visual_innovation,
)
from research.reframe_v3.increment_response import (
    collect_increment_evidence, fit_increment_response, increment_response_context,
)
from research.reframe_v3.increment_response_evaluation import _cost, _error
from research.reframe_v3.round6_reference import (
    _local_hub_loader, _score_pool, _sealed_candidates, _write_csv, _write_json,
)
from research.reframe_v3.shadow_selection_audit import (
    _load_anchor_observations, _load_runtime, _load_segments, _seed_all, _transform_anchor_obs,
)


ARMS = ("FROZEN", "INCREMENT-MEAN", "FINE-MEAN", "FACT-PHASE",
        "ACTUATOR-READOUT", "ACTUATOR-FEEDBACK")
INNOVATION_ARMS = ("FROZEN", "ACTUATOR-FEEDBACK", "MEAN-VISUAL-READOUT",
                   "MEAN-VISUAL-FEEDBACK", "INNOVATION-READOUT",
                   "INNOVATION-FEEDBACK", "FACT-PHASE")


def _arms(args):
    return INNOVATION_ARMS if args.arm_set == "innovation" else ARMS


def _slice(observed, start, end):
    return {key: value[:, start:end] for key, value in observed.items()}


def _reset(model, selected, initial):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    with torch.no_grad():
        for parameter, value in zip(selected, initial):
            parameter.copy_(value)


def fit_arm(model, selected, arm, fine, encoded, low_actions, preprocessor, steps_per_event, lr):
    """Only the supplied completed prefix is accessible to this function."""
    if arm == "FROZEN":
        return {"arm": arm, "updates": 0}
    if arm.startswith("ACTUATOR-"):
        return {"arm": arm, "updates": 0,
                "actuator": fit_actuator_response(fine["proprio"], low_actions, preprocessor)}
    if arm.startswith(("MEAN-VISUAL-", "INNOVATION-")):
        actuator = fit_actuator_response(fine["proprio"], low_actions, preprocessor)
        x, y, evidence = collect_innovation_evidence(model, fine, encoded, low_actions, actuator)
        return {"arm": arm, "updates": 0, "actuator": actuator,
                "visual_innovation": fit_visual_innovation(x, y),
                "visual_kind": "mean" if arm.startswith("MEAN-") else "conditional",
                "evidence": evidence, "innovation_X": x, "innovation_Y": y}
    chains = phase_chains(encoded, low_actions)
    if arm in ("INCREMENT-MEAN", "FINE-MEAN"):
        evidence = [collect_increment_evidence(model, chain["observed"], chain["actions"])
                    for chain in (chains[:1] if arm == "INCREMENT-MEAN" else chains)]
        response = fit_increment_response(torch.cat([item[0] for item in evidence], dim=1),
                                          torch.cat([item[1] for item in evidence], dim=1),
                                          evidence[0][2], "mean_residual")
        return {"arm": arm, "updates": 0, "response": response}
    if arm != "FACT-PHASE":
        raise ValueError("unknown actuator control")
    for parameter in selected:
        parameter.requires_grad_(True)
    trace = []
    for event in range(1, low_actions.shape[1] // 5 + 1):
        # Same arrival/update opportunities as FACT-MATCHED, now all real phases.
        event_chains = phase_chains(_slice(encoded, 0, 5*event+1), low_actions[:, :5*event])
        optimizer = torch.optim.Adam(selected, lr=lr, betas=(.9, .999), eps=1e-8)
        for _ in range(steps_per_event):
            optimizer.zero_grad(set_to_none=True)
            residuals = torch.cat([_residuals(model, chain["observed"], chain["actions"])
                                   for chain in event_chains], dim=0)
            loss = residuals.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite phase factual loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in selected):
                raise FloatingPointError("invalid phase factual gradient")
            optimizer.step()
            trace.append({"event": event, "loss_before_step": float(loss.detach()),
                          "windows": int(residuals.shape[0])})
    for parameter in selected:
        parameter.requires_grad_(False)
    return {"arm": arm, "updates": len(trace), "trace": trace,
            "parameters": [p.detach().cpu().clone() for p in selected]}


@contextmanager
def arm_context(model, fitted):
    if "response" in fitted:
        with increment_response_context(model, fitted["response"]):
            yield
    elif "actuator" in fitted:
        original = model.rollout
        existed = "rollout" in model.__dict__
        instance = model.__dict__.get("rollout")
        options = {"proprio_transition": make_actuator_transition(fitted["actuator"]),
                   "proprio_feedback": fitted["arm"] != "ACTUATOR-READOUT"}
        if "visual_innovation" in fitted:
            options.update(visual_revision=make_visual_innovation(
                fitted["visual_innovation"], fitted["visual_kind"]),
                visual_feedback=fitted["arm"].endswith("-FEEDBACK"))
        model.rollout = partial(original, **options)
        try:
            yield
        finally:
            if existed:
                model.rollout = instance
            else:
                del model.rollout
    else:
        yield


def _native_last(model, observed, actions):
    path, _ = model.rollout(observed, actions)
    return {key: value[:, -1:] for key, value in path.items()}


@torch.no_grad()
def completed_metrics(model, observed, actions, encoded, goal):
    """Uniform last-token metrics on the same coarse true subpaths for all arms.

    This raw-observation call reaches the exact revised native rollout. Calling
    the legacy encoded-only mirror would bypass the raw proprio subsystem.
    """
    values = {}
    factual = []
    for teacher in (True, False):
        states, tasks = [], []
        for start in range(actions.shape[1]):
            for end in range(start+1, actions.shape[1]+1):
                first = max(start, end-model.num_hist) if teacher else start
                source = _slice(observed, first, end if teacher else first+1)
                prediction = _native_last(model, source, actions[:, first:end])
                target = _slice(encoded, end, end+1)
                state_error = sum((prediction[k]-target[k]).square().sum() for k in prediction) / 394
                states.append(float(state_error))
                tasks.append(float((_cost(prediction, goal)-_cost(target, goal)).square()))
                if teacher and start == 0:
                    factual.append(float(state_error))
        label = "teacher" if teacher else "composed"
        values[f"completed_{label}_state_loss"] = sum(states) / len(states)
        values[f"completed_{label}_task_loss"] = sum(tasks) / len(tasks)
    values["factual_support_loss"] = sum(factual) / len(factual)
    return values


@torch.no_grad()
def _held_predictions(model, observed, actions, cut):
    outputs = {}
    for end in range(cut+1, actions.shape[1]+1):
        first = max(0, end-model.num_hist)
        outputs[("real_history_transition", end)] = _native_last(
            model, _slice(observed, first, end), actions[:, first:end])
    path, _ = model.rollout(_slice(observed, cut, cut+1), actions[:, cut:])
    for end in range(cut+1, actions.shape[1]+1):
        outputs[("composed_continuation", end)] = _slice(path, end-cut, end-cut+1)
    return outputs


def held_forward(model, selected, initial, fine, encoded_fine, low_actions,
                 observed, actions, encoded, goal, preprocessor, args):
    cut = actions.shape[1] // 2
    _reset(model, selected, initial)
    frozen = _held_predictions(model, observed, actions, cut)
    rows, fit_states = [], {}
    for arm in _arms(args):
        _reset(model, selected, initial)
        fitted = fit_arm(model, selected, arm, _slice(fine, 0, cut*5+1),
                         _slice(encoded_fine, 0, cut*5+1), low_actions[:, :cut*5],
                         preprocessor, args.steps_per_event, args.matched_lr)
        fit_states[arm] = fitted
        with arm_context(model, fitted):
            outputs = _held_predictions(model, observed, actions, cut)
        for (scope, end), prediction in outputs.items():
            actual = _slice(encoded, end, end+1)
            base = frozen[(scope, end)]
            truth, base_cost, revised_cost = (_cost(x, goal) for x in (actual, base, prediction))
            rows.append({"split": args.split, "sample": args.sample, "mpc": args.mpc,
                         "fit_transitions": cut, "scope": scope, "endpoint": end, "arm": arm,
                         "base_state_error": float(_error(base, actual)),
                         "state_error": float(_error(prediction, actual)),
                         "base_signed_task_error": float(truth-base_cost),
                         "signed_task_error": float(truth-revised_cost),
                         "K": float(revised_cost-base_cost),
                         "closure_error": float(((truth-base_cost)-(revised_cost-base_cost)-(truth-revised_cost)).abs()),
                         **{f"{key}_signed_task_error": float(
                             (actual[key]-goal[key]).square().mean()
                             -(prediction[key]-goal[key]).square().mean()) for key in actual}})
    # Free-running physical self-motion check; no true intermediate state input.
    transition = make_actuator_transition(fit_states["ACTUATOR-FEEDBACK"]["actuator"])
    predicted = observed["proprio"][:, cut:cut+1]
    physical_rows = []
    std = preprocessor.proprio_std.to(predicted.device)
    for end in range(cut+1, actions.shape[1]+1):
        predicted = transition(predicted, actions[:, end-1:end])
        error = (predicted-observed["proprio"][:, end:end+1]) * std
        physical_rows.append({"fit_lowlevel_transitions": cut*5, "endpoint_chunk": end,
                              "position_max_abs": float(error[..., :2].abs().max()),
                              "velocity_max_abs": float(error[..., 2:].abs().max())})
    return rows, fit_states, physical_rows


def run_actuator_controls(args):
    if args.out.exists() or args.pool_source is None:
        raise ValueError("actuator controls need a new output path and sealed original pool")
    if args.mpc not in (2, 4) or args.steps_per_event != 4 or args.matched_lr != 1e-4:
        raise ValueError("actuator contract freezes MPC2/4 and phase factual 4 steps/event at lr1e-4")
    _seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    started = time.perf_counter()
    with patch.object(torch.hub, "load", _local_hub_loader()):
        model, preprocessor, config = _load_runtime(args.checkpoint_dir, device)
    if int(config.frameskip) != 5:
        raise ValueError("actuator control is contracted for five commands per token")
    trainer = AdaJEPATrainer(model, lr=5e-4, steps=1, optimizer_name="adam",
                            finetune_encoder=False, last_layer_only=True)
    selected = trainer._ada_predictor_params
    initial = [p.detach().clone() for p in selected]
    segments = _load_segments(args.donor, args.sample, args.mpc, 5, preprocessor, full_resolution=True)
    fine, low_actions = merge_completed_lowlevel(segments)
    fine = {key: value.to(device) for key, value in fine.items()}
    low_actions = low_actions.to(device)
    observed = {key: value[:, ::5] for key, value in fine.items()}
    actions = low_actions.reshape(1, args.mpc, 10)
    encoding_started = time.perf_counter()
    with torch.no_grad():
        encoded_fine = model.encode_obs(fine)
    encoded = {key: value[:, ::5] for key, value in encoded_fine.items()}
    current_raw, goal_raw = _load_anchor_observations(args.donor, args.sample, args.mpc)
    current, goal = _transform_anchor_obs(preprocessor, current_raw, goal_raw, device)
    with torch.no_grad():
        goal_z = model.encode_obs(goal)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encoding_seconds = time.perf_counter() - encoding_started
    candidates = _sealed_candidates(args.pool_source, args)
    arms = _arms(args)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "RUN_CONTRACT.json", {
        "status": "FROZEN_BEFORE_ANY_CANDIDATE_TRUTH_READ", "split": args.split,
        "sample": args.sample, "mpc": args.mpc, "arm_set": args.arm_set, "arms": arms,
        "donor": str(args.donor), "checkpoint_dir": str(args.checkpoint_dir),
        "pool_source": str(args.pool_source), "candidate_ids": [c[0] for c in candidates],
        "data_cutoff": f"completed lowlevel transitions 0..{5*args.mpc-1}",
        "lowlevel_transitions": 5*args.mpc, "native_phase_windows": 5*args.mpc-4,
        "oracle_truth_read": False, "counterfactual_environment_calls": 0,
        "task_weight_multiplier": args.task_weight_multiplier,
        "old_development_pool_only": True, "steps_per_event": 4, "phase_lr": 1e-4,
        "actuator_law": "raw [delta_position,next_velocity]=[velocity,action] M, FP64 LS rank4; no simulator coefficients",
        "information": "fine RGB/proprio and raw actions are completed observations; no privileged object state",
        "support_metric": "last-token394-coordinate MSE, all coarse transitions with native real history; all start/end subpaths separately",
        "paired_intervention": "identical self-motion and fitted constants, readout-only versus recurrent feedback; paired first visual prediction identical",
        "prediction_operator": "existing models/visual_world_model.py:rollout optional proprio_transition and visual_revision; original default retained",
    })
    training, predictions, budgets, states, paths = [], [], [], {}, {}
    fiber_candidates, fiber_rows, fiber_paths = [], [], {}
    probe_candidates, probe_rows, probe_paths = [], [], {}
    if args.arm_set == "innovation" and args.sample in (0, 1) and args.mpc == 2:
        probe_dir = args.out / "endpoint_probe"
        probe_dir.mkdir()
        base = next(action for name, action, _, _ in candidates if name == "g99_after")
        actuator = fit_actuator_response(fine["proprio"], low_actions, preprocessor)
        probe_actions, algebra = endpoint_matched_actions(actuator, base, low_actions)
        for name, action in probe_actions:
            filename = probe_dir / f"{name}.pt"
            torch.save({"u_after": action}, filename)
            probe_candidates.append((name, action, str(filename), "u_after"))
        _write_json(probe_dir / "RUN_CONTRACT.json", {
            "status": "D_ONLY_ENDPOINT_MATCHED_ACTION_PROBE", "split": args.split,
            "sample": args.sample, "mpc": args.mpc, "donor": str(args.donor),
            "checkpoint_dir": str(args.checkpoint_dir), "oracle_truth_read": False,
            "counterfactual_environment_calls": 0, "arms": arms, "algebra": algebra,
            "interpretation": "predicted p/v matched; actual endpoint error must be measured, no outcome filtering",
        })
    if args.arm_set == "actuator" and args.sample in (0, 1) and args.mpc == 2:
        source = args.out.parent / f"fiber_{args.split}_s{args.sample}_m2"
        if json.loads((source / "RUN_SUMMARY.json").read_text())["status"] != "PREDICTIONS_SEALED_NO_ORACLE_READ":
            raise ValueError("unsealed fiber candidates")
        with (source / "PREDICTIONS.csv").open(encoding="utf-8", newline="") as handle:
            fiber_manifest = list(csv.DictReader(handle))
        for row in fiber_manifest:
            action = torch.load(row["candidate_tensor"], map_location="cpu")[row["tensor_key"]]
            fiber_candidates.append((row["candidate_id"], action, row["candidate_tensor"], row["tensor_key"]))
        if len(fiber_candidates) != 5:
            raise ValueError("expected five fiber plans")
    for arm in arms:
        _reset(model, selected, initial)
        fit_start = time.perf_counter()
        fitted = fit_arm(model, selected, arm, fine, encoded_fine, low_actions, preprocessor,
                         args.steps_per_event, args.matched_lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter()-fit_start
        states[arm], paths[arm], fiber_paths[arm] = fitted, {}, {}
        probe_paths[arm] = {}
        with arm_context(model, fitted):
            metrics = completed_metrics(model, observed, actions, encoded, goal_z)
            rows = _score_pool(model, current, goal, candidates, args.mpc, path_sink=paths[arm])
            if fiber_candidates:
                fiber_rows += [{"split": args.split, "sample": args.sample, "mpc": args.mpc,
                                "arm": arm, **row} for row in _score_pool(
                                    model, current, goal, fiber_candidates, args.mpc, path_sink=fiber_paths[arm])]
            if probe_candidates:
                probe_rows += [{"split": args.split, "sample": args.sample, "mpc": args.mpc,
                                "arm": arm, **row} for row in _score_pool(
                                    model, current, goal, probe_candidates, args.mpc, path_sink=probe_paths[arm])]
        predictions += [{"split": args.split, "sample": args.sample, "mpc": args.mpc,
                         "arm": arm, **row} for row in rows]
        training.append({"split": args.split, "sample": args.sample, "mpc": args.mpc,
                         "arm": arm, "updates": fitted["updates"], **metrics})
        budgets.append({"arm": arm, "fit_wall_seconds": fit_seconds,
                        "fit_and_scoring_wall_seconds": time.perf_counter()-fit_start,
                        "updated_parameters": (16 + (10*384+10+384 if "visual_innovation" in fitted else 0)) if "actuator" in fitted else
                        sum(p.numel() for p in selected) if arm == "FACT-PHASE" else 0})
        print(arm, "support", metrics["factual_support_loss"], "seconds", budgets[-1]["fit_and_scoring_wall_seconds"], flush=True)
    for bank in (paths, fiber_paths, probe_paths):
        for candidate in bank["FROZEN"]:
            if args.arm_set == "innovation":
                base = bank["ACTUATOR-FEEDBACK"][candidate]
                for prefix in ("MEAN-VISUAL", "INNOVATION"):
                    readout, feedback = [bank[f"{prefix}-{suffix}"][candidate]
                                         for suffix in ("READOUT", "FEEDBACK")]
                    if (not torch.equal(readout["proprio"], base["proprio"])
                            or not torch.equal(feedback["proprio"], base["proprio"])
                            or not torch.equal(readout["visual"][:, :2], feedback["visual"][:, :2])):
                        raise AssertionError("joint visual intervention parity failed")
                continue
            base, readout, feedback = [bank[arm][candidate] for arm in
                                      ("FROZEN", "ACTUATOR-READOUT", "ACTUATOR-FEEDBACK")]
            if (not torch.equal(readout["visual"], base["visual"])
                    or not torch.equal(readout["proprio"], feedback["proprio"])
                    or not torch.equal(feedback["visual"][:, :2], base["visual"][:, :2])):
                raise AssertionError("readout/feedback causal intervention parity failed")
    held_rows, held_states, physical = held_forward(model, selected, initial, fine, encoded_fine,
        low_actions, observed, actions, encoded, goal_z, preprocessor, args)
    if args.arm_set == "innovation":
        for state_bank in (states, held_states):
            reference = state_bank["INNOVATION-FEEDBACK"]["visual_innovation"]
            for arm in ("MEAN-VISUAL-READOUT", "MEAN-VISUAL-FEEDBACK", "INNOVATION-READOUT"):
                if any(not torch.equal(reference[key], state_bank[arm]["visual_innovation"][key])
                       for key in ("C", "mean_p", "mean_v")):
                    raise AssertionError("visual control fits are not identical")
    for name, table in (("PREDICTIONS", predictions), ("TRAINING", training), ("BUDGET", budgets),
                        ("HELD_FORWARD", held_rows), ("HELD_ACTUATOR", physical)):
        _write_csv(args.out / f"{name}.csv", table)
    if fiber_rows:
        _write_csv(args.out / "FIBER_PREDICTIONS.csv", fiber_rows)
        torch.save(fiber_paths, args.out / "FIBER_PATHS.pt")
    if probe_rows:
        _write_csv(probe_dir / "PREDICTIONS.csv", probe_rows)
        torch.save(probe_paths, probe_dir / "PREDICTED_PATHS.pt")
        _write_json(probe_dir / "RUN_SUMMARY.json", {
            "status": "PREDICTIONS_SEALED_NO_ORACLE_READ", "environment_branches": 0,
            "candidate_count": len(probe_candidates), "arms": len(arms), "prediction_rows": len(probe_rows),
        })
    torch.save(states, args.out / "actuator_responses.pt")
    torch.save(held_states, args.out / "held_responses.pt")
    torch.save(paths, args.out / "PREDICTED_PATHS.pt")
    if len(predictions) != 8*len(arms):
        raise AssertionError("incomplete actuator predictions")
    _write_json(args.out / "RUN_SUMMARY.json", {
        "status": "PREDICTIONS_SEALED_NO_ORACLE_READ", "candidate_count": 8, "arms": len(arms),
        "prediction_rows": len(predictions), "environment_branches": 0,
        "wall_seconds": time.perf_counter()-started, "fine_encoding_wall_seconds": encoding_seconds,
        "paired_intervention_parity": True, "actual_chunk_count": args.mpc,
    })
