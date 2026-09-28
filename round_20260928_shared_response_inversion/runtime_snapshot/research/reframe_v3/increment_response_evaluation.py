"""Time-ordered falsification using completed data only, never query outcomes."""

from __future__ import annotations

from contextlib import ExitStack

import torch

from research.reframe_v3.finite_task_reference import successor_from_history
from research.reframe_v3.increment_response import (
    collect_increment_evidence, fit_increment_response, increment_response_context,
)
from research.reframe_v3.matched_feedback_forecast import rollout_from_zobs


def _cost(observation, goal):
    return sum((observation[key] - goal[key]).square().mean()
               for key in ("visual", "proprio"))


def _error(prediction, target):
    return sum((prediction[key] - target[key]).square().mean()
               for key in ("visual", "proprio"))


@torch.no_grad()
def held_forward_probe(model, observed, actions, goal, split, sample, mpc,
                       arm_set="increment"):
    """Fit first half, inspect later transitions and a free-running continuation.

    These later labels were already observed before the main query, but they are
    never used in this internal fit. This is not an independent blind test.
    """
    cut = actions.shape[1] // 2
    if cut < 1 or cut == actions.shape[1]:
        raise ValueError("held-forward probe requires nonempty past and future")
    prefix = {key: value[:, :cut+1] for key, value in observed.items()}
    evidence = collect_increment_evidence(model, prefix, actions[:, :cut])
    frozen_predictions = {}

    def predict_held():
        outputs = {}
        for transition in range(cut, actions.shape[1]):
            outputs[("real_history_transition", transition+1)] = successor_from_history(
                model, observed, actions, transition)
        start = {key: value[:, cut:cut+1] for key, value in observed.items()}
        path, _ = rollout_from_zobs(model, start, actions[:, cut:])
        for end in range(cut+1, actions.shape[1]+1):
            outputs[("composed_continuation", end)] = {
                key: value[:, end-cut:end-cut+1] for key, value in path.items()}
        return outputs

    frozen_predictions = predict_held()
    rows = []
    arms = [("FROZEN", None), ("INCREMENT-MEAN", "mean_residual")]
    if arm_set == "increment":
        arms += [("INCREMENT-SCALAR", "scalar_gain"), ("INCREMENT-SECANT", "multisecant")]
    elif arm_set == "input":
        from research.reframe_v3.action_fiber_probe import action_fiber
        from research.reframe_v3.action_input_revision import (
            action_input_revision_context, fit_action_input_revision,
        )
        direction, _ = action_fiber(model.action_encoder)
        center = actions[:, :cut].mean(dim=(0, 1))
        arms += [(f"INPUT-{label}-{steps}", (kind, steps)) for steps in (20, 100)
                 for label, kind in (("BIAS", "constant"), ("FIBER", "fiber"))]
    else:
        raise ValueError("unknown held-forward arm set")
    for arm, kind in arms:
        with ExitStack() as stack:
            if isinstance(kind, tuple):
                with torch.enable_grad():
                    fitted = fit_action_input_revision(
                        model, prefix, actions[:, :cut], kind=kind[0], steps=kind[1],
                        n=direction, center=center)
                stack.enter_context(action_input_revision_context(
                    model, fitted["b"], kind=kind[0], n=fitted["n"], center=fitted["center"]))
                stack.enter_context(increment_response_context(model, fitted["output_response"]))
            elif kind is not None:
                fitted = fit_increment_response(*evidence, kind=kind)
                stack.enter_context(increment_response_context(model, fitted))
            predicted = frozen_predictions if kind is None else predict_held()
        for (scope, end), current in predicted.items():
            actual = {key: value[:, end:end+1] for key, value in observed.items()}
            frozen = frozen_predictions[(scope, end)]
            true_cost, base_cost, updated_cost = (_cost(z, goal) for z in (actual, frozen, current))
            base_error = true_cost - base_cost
            updated_error = true_cost - updated_cost
            correction = updated_cost - base_cost
            rows.append({"split": split, "sample": sample, "mpc": mpc,
                         "fit_transitions": cut, "scope": scope, "endpoint": end,
                         "arm": arm, "base_state_error": float(_error(frozen, actual)),
                         "state_error": float(_error(current, actual)),
                         "base_signed_task_error": float(base_error),
                         "signed_task_error": float(updated_error),
                         "K": float(correction),
                         "closure_error": float((base_error-correction-updated_error).abs())})
    return rows
