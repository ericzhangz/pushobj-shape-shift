"""Non-novel Round-6 finite task-function reference on the native predictor.

This module only transforms completed, encoded experience into losses.  It has
no candidate/oracle loader and never updates the frozen suffix model.
"""

from __future__ import annotations

import torch

from research.reframe_v3.matched_feedback_forecast import (
    _join_encoded_observation_action, rollout_from_zobs,
)


def _slice_obs(obs: dict[str, torch.Tensor], start: int, stop: int):
    return {key: value[:, start:stop] for key, value in obs.items()}


def _append_obs(prefix: dict[str, torch.Tensor], frame: dict[str, torch.Tensor]):
    return {key: torch.cat((prefix[key], frame[key]), dim=1) for key in prefix}


def _validate_chain(obs, actions, goal):
    length = int(actions.shape[1])
    if length < 1 or any(value.shape[1] != length + 1 for value in obs.values()):
        raise ValueError("completed factual chain must contain T actions and T+1 observations")
    if any(value.shape[1] != 1 for value in goal.values()):
        raise ValueError("task goal must contain one encoded observation")
    if not all(torch.isfinite(value).all() for value in (*obs.values(), actions, *goal.values())):
        raise FloatingPointError("nonfinite encoded factual chain, action, or goal")


def successor_from_history(model, observed, actions, transition: int):
    """Predict one next observation using the same observed cache and action slot."""
    start = max(0, transition - int(model.num_hist) + 1)
    prefix = _slice_obs(observed, start, transition + 1)
    relevant_actions = actions[:, start : transition + 1]
    completed, _ = rollout_from_zobs(model, prefix, relevant_actions)
    return _slice_obs(completed, completed["visual"].shape[1] - 1,
                      completed["visual"].shape[1])


def suffix_actions(actions: torch.Tensor, transition: int, kind: str):
    """Deterministic, legal actions drawn only from completed experience."""
    count = int(actions.shape[1])
    if kind == "zero":
        return actions[:, :0]
    if kind == "forward2":
        offsets = (1, 2)
    elif kind == "forward4":
        offsets = (1, 2, 3, 4)
    elif kind == "reverse4":
        offsets = (-1, -2, -3, -4)
    else:
        raise ValueError(f"unknown suffix kind {kind!r}")
    indices = [(transition + offset) % count for offset in offsets]
    return torch.cat([actions[:, index : index + 1] for index in indices], dim=1)


def task_readout(reference_model, observed, actions, transition, successor,
                 suffix, goal, channel: str):
    """Frozen terminal task readout after cache shift and a specified suffix."""
    if channel not in ("visual", "proprio"):
        raise ValueError("channel must be visual or proprio")
    start = max(0, transition - int(reference_model.num_hist) + 1)
    source = _slice_obs(observed, start, transition + 1)
    cache = _append_obs(source, successor)
    if suffix.shape[1]:
        skipped = max(0, cache["visual"].shape[1] - int(reference_model.num_hist))
        cache = _slice_obs(cache, skipped, cache["visual"].shape[1])
        # The action at the new successor is the first suffix action.
        action_slots = torch.cat((actions[:, start : transition + 1], suffix), dim=1)
        action_slots = action_slots[:, skipped:]
        path, _ = rollout_from_zobs(reference_model, cache, action_slots)
    else:
        path = cache
    return (path[channel][:, -1:] - goal[channel]).square().mean()


def make_functionals(reference_model, observed, actions, goal, *, linear: bool):
    """Build exactly eight fixed tests per completed transition.

    The native goal is split into its visual/proprio components; four suffixes
    are used per component.  No second goal or future environment label enters.
    """
    _validate_chain(observed, actions, goal)
    tests = []
    for transition in range(actions.shape[1]):
        with torch.no_grad():
            predicted = successor_from_history(reference_model, observed, actions, transition)
        actual = _slice_obs(observed, transition + 1, transition + 2)
        for kind in ("zero", "forward2", "forward4", "reverse4"):
            suffix = suffix_actions(actions, transition, kind)
            for channel in ("visual", "proprio"):
                with torch.no_grad():
                    target = task_readout(reference_model, observed, actions, transition,
                                          actual, suffix, goal, channel).detach()
                gradient = None
                if linear:
                    leaf = {key: value.detach().clone().requires_grad_(True)
                            for key, value in predicted.items()}
                    value = task_readout(reference_model, observed, actions, transition,
                                         leaf, suffix, goal, channel)
                    partial = torch.autograd.grad(value, tuple(leaf.values()),
                                                  allow_unused=True)
                    gradient = {
                        key: torch.zeros_like(leaf[key]) if item is None else item.detach()
                        for key, item in zip(leaf, partial)
                    }
                tests.append({"transition": transition, "kind": kind,
                              "channel": channel, "suffix": suffix,
                              "target": target, "gradient": gradient,
                              "actual": actual})
    return tests


def task_reference_loss(student, reference_model, observed, actions, goal,
                        tests: list[dict], *, linear: bool):
    if not tests:
        raise ValueError("no completed-experience test functionals")
    predicted = {}
    errors = []
    for test in tests:
        transition = test["transition"]
        if transition not in predicted:
            predicted[transition] = successor_from_history(
                student, observed, actions, transition)
        successor = predicted[transition]
        if linear:
            # Cache/action slots do not vary: the VJP is only on successor obs.
            signed = sum((test["gradient"][key] *
                          (successor[key] - test["actual"][key])).sum()
                         for key in successor)
        else:
            signed = task_readout(reference_model, observed, actions, transition,
                                  successor, test["suffix"], goal,
                                  test["channel"]) - test["target"]
        errors.append(signed.square())
    loss = torch.stack(errors).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite task-functional loss")
    return loss


def completed_path_loss(model, observed, actions, *, teacher_forced: bool,
                        task_goal=None):
    """Standard prediction/task losses on every completed start-end subpath.

    Every pair ``0 <= start < end <= T`` has equal weight. Both modes start
    with one real frame and use ``min(end-start, num_hist)`` source frames.
    Composed mode feeds predictions back; teacher mode uses the corresponding
    real cache. Action coordinates never enter the target loss. With a goal,
    visual and proprio task-cost differences are summed before squaring.
    This is a multi-step reference objective, not a new mechanism.
    """
    validation_goal = (task_goal if task_goal is not None
                       else _slice_obs(observed, 0, 1))
    _validate_chain(observed, actions, validation_goal)
    if int(model.num_hist) < 1:
        raise ValueError("num_hist must be positive")
    if model.concat_dim not in (0, 1):
        raise ValueError("unsupported native token layout")
    if model.concat_dim == 1 and int(model.action_dim) < 1:
        raise ValueError("concat_dim=1 requires a positive action_dim")
    length = int(actions.shape[1])
    with torch.no_grad():
        padded = torch.cat((actions, torch.zeros_like(actions[:, :1])), dim=1)
        targets = _join_encoded_observation_action(model, observed, padded)
    losses = []
    for start in range(length):
        if not teacher_forced:
            _, composed = rollout_from_zobs(
                model, _slice_obs(observed, start, start + 1), actions[:, start:])
        for end in range(start + 1, length + 1):
            if teacher_forced:
                first = max(start, end - int(model.num_hist))
                source = _join_encoded_observation_action(
                    model, _slice_obs(observed, first, end), actions[:, first:end])
                predicted = model.predict(source)[:, -1:]
            else:
                predicted = composed[:, end-start:end-start+1]
            actual = targets[:, end:end+1]
            if task_goal is None:
                if model.concat_dim == 0:
                    difference = predicted[:, :, :-1] - actual[:, :, :-1]
                else:
                    drop = int(model.action_dim)
                    difference = predicted[..., :-drop] - actual[..., :-drop]
                losses.append(difference.square().mean())
            else:
                predicted_obs, _ = model.separate_emb(predicted)
                signed = sum(
                    ((predicted_obs[key] - task_goal[key]).square()
                     - (observed[key][:, end:end+1] - task_goal[key]).square())
                    .flatten(start_dim=1).mean(dim=1)
                    for key in ("visual", "proprio"))
                losses.append(signed.square().mean())
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite completed-path loss")
    return loss
