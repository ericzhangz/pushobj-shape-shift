"""Repack completed fine-resolution evidence into native stride-five chains.

This module changes only data sampling, never the model's time step. Overlapping
windows and separate phases are not independent samples or one joined history.
"""

from __future__ import annotations

import torch


def _frameskip(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("frameskip must be a positive integer")
    return value


def _observations(observed, length):
    if not isinstance(observed, dict) or set(observed) != {"visual", "proprio"}:
        raise ValueError("expected visual/proprio observation tensors")
    for key, value in observed.items():
        if (not torch.is_tensor(value) or value.ndim < 3
                or value.shape[:2] != (1, length)):
            raise ValueError(f"{key} must have leading shape [1,{length}]")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite completed {key}")


def _actions(actions, length, width):
    if (not torch.is_tensor(actions) or actions.shape != (1, length, width)
            or not actions.is_floating_point()):
        raise ValueError(f"expected floating actions [1,{length},{width}]")
    if not torch.isfinite(actions).all():
        raise FloatingPointError("nonfinite completed actions")


def merge_completed_lowlevel(segments, frameskip=5):
    """Join full-resolution completed chunks, removing equal boundary copies.

    Each input is ``(transformed_observed, normalized_packed_actions)`` with
    ``frameskip+1`` real frames and one packed ``2*frameskip`` action. Exact
    boundary equality is required; missing or inconsistent frames fail closed.
    Returns the original transformed coordinates at all N+1 times and actions
    of shape [1,N,2], without re-normalization or any future observations.
    """
    frameskip = _frameskip(frameskip)
    segments = list(segments)
    if not segments:
        raise ValueError("at least one completed chunk is required")
    first_observed, first_actions = segments[0]
    for index, (observed, actions) in enumerate(segments):
        _observations(observed, frameskip + 1)
        _actions(actions, 1, 2 * frameskip)
        if actions.dtype != first_actions.dtype or actions.device != first_actions.device:
            raise ValueError("completed action dtype/device changed across chunks")
        for key, value in observed.items():
            reference = first_observed[key]
            if (value.shape != reference.shape or value.dtype != reference.dtype
                    or value.device != reference.device):
                raise ValueError(f"completed {key} layout changed across chunks")
            if index and not torch.equal(segments[index - 1][0][key][:, -1], value[:, 0]):
                raise ValueError(f"noncontinuous completed {key} boundary at chunk {index}")
    fine_observed = {
        key: torch.cat([first_observed[key]] + [observed[key][:, 1:]
                       for observed, _ in segments[1:]], dim=1)
        for key in first_observed
    }
    low_actions = torch.cat([actions.reshape(1, frameskip, 2)
                             for _, actions in segments], dim=1)
    return fine_observed, low_actions


def phase_chains(encoded_fine, low_actions, frameskip=5):
    """Return separate native chains for every legal phase of completed data.

    A transition at low-level start s uses actions [s,s+frameskip) and the real
    observation at s+frameskip. Only windows fully inside supplied data exist.
    The returned observation/action chains can use the existing native history
    interface unchanged; do not concatenate different phases before prediction.
    Metadata times are relative to the beginning of the supplied evidence.
    """
    frameskip = _frameskip(frameskip)
    if not torch.is_tensor(low_actions) or low_actions.ndim != 3:
        raise ValueError("expected low-level actions [1,N,2]")
    length = low_actions.shape[1]
    _actions(low_actions, length, 2)
    _observations(encoded_fine, length + 1)
    chains = []
    for phase in range(frameskip):
        starts = list(range(phase, length - frameskip + 1, frameskip))
        if not starts:
            continue
        ends = [start + frameskip for start in starts]
        times = [phase] + ends
        chains.append({
            "phase": phase,
            "observed": {key: value[:, times] for key, value in encoded_fine.items()},
            "actions": torch.cat([low_actions[:, start:start + frameskip].reshape(
                1, 1, 2 * frameskip) for start in starts], dim=1),
            "lowlevel_starts": starts,
            "lowlevel_ends": ends,
            "observation_times": times,
        })
    return chains
