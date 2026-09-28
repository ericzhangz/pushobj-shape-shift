"""Observed actuator response: a bounded, non-original identification reference.

The explicit hypothesis is a shared translation-equivariant linear relation
``[p_next-p, v_next] = [v, u] @ M`` on raw proprio and low-level actions.
Only caller-supplied completed transitions identify M. There is no intercept,
regularization, clipping, environment-parameter access, or alternate predictor.
Whether this relation transfers to new actions is an experimental question.
"""

from __future__ import annotations

import torch


CHUNK_LENGTH = 5


def _constant(value, size, name, *, positive=False):
    result = torch.as_tensor(value).detach().cpu().to(torch.float64).clone()
    if result.shape != (size,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [{size}] normalization constant")
    if positive and not (result > 0).all():
        raise ValueError(f"{name} must be strictly positive")
    return result


def _normalization(source):
    getter = source.__getitem__ if isinstance(source, dict) else lambda name: getattr(source, name)
    return {
        "proprio_mean": _constant(getter("proprio_mean"), 4, "proprio_mean"),
        "proprio_std": _constant(getter("proprio_std"), 4, "proprio_std", positive=True),
        "action_mean": _constant(getter("action_mean"), 2, "action_mean"),
        "action_std": _constant(getter("action_std"), 2, "action_std", positive=True),
    }


def fit_actuator_response(normalized_proprio, normalized_low_actions, preprocessor):
    """Fit M on exactly N actions and N+1 real proprio observations, CPU FP64.

    Arguments are [1,N+1,4] and [1,N,2], respectively. Current complete query
    prefixes contain N=10 or N=20; shorter completed prefixes may be used for
    time-ordered checks if their design matrix has full rank four. Extra future
    observations are rejected rather than silently truncated.
    """
    if (not torch.is_tensor(normalized_proprio)
            or not torch.is_tensor(normalized_low_actions)
            or not torch.is_floating_point(normalized_proprio)
            or not torch.is_floating_point(normalized_low_actions)):
        raise ValueError("normalized observations and actions must be floating tensors")
    if (normalized_low_actions.ndim != 3 or normalized_low_actions.shape[0] != 1
            or normalized_low_actions.shape[2] != 2):
        raise ValueError("completed low-level actions must have shape [1,N,2]")
    count = normalized_low_actions.shape[1]
    if count < 4 or normalized_proprio.shape != (1, count + 1, 4):
        raise ValueError("fit requires exactly N+1 proprio observations for N>=4 completed actions")
    if (not torch.isfinite(normalized_proprio).all()
            or not torch.isfinite(normalized_low_actions).all()):
        raise FloatingPointError("nonfinite completed actuator evidence")
    constants = _normalization(preprocessor)
    proprio = (normalized_proprio.detach().cpu().double()[0]
               * constants["proprio_std"] + constants["proprio_mean"])
    actions = (normalized_low_actions.detach().cpu().double()[0]
               * constants["action_std"] + constants["action_mean"])
    design = torch.cat((proprio[:-1, 2:], actions), dim=1)
    targets = torch.cat((proprio[1:, :2] - proprio[:-1, :2], proprio[1:, 2:]), dim=1)
    result = torch.linalg.lstsq(design, targets, rcond=None, driver="gelsd")
    rank = int(result.rank)
    if rank != 4:
        raise ValueError(f"actuator response requires rank four; observed design rank is {rank}")
    matrix = result.solution
    residual = design @ matrix - targets
    if not torch.isfinite(matrix).all() or not torch.isfinite(residual).all():
        raise FloatingPointError("nonfinite actuator least-squares fit")
    return {
        "M": matrix.detach().clone(), "rank": rank,
        "singular_values": result.singular_values.detach().clone(),
        "rank_tolerance": float(result.singular_values[0]) * max(design.shape)
                          * torch.finfo(torch.float64).eps,
        "fit_rms": float(residual.square().mean().sqrt()),
        "fit_rms_by_output": residual.square().mean(dim=0).sqrt(),
        "fit_max_abs": float(residual.abs().max()),
        "num_transitions": count, "chunk_length": CHUNK_LENGTH,
        "input_order": ["v_x", "v_y", "u_x", "u_y"],
        "output_order": ["delta_p_x", "delta_p_y", "v_next_x", "v_next_y"],
        "hypothesis": "one translation-equivariant linear actuator response shared across actions",
        "fit_error_units": "raw coordinates; RMS across transitions and four output coordinates",
        "solver": "CPU FP64 torch.linalg.lstsq gelsd with default machine-precision rank cutoff",
        **constants,
    }


def make_actuator_transition(fitted, *, path_observer=None):
    """Return a differentiable normalized [B,1,4] -> [B,1,4] five-action callback.

    The packed input action is [B,1,10], ordered as five consecutive raw-action
    pairs after denormalization. All five transitions use one fitted matrix.
    Constants are copied when this callback is created; no data are refitted.
    Optional path_observer receives raw [B,6,4] states (start plus five steps).
    This exposes the SAME recurrence to a contact readout, not a second actuator
    implementation. No observation buffer is constructed on the default path.
    """
    if fitted.get("rank") != 4 or fitted.get("chunk_length") != CHUNK_LENGTH:
        raise ValueError("callback requires a full-rank fitted five-action actuator response")
    if path_observer is not None and not callable(path_observer):
        raise ValueError("path_observer must be callable or None")
    matrix = torch.as_tensor(fitted["M"]).detach().cpu().double().clone()
    if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
        raise ValueError("fitted actuator matrix must be finite [4,4]")
    constants = _normalization(fitted)
    devices = {}

    def transition(normalized_prop, normalized_chunk):
        if (not torch.is_tensor(normalized_prop) or not torch.is_tensor(normalized_chunk)
                or not torch.is_floating_point(normalized_prop)
                or not torch.is_floating_point(normalized_chunk)):
            raise ValueError("actuator callback inputs must be floating tensors")
        if (normalized_prop.ndim != 3 or normalized_prop.shape[0] < 1
                or normalized_prop.shape[1:] != (1, 4)
                or normalized_chunk.shape != (normalized_prop.shape[0], 1, 10)):
            raise ValueError("actuator callback requires [batch,1,4] proprio and [batch,1,10] action")
        if (normalized_prop.device != normalized_chunk.device
                or normalized_prop.dtype != normalized_chunk.dtype):
            raise ValueError("actuator callback inputs must share device and dtype")
        if not torch.isfinite(normalized_prop).all() or not torch.isfinite(normalized_chunk).all():
            raise FloatingPointError("nonfinite actuator callback input")
        device = normalized_prop.device
        if device not in devices:
            devices[device] = {key: value.to(device) for key, value in constants.items()}
            devices[device]["M"] = matrix.to(device)
        local = devices[device]
        raw_prop = normalized_prop.double() * local["proprio_std"] + local["proprio_mean"]
        raw_actions = (normalized_chunk.double().reshape(-1, CHUNK_LENGTH, 2)
                       * local["action_std"] + local["action_mean"])
        raw_prop = raw_prop[:, 0]
        path = [raw_prop] if path_observer is not None else None
        for step in range(CHUNK_LENGTH):
            response = torch.cat((raw_prop[:, 2:], raw_actions[:, step]), dim=1) @ local["M"]
            raw_prop = torch.cat((raw_prop[:, :2] + response[:, :2], response[:, 2:]), dim=1)
            if path is not None:
                path.append(raw_prop)
        predicted = ((raw_prop[:, None] - local["proprio_mean"]) / local["proprio_std"])
        predicted = predicted.to(normalized_prop.dtype)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("actuator response produced nonfinite proprio")
        if path is not None:
            path_observer(torch.stack(path, dim=1))
        return predicted

    return transition
