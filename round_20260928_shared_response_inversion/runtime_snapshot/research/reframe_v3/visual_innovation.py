"""D-only conditional visual residual reference, not an originality claim.

The centered least-squares relation is ``r_v = mean_v + (r_p-mean_p) @ C``.
Here r_p is the same-step actuator-revised proprio prediction minus the native
prediction, not a true query-time residual. No goal, object state, candidate
outcome, rollout implementation, ridge, or clipping is used in this module.
Transfer of this classical conditional residual relation is unverified.
"""

from __future__ import annotations

import math

import torch

from research.reframe_v3.actuator_response import make_actuator_transition
from research.reframe_v3.completed_phase_evidence import phase_chains
from research.reframe_v3.matched_feedback_forecast import _join_encoded_observation_action


VISUAL_DIM = 384
PROPRIO_DIM = 10
ACTION_DIM = 10
FRAMESKIP = 5


def _floating(value, shape, name):
    if (not torch.is_tensor(value) or not value.is_floating_point()
            or tuple(value.shape) != tuple(shape)):
        raise ValueError(f"{name} must be a floating tensor with shape {tuple(shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {name}")


@torch.no_grad()
def collect_innovation_evidence(model, fine_normalized_obs, encoded_fine, low_actions,
                                fitted_actuator):
    """Collect X=[windows,10], Y=[windows,384] in phase_chains window order.

    Each native prediction uses only real same-phase stride-five history ending
    at that window's start, capped at the host's three history tokens. The
    actuator is advanced from that same real start through that window's five
    actions. Only Y uses its real successor visual encoding. All supplied
    observations/actions must belong to one completed prefix before the query.
    """
    expected = {"concat_dim": 1, "num_hist": 3, "proprio_dim": PROPRIO_DIM,
                "action_dim": ACTION_DIM, "num_proprio_repeat": 1, "num_action_repeat": 1}
    if any(getattr(model, key, None) != value for key, value in expected.items()):
        raise ValueError("visual innovation supports only the native single-token PushObj layout/history")
    if model.training:
        raise ValueError("innovation evidence requires the native base model in eval mode")
    if (not torch.is_tensor(low_actions) or low_actions.ndim != 3
            or low_actions.shape[0] != 1 or low_actions.shape[1] < FRAMESKIP):
        raise ValueError("at least five completed low-level actions [1,N,2] are required")
    length = low_actions.shape[1]
    _floating(low_actions, (1, length, 2), "low-level actions")
    for observed in (fine_normalized_obs, encoded_fine):
        if not isinstance(observed, dict) or set(observed) != {"visual", "proprio"}:
            raise ValueError("only completed visual/proprio observations are accepted; no goal or state fields")
    fine_visual = fine_normalized_obs["visual"]
    if (not torch.is_tensor(fine_visual) or fine_visual.ndim < 3
            or fine_visual.shape[:2] != (1, length + 1)):
        raise ValueError("exactly N+1 completed fine visual observations are required")
    _floating(fine_visual, fine_visual.shape, "fine visual observations")
    _floating(fine_normalized_obs["proprio"], (1, length + 1, 4), "fine proprio observations")
    _floating(encoded_fine["visual"], (1, length + 1, 1, VISUAL_DIM), "encoded visual observations")
    _floating(encoded_fine["proprio"], (1, length + 1, PROPRIO_DIM), "encoded proprio observations")
    for value in (*fine_normalized_obs.values(), *encoded_fine.values()):
        if value.device != low_actions.device or value.dtype != low_actions.dtype:
            raise ValueError("completed innovation observations/actions must share dtype and device")
    if (not isinstance(fitted_actuator.get("num_transitions"), int)
            or not 4 <= fitted_actuator["num_transitions"] <= length):
        raise ValueError("actuator fitting cutoff must lie within the supplied completed prefix")
    transition = make_actuator_transition(fitted_actuator)
    xs, ys, windows = [], [], []
    for chain in phase_chains(encoded_fine, low_actions, FRAMESKIP):
        for index, (start, end) in enumerate(zip(chain["lowlevel_starts"], chain["lowlevel_ends"])):
            first = max(0, index + 1 - model.num_hist)
            history = {key: value[:, first:index + 1] for key, value in chain["observed"].items()}
            source = _join_encoded_observation_action(model, history, chain["actions"][:, first:index + 1])
            _floating(source, (1, index + 1 - first, 1, VISUAL_DIM + PROPRIO_DIM + ACTION_DIM),
                      "native history tokens")
            predicted = model.predict(source)
            _floating(predicted, source.shape, "native predictions")
            revised_raw = transition(fine_normalized_obs["proprio"][:, start:start + 1],
                                     chain["actions"][:, index:index + 1])
            revised_prop = model.encode_proprio(revised_raw)
            _floating(revised_prop, (1, 1, PROPRIO_DIM), "actuator-revised proprio encoding")
            last = predicted[0, -1, 0].detach().cpu().double()
            xs.append(revised_prop[0, 0].detach().cpu().double() - last[VISUAL_DIM:VISUAL_DIM + PROPRIO_DIM])
            ys.append(encoded_fine["visual"][0, end, 0].detach().cpu().double() - last[:VISUAL_DIM])
            windows.append({"phase": chain["phase"], "phase_transition": index,
                            "lowlevel_start": start, "lowlevel_end": end,
                            "history_observation_times": chain["observation_times"][first:index + 1],
                            "history_action_starts": chain["lowlevel_starts"][first:index + 1]})
    if len(xs) != length - FRAMESKIP + 1:
        raise AssertionError("innovation collection omitted a legal completed window")
    return torch.stack(xs), torch.stack(ys), {
        "num_windows": len(xs), "completed_lowlevel_transitions": length,
        "actuator_fit_transitions": fitted_actuator["num_transitions"], "frameskip": FRAMESKIP,
        "num_hist": model.num_hist, "windows": windows,
        "x_definition": "encode_proprio(actuator(real_start, five_actions)) minus native predicted proprio",
        "y_definition": "real successor encoded visual minus native predicted visual",
        "sampling": "phase_chains order; overlapping completed windows are not independent samples",
        "goal_used": False, "candidate_outcomes_used": False,
    }


def fit_visual_innovation(X, Y):
    """CPU FP64 centered min-Frobenius-norm least squares, including rank zero.

    ``condition`` is infinite when any of the ten input directions is absent;
    ``retained_condition`` describes only the nonzero retained spectrum (None
    at rank zero). Neither is an acceptance gate. A single observation has
    exactly zero centered design and C=0, i.e. the explicit mean-only solution.
    """
    if not torch.is_tensor(X) or X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("X must contain at least one completed innovation row")
    count = X.shape[0]
    _floating(X, (count, PROPRIO_DIM), "X")
    _floating(Y, (count, VISUAL_DIM), "Y")
    X, Y = [value.detach().cpu().double().clone() for value in (X, Y)]
    mean_p, mean_v = X.mean(dim=0), Y.mean(dim=0)
    centered_x, centered_y = X - mean_p, Y - mean_v
    solved = torch.linalg.lstsq(centered_x, centered_y, rcond=None, driver="gelsd")
    C, rank, singular_values = solved.solution, int(solved.rank), solved.singular_values
    residual = centered_x @ C - centered_y
    if not torch.isfinite(C).all() or not torch.isfinite(residual).all():
        raise FloatingPointError("nonfinite centered innovation fit")
    if rank == 0 and torch.count_nonzero(C):
        raise AssertionError("rank-zero minimum-norm innovation solution must be exactly zero")
    values = {
        "fit_rms": float(residual.square().mean().sqrt()),
        "fit_max_abs": float(residual.abs().max()),
        "mean_only_rms": float(centered_y.square().mean().sqrt()),
        "coefficient_frobenius": float(torch.linalg.vector_norm(C)),
        "coefficient_operator_norm": float(torch.linalg.matrix_norm(C, ord=2)),
        "mean_p_norm": float(torch.linalg.vector_norm(mean_p)),
        "mean_v_norm": float(torch.linalg.vector_norm(mean_v)),
        "centered_x_frobenius": float(torch.linalg.vector_norm(centered_x)),
        "centered_y_frobenius": float(torch.linalg.vector_norm(centered_y)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("nonfinite innovation fit diagnostics")
    return {"C": C.detach().clone(), "mean_p": mean_p, "mean_v": mean_v,
            "rank": rank, "singular_values": singular_values.detach().clone(),
            "num_windows": count, "rank_tolerance": float(singular_values[0]) * max(centered_x.shape)
                                                         * torch.finfo(torch.float64).eps,
            "condition": float(singular_values[0] / singular_values[-1]) if rank == PROPRIO_DIM else math.inf,
            "retained_condition": float(singular_values[0] / singular_values[rank - 1]) if rank else None,
            "solver": "centered CPU FP64 torch.linalg.lstsq gelsd rcond=None; minimum Frobenius norm",
            "condition_definition": "infinite if input column rank is below ten; not a fitting threshold",
            "fit_error_definition": "RMS across completed windows and 384 visual coordinates",
            **values}


def make_visual_innovation(fitted, kind="conditional"):
    """Return a differentiable last-token visual correction with frozen constants."""
    if kind not in ("mean", "conditional"):
        raise ValueError("visual innovation kind must be mean or conditional")
    constants = {}
    for key, shape in (("C", (PROPRIO_DIM, VISUAL_DIM)), ("mean_p", (PROPRIO_DIM,)),
                       ("mean_v", (VISUAL_DIM,))):
        _floating(fitted[key], shape, f"fitted {key}")
        constants[key] = fitted[key].detach().cpu().double().clone()
    devices = {}

    def correct(pred_visual, pred_prop, revised_prop):
        if not torch.is_tensor(pred_visual) or pred_visual.ndim != 4 or pred_visual.shape[0] < 1:
            raise ValueError("visual innovation requires nonempty [batch,1,1,384] visual tokens")
        batch = pred_visual.shape[0]
        _floating(pred_visual, (batch, 1, 1, VISUAL_DIM), "predicted visual")
        _floating(pred_prop, (batch, 1, 1, PROPRIO_DIM), "predicted proprio")
        _floating(revised_prop, (batch, 1, 1, PROPRIO_DIM), "revised proprio")
        if any(value.device != pred_visual.device or value.dtype != pred_visual.dtype
               for value in (pred_prop, revised_prop)):
            raise ValueError("innovation callback inputs must share dtype and device")
        if pred_visual.device not in devices:
            devices[pred_visual.device] = {key: value.to(pred_visual.device) for key, value in constants.items()}
        local = devices[pred_visual.device]
        correction = local["mean_v"]
        if kind == "conditional":
            innovation = revised_prop.double() - pred_prop.double()
            correction = correction + (innovation - local["mean_p"]) @ local["C"]
        result = (pred_visual.double() + correction).to(pred_visual.dtype)
        if not torch.isfinite(result).all():
            raise FloatingPointError("visual innovation produced nonfinite visual tokens")
        return result

    return correct
