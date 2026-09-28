"""Falsifiable shared-increment reference, not an original learning algorithm.

Hypothesis: within completed experience and subsequent queries, one shared
linear map converts the native model's observation increments into real ones.
The map is a standard least-change multisecant construction. Neither fitting
its historical secants nor a small training error establishes that hypothesis
for new actions. The mean-residual and scalar-gain arms are explicit controls.

Only the released PushObj layout is supported: one token, 384 visual and 10
proprio coordinates, followed by 10 action coordinates, without repetition.
No rollout implementation, candidate truth, or query-dependent metric occurs
here. The context wraps the existing predictor and retains native cache logic.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from research.reframe_v3.matched_feedback_forecast import (
    _join_encoded_observation_action,
)


VISUAL_DIM = 384
PROPRIO_DIM = 10
OBS_DIM = VISUAL_DIM + PROPRIO_DIM
ACTION_DIM = 10
KINDS = ("mean_residual", "scalar_gain", "multisecant")


def _task_scale():
    # sum_c mean((z_c-goal_c)^2) = ||scale * (z-goal)||^2.
    return torch.cat((torch.full((VISUAL_DIM,), 1 / math.sqrt(VISUAL_DIM),
                                 dtype=torch.float64),
                      torch.full((PROPRIO_DIM,), 1 / math.sqrt(PROPRIO_DIM),
                                 dtype=torch.float64)))


def _validate_model(model):
    expected = {"concat_dim": 1, "proprio_dim": PROPRIO_DIM,
                "action_dim": ACTION_DIM, "num_proprio_repeat": 1,
                "num_action_repeat": 1}
    if any(getattr(model, key, None) != value for key, value in expected.items()):
        raise ValueError("increment response supports only native single-token PushObj layout")
    if int(model.num_hist) < 1:
        raise ValueError("native num_hist must be positive")


def _validate_packed(value, num_hist):
    if (value.ndim != 4 or value.shape[2:] != (1, OBS_DIM + ACTION_DIM)
            or not 1 <= value.shape[1] <= num_hist):
        raise ValueError("expected native [batch, history<=num_hist, 1, 404] tokens")
    if not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite native predictor tokens")


@torch.no_grad()
def collect_increment_evidence(model, observed, actions):
    """Return CPU FP64 V,R=[394,T] and task scale=[394] from one real chain.

    ``observed`` is already encoded, with T+1 observations. Each transition
    uses all of its available real history, truncated only by native num_hist.
    The model's last output token predicts that transition's real successor.
    """
    _validate_model(model)
    if set(observed) != {"visual", "proprio"}:
        raise ValueError("encoded observation must have visual and proprio only")
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] < 1:
        raise ValueError("evidence must be one episode with nonempty completed actions")
    length = actions.shape[1]
    if (observed["visual"].shape != (1, length + 1, 1, VISUAL_DIM)
            or observed["proprio"].shape != (1, length + 1, PROPRIO_DIM)):
        raise ValueError("expected T+1 native encoded observations for T actions")
    if not all(torch.isfinite(x).all() for x in (*observed.values(), actions)):
        raise FloatingPointError("nonfinite completed increment evidence")
    increments, residuals = [], []
    for transition in range(length):
        start = max(0, transition + 1 - int(model.num_hist))
        history = {key: value[:, start:transition + 1]
                   for key, value in observed.items()}
        source = _join_encoded_observation_action(
            model, history, actions[:, start:transition + 1])
        _validate_packed(source, int(model.num_hist))
        predicted = model.predict(source)
        if predicted.shape != source.shape or not torch.isfinite(predicted).all():
            raise ValueError("native predictor returned invalid tokens")
        predicted_obs = predicted[0, -1, 0, :OBS_DIM].detach().cpu().double()
        last_obs = source[0, -1, 0, :OBS_DIM].detach().cpu().double()
        actual_obs = torch.cat((observed["visual"][0, transition + 1, 0],
                                observed["proprio"][0, transition + 1])).detach().cpu().double()
        increments.append(predicted_obs - last_obs)
        residuals.append(actual_obs - predicted_obs)
    return torch.stack(increments, dim=1), torch.stack(residuals, dim=1), _task_scale()


def fit_increment_response(V, R, scale, kind):
    """Fit the declared reference without clipping, ridge, or tuned rank.

    Matrix columns are completed transitions. ``fit_rms`` is the root mean
    across transitions of squared task-normalized residual error. The reported
    operator_gain is the spectral norm of the linear increment map in the same
    normalized coordinates; mean_residual has identity linear part.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown increment response kind {kind!r}")
    V, R, scale = [x.detach().cpu().to(torch.float64).clone() for x in (V, R, scale)]
    if V.ndim != 2 or V.shape[0] != OBS_DIM or V.shape[1] < 1 or R.shape != V.shape:
        raise ValueError("V and R must have equal nonempty [394, transitions] shapes")
    if scale.shape != (OBS_DIM,) or not torch.equal(scale, _task_scale()):
        raise ValueError("scale must be the fixed native visual/proprio task mean scale")
    if not all(torch.isfinite(x).all() for x in (V, R, scale)):
        raise FloatingPointError("nonfinite increment fitting input")
    normalized_v, normalized_r = scale[:, None] * V, scale[:, None] * R
    U, singular_values, Vh = torch.linalg.svd(normalized_v, full_matrices=False)
    tolerance = (max(normalized_v.shape) * torch.finfo(torch.float64).eps
                 * float(singular_values[0]))
    retained = singular_values > tolerance
    inverse = torch.zeros_like(singular_values)
    inverse[retained] = singular_values[retained].reciprocal()
    pseudoinverse = (Vh.T * inverse) @ U.T
    fitted = {"kind": kind, "scale": scale, "rank": int(retained.sum()),
              "rank_tolerance": tolerance, "singular_values": singular_values,
              "num_evidence": int(V.shape[1]),
              "fit_rms_definition": "sqrt(mean_transition(sum_coordinate((scale * residual_error)^2)))"}
    identity = torch.eye(OBS_DIM, dtype=torch.float64)
    if kind == "mean_residual":
        fitted["mean_residual"] = R.mean(dim=1)
        fitted_r = fitted["mean_residual"][:, None].expand_as(R)
        operator = identity
    elif kind == "scalar_gain":
        denominator = normalized_v.square().sum()
        if denominator == 0:
            raise ValueError("scalar gain is unidentifiable from zero predicted increments")
        gain = ((normalized_v * (normalized_v + normalized_r)).sum() / denominator)
        fitted["gain"] = float(gain)
        fitted_r = (gain - 1) * V
        operator = gain * identity
    else:
        fitted["left"] = normalized_r
        fitted["right"] = pseudoinverse
        normalized_correction = normalized_r @ pseudoinverse
        fitted_r = (normalized_correction @ normalized_v) / scale[:, None]
        operator = identity + normalized_correction
    fitted["fit_rms"] = float(((scale[:, None] * (fitted_r - R)).square()
                               .sum(dim=0).mean()).sqrt())
    fitted["uncorrected_rms"] = float(normalized_r.square().sum(dim=0).mean().sqrt())
    fitted["operator_gain"] = float(torch.linalg.svdvals(operator)[0])
    if not all(math.isfinite(fitted[key])
               for key in ("fit_rms", "uncorrected_rms", "operator_gain")):
        raise FloatingPointError("nonfinite fitted increment response")
    return fitted


@contextmanager
def increment_response_context(model, fitted):
    """Apply one fitted shared response at every native predicted obs token.

    No parameters, input action slots, or output action coordinates are changed.
    Native autoregressive code still owns history truncation and action insertion.
    Gradient flow through the original prediction and correction is retained.
    """
    _validate_model(model)
    kind = fitted["kind"]
    if kind not in KINDS:
        raise ValueError("invalid fitted increment response kind")
    if not torch.equal(fitted["scale"].cpu().double(), _task_scale()):
        raise ValueError("fitted response uses an unsupported task scale")
    original = model.predict
    had_instance_predict = "predict" in model.__dict__
    instance_predict = model.__dict__.get("predict")
    cached_device = {}

    def predict(source):
        _validate_packed(source, int(model.num_hist))
        predicted = original(source)
        if predicted.shape != source.shape:
            raise ValueError("native predictor changed the packed token layout")
        if source.device not in cached_device:
            cached_device[source.device] = {
                key: value.to(device=source.device, dtype=torch.float64)
                for key, value in fitted.items() if torch.is_tensor(value)}
        tensors = cached_device[source.device]
        increment = (predicted[..., :OBS_DIM].double()
                     - source[..., :OBS_DIM].double())
        if kind == "mean_residual":
            correction = tensors["mean_residual"]
        elif kind == "scalar_gain":
            correction = (fitted["gain"] - 1) * increment
        else:
            normalized = increment * tensors["scale"]
            coefficients = normalized @ tensors["right"].T
            correction = (coefficients @ tensors["left"].T) / tensors["scale"]
        corrected_obs = (predicted[..., :OBS_DIM].double() + correction).to(predicted.dtype)
        result = torch.cat((corrected_obs, predicted[..., OBS_DIM:]), dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("shared increment response produced nonfinite tokens")
        return result

    model.predict = predict
    try:
        yield model
    finally:
        if had_instance_predict:
            model.predict = instance_predict
        else:
            del model.predict
