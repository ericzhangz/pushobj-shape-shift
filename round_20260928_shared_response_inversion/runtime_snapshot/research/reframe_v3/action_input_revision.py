"""Small action-input calibration reference on the native prediction path.

Only completed encoded experience enters fitting. The 10-dimensional input
lift is compared under constant and declared fiber features; a shared output
translation is eliminated analytically. This is not a novelty claim.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from research.reframe_v3.increment_response import (
    ACTION_DIM, OBS_DIM, PROPRIO_DIM, VISUAL_DIM, _task_scale,
    _validate_model, _validate_packed,
)
from research.reframe_v3.matched_feedback_forecast import _join_encoded_observation_action


KINDS = ("constant", "fiber")
LEARNING_RATE = .01
ALLOWED_STEPS = (20, 100)


def _fixed_vector(value, reference, name):
    if not torch.is_tensor(value) or value.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape [10]")
    value = value.detach().to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {name}")
    return value


@contextmanager
def action_input_revision_context(model, b, *, kind, n, center):
    """Temporarily use e_b(a)=e_0(a)+phi(a)b without changing any weights.

    ``b`` is not detached: gradients through the original encoder, raw actions,
    and the lift remain available. Fixed n/center and saved CPU states are moved
    to the native action device and dtype. Exit restores the exact prior method.
    """
    _validate_model(model)
    if kind not in KINDS or not torch.is_tensor(b) or b.shape != (ACTION_DIM,):
        raise ValueError("expected constant/fiber kind and a [10] input lift")
    if not torch.isfinite(b).all():
        raise FloatingPointError("nonfinite input lift")
    original = model.encode_act
    had_instance_method = "encode_act" in model.__dict__
    instance_method = model.__dict__.get("encode_act")

    def encode_act(action):
        if action.ndim != 3 or action.shape[-1] != ACTION_DIM or not action.is_floating_point():
            raise ValueError("expected native [batch,time,10] floating actions")
        native = original(action)
        if native.shape != action.shape:
            raise ValueError("input revision requires native 10-to-10 action encoding")
        fixed_n = _fixed_vector(n, action, "n")
        fixed_center = _fixed_vector(center, action, "center")
        phi = (torch.ones_like(action[..., :1]) if kind == "constant"
               else ((action - fixed_center) * fixed_n).sum(dim=-1, keepdim=True))
        result = native + phi.to(native.dtype) * b.to(device=native.device, dtype=native.dtype)
        if not torch.isfinite(result).all():
            raise FloatingPointError("action-input revision produced nonfinite embeddings")
        return result

    model.encode_act = encode_act
    try:
        yield model
    finally:
        if had_instance_method:
            model.encode_act = instance_method
        else:
            del model.encode_act


def _validate_fit(model, observed, actions, n, center, kind, steps):
    _validate_model(model)
    if kind not in KINDS or steps not in ALLOWED_STEPS:
        raise ValueError("use constant/fiber and one of the fixed 20/100-step budgets")
    if torch.is_inference_mode_enabled():
        raise ValueError("input calibration requires autograd, not inference_mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all model parameters must already be frozen")
    if any(module.training for module in model.modules()):
        raise ValueError("model must already be in deterministic eval mode")
    if (actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] < 1
            or actions.shape[2] != ACTION_DIM or not actions.is_floating_point()):
        raise ValueError("fit needs one nonempty completed [1,T,10] action chain")
    length = actions.shape[1]
    if (set(observed) != {"visual", "proprio"}
            or observed["visual"].shape != (1, length + 1, 1, VISUAL_DIM)
            or observed["proprio"].shape != (1, length + 1, PROPRIO_DIM)):
        raise ValueError("fit accepts exactly T+1 native encoded observations, without future labels")
    if any(value.device != actions.device or value.dtype != actions.dtype for value in observed.values()):
        raise ValueError("encoded observations and actions must share native dtype and device")
    if not all(torch.isfinite(value).all() for value in (*observed.values(), actions)):
        raise FloatingPointError("nonfinite completed input-calibration evidence")
    n = _fixed_vector(n, actions, "n")
    center = _fixed_vector(center, actions, "center")
    if not torch.isclose(n.norm(), n.new_tensor(1.), rtol=1e-5, atol=1e-6):
        raise ValueError("n must be the supplied unit fiber direction")
    if not torch.allclose(center, actions.mean(dim=(0, 1)), rtol=0., atol=1e-6):
        raise ValueError("center must be the mean of exactly the supplied completed actions")
    return n, center


def _residuals(model, observed, actions):
    """Actual minus native prediction, with each transition's real cache."""
    residuals = []
    for transition in range(actions.shape[1]):
        first = max(0, transition + 1 - int(model.num_hist))
        source = _join_encoded_observation_action(
            model, {key: value[:, first:transition+1] for key, value in observed.items()},
            actions[:, first:transition+1])
        _validate_packed(source, int(model.num_hist))
        prediction = model.predict(source)
        if prediction.shape != source.shape or not torch.isfinite(prediction).all():
            raise ValueError("native predictor returned invalid observation/action tokens")
        actual = torch.cat((observed["visual"][0, transition+1, 0],
                            observed["proprio"][0, transition+1]))
        residuals.append(actual - prediction[0, -1, 0, :OBS_DIM])
    return torch.stack(residuals)


def _losses(residuals):
    mean = residuals.mean(dim=0)
    return residuals.square().mean(), (residuals - mean).square().mean(), mean


def fit_action_input_revision(model, observed, actions, *, kind, n, center, steps=20):
    """Fit only b; profile out c_b=mean(actual-prediction) over real transitions.

    Native observation coordinates (394) and transitions receive equal weight.
    The two prescribed budgets share Adam lr=.01 and zero initialization.
    A single transition has identically zero centered objective; it still runs
    the stated budget and reports that degeneracy without a transfer claim.
    Returned tensors are detached CPU states. ``output_response`` can be passed
    directly to the existing ``increment_response_context`` after input fitting.
    """
    n, center = _validate_fit(model, observed, actions, n, center, kind, steps)
    observed = {key: value.detach() for key, value in observed.items()}
    actions = actions.detach()
    with torch.enable_grad():
        b = torch.nn.Parameter(actions.new_zeros(ACTION_DIM))
        optimizer = torch.optim.Adam([b], lr=LEARNING_RATE)
        loss_trace, grad_norms = [], []
        with action_input_revision_context(model, b, kind=kind, n=n, center=center):
            with torch.no_grad():
                before_full, before_centered, _ = _losses(_residuals(model, observed, actions))
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                full, centered, _ = _losses(_residuals(model, observed, actions))
                if not torch.isfinite(centered):
                    raise FloatingPointError("nonfinite centered factual objective")
                centered.backward()
                if b.grad is None or not torch.isfinite(b.grad).all():
                    raise FloatingPointError("missing or nonfinite input-lift gradient")
                loss_trace.append({"full": float(full.detach()), "centered": float(centered.detach())})
                grad_norms.append(float(b.grad.norm()))
                optimizer.step()
            with torch.no_grad():
                after_full, after_centered, c = _losses(_residuals(model, observed, actions))
        if not all(torch.isfinite(value).all() for value in (b, c, after_full, after_centered)):
            raise FloatingPointError("nonfinite fitted input revision")
    singleton = actions.shape[1] == 1
    if singleton and (torch.count_nonzero(b).item() or after_centered.item() != 0):
        raise AssertionError("one-transition centered fit must retain zero b exactly")
    return {
        "kind": kind, "b": b.detach().cpu().clone(), "c": c.detach().cpu().clone(),
        "n": n.detach().cpu().clone(), "center": center.detach().cpu().clone(),
        "steps": steps, "lr": LEARNING_RATE, "num_evidence": int(actions.shape[1]),
        "status": "single_transition_centered_objective_zero" if singleton else "fitted",
        "before_full_loss": float(before_full), "before_centered_loss": float(before_centered),
        "after_full_loss": float(after_full), "after_centered_loss": float(after_centered),
        "input_b_norm": float(b.detach().norm()), "grad_norms": grad_norms, "loss_trace": loss_trace,
        "loss_definition": "mean_transition_coordinate residual^2 on native 394 observation coordinates; output mean analytically removed",
        "output_response": {"kind": "mean_residual", "scale": _task_scale(),
                            "mean_residual": c.detach().cpu().double().clone()},
    }
