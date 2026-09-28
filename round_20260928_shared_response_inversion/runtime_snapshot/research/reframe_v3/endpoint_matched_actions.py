"""Fixed endpoint-matched action probes for an already fitted actuator.

This is local linear algebra for evaluation, not another world model or an
update rule. Only completed commands set amplitude; no outcomes are accepted.
FP64 predicted endpoint equality is not a claim about the real environment.
"""

from __future__ import annotations

import math

import torch

from research.reframe_v3.actuator_response import CHUNK_LENGTH, _normalization


def actuator_matrices(fitted):
    """Column-state x' = A x + B u from row-layout [v,u] M=[delta_p,v']."""
    if fitted.get("rank") != 4 or fitted.get("chunk_length") != CHUNK_LENGTH:
        raise ValueError("a full-rank fitted five-command actuator is required")
    matrix = torch.as_tensor(fitted["M"]).detach().cpu().double().clone()
    if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
        raise ValueError("actuator M must be finite [4,4]")
    constants = _normalization(fitted)
    A = torch.zeros(4, 4, dtype=torch.float64)
    A[:2, :2] = torch.eye(2, dtype=torch.float64)
    A[:2, 2:] = matrix[:2, :2].T
    A[2:, 2:] = matrix[:2, 2:].T
    B = torch.cat((matrix[2:, :2].T, matrix[2:, 2:].T), dim=0)
    return A, B, constants


def endpoint_jacobian(fitted):
    """Raw end-state derivative with respect to ten normalized commands."""
    A, B, constants = actuator_matrices(fitted)
    normalized_B = B * constants["action_std"][None, :]
    G = torch.cat([torch.linalg.matrix_power(A, CHUNK_LENGTH-1-step) @ normalized_B
                   for step in range(CHUNK_LENGTH)], dim=1)
    if not torch.isfinite(G).all():
        raise FloatingPointError("nonfinite endpoint Jacobian")
    return G


def _kernel_direction(row_basis, pattern):
    direction = pattern - row_basis.T @ (row_basis @ pattern)
    norm = torch.linalg.vector_norm(direction)
    zero_tolerance = 256 * torch.finfo(torch.float64).eps * float(pattern.norm())
    if not torch.isfinite(norm) or float(norm) <= zero_tolerance:
        raise ValueError("fixed temporal pattern has zero numerical kernel projection; no replacement direction")
    return direction * (math.sqrt(10) / norm), float(norm), zero_tolerance


def _difference_path(A, normalized_B, change):
    """Raw p/v differences; affine normalization means and initial state cancel."""
    state = torch.zeros(4, dtype=torch.float64)
    path = []
    for action in change.reshape(CHUNK_LENGTH, 2):
        state = A @ state + normalized_B @ action
        path.append(state)
    return torch.stack(path)


def _path_metadata(path, suffix):
    return {
        f"raw_state_difference_path_{suffix}": path.tolist(),
        f"raw_endpoint_difference_{suffix}": path[-1].tolist(),
        f"raw_endpoint_max_abs_{suffix}": float(path[-1].abs().max()),
        f"internal_position_max_l2_{suffix}": float(path[:-1, :2].norm(dim=1).max()),
        f"internal_velocity_max_l2_{suffix}": float(path[:-1, 2:].norm(dim=1).max()),
    }


def endpoint_matched_actions(fitted, base, completed_low_actions):
    """Return nine fixed ``(name, CPU action tensor)`` pairs plus JSON metadata.

    Input base is [1,5,10]; completed normalized low-level commands are [1,N,2].
    Four later chunks remain exactly unchanged. Amplitude is the overall RMS
    of completed commands after subtracting the per-axis temporal mean, with
    no floor. Directions are projected fixed (+1,-1,0,0,0) x/y patterns, each
    normalized to L2 sqrt(10). Both signs at full/half amplitude are retained.

    All construction/endpoint checks use CPU FP64 without callback output
    casts. Returned plans use base.dtype; their cast-induced endpoint errors
    are separately recorded, never corrected by selecting different probes.
    """
    if (not torch.is_tensor(base) or base.shape != (1, 5, 10)
            or not base.is_floating_point() or not torch.isfinite(base).all()):
        raise ValueError("base must be finite floating [1,5,10]")
    if (not torch.is_tensor(completed_low_actions) or completed_low_actions.ndim != 3
            or completed_low_actions.shape[0] != 1 or completed_low_actions.shape[2] != 2
            or completed_low_actions.shape[1] < 1 or not completed_low_actions.is_floating_point()
            or not torch.isfinite(completed_low_actions).all()):
        raise ValueError("completed commands must be finite floating [1,N,2]")
    base_native = base.detach().cpu().clone()
    base64 = base_native.double()
    completed = completed_low_actions.detach().cpu().double()
    center = completed.mean(dim=(0, 1))
    amplitude = float((completed-center).square().mean().sqrt())
    if not math.isfinite(amplitude) or amplitude == 0.:
        raise ValueError("completed centered command RMS is zero/nonfinite; no amplitude floor is allowed")
    A, B, constants = actuator_matrices(fitted)
    normalized_B = B * constants["action_std"][None, :]
    G = endpoint_jacobian(fitted)
    _, singular_values, Vh = torch.linalg.svd(G, full_matrices=True)
    rank_tolerance = max(G.shape) * torch.finfo(torch.float64).eps * float(singular_values[0])
    rank = int((singular_values > rank_tolerance).sum())
    if rank != 4:
        raise ValueError(f"five-command endpoint Jacobian must have rank four; got {rank}")
    row_basis = Vh[:rank]
    directions, direction_metadata = {}, {}
    for axis, coordinate in (("x", 0), ("y", 1)):
        pattern = torch.zeros(CHUNK_LENGTH, 2, dtype=torch.float64)
        pattern[0, coordinate], pattern[1, coordinate] = 1., -1.
        direction, projected_norm, zero_tolerance = _kernel_direction(row_basis, pattern.flatten())
        directions[axis] = direction
        direction_metadata[axis] = {
            "fixed_pattern": pattern.flatten().tolist(), "direction": direction.tolist(),
            "projected_pattern_norm": projected_norm, "projection_zero_tolerance": zero_tolerance,
            "direction_l2": float(direction.norm()), "kernel_residual": (G @ direction).tolist(),
            "kernel_residual_l2": float((G @ direction).norm()),
        }
    candidates = [("base", base_native)]
    metadata_rows = []
    specifications = [("base", "none", 0., 0.)] + [
        (f"{axis}_{sign_name}_{scale_name}", axis, sign, fraction)
        for scale_name, fraction in (("full", 1.), ("half", .5))
        for axis in ("x", "y") for sign_name, sign in (("plus", 1.), ("minus", -1.))]
    for name, axis, sign, fraction in specifications:
        delta = (torch.zeros(10, dtype=torch.float64) if name == "base"
                 else sign*fraction*amplitude*directions[axis])
        plan64 = base64.clone()
        plan64[:, 0] += delta
        fp64_delta = plan64[0, 0]-base64[0, 0]
        path64 = _difference_path(A, normalized_B, fp64_delta)
        tolerance = 256 * torch.finfo(torch.float64).eps * max(
            1., float(singular_values[0])*float(delta.norm()))
        if float(path64[-1].norm()) > tolerance or float((G @ fp64_delta).norm()) > tolerance:
            raise ValueError("FP64 endpoint kernel constraint failed before casting")
        native = plan64.to(base_native.dtype)
        if not torch.isfinite(native).all():
            raise FloatingPointError("probe became nonfinite in base dtype")
        if not torch.equal(native[:, 1:], base_native[:, 1:]):
            raise AssertionError("probe changed a later chunk")
        actual_delta = native[0, 0].double()-base64[0, 0]
        cast_path = _difference_path(A, normalized_B, actual_delta)
        if name != "base":
            if torch.equal(native[:, :1], base_native[:, :1]):
                raise ValueError("base dtype erased the nonzero probe; no amplitude inflation is allowed")
            candidates.append((name, native))
        raw_commands = native.double().reshape(1, 25, 2)*constants["action_std"]+constants["action_mean"]
        row = {
            "name": name, "axis": axis, "sign": sign, "amplitude_fraction": fraction,
            "requested_normalized_delta_rms": float(delta.square().mean().sqrt()),
            "fp64_addition_delta_max_abs": float((fp64_delta-delta).abs().max()),
            "actual_normalized_delta_rms_after_cast": float(actual_delta.square().mean().sqrt()),
            "cast_delta_max_abs": float((actual_delta-delta).abs().max()),
            "fp64_endpoint_l2_tolerance": tolerance,
            "first_chunk_normalized_min": native[0, 0].reshape(5, 2).amin(dim=0).tolist(),
            "first_chunk_normalized_max": native[0, 0].reshape(5, 2).amax(dim=0).tolist(),
            "whole_plan_raw_command_min": raw_commands[0].amin(dim=0).tolist(),
            "whole_plan_raw_command_max": raw_commands[0].amax(dim=0).tolist(),
            **_path_metadata(path64, "fp64"), **_path_metadata(cast_path, "after_cast"),
        }
        metadata_rows.append(row)
    metadata = {
        "kind": "fixed_endpoint_matched_first_chunk_probe_not_a_method",
        "candidate_count": 9, "horizon_chunks": 5, "commands_per_chunk": CHUNK_LENGTH,
        "changed_chunk": 0, "base_dtype": str(base.dtype), "output_device": "cpu",
        "num_completed_lowlevel_commands": completed.shape[1],
        "completed_per_axis_mean": center.tolist(),
        "completed_normalized_command_min": completed[0].amin(dim=0).tolist(),
        "completed_normalized_command_max": completed[0].amax(dim=0).tolist(),
        "completed_centered_normalized_action_rms": amplitude,
        "amplitude_rule": "RMS over all completed scalar normalized commands after per-axis temporal centering; no floor",
        "endpoint_jacobian": G.tolist(), "endpoint_rank": rank,
        "endpoint_singular_values": singular_values.tolist(), "endpoint_rank_tolerance": rank_tolerance,
        "endpoint_condition": float(singular_values[0]/singular_values[-1]),
        "directions": direction_metadata,
        "direction_cosine": float(torch.dot(directions["x"], directions["y"])/10.),
        "action_mean": constants["action_mean"].tolist(), "action_std": constants["action_std"].tolist(),
        "candidates": metadata_rows,
        "information_boundary": "fitted completed-experience actuator, base commands, completed commands only; no future/outcome argument or screening",
        "matching_claim": "predicted raw p/v first-chunk endpoint differences under fitted FP64 linear dynamics; actual dtype errors reported; real environment matching not asserted",
        "internal_path_scope": "four interior command boundaries; not a predicted visual/contact trajectory",
        "command_range_policy": "record only; no clipping, rescaling, alternative direction, or outcome-dependent selection",
    }
    return candidates, metadata
