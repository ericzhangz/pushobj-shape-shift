"""Seal a bounded falsification of the checkpoint's action-LN equivalence.

This is an evaluation module, not a new world-model path or a learned method.
No environment results are read. The existing isolated branch evaluator owns
all real consequences. Directions and amplitudes use checkpoint algebra only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from research.reframe_v3.round6_reference import (
    _local_hub_loader, _score_pool, _write_csv, _write_json,
)
from research.reframe_v3.shadow_selection_audit import (
    _load_anchor_observations, _load_runtime, _transform_anchor_obs,
)


def action_fiber(encoder):
    """Unit raw normalized-action direction removed exactly by LN centering."""
    weight = encoder.patch_embed.weight.detach().cpu().double().squeeze(-1)
    bias = encoder.patch_embed.bias.detach().cpu().double()
    if weight.shape != (10, 10) or not isinstance(encoder.norm, torch.nn.LayerNorm):
        raise ValueError("requires released 10-to-10 action Conv1d plus LayerNorm")
    singular = torch.linalg.svdvals(weight)
    if singular[-1] <= singular[0] * 10 * torch.finfo(torch.float64).eps:
        raise ValueError("the declared inverse direction requires full-rank W")
    direction = torch.linalg.solve(weight, torch.ones(10, dtype=torch.float64))
    direction /= direction.norm()
    projector = (torch.eye(10, dtype=torch.float64)
                 - torch.ones(10, 10, dtype=torch.float64) / 10)
    center = -torch.linalg.pinv(projector @ weight) @ (projector @ bias)
    return direction, {"weight_singular_values": singular.tolist(),
                       "unit_direction": direction.tolist(),
                       "centered_W_direction_norm": float((projector @ weight @ direction).norm()),
                       "scale_ray_minimum_norm_center": center.tolist(),
                       "scale_ray_center_rms": float(center.square().mean().sqrt())}


def probe_actions(base, direction):
    if base.shape != (1, 5, 10) or direction.shape != (10,):
        raise ValueError("expected native H5 action and a 10-coordinate direction")
    result = [("base_g99", base.clone())]
    shift = math.sqrt(10) * direction.to(base)
    for scope in ("first", "all"):
        for sign in (-1, 1):
            action = base.clone()
            action[:, :1 if scope == "first" else 5] += sign * shift
            result.append((f"fiber_{scope}_{'minus' if sign < 0 else 'plus'}", action))
    return result


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    with patch.object(torch.hub, "load", _local_hub_loader()):
        model, preprocessor, _ = _load_runtime(args.checkpoint_dir, device)
    direction, algebra = action_fiber(model.action_encoder)
    base = torch.load(args.donor / "tensors" / f"s{args.sample}_m{args.mpc}_g99.pt",
                      map_location="cpu")["u_after"].detach().clone()
    actions = probe_actions(base, direction)
    current_raw, goal_raw = _load_anchor_observations(args.donor, args.sample, args.mpc)
    current, goal = _transform_anchor_obs(preprocessor, current_raw, goal_raw, device)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "RUN_CONTRACT.json", {
        "status": "CHECKPOINT_ONLY_FINITE_FIBER_PROBE_NOT_METHOD_SELECTION",
        "donor": str(args.donor), "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split, "sample": args.sample, "mpc": args.mpc,
        "oracle_truth_read": False, "normalized_shift_rms_per_modified_chunk": 1.,
        "amplitude_selection": "fixed before outcomes; no amplitude or direction search",
        "query_history_horizon_scoring": "unchanged native single-frame H5 terminal",
        "algebra": algebra,
    })
    candidates, encodings = [], []
    with torch.no_grad():
        base_encoding = model.encode_act(base.to(device))
        for name, action in actions:
            path = args.out / f"{name}.pt"
            torch.save({"u_after": action}, path)
            candidates.append((name, action, str(path), "u_after"))
            encoded = model.encode_act(action.to(device))
            raw = preprocessor.denormalize_actions(action.reshape(1, 25, 2))
            encodings.append({"candidate_id": name,
                              "action_difference_l2": float((action - base).norm()),
                              "embedding_difference_max": float((encoded - base_encoding).abs().max()),
                              "raw_action_abs_max": float(raw.abs().max()),
                              "raw_action_rms": float(raw.square().mean().sqrt())})
    scores = [{"arm": "FROZEN", **row}
              for row in _score_pool(model, current, goal, candidates, args.mpc)]
    _write_csv(args.out / "PREDICTIONS.csv", scores)
    _write_csv(args.out / "ENCODING.csv", encodings)
    _write_json(args.out / "RUN_SUMMARY.json", {
        "status": "PREDICTIONS_SEALED_NO_ORACLE_READ", "environment_branches": 0,
        "candidate_count": len(candidates), "mechanism_training_steps": 0,
    })
    print(json.dumps({"out": str(args.out), "algebra": algebra,
                      "encodings": encodings, "scores": scores}, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("T", "L"), required=True)
    parser.add_argument("--sample", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mpc", type=int, choices=(2,), default=2)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("D:/EV-TTT/pushobj_shape_shift"))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
