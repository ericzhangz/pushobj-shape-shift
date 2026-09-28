"""Evaluation only: full future tokens and directed action contrasts.

Inputs have axes [root, candidate, future_step, feature].  No observation at
the root is included.  Root is the aggregation unit; candidate pairs and time
steps are descriptive repeated measurements, never independent samples.
"""
from itertools import combinations

import numpy as np


def evaluate_response(pred, truth):
    """Return JSON-safe metrics; use 384 features for the visual primary score.

    The same dimension-normalized calculation can be applied separately to
    proprio embeddings.  Do not concatenate modalities or interpret proprio
    embeddings as physical coordinates.  Exactly zero true contrasts have no
    direction: their gain/projection is None and spurious magnitude is retained.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 4:
        raise ValueError("matching [root,candidate,future_step,feature] required")
    roots, candidates, steps, dim = pred.shape
    if min(roots, steps, dim) < 1 or candidates < 2:
        raise ValueError("nonempty axes and at least two candidates required")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise ValueError("all tokens must be finite; invalid runs cannot be dropped")

    error = pred - truth
    full = np.mean(error * error, axis=(1, 3))
    bias = np.mean(np.mean(error, axis=1) ** 2, axis=-1)
    pairs = np.asarray(list(combinations(range(candidates), 2)))
    td = truth[:, pairs[:, 0]] - truth[:, pairs[:, 1]]
    pd = pred[:, pairs[:, 0]] - pred[:, pairs[:, 1]]
    true_energy = np.mean(td * td, axis=-1)
    pred_energy = np.mean(pd * pd, axis=-1)
    dot = np.mean(td * pd, axis=-1)
    pair_error = np.mean((pd - td) ** 2, axis=-1)
    contrast = pair_error.mean(axis=1)

    # All unordered-pair mean is exactly 2*C/(C-1) times centered-error MSE.
    centered = error - error.mean(axis=1, keepdims=True)
    equivalent = 2.0 * candidates / (candidates - 1) * np.mean(centered ** 2, axis=(1, 3))
    identity_error = float(np.max(np.abs(contrast - equivalent)))
    if not np.allclose(contrast, equivalent, rtol=1e-11, atol=1e-13):
        raise ArithmeticError("pair versus centered contrast identity failed")

    rows = []
    for r in range(roots):
        for p, (left, right) in enumerate(pairs):
            for t in range(steps):
                te, pe, cross = (float(a[r, p, t]) for a in (true_energy, pred_energy, dot))
                x = float(np.sqrt(te))
                nonzero = te > 0.0
                y = cross / x if nonzero else None
                gain = cross / te if nonzero else None
                orthogonal = max(0.0, pe - y * y) if nonzero else None
                rows.append({
                    "root": r, "left": int(left), "right": int(right), "step": t + 1,
                    "true_contrast_rms": x,
                    "predicted_contrast_rms": float(np.sqrt(pe)),
                    "directed_prediction_rms": y, "directional_gain": gain,
                    "orthogonal_mse": orthogonal,
                    "contrast_mse": float(pair_error[r, p, t]),
                    "zero_true_contrast": not nonzero,
                    "spurious_mse_at_zero": pe if not nonzero else None,
                })

    root_rows = []
    for r in range(roots):
        te = float(true_energy[r].mean())
        pe = float(pred_energy[r].mean())
        cross = float(dot[r].mean())
        root_rows.append({
            "root": r, "full_mse": float(full[r].mean()),
            "terminal_full_mse": float(full[r, -1]),
            "contrast_mse": float(contrast[r].mean()),
            "terminal_contrast_mse": float(contrast[r, -1]),
            "shared_bias_mse": float(bias[r].mean()),
            "true_contrast_energy": te, "predicted_contrast_energy": pe,
            "energy_weighted_directional_gain": cross / te if te > 0.0 else None,
            "full_mse_by_step": full[r].tolist(),
            "contrast_mse_by_step": contrast[r].tolist(),
        })
    scalar_keys = ("full_mse", "terminal_full_mse", "contrast_mse",
                   "terminal_contrast_mse", "shared_bias_mse",
                   "true_contrast_energy", "predicted_contrast_energy")
    aggregate = {key: float(np.mean([r[key] for r in root_rows])) for key in scalar_keys}
    aggregate["full_mse_by_step"] = full.mean(axis=0).tolist()
    aggregate["contrast_mse_by_step"] = contrast.mean(axis=0).tolist()
    return {
        "contract": {"axes": ["root", "candidate", "future_step", "feature"],
                     "shape": list(pred.shape), "aggregation_unit": "root",
                     "pairs_per_root": len(pairs),
                     "pair_rows_are_independent_samples": False,
                     "pair_centering_identity_max_abs": identity_error,
                     "root_full_bias_contrast_identity_max_abs": float(np.max(np.abs(
                         full - bias - (candidates - 1) / (2.0 * candidates) * contrast)))},
        "aggregate": aggregate, "per_root": root_rows, "pair_rows": rows,
    }


def verify_contract():
    """Synthetic algebra checks only; no model, files, or outcome data."""
    rng = np.random.default_rng(20260928)
    truth = rng.normal(size=(3, 4, 5, 384))
    exact = evaluate_response(truth, truth)
    assert exact["aggregate"]["full_mse"] == 0.0
    assert exact["aggregate"]["contrast_mse"] == 0.0
    assert np.allclose([r["directional_gain"] for r in exact["pair_rows"]], 1.0)
    shifted = evaluate_response(truth + 2.0, truth)
    assert np.isclose(shifted["aggregate"]["full_mse"], 4.0)
    assert shifted["aggregate"]["contrast_mse"] < 1e-28
    collapsed = evaluate_response(np.repeat(truth.mean(axis=1, keepdims=True), 4, axis=1), truth)
    assert all(r["directed_prediction_rms"] == 0.0 for r in collapsed["pair_rows"])
    flipped = evaluate_response(-truth, truth)
    assert np.allclose([r["directional_gain"] for r in flipped["pair_rows"]], -1.0)
    zero = evaluate_response(truth, np.zeros_like(truth))
    assert all(r["directional_gain"] is None and r["spurious_mse_at_zero"] >= 0
               for r in zero["pair_rows"])
    random = evaluate_response(rng.normal(size=truth.shape), truth)
    for r in random["pair_rows"]:
        assert np.isclose(r["contrast_mse"],
                          (r["directed_prediction_rms"] - r["true_contrast_rms"]) ** 2
                          + r["orthogonal_mse"])
    assert random["contract"]["root_full_bias_contrast_identity_max_abs"] < 1e-12
    return {"status": "PASS", "kind": "synthetic_algebra_only",
            "checks": ["exact", "common_bias", "collapse", "reverse", "zero_truth",
                       "pair_centering", "parallel_orthogonal", "full_bias_contrast"],
            "random_identity_max_abs": random["contract"]["pair_centering_identity_max_abs"]}


if __name__ == "__main__":
    import json
    print(json.dumps(verify_contract(), indent=2, allow_nan=False))
