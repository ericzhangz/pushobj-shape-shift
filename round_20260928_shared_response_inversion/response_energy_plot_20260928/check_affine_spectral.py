"""Independent numerical check of the frozen affine spectral equations.

Synthetic algebra only: no visual model, environment, training data or metrics.
"""
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(1907)
    n, r, d, q = 34, 4, 12, 5
    s = rng.normal(size=(r, n)) + rng.normal(size=(r, 1))
    c_true = rng.normal(size=(d, r))
    bias_true = rng.normal(size=(d, 1))
    actions = [(0.7 * np.eye(r) + 0.08 * rng.normal(size=(r, r)),
                0.15 * rng.normal(size=(r, 1))) for _ in range(q)]

    def output(state):
        return c_true @ state + bias_true

    def response_table(state):
        return np.concatenate([output(state)] + [
            output(a @ state + b) for a, b in actions
        ], axis=0)

    h = response_table(s)
    shifted = [response_table(a @ s + b) for a, b in actions]
    mu = h.mean(axis=1, keepdims=True)
    u, singular, vt = np.linalg.svd(h - mu, full_matrices=False)
    o = u[:, :r] * (singular[:r] / np.sqrt(n))[None, :]
    x = np.sqrt(n) * vt[:r]
    x_aug = np.vstack([np.ones((1, n)), x])
    learned = []
    residuals = []
    split_errors = []
    for ha in shifted:
        target = ha - mu
        b = np.linalg.pinv(o, rcond=1e-8) @ target @ np.linalg.pinv(x_aug, rcond=1e-8)
        a = np.vstack([np.eye(1, r + 1), b])
        learned.append(a)
        fitted = o @ b @ x_aug
        residuals.append(float(np.max(np.abs(target - fitted))))
        po = o @ np.linalg.pinv(o)
        px = np.linalg.pinv(x_aug) @ x_aug
        lhs = np.linalg.norm(target - fitted) ** 2
        rhs = np.linalg.norm((np.eye(h.shape[0]) - po) @ target) ** 2
        rhs += np.linalg.norm(po @ target @ (np.eye(n) - px)) ** 2
        split_errors.append(float(abs(lhs - rhs)))

    c = np.concatenate([mu[:d], o[:d]], axis=1)
    max_recursive_error = 0.0
    max_constant_error = 0.0
    for _ in range(100):
        j = rng.integers(n)
        actual = s[:, j:j + 1].copy()
        state = x_aug[:, j:j + 1].copy()
        for a_id in rng.integers(q, size=5):
            a_true, b_true = actions[a_id]
            actual = a_true @ actual + b_true
            state = learned[a_id] @ state
            max_recursive_error = max(max_recursive_error,
                                      float(np.max(np.abs(c @ state - output(actual)))))
            max_constant_error = max(max_constant_error, float(abs(state[0, 0] - 1)))

    checks = {
        "kind": "synthetic algebra check, not native experiment",
        "dimensions": {"roots": n, "dynamic_rank": r, "visual_output": d, "actions": q},
        "centered_h_reconstruction_max_error": float(np.max(np.abs(h - mu - o @ x))),
        "state_population_covariance_error": float(np.max(np.abs(x @ x.T / n - np.eye(r)))),
        "state_absolute_max": float(np.max(np.abs(x))),
        "state_fraction_outside_tanh_range": float(np.mean(np.abs(x) > 1)),
        "shift_fit_max_errors": residuals,
        "orthogonal_error_split_absolute_errors": split_errors,
        "h5_recursive_max_error_100_words": max_recursive_error,
        "homogeneous_coordinate_max_error": max_constant_error,
        "native_model_calls": 0,
        "environment_calls": 0,
    }
    assert checks["centered_h_reconstruction_max_error"] < 1e-10
    assert max(residuals) < 1e-10
    assert max_recursive_error < 1e-10
    assert max_constant_error == 0
    assert checks["state_fraction_outside_tanh_range"] > 0
    output_path = Path(__file__).with_name("AFFINE_SPECTRAL_CHECKS.json")
    output_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
