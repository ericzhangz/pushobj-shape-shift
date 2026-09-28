"""CPU-only checks of the proposed shared-response inverse formulation.

No model, trajectory cache, environment, or future observation is accessed.
Run with Python + NumPy. Results are written next to this script.
"""
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(20260928)
    profile_checks = []
    for trial in range(128):
        n = int(rng.integers(2, 12))
        m = int(rng.integers(2, 12))
        a = np.eye(n) - np.tril(rng.normal(scale=0.25, size=(n, n)), -1)
        k = rng.normal(size=(m, n)) / np.sqrt(n)
        r0 = rng.normal(size=(m, m)) / np.sqrt(m)
        r = r0 @ r0.T + 0.4 * np.eye(m)
        b = rng.normal(size=n)
        d = rng.normal(size=m)
        mu = float(10 ** rng.uniform(-2, 2))
        w_deploy = np.linalg.solve(a, b)
        e = d - k @ w_deploy
        normal = k.T @ np.linalg.solve(r, k) + mu * a.T @ a
        w = np.linalg.solve(normal, k.T @ np.linalg.solve(r, d) + mu * a.T @ b)
        residual = d - k @ w
        profile = float(residual @ np.linalg.solve(r, residual) + mu * np.sum((a @ w - b) ** 2))
        ainv = np.linalg.solve(a, np.eye(n))
        sigma = r + (k @ ainv @ ainv.T @ k.T) / mu
        closed = float(e @ np.linalg.solve(sigma, e))
        error = abs(profile - closed)
        assert np.isclose(profile, closed, rtol=1e-10, atol=1e-10)
        profile_checks.append({"trial": trial, "n": n, "m": m, "mu": mu, "absolute_error": error})

    scalar = []
    for mu in (0.01, 0.1, 1.0, 10.0, 100.0):
        # J=(1-w)^2 + mu*(w-theta)^2 + theta^2.
        solution = np.linalg.solve(np.array([[1 + mu, -mu], [-mu, 1 + mu]]), np.array([1.0, 0.0]))
        w, theta = map(float, solution)
        assert np.allclose(solution, [(mu + 1) / (2 * mu + 1), mu / (2 * mu + 1)])
        scalar.append({"mu": mu, "w_aux": w, "theta_deploy": theta,
                       "auxiliary_data_mse": (1 - w) ** 2,
                       "deployed_data_mse": (1 - theta) ** 2,
                       "penalized_objective": (1-w)**2 + mu*(w-theta)**2 + theta**2})

    amplification = []
    h = np.array([[0.0, 0.0], [1.0, 0.0]])
    d = np.array([0.0, 1.0])
    for theta in (0.0, 1.0, 10.0, 100.0):
        # v0=0, K=R=I, mu=1. Exact causal deployment is always zero.
        a = np.eye(2) - theta * h
        w = np.linalg.solve(np.eye(2) + a.T @ a, d)
        profile = float(np.sum((d - w)**2) + np.sum((a @ w)**2))
        assert np.isclose(profile, 2 / (theta**2 + 4))
        amplification.append({"theta": theta, "w_aux": w.tolist(), "w_deploy": [0.0, 0.0],
                              "profile_loss": profile, "deployed_data_mse": 1.0,
                              "condition_number_A": float(np.linalg.cond(a))})

    # Observing every directly injected output gives block-unit-lower-triangular K.
    inversion = []
    horizon, state_dim, visual_dim = 5, 7, 3
    b = np.zeros((state_dim, visual_dim))
    b[:visual_dim] = np.eye(visual_dim)
    c = b.T
    for trial in range(32):
        transitions = rng.normal(scale=0.35, size=(horizon, state_dim, state_dim))
        k = np.zeros((horizon * visual_dim, horizon * visual_dim))
        for j in range(horizon):
            propagator = b.copy()
            for t in range(j, horizon):
                if t > j:
                    propagator = transitions[t] @ propagator
                k[t*visual_dim:(t+1)*visual_dim, j*visual_dim:(j+1)*visual_dim] = c @ propagator
        w_true = rng.normal(size=horizon * visual_dim)
        error = float(np.max(np.abs(np.linalg.solve(k, k @ w_true) - w_true)))
        assert np.array_equal(np.diag(k), np.ones(horizon * visual_dim))
        assert np.isclose(np.linalg.det(k), 1.0)
        assert error < 1e-10
        inversion.append({"trial": trial, "max_source_recovery_error": error})

    # First-order kernel inclusion can pass while nonlinear targets differ.
    nonlinear = {"support_map": "F_D(theta)=theta[0]", "query_map": "F_U(theta)=theta[0]+theta[1]^2",
                 "reference_theta": [0.0, 0.0], "S_D": [[1.0, 0.0]], "L_U": [[1.0, 0.0]],
                 "theta_alternative": [0.0, 0.1], "same_support_output": 0.0,
                 "different_query_output": 0.01, "first_order_kernel_inclusion": True}
    result = {"seed": 20260928, "model_calls": 0, "environment_calls": 0,
              "profile_identity_trials": len(profile_checks),
              "max_profile_identity_absolute_error": max(x["absolute_error"] for x in profile_checks),
              "scalar_relaxation": scalar,
              "hard_deployment_scalar_optimum": {"theta": 0.5, "deployed_data_mse": 0.25, "objective": 0.5},
              "homogeneous_response_counterexample": amplification,
              "unit_triangular_inversion_trials": len(inversion),
              "max_source_recovery_error": max(x["max_source_recovery_error"] for x in inversion),
              "nonlinear_identifiability_counterexample": nonlinear}
    path = Path(__file__).with_name("MATH_CHECK.json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
