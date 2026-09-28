"""Exact, synthetic checks of modeling distinctions; no visual-model experiment.

Run with the existing adajepa-pilot Python. Uses no model, dataset, or network.
All constants define illustrative systems, not tuned research hyperparameters.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def balanced_example():
    # Independent input and output channels; C is part of the toy system.
    a = np.array([0.2, 0.9])
    b = np.array([5.0, 0.2])
    c = np.array([0.02, 5.0])
    wc = b**2 / (1 - a**2)
    wo = c**2 / (1 - a**2)
    hsv = np.sqrt(wc * wo)
    pod_keep = int(np.argmax(wc))
    balanced_keep = int(np.argmax(hsv))
    # For these positive diagonal scalar transfers, the maximum is at DC.
    hinf_discard = abs(b * c) / (1 - a)
    omega = np.linspace(0.0, np.pi, 4097)
    frequency_max = np.max(
        abs((b * c)[:, None] / (np.exp(1j * omega)[None, :] - a[:, None])),
        axis=1,
    )
    assert np.allclose(hinf_discard, frequency_max, atol=1e-12)
    assert pod_keep == 0 and balanced_keep == 1
    assert max(abs(a)) < 1
    # Actual Markov parameters factor into observability x controllability.
    horizon = 32
    A, B, C = np.diag(a), np.diag(b), np.diag(c)
    reach = np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(horizon)])
    observe = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(horizon)])
    hankel = np.block([
        [C @ np.linalg.matrix_power(A, i + j) @ B for j in range(horizon)]
        for i in range(horizon)
    ])
    factor_error = float(np.max(abs(hankel - observe @ reach)))
    assert factor_error < 1e-12
    # +/- unit impulses on the second input: true first-future output gap=2.
    true_gap = 2 * B[1, 1] * C[1, 1]
    return {
        "A_diagonal": a.tolist(), "B_diagonal": b.tolist(),
        "C_diagonal": c.tolist(), "controllability_diagonal": wc.tolist(),
        "observability_diagonal": wo.tolist(), "hankel_singular_values_by_mode": hsv.tolist(),
        "POD_kept_mode_one_based": pod_keep + 1,
        "balanced_kept_mode_one_based": balanced_keep + 1,
        "POD_discarded_transfer_Hinf": float(hinf_discard[1]),
        "balanced_discarded_transfer_Hinf": float(hinf_discard[0]),
        "both_reduced_models_stable": True,
        "hankel_factorization_max_error": factor_error,
        "second_channel_opposite_impulse_output_gap": float(true_gap),
        "POD_predicted_gap": 0.0, "balanced_predicted_gap": float(true_gap),
        "scope": "Known stable linear toy; not evidence of visual encoder failure or a new algorithm.",
    }


def path_example():
    # Smooth controlled system: dx=dU1, dy=x dU2. Straight segments integrate exactly.
    def solve(increments):
        x, y = 0.0, 0.0
        states = [[x, y]]
        for dx, du2 in increments:
            y += x * du2 + 0.5 * dx * du2
            x += dx
            states.append([x, y])
        return np.asarray(states)

    ab = np.array([[1.0, 0.0], [0.0, 1.0]])
    ba = ab[::-1]
    ab_states, ba_states = solve(ab), solve(ba)
    assert np.array_equal(ab.sum(axis=0), ba.sum(axis=0))
    assert ab_states[-1, 1] == 1 and ba_states[-1, 1] == 0
    signed_loop = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    inverse_word = np.array([[1., 0.], [0., 1.], [0., -1.], [-1., 0.]])
    assert solve(signed_loop)[-1, 1] == 1
    assert solve(inverse_word)[-1, 1] == 0
    # Splitting an identical straight control segment preserves this exact solution.
    split_error = 0.0
    for sequence in [ab, ba, signed_loop]:
        for subdivisions in [2, 4, 8]:
            refined = np.repeat(sequence / subdivisions, subdivisions, axis=0)
            split_error = max(split_error, float(np.max(abs(solve(refined)[-1] - solve(sequence)[-1]))))
    assert split_error < 1e-12
    return {
        "AB_final_state": ab_states[-1].tolist(), "BA_final_state": ba_states[-1].tolist(),
        "same_first_level_input_increment": ab.sum(axis=0).tolist(),
        "ordered_second_level_U1_then_U2": [1.0, 0.0],
        "net_zero_input_loop_final_state": solve(signed_loop)[-1].tolist(),
        "actual_inverse_word_final_state": solve(inverse_word)[-1].tolist(),
        "same_path_subdivision_max_error": split_error,
        "scope": "Classical order-dependent response. An ordinary recurrent state can implement the exact same rule; no superiority claim.",
    }


def sweeping_example():
    # Bilateral scalar play, not the physical free space around a single pusher.
    def solve(path, q0=0.0):
        q = float(q0)
        states = [q]
        for p in path[1:]:
            q = float(np.clip(q, p - 1.0, p + 1.0))
            states.append(q)
        return np.asarray(states)

    plus, minus = [0., 2., 0.], [0., -2., 0.]
    plus_q, minus_q = solve(plus), solve(minus)
    assert plus_q[-1] == 1.0 and minus_q[-1] == -1.0
    # Both final outputs meet exactly the same final interval constraint.
    assert abs(plus_q[-1]) <= 1 and abs(minus_q[-1]) <= 1
    max_violation = 0.0
    for path in [plus, minus, [0., 2., 0., -2., 0.]]:
        for x in np.linspace(-1, 1, 21):
            for y in np.linspace(-1, 1, 21):
                max_violation = max(max_violation, float(np.max(abs(solve(path, x) - solve(path, y))) - abs(x - y)))
    assert max_violation < 1e-12
    # Repeated fixed input does not move the pure play output (no inertia).
    assert np.array_equal(solve([0., 2., 2., 2.]), np.array([0., 1., 1., 1.]))
    # A 90-degree rotation is nonexpansive but fails firm nonexpansiveness.
    d, rotated = np.array([1., 0.]), np.array([0., 1.])
    firm_residual = float(rotated @ rotated - rotated @ d)
    assert firm_residual == 1.0
    return {
        "positive_excursion_response": plus_q.tolist(),
        "negative_excursion_response": minus_q.tolist(),
        "same_actuator_endpoint": 0.0, "object_endpoint_gap": 2.0,
        "endpoint_only_projection_output": 0.0,
        "fixed_path_nonexpansiveness_max_violation": max_violation,
        "fixed_input_continuation": solve([0., 2., 2., 2.]).tolist(),
        "rotation_firm_nonexpansiveness_violation": firm_residual,
        "scope": "Rechecks an already discussed classical model. Convex quasistatic play excludes free inertial continuation; not a new proposed PushObj mechanism.",
    }


def main():
    result = {
        "experiment_type": "synthetic_mathematical_checks_only",
        "native_model_calls": 0, "training_steps": 0, "environment_calls": 0,
        "balanced": balanced_example(), "ordered_path": path_example(),
        "sweeping": sweeping_example(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = Path(__file__).with_name("MATH_CHECKS.json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
