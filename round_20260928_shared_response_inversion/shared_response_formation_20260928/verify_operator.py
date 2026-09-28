"""Post-run invariants only; no training or selection from query outcomes."""
import json
import torch
from formation import (OUT, CASES, Q, runtime, Counter, support_view, cached_rollout,
                       hard, auxiliary, phi_response_jac, ridge, dump, cpu)


@torch.no_grad()
def main():
    data = torch.load(OUT / "PREPARED.pt", map_location="cpu", weights_only=False)
    info = json.loads((OUT / "PRIORS.json").read_text())
    model, prep = runtime()
    counter = Counter(model)
    counter.phase = "post_run_operator_checks"
    checks = []
    for shape, sample in CASES:
        case = f"{shape}{sample}"
        initial, actions, target = support_view(data[case])
        prior = torch.load(OUT / f"PRIOR_{shape}.pt", weights_only=False).cuda()
        lam = info[shape]["alpha"]
        trace = []
        hard(model, initial, actions, target, prior, prior, lam, trace)
        w = torch.cat([t["source"] for t in trace])
        direction = torch.randn(Q, device="cuda")
        direction /= direction.norm()
        h = 1e-3
        wp, wm = w.clone(), w.clone()
        wp[0] += h * direction
        wm[0] -= h * direction
        vp = auxiliary(model, initial, actions, target, prior, wp, prior, lam, 1.)[3][:, 1]
        vm = auxiliary(model, initial, actions, target, prior, wm, prior, lam, 1.)[3][:, 1]
        finite = prior @ ((vp - vm) / (2 * h))
        analytical = phi_response_jac(prior, actions[0, 3]) @ direction
        feedback_error = float((finite - analytical).norm() / max(1., float(analytical.norm())))
        assert feedback_error < 0.005
        true_sources = torch.zeros_like(w)
        for step in range(2):
            trace = []
            cached_rollout(model, initial, actions, sources=true_sources, trace=trace)
            true_sources[step] = target[step] - trace[step]["native"][0, 0, 0, :Q]
        before, _, _, V, _ = auxiliary(model, initial, actions, target, prior,
                                      true_sources, prior, lam, 1.)
        revised = ridge(true_sources.T, V, lam, prior)
        after = auxiliary(model, initial, actions, target, revised, true_sources, prior, lam, 1.)[0]
        assert float(after) <= float(before) + 1e-5
        prefixes, support = {}, {}
        alt = actions.clone()
        alt[:, 3] += 0.5
        for arm in ("PRIOR", "STEP_RIDGE", "DIRECT", "SOURCE_MU1", "SOURCE_MU10"):
            theta = torch.load(OUT / f"{case}_{arm}.pt", weights_only=False).cuda()
            left = cached_rollout(model, initial, actions, theta=theta)
            right = cached_rollout(model, initial, alt, theta=theta)
            error = max(float((left[k][:, :4] - right[k][:, :4]).abs().max()) for k in left)
            assert error == 0.
            prefixes[arm] = error
            objective, mse_sum = hard(model, initial, actions, target, theta, prior, lam)
            support[arm] = {"hard_objective": float(objective), "mse": float(mse_sum) / (2 * Q)}
        native = cached_rollout(model, initial, actions)
        support["NATIVE"] = {"mse": float(
            (native["visual"][0, 3:, 0] - target).square().mean())}
        checks.append({"case": case, "feedback_jac_relative_error": feedback_error,
                       "ridge_joint_before": float(before), "ridge_joint_after": float(after),
                       "prefix_max_abs": prefixes, "support": support})
    dump("OPERATOR_CHECKS.json", {"checks": checks, "counts": counter.rows,
                                 "new_training_steps": 0, "environment_calls": 0})
    print(json.dumps(cpu(checks), indent=2), flush=True)


if __name__ == "__main__":
    main()
