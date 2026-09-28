"""Shared response formation on the existing native observation_transition path.

No environment access. Predict phase never opens query outcomes.
The support is now MPC0..3; the query is MPC4, not the old MPC2.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
REPO = Path("D:/EV-TTT/adajepa_official_51d8665")
DATA = REPO / "artifacts/round_6_shared_revision/results"
CHECKPOINT = Path("D:/EV-TTT/pushobj_shape_shift")
sys.path.insert(0, str(REPO))
os.environ["TORCH_HOME"] = "D:/EV-TTT/adajepa_runtime/torch"
from research.reframe_v3.round6_reference import _local_hub_loader, _sealed_candidates
from research.reframe_v3.shadow_selection_audit import _load_runtime, _load_anchor_observations
from research.reframe_v3.rgb_geometry_probe import load_completed
from research.reframe_v3.matched_feedback_forecast import _join_encoded_observation_action
from research.contrast_probe import native_objective_breakdown

Q, NS, NF = 384, 1212, 13343
CASES = [(s, i) for s in ("T", "L") for i in (0, 1)]


def dump(name, value):
    path = OUT / name
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu(value):
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value.detach().cpu() if torch.is_tensor(value) else value


def dev(value):
    return {k: v.cuda() for k, v in value.items()}


def runtime():
    torch.set_num_threads(4)
    torch.manual_seed(260928)
    np.random.seed(260928)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with patch.object(torch.hub, "load", _local_hub_loader()):
        model, prep, cfg = _load_runtime(CHECKPOINT, torch.device("cuda:0"))
    assert model.num_hist == 3 and cfg.frameskip == 5
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    return model, prep


class Counter:
    def __init__(self, model):
        self.phase = "setup"
        self.rows = {}
        self.hook = model.predictor.register_forward_pre_hook(self.count)

    def count(self, module, inputs):
        row = self.rows.setdefault(self.phase, {"forward_calls": 0, "batch_transitions": 0,
                                               "vjp_seed_vectors": 0, "backward_calls": 0})
        row["forward_calls"] += 1
        row["batch_transitions"] += int(inputs[0].shape[0])

    def grad(self, seeds=1):
        row = self.rows[self.phase]
        row["vjp_seed_vectors"] += seeds
        row["backward_calls"] += 1


def phi(history, action):
    s = history.reshape(history.shape[0], -1)
    a = action.reshape(action.shape[0], -1)
    assert s.shape[1] == NS and a.shape[1] == 10
    return torch.cat([torch.ones_like(a[:, :1]), s / math.sqrt(NS),
                      a / math.sqrt(10),
                      (a[:, :, None] * s[:, None, :]).flatten(1) / math.sqrt(10 * NS)], 1)


def phi_response_jac(theta, action):
    """Exact derivative of second-step features wrt first injected visual."""
    indices = torch.arange(2 * 404, 2 * 404 + Q, device=theta.device)
    result = theta[:, 1 + indices] / math.sqrt(NS)
    offset = 1 + NS + 10
    for j in range(10):
        result = result + action[j] * theta[:, offset + j * NS + indices] / math.sqrt(10 * NS)
    return result


def cached_rollout(model, initial, actions, theta=None, sources=None, trace=None):
    """Reuse frozen encodings, retaining the single native rollout/cache path."""
    if theta is not None and sources is not None:
        raise ValueError("choose deploy law or auxiliary sources")
    step = 0

    def transition(history, action):
        nonlocal step
        native = model.predict(history)[:, -1:, ..., :394]
        feature = phi(history, action)
        w = (feature @ theta.T if theta is not None else
             sources[step].reshape(1, Q) if sources is not None else
             torch.zeros_like(native[:, 0, 0, :Q]))
        if trace is not None:
            trace.append({"feature": feature, "native": native, "source": w})
        step += 1
        return torch.cat([native[..., :Q] + w[:, None, None, :], native[..., Q:]], -1)

    callback = transition if theta is not None or sources is not None or trace is not None else None
    # encode_obs is frozen; this temporary cache adapter changes no recurrent path.
    with patch.object(model, "encode_obs", lambda obs: initial):
        result, _ = model.rollout(initial, actions, observation_transition=callback)
    return result


@torch.no_grad()
def encode_episode(model, prep, images, props):
    pieces = []
    for start in range(0, len(images), 8):
        obs = prep.transform_obs({"visual": images[None, start:start + 8],
                                  "proprio": props[None, start:start + 8]})
        pieces.append(cpu(model.encode_obs(dev(obs))))
    return {k: torch.cat([v[k] for v in pieces], 1) for k in pieces[0]}


def episode_view(episode, indices):
    return {k: v[:, indices].cuda() for k, v in episode["z"].items()}


def support_view(episode):
    initial = episode_view(episode, [0, 5, 10])
    actions = episode["actions"][:20].reshape(1, 4, 10).cuda()
    target = episode["z"]["visual"][0, [15, 20], 0].cuda()
    return initial, actions, target


def data_error(result, target):
    return (result["visual"][0, 3:, 0] - target).square().sum()


def hard(model, initial, actions, target, theta, prior, lam, trace=None):
    result = cached_rollout(model, initial, actions, theta=theta, trace=trace)
    err = data_error(result, target)
    return err + lam * (theta - prior).square().sum(), err


def ridge(W, V, alpha, prior=None):
    """Raw-sum objective ||W-Theta V||^2 + alpha||Theta-prior||^2."""
    assert alpha > 0
    residual = W if prior is None else W - prior @ V
    gram = V.double().T @ V.double()
    system = gram + alpha * torch.eye(V.shape[1], device=V.device, dtype=torch.float64)
    coefficient = torch.linalg.solve(system, residual.double().T).T.float()
    delta = coefficient @ V.T
    return delta if prior is None else prior + delta


@torch.no_grad()
def prepare():
    if (OUT / "PREPARED.pt").exists():
        raise FileExistsError("prepared cache exists")
    model, prep = runtime()
    counter = Counter(model)
    counter.phase = "prepare"
    episodes, manifest = {}, {}
    protected = [Path(__file__), REPO / "models/visual_world_model.py",
                 CHECKPOINT / "checkpoints/model_latest.pth", CHECKPOINT / "hydra.yaml"]
    for shape in ("T", "L"):
        donor = DATA / f"donor_{shape}_seed101_n12_capture"
        for sample in range(12):
            files = sorted((donor / "real_evidence").glob(f"s{sample}_executed_mpc*.npz"))
            count = 4 if sample < 2 else len(files)
            assert count >= 3
            images, props, actions = load_completed(donor, sample, count=count)
            episodes[f"{shape}{sample}"] = {
                "shape": shape, "sample": sample, "z": encode_episode(model, prep, images, props),
                "actions": torch.from_numpy(actions).float(), "frames": len(images), "chunks": count}
            protected += [donor / "real_evidence" / f"s{sample}_executed_mpc{i}.npz"
                          for i in range(count)]
            print("encoded", shape, sample, len(images), flush=True)
    probe = episodes["T0"]
    donor = DATA / "donor_T_seed101_n12_capture"
    images, props, _ = load_completed(donor, 0, count=4)
    observed = dev(prep.transform_obs({"visual": images[None, [0, 5, 10]],
                                      "proprio": props[None, [0, 5, 10]]}))
    initial, acts, target = support_view(probe)
    original, _ = model.rollout(observed, acts)
    cached = cached_rollout(model, initial, acts)
    zero = cached_rollout(model, initial, acts, theta=torch.zeros(Q, NF, device="cuda"))
    audits = {f"cache_{k}": float((original[k] - cached[k]).abs().max()) for k in original}
    audits.update({f"zero_{k}": float((cached[k] - zero[k]).abs().max()) for k in cached})
    assert max(audits.values()) < 2e-5, audits
    for path in protected:
        manifest[str(path)] = sha(path)
    torch.save(episodes, OUT / "PREPARED.pt")
    dump("PREPARE.json", {"sha256": sha(OUT / "PREPARED.pt"), "input_hashes": manifest,
                          "native_path_checks": audits, "counts": counter.rows,
                          "environment_calls": 0, "query_outcomes_read": False})


@torch.no_grad()
def fit_prior(model, episodes, shape):
    features, residuals, owners = [], [], []
    for sample in range(2, 12):
        ep = episodes[f"{shape}{sample}"]
        for start in range(ep["frames"] - 15):
            initial = episode_view(ep, [start, start + 5, start + 10])
            actions = ep["actions"][start:start + 15].reshape(1, 3, 10).cuda()
            joined = _join_encoded_observation_action(model, initial, actions)
            feature = phi(joined, actions[:, -1:])
            native = cached_rollout(model, initial, actions)["visual"][0, -1, 0]
            target = ep["z"]["visual"][0, start + 15, 0].cuda()
            features.append(feature[0])
            residuals.append(target - native)
            owners.append(sample)
    V, W = torch.stack(features, 1), torch.stack(residuals, 1)
    gram = (V.double().T @ V.double()).cpu()
    responses = W.double().cpu()
    owners = torch.tensor(owners)
    cv = []
    for alpha in (0.1, 1.0, 10.0):
        errors = []
        for sample in range(2, 12):
            train, test = owners != sample, owners == sample
            K = gram[train][:, train] + alpha * torch.eye(int(train.sum()), dtype=torch.float64)
            beta = torch.linalg.solve(K, responses[:, train].T)
            predicted = gram[test][:, train] @ beta
            errors.append(float((predicted - responses[:, test].T).square().mean()))
        cv.append({"alpha": alpha, "mean_episode_mse": float(np.mean(errors)),
                   "episode_mse": errors})
    alpha = min(cv, key=lambda r: r["mean_episode_mse"])["alpha"]
    prior = ridge(W, V, alpha)
    return prior, alpha, {"shape": shape, "windows": V.shape[1], "cv": cv, "alpha": alpha,
                         "native_one_step_mse": float(W.square().mean())}


def direct_fit(model, initial, actions, target, prior, lam, counter, log):
    theta = prior.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=100, max_eval=120,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=8, line_search_fn="strong_wolfe")
    best = {"objective": float("inf"), "theta": prior.clone()}
    start = time.perf_counter()

    def closure():
        opt.zero_grad()
        objective, err = hard(model, initial, actions, target, theta, prior, lam)
        loss = objective / (2 * Q)
        loss.backward()
        counter.grad()
        value = float(objective.detach())
        if value < best["objective"]:
            best.update(objective=value, theta=theta.detach().clone())
        log.append({"method": "DIRECT", "iteration": len(log), "hard_objective": value,
                    "deployed_mse": float(err.detach()) / (2 * Q),
                    "elapsed": time.perf_counter() - start})
        return loss
    opt.step(closure)
    return best["theta"], {"seconds": time.perf_counter() - start, "closures": len(log),
                           "best_objective": best["objective"],
                           "optimizer_iterations": opt.state[theta].get("n_iter", 0)}


def auxiliary(model, initial, actions, target, theta, sources, prior, lam, mu):
    trace = []
    result = cached_rollout(model, initial, actions, sources=sources, trace=trace)
    V = torch.cat([t["feature"] for t in trace], 0).T
    consistency = (sources.T - theta @ V).square().sum()
    obs = data_error(result, target)
    return obs + mu * consistency + lam * (theta - prior).square().sum(), obs, consistency, V, trace


def source_fit(model, initial, actions, target, prior, lam, mu, counter, log):
    theta = prior.clone()
    with torch.no_grad():
        trace = []
        initial_hard, _ = hard(model, initial, actions, target, theta, prior, lam, trace)
        w = torch.cat([t["source"] for t in trace], 0).detach()
    best_theta, best_hard = theta.clone(), float(initial_hard)
    start = time.perf_counter()
    last_theta = theta
    for iteration in range(8):
        wr = w.detach().requires_grad_(True)
        _, _, _, _, trace = auxiliary(model, initial, actions, target, theta, wr, prior, lam, mu)
        native2 = trace[1]["native"][0, 0, 0, :Q]
        rows = []
        for offset in range(0, Q, 16):
            seeds = torch.eye(Q, device="cuda")[offset:offset + 16]
            jac = torch.autograd.grad(native2, wr, grad_outputs=seeds,
                                      is_grads_batched=True, retain_graph=True)[0]
            counter.grad(seeds.shape[0])
            assert float(jac[:, 1].abs().max()) == 0.0
            rows.append(jac[:, 0].detach())
        J = torch.cat(rows, 0)
        Qtheta = phi_response_jac(theta, actions[0, 3])
        with torch.no_grad():
            I = torch.eye(Q, dtype=torch.float64)
            K = torch.eye(2 * Q, dtype=torch.float64)
            A = torch.eye(2 * Q, dtype=torch.float64)
            K[Q:, :Q] = J.double().cpu()
            A[Q:, :Q] = -Qtheta.double().cpu()
            d1 = target[0] - trace[0]["native"][0, 0, 0, :Q]
            d2 = target[1] - native2 + J @ w[0]
            d = torch.cat([d1, d2]).double().cpu()
            b1 = theta @ trace[0]["feature"][0]
            b2 = theta @ trace[1]["feature"][0] - Qtheta @ w[0]
            b = torch.cat([b1, b2]).double().cpu()
            proposal = torch.linalg.solve(K.T @ K + mu * A.T @ A,
                                           K.T @ d + mu * A.T @ b).float().cuda().reshape(2, Q)
            old_joint = float(auxiliary(model, initial, actions, target, theta, w, prior, lam, mu)[0])
            accepted = 0.0
            for step in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = w + step * (proposal - w)
                trial_joint = float(auxiliary(model, initial, actions, target, theta, trial, prior, lam, mu)[0])
                if trial_joint <= old_joint + 1e-6 * max(1.0, old_joint):
                    w, accepted = trial, step
                    break
            _, _, _, V, _ = auxiliary(model, initial, actions, target, theta, w, prior, lam, mu)
            theta = ridge(w.T, V, lam / mu, prior)
            joint, obs, cons, _, _ = auxiliary(model, initial, actions, target, theta, w, prior, lam, mu)
            hard_value, hard_err = hard(model, initial, actions, target, theta, prior, lam)
            if float(hard_value) < best_hard:
                best_hard, best_theta = float(hard_value), theta.clone()
            row = {"method": f"SOURCE_MU{mu:g}", "iteration": iteration,
                   "joint_objective": float(joint), "auxiliary_mse": float(obs) / (2 * Q),
                   "consistency_mse": float(cons) / (2 * Q),
                   "hard_objective": float(hard_value), "deployed_mse": float(hard_err) / (2 * Q),
                   "source_step": accepted, "elapsed": time.perf_counter() - start}
            log.append(row)
            print(json.dumps(row), flush=True)
            last_theta = theta
        del wr, trace, native2, J, rows
    return best_theta, last_theta, {"seconds": time.perf_counter() - start,
                                   "best_objective": best_hard, "outer_steps": 8}


@torch.no_grad()
def diagnostics(model, initial, actions, target, theta, counter):
    trace = []
    hard(model, initial, actions, target, theta, theta, 1.0, trace)
    w = torch.cat([v["source"] for v in trace], 0)
    _, _, consistency, _, _ = auxiliary(model, initial, actions, target, theta, w, theta, 1., 1.)
    direction = torch.randn_like(w)
    direction /= direction.norm()
    delta = 1e-3
    with torch.enable_grad():
        v = w.detach().requires_grad_(True)
        out = cached_rollout(model, initial, actions, sources=v)["visual"][0, 3:, 0]
        probe = torch.randn_like(out)
        grad = torch.autograd.grad((out * probe).sum(), v)[0]
        counter.grad()
        analytical = float((grad * direction).sum())
    plus = cached_rollout(model, initial, actions, sources=w + delta * direction)["visual"][0, 3:, 0]
    minus = cached_rollout(model, initial, actions, sources=w - delta * direction)["visual"][0, 3:, 0]
    numerical = float(((plus - minus) * probe).sum() / (2 * delta))
    report = {"initial_source_consistency_mse": float(consistency) / (2 * Q),
              "gradient_analytic": analytical, "gradient_finite_difference": numerical,
              "gradient_relative_error": abs(analytical - numerical) / max(1., abs(analytical))}
    assert report["initial_source_consistency_mse"] < 1e-9
    assert report["gradient_relative_error"] < 0.03, report
    return report


def predict():
    if (OUT / "PREDICTIONS.pt").exists():
        raise FileExistsError("predictions already exist")
    prepared = json.loads((OUT / "PREPARE.json").read_text(encoding="utf8"))
    assert sha(OUT / "PREPARED.pt") == prepared["sha256"]
    for path, digest in prepared["input_hashes"].items():
        assert sha(path) == digest, path
    episodes = torch.load(OUT / "PREPARED.pt", map_location="cpu", weights_only=False)
    model, prep = runtime()
    versions = [(t, t._version) for t in list(model.parameters()) + list(model.buffers())]
    counter = Counter(model)
    priors, prior_info, training, fit_info, checks, predictions = {}, {}, {}, {}, {}, {}
    for shape in ("T", "L"):
        counter.phase = f"prior_{shape}"
        prior, lam, info = fit_prior(model, episodes, shape)
        priors[shape] = (prior, lam)
        prior_info[shape] = info
        torch.save(cpu(prior), OUT / f"PRIOR_{shape}.pt")
        print("prior", shape, json.dumps(info), flush=True)
    dump("PRIORS.json", prior_info)
    for shape, sample in CASES:
        case = f"{shape}{sample}"
        episode = episodes[case]
        initial, actions, target = support_view(episode)
        prior, lam = priors[shape]
        counter.phase = f"{case}_checks"
        checks[case] = diagnostics(model, initial, actions, target, prior, counter)
        arms = {"NATIVE": None, "PRIOR": prior}
        with torch.no_grad():
            sources = torch.zeros(2, Q, device="cuda")
            # Sequentially recover support sources with true visual clamped.
            for step in range(2):
                trace = []
                cached_rollout(model, initial, actions, sources=sources, trace=trace)
                sources[step] = target[step] - trace[step]["native"][0, 0, 0, :Q]
            trace = []
            cached_rollout(model, initial, actions, sources=sources, trace=trace)
            V = torch.cat([v["feature"] for v in trace], 0).T
            arms["STEP_RIDGE"] = ridge(sources.T, V, lam, prior)
        counter.phase = f"{case}_DIRECT"
        training[f"{case}_DIRECT"] = []
        arms["DIRECT"], fit_info[f"{case}_DIRECT"] = direct_fit(
            model, initial, actions, target, prior, lam, counter, training[f"{case}_DIRECT"])
        for mu in (1.0, 10.0):
            tag = f"SOURCE_MU{mu:g}"
            counter.phase = f"{case}_{tag}"
            training[f"{case}_{tag}"] = []
            arms[tag], last, fit_info[f"{case}_{tag}"] = source_fit(
                model, initial, actions, target, prior, lam, mu, counter, training[f"{case}_{tag}"])
            torch.save(cpu(last), OUT / f"{case}_{tag}_LAST.pt")
        for arm, theta in arms.items():
            if theta is not None:
                torch.save(cpu(theta), OUT / f"{case}_{arm}.pt")
        donor = DATA / f"donor_{shape}_seed101_n12_capture"
        source = DATA / f"reference_{shape}_seed101_s{sample}_m4_eight"
        plans = _sealed_candidates(source, SimpleNamespace(split=shape, sample=sample, mpc=4, donor=donor))
        _, goal_raw = _load_anchor_observations(donor, sample, 4)
        query_initial = episode_view(episode, [10, 15, 20])
        historical_actions = episode["actions"][10:20].reshape(1, 2, 10).cuda()
        counter.phase = f"{case}_query"
        with torch.no_grad():
            goal_z = model.encode_obs(dev(prep.transform_obs(goal_raw)))
            predictions[case] = {"goal_z": cpu(goal_z), "plans": {}}
            for cid, action, path, key in plans:
                aligned = torch.cat([historical_actions, action.cuda()], 1)
                entry = {"actions": action.cpu(), "arms": {}, "tensor_path": path, "tensor_key": key}
                for arm, theta in arms.items():
                    result = cached_rollout(model, query_initial, aligned, theta=theta)
                    result = {k: v[:, 2:] for k, v in result.items()}
                    score = native_objective_breakdown(result, goal_z, 4, alpha=1., base=2.)
                    assert all(torch.isfinite(v).all() for v in result.values())
                    entry["arms"][arm] = {"z": cpu(result),
                        "cost": {k: float(score[k]) for k in ("visual", "proprio", "total")}}
                predictions[case]["plans"][cid] = entry
            # Future suffix change must not change the first generated observation.
            base = torch.cat([historical_actions, plans[0][1].cuda()], 1)
            altered = base.clone()
            altered[:, 3:] = 0.
            check_a = cached_rollout(model, query_initial, base, theta=arms["DIRECT"])
            check_b = cached_rollout(model, query_initial, altered, theta=arms["DIRECT"])
            checks[case]["common_visual_prefix_max_abs"] = float(
                (check_a["visual"][:, :4] - check_b["visual"][:, :4]).abs().max())
            assert checks[case]["common_visual_prefix_max_abs"] < 1e-6
        torch.save(predictions[case], OUT / f"{case}_PREDICTIONS.pt")
        dump(f"{case}_TRAINING.json", {k: v for k, v in training.items() if k.startswith(case)})
        print("FINISHED", case, flush=True)
    assert all(t._version == old for t, old in versions)
    torch.save(predictions, OUT / "PREDICTIONS.pt")
    dump("TRAINING.json", training)
    dump("FIT_INFO.json", fit_info)
    dump("CHECKS.json", checks)
    dump("SEALED.json", {"prediction_sha256": sha(OUT / "PREDICTIONS.pt"),
                        "prepared_sha256": prepared["sha256"], "runner_sha256": sha(__file__),
                        "status": "PREDICTIONS_SEALED_NO_QUERY_OUTCOMES_READ",
                        "query_outcomes_read": False, "environment_calls": 0,
                        "backbone_parameters_buffers_unchanged": True,
                        "counts": counter.rows, "priors": prior_info,
                        "fit_info": fit_info, "checks": checks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "predict"])
    globals()[parser.parse_args().phase]()
