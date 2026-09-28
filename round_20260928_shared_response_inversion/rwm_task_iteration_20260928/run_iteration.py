"""Learn a four-block revision metric using the existing STEP_RIDGE and rollout.

All meta targets are completed offline episodes. MPC4 query outcomes are never
opened here. This is a small meta-adaptation baseline, not a new native operator.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import torch


OUT = Path(__file__).resolve().parent
FORMATION = OUT.parent / "shared_response_formation_20260928"
SPEC = importlib.util.spec_from_file_location("existing_formation", FORMATION / "formation.py")
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)
BLOCKS = [(0, 1), (1, 1213), (1213, 1223), (1223, 13343)]
BLOCK_NAMES = ["constant", "history", "action", "action_history"]
FIXED_STEPS = (0, 25, 50, 100)
NEW_ARMS = ("SHARED_PRIOR", "SHARED_RIDGE_FIXED", "SHARED_RIDGE_SCALE",
            "SHARED_RIDGE_METRIC", "SHARED_RIDGE_METRIC_SCALE_ONLY")


def dump(name, value):
    with (OUT / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save(name, value):
    with (OUT / name).open("xb") as stream:
        torch.save(f.cpu(value), stream)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(name)


@torch.no_grad()
def support_cache(model, episode):
    """Two sequential two-step calls: true visual, always native proprio.

The second call's trace already has both correct support features: changing the
second source cannot change either feature. No third redundant call is needed.
"""
    initial, actions, target = f.support_view(episode)
    sources = torch.zeros(2, f.Q, device="cuda")
    for step in range(2):
        trace = []
        f.cached_rollout(model, initial, actions, sources=sources, trace=trace)
        sources[step] = target[step] - trace[step]["native"][0, 0, 0, :f.Q]
    V = torch.cat([row["feature"] for row in trace], 0).T
    return sources.T.cpu(), V.cpu()


@torch.no_grad()
def one_step_cache(model, episode, start):
    initial = f.episode_view(episode, [start, start + 5, start + 10])
    actions = episode["actions"][start:start + 15].reshape(1, 3, 10).cuda()
    trace = []
    result = f.cached_rollout(model, initial, actions, trace=trace)
    assert len(trace) == 1
    native = result["visual"][0, -1, 0].cpu()
    target = episode["z"]["visual"][0, start + 15, 0]
    return trace[0]["feature"][0].cpu(), native, target


@torch.no_grad()
def offline_cache(model, episodes, counter):
    cache = {}
    for shape in ("T", "L"):
        counter.phase = "offline_H1_" + shape
        features, responses, owners, starts = [], [], [], []
        for sample in range(2, 10):
            ep = episodes[f"{shape}{sample}"]
            for start in range(ep["frames"] - 15):
                feature, native, target = one_step_cache(model, ep, start)
                features.append(feature)
                responses.append(target - native)
                owners.append(sample)
                starts.append(start)
        cache[shape] = {"V": torch.stack(features, 1), "W": torch.stack(responses, 1),
                        "owners": torch.tensor(owners), "starts": torch.tensor(starts)}
    assert sum(row["V"].shape[1] for row in cache.values()) == 161
    return cache


def prior_from_cache(cache, excluded=None):
    keep = (cache["owners"] != excluded if excluded is not None else
            torch.ones_like(cache["owners"], dtype=torch.bool))
    return f.ridge(cache["W"][:, keep].cuda(), cache["V"][:, keep].cuda(), 10.)


def compressed_record(W, V, feature, native, target, prior, owner, split):
    prior_cpu = prior.detach().cpu()
    # Match the float32 residual used by formation.ridge; the solve is float64.
    R = (W - prior_cpu @ V).double()
    gram = torch.stack([V[a:b].double().T @ V[a:b].double() for a, b in BLOCKS])
    cross = torch.stack([V[a:b].double().T @ feature[a:b].double() for a, b in BLOCKS])
    return {"owner": owner, "split": split, "W": W, "V": V,
            "query_feature": feature, "query_native": native, "query_target": target,
            "R": R, "block_gram": gram, "block_cross": cross,
            "base_query": (native + prior_cpu @ feature).double(),
            "prior_sha256": tensor_sha(prior_cpu)}


def input_coverage(V, feature, support_actions, query_action):
    support = support_actions.double().reshape(2, 10)
    query = query_action.double().reshape(10)
    cosine = (V.double().T @ feature.double()) / (V.double().norm(dim=0) * feature.double().norm())
    finite(cosine, "undefined feature cosine")
    return {"packed_action_l2_from_each_support": (support-query).norm(dim=1).tolist(),
            "feature_cosine_with_each_support": cosine.tolist(),
            "same_initial_state_interventions": False}


def stack_records(records):
    keys = ("R", "block_gram", "block_cross", "base_query", "query_target")
    return {key: torch.stack([row[key] for row in records]).double() for key in keys}


def weights_from(logits, loglambda):
    metric = torch.exp(logits - logits.mean())
    lam = torch.exp(loglambda)
    finite(metric, "non-finite metric")
    finite(lam, "non-finite lambda")
    if bool((metric <= 0).any()) or float(lam.detach()) <= 0:
        raise FloatingPointError("metric and lambda must be positive")
    return metric, lam


def meta_predictions(batch, logits, loglambda):
    metric, lam = weights_from(logits, loglambda)
    gram = torch.einsum("b,nbij->nij", metric, batch["block_gram"])
    cross = torch.einsum("b,nbi->ni", metric, batch["block_cross"])
    solved = torch.linalg.solve(gram + lam * torch.eye(2, dtype=torch.float64), cross)
    predicted = batch["base_query"] + torch.einsum("nqi,ni->nq", batch["R"], solved)
    finite(predicted, "non-finite meta predictions")
    return predicted


def meta_loss(batch, logits, loglambda):
    return (meta_predictions(batch, logits, loglambda) - batch["query_target"]).square().mean()


def state_row(step, loss, logits, loglambda):
    metric, lam = weights_from(logits, loglambda)
    return {"step": step, "train_loss": float(loss), "logits": logits.detach().tolist(),
            "loglambda": float(loglambda.detach()), "metric": metric.detach().tolist(),
            "lambda": float(lam.detach())}


def learn_metric(batch, name):
    logits = torch.zeros(4, dtype=torch.float64, requires_grad=name == "METRIC")
    loglambda = torch.tensor(math.log(10.), dtype=torch.float64, requires_grad=True)
    params = [loglambda] if name == "SCALE" else [logits, loglambda]
    optimizer = torch.optim.Adam(params, lr=.05)
    rows, checkpoints, best = [], {}, None
    for step in range(101):
        optimizer.zero_grad()
        loss = meta_loss(batch, logits, loglambda)
        finite(loss, name + " loss")
        row = state_row(step, loss.detach(), logits, loglambda)
        rows.append(row)
        if step in FIXED_STEPS:
            checkpoints[str(step)] = copy.deepcopy(row)
        if best is None or row["train_loss"] < best["train_loss"]:
            best = copy.deepcopy(row)
        if step == 100:
            break
        loss.backward()
        for parameter in params:
            if parameter.grad is None:
                raise RuntimeError("missing outer gradient")
            finite(parameter.grad, name + " gradient")
        optimizer.step()
    return {"history": rows, "checkpoints": checkpoints, "selected": best,
            "selection": "minimum train loss over steps 0..100; no evaluation outcomes"}


@torch.no_grad()
def metric_ridge(W, V, prior, metric, lam):
    """Closed form in original phi coordinates; deploy via cached_rollout."""
    W, V = W.to(prior.device), V.to(prior.device)
    diagonal = torch.cat([torch.full((b-a,), float(metric[i]), dtype=torch.float64,
                                    device=prior.device) for i, (a, b) in enumerate(BLOCKS)])
    MV = diagonal[:, None] * V.double()
    gram = V.double().T @ MV
    residual = (W - prior @ V).double()
    coefficient = torch.linalg.solve(gram + float(lam) * torch.eye(2,
                                    dtype=torch.float64, device=prior.device), residual.T).T
    theta = prior + (coefficient @ MV.T).float()
    finite(theta, "non-finite fitted theta")
    return theta


@torch.no_grad()
def evaluate_meta(records, learned):
    batch = stack_records(records)
    result = {}

    def summarize(predictions):
        errors = (predictions-batch["query_target"]).square().mean(1)
        return {"mean_mse": float(errors.mean()),
                "episode_mse": {r["owner"]: float(err) for r, err in zip(records, errors)}}

    for method, report in learned.items():
        result[method] = {}
        checkpoints = dict(report["checkpoints"])
        checkpoints["selected_by_train"] = report["selected"]
        for label, row in checkpoints.items():
            logits = torch.tensor(row["logits"], dtype=torch.float64)
            loglambda = torch.tensor(row["loglambda"], dtype=torch.float64)
            result[method][label] = summarize(meta_predictions(batch, logits, loglambda))
    result["PRIOR_ONLY"] = {"no_adaptation": summarize(batch["base_query"])}
    state = learned["METRIC"]["selected"]
    result["METRIC_SCALE_ONLY"] = {"selected_by_train": summarize(meta_predictions(batch,
        torch.zeros(4, dtype=torch.float64), torch.tensor(state["loglambda"], dtype=torch.float64)))}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generated = [p for p in OUT.iterdir() if p.suffix == ".pt" or p.name in
                 ("META_LOG.json", "META_EVAL.json", "COORDINATES.json", "SEALED.json", "CHECKS.json")]
    if generated:
        raise FileExistsError("refusing to overwrite outputs: " + str(generated))
    started = time.perf_counter()
    prepared = json.loads((FORMATION / "PREPARE.json").read_text(encoding="utf8"))
    assert f.sha(FORMATION / "PREPARED.pt") == prepared["sha256"]
    for path, digest in prepared["input_hashes"].items():
        assert f.sha(path) == digest, path
    old_seal = json.loads((FORMATION / "SEALED.json").read_text(encoding="utf8"))
    assert f.sha(FORMATION / "PREDICTIONS.pt") == old_seal["prediction_sha256"]
    episodes = torch.load(FORMATION / "PREPARED.pt", map_location="cpu", weights_only=False)
    legacy = torch.load(FORMATION / "PREDICTIONS.pt", map_location="cpu", weights_only=False)
    model, _ = f.runtime()
    versions = [(t, t._version) for t in list(model.parameters()) + list(model.buffers())]
    counter = f.Counter(model)
    def enforce_budget(module, inputs):
        if sum(row["batch_transitions"] for row in counter.rows.values()) > 1200:
            raise RuntimeError("predictor sample-transition budget exceeded")
    budget_hook = model.predictor.register_forward_pre_hook(enforce_budget)
    cache = offline_cache(model, episodes, counter)
    save("OFFLINE_H1.pt", cache)
    with torch.no_grad():
        priors = {shape: prior_from_cache(cache[shape]) for shape in ("T", "L")}
    for shape, prior in priors.items():
        save(f"SHARED_PRIOR_{shape}.pt", prior)
    meta = []
    checks = {"initial_metric_vs_formation_ridge": {}, "compressed_vs_dense_initial": {},
              "compressed_vs_native_query": {}}
    for shape in ("T", "L"):
        for sample in range(2, 12):
            owner = f"{shape}{sample}"
            ep = episodes[owner]
            if ep["frames"] < 26:
                continue
            split = "train_crossfit" if sample < 10 else "evaluation_only"
            counter.phase = "meta_cache_" + split
            with torch.no_grad():
                prior = prior_from_cache(cache[shape], sample) if sample < 10 else priors[shape]
                W, V = support_cache(model, ep)
                feature, native, target = one_step_cache(model, ep, 10)
                record = compressed_record(W, V, feature, native, target, prior, owner, split)
                record["support_packed_actions"] = ep["actions"][10:20].reshape(2, 10)
                record["query_packed_action"] = ep["actions"][20:25].reshape(10)
                record["input_coverage"] = input_coverage(V, feature,
                    record["support_packed_actions"], record["query_packed_action"])
                dense = metric_ridge(W, V, prior, [1.] * 4, 10.)
                reference = f.ridge(W.cuda(), V.cuda(), 10., prior)
                error = float((dense - reference).abs().max())
                assert error < 2e-5, (owner, error)
                checks["initial_metric_vs_formation_ridge"][owner] = error
                compressed = meta_predictions(stack_records([record]), torch.zeros(4,
                    dtype=torch.float64), torch.tensor(math.log(10.), dtype=torch.float64))[0]
                direct = native.double() + (dense @ feature.cuda()).cpu().double()
                error = float((compressed - direct).abs().max())
                assert error < 2e-5, (owner, error)
                checks["compressed_vs_dense_initial"][owner] = error
                meta.append(record)
    train = [row for row in meta if row["split"] == "train_crossfit"]
    evaluation = [row for row in meta if row["split"] == "evaluation_only"]
    assert len(train) == 14 and len(evaluation) == 4
    save("META_CACHE.pt", {"episodes": meta, "train_owners": [r["owner"] for r in train],
                           "evaluation_owners": [r["owner"] for r in evaluation]})
    batch = stack_records(train)
    learned = {name: learn_metric(batch, name) for name in ("SCALE", "METRIC")}
    dump("META_LOG.json", learned)
    # Evaluation is performed only after both training choices are fixed.
    dump("META_EVAL.json", {"training": evaluate_meta(train, learned),
                            "evaluation_only": evaluate_meta(evaluation, learned),
                            "used_for_selection": False})
    scale = learned["SCALE"]["selected"]
    metric = learned["METRIC"]["selected"]
    settings = {"SHARED_RIDGE_FIXED": ([1.] * 4, 10.),
                "SHARED_RIDGE_SCALE": ([1.] * 4, scale["lambda"]),
                "SHARED_RIDGE_METRIC": (metric["metric"], metric["lambda"]),
                "SHARED_RIDGE_METRIC_SCALE_ONLY": ([1.] * 4, metric["lambda"])}
    # The compressed outer computation must equal actual native deployment.
    counter.phase = "meta_native_equivalence"
    with torch.no_grad():
        rule_states = {"FIXED": {"logits": [0.] * 4, "loglambda": math.log(10.)},
                       "SCALE": scale, "METRIC": metric}
        for owner in ("T2", "L2"):
            record = next(row for row in meta if row["owner"] == owner)
            prior = prior_from_cache(cache[owner[0]], 2)
            ep = episodes[owner]
            initial = f.episode_view(ep, [10, 15, 20])
            actions = ep["actions"][10:25].reshape(1, 3, 10).cuda()
            for rule, state in rule_states.items():
                logits = torch.tensor(state["logits"], dtype=torch.float64)
                loglambda = torch.tensor(state["loglambda"], dtype=torch.float64)
                m, lam = weights_from(logits, loglambda)
                theta = metric_ridge(record["W"], record["V"], prior, m.tolist(), float(lam))
                result = f.cached_rollout(model, initial, actions, theta=theta)
                native_deployed = result["visual"][0, -1, 0].cpu().double()
                compressed = meta_predictions(stack_records([record]), logits, loglambda)[0]
                error = float((compressed-native_deployed).abs().max())
                assert error < 2e-5, (owner, rule, error)
                checks["compressed_vs_native_query"][owner + "_" + rule] = error
    predictions, coordinates = {}, {"block_names": BLOCK_NAMES, "arms": settings,
        "meta_input_coverage": {row["owner"]: row["input_coverage"] for row in meta}, "cases": {}}
    for shape, sample in f.CASES:
        case = f"{shape}{sample}"
        ep, prior = episodes[case], priors[shape]
        counter.phase = case + "_support"
        with torch.no_grad():
            W, V = support_cache(model, ep)
            arms = {"SHARED_PRIOR": prior}
            arms.update({name: metric_ridge(W, V, prior, m, lam)
                         for name, (m, lam) in settings.items()})
            coordinates["cases"][case] = {"W_sha256": tensor_sha(W), "V_sha256": tensor_sha(V),
                "query_input_coverage": {},
                "laws": {name: {"theta_sha256": tensor_sha(theta),
                                 "support_linear_residual_mse": float((W.cuda() - theta @ V.cuda()).square().mean()),
                                 "prior_delta_norm": float((theta-prior).norm())}
                         for name, theta in arms.items()}}
            save(case + "_SUPPORT.pt", {"W": W, "V": V})
            for name, theta in arms.items():
                save(case + "_" + name + ".pt", theta)
            original = legacy[case]
            goal = f.dev(original["goal_z"])
            query_initial = f.episode_view(ep, [10, 15, 20])
            historical = ep["actions"][10:20].reshape(1, 2, 10).cuda()
            predictions[case] = {"goal_z": original["goal_z"], "plans": {},
                "legacy_arms": ["NATIVE", "PRIOR", "STEP_RIDGE", "DIRECT"],
                "new_arms": list(NEW_ARMS)}
            counter.phase = case + "_query"
            for cid, old_entry in original["plans"].items():
                entry = {key: copy.deepcopy(value) for key, value in old_entry.items() if key != "arms"}
                entry["arms"] = {name: copy.deepcopy(old_entry["arms"][name])
                                 for name in predictions[case]["legacy_arms"]}
                aligned = torch.cat([historical, old_entry["actions"].cuda()], 1)
                joined = f._join_encoded_observation_action(model, query_initial, aligned[:, :3])
                query_feature = f.phi(joined, aligned[:, 2:3])[0].cpu()
                coordinates["cases"][case]["query_input_coverage"][cid] = input_coverage(
                    V, query_feature, ep["actions"][10:20], old_entry["actions"][0, 0])
                for name, theta in arms.items():
                    result = f.cached_rollout(model, query_initial, aligned, theta=theta)
                    result = {key: value[:, 2:] for key, value in result.items()}
                    for value in result.values():
                        finite(value, case + "/" + cid + "/" + name)
                    score = f.native_objective_breakdown(result, goal, 4, alpha=1., base=2.)
                    entry["arms"][name] = {"z": f.cpu(result), "cost":
                        {key: float(score[key]) for key in ("visual", "proprio", "total")}}
                predictions[case]["plans"][cid] = entry
        print("completed", case, flush=True)
    assert all(t._version == version for t, version in versions)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    calls = sum(row["forward_calls"] for row in counter.rows.values())
    transitions = sum(row["batch_transitions"] for row in counter.rows.values())
    assert transitions <= 1200, counter.rows
    assert calls == 1073 and transitions == 1073, counter.rows
    budget_hook.remove()
    save("PREDICTIONS.pt", predictions)
    dump("COORDINATES.json", coordinates)
    dump("CHECKS.json", checks)
    dump("SEALED.json", {"status": "PREDICTIONS_SEALED_NO_MPC4_QUERY_OUTCOMES_READ",
        "query_outcomes_read": False, "old_development_cases_not_blind": True,
        "environment_calls": 0, "backbone_parameters_buffers_unchanged": True,
        "outer_training": {"SCALE_parameters": 1, "METRIC_parameters": 5,
                           "METRIC_effective_parameters": 4, "Adam_steps_per_arm": 100,
                           "total_Adam_steps": 200,
                           "learning_rate": .05, "selection": "minimum meta train loss"},
        "online_fitting": "closed-form weighted ridge on two completed support steps",
        "prior_train_samples": list(range(2, 10)), "prior_alpha": 10.,
        "meta_train_episodes": [row["owner"] for row in train],
        "meta_evaluation_episodes": [row["owner"] for row in evaluation],
        "counts": counter.rows, "predictor_calls": calls, "sample_transitions": transitions,
        "legacy_prediction_sha256": old_seal["prediction_sha256"],
        "prepared_sha256": prepared["sha256"], "formation_sha256": f.sha(FORMATION / "formation.py"),
        "runner_sha256": f.sha(__file__), "protocol_sha256": f.sha(OUT / "RUN_PROTOCOL.md"),
        "prediction_sha256": f.sha(OUT / "PREDICTIONS.pt"),
        "artifact_sha256": {p.name: f.sha(p) for p in sorted(OUT.iterdir()) if p.suffix == ".pt"},
        "seconds": time.perf_counter() - started})
    print("sealed", calls, transitions, flush=True)


if __name__ == "__main__":
    main()
