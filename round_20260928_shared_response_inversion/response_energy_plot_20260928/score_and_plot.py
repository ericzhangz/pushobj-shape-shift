"""Score sealed visual forecasts; never trains/selects models using test outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from metric_contract import evaluate_response, verify_contract

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def write_csv(path, rows):
    with path.open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def root_step_metrics(pred, truth, goal, metric, step, root):
    pr, tr = pred[root, :, step - 1], truth[root, :, step - 1]
    pc = np.mean((pr - goal[root]) ** 2, axis=-1)
    tc = np.mean((tr - goal[root]) ** 2, axis=-1)
    ii, jj = np.triu_indices(len(pc), 1)
    pair_rows = [x for x in metric["pair_rows"] if x["root"] == root and x["step"] == step]
    te = np.asarray([x["true_contrast_rms"] ** 2 for x in pair_rows])
    valid = te > 0
    directional = np.asarray([0.0 if x["directed_prediction_rms"] is None
                              else x["directed_prediction_rms"] for x in pair_rows])
    cross = np.sqrt(te) * directional
    orth = [x["orthogonal_mse"] for x in pair_rows if x["orthogonal_mse"] is not None]
    per_root = metric["per_root"][root]
    return {
        "prefix_full_mse": float(np.mean(per_root["full_mse_by_step"][:step])),
        "endpoint_full_mse": float(per_root["full_mse_by_step"][step - 1]),
        "prefix_contrast_mse": float(np.mean(per_root["contrast_mse_by_step"][:step])),
        "endpoint_contrast_mse": float(per_root["contrast_mse_by_step"][step - 1]),
        "directional_gain": float(cross.sum() / te.sum()) if te.sum() > 0 else None,
        "orthogonal_mse": float(np.mean(orth)) if orth else None,
        "zero_true_pairs": int(np.sum(~valid)),
        "true_contrast_energy": float(te.mean()),
        "visual_cost_gap_mae": float(np.mean(np.abs((pc[ii] - pc[jj]) - (tc[ii] - tc[jj])))),
        "visual_selection_regret": float(tc[int(np.argmin(pc))] - tc.min()),
        "selected_word_index": int(np.argmin(pc)),
        "best_word_index": int(np.argmin(tc)),
    }


METRICS = ("prefix_full_mse", "endpoint_full_mse", "prefix_contrast_mse",
           "endpoint_contrast_mse", "directional_gain", "orthogonal_mse",
           "true_contrast_energy", "visual_cost_gap_mae", "visual_selection_regret")


def summarize(rows):
    result = {}
    rng = np.random.default_rng(260928)
    for arm in sorted(set(r["arm"] for r in rows)):
        result[arm] = {}
        for horizon in (1, 2, 3, 5):
            group = [r for r in rows if r["arm"] == arm and r["horizon"] == horizon]
            seeds = sorted(set(r["seed"] for r in group))
            episodes = sorted(set(r["episode"] for r in group))
            if not group:
                continue
            summary = {"episodes": episodes, "seeds": seeds, "metrics": {}}
            for name in METRICS:
                if any(r[name] is None for r in group):
                    summary["metrics"][name] = {"mean": None, "reason": "undefined for at least one root"}
                    continue
                by_seed = []
                ep_values = []
                for seed in seeds:
                    by_seed.append(float(np.mean([np.mean([r[name] for r in group
                                     if r["seed"] == seed and r["episode"] == ep]) for ep in episodes])))
                for ep in episodes:
                    ep_values.append(float(np.mean([np.mean([r[name] for r in group
                                      if r["episode"] == ep and r["seed"] == seed]) for seed in seeds])))
                sampled = np.asarray(ep_values)[rng.integers(0, len(episodes), size=(2000, len(episodes)))].mean(1)
                summary["metrics"][name] = {
                    "mean": float(np.mean(ep_values)), "seed_values": by_seed,
                    "seed_min": min(by_seed), "seed_max": max(by_seed),
                    "episode_values_seed_mean": dict(zip(episodes, ep_values)),
                    "episode_bootstrap_95_descriptive": np.quantile(sampled, [.025, .975]).tolist(),
                }
            result[arm][str(horizon)] = summary
    paired = {}
    if "GRU" in result:
        for arm, values in result.items():
            if arm == "GRU":
                continue
            paired[arm] = {}
            for horizon in values:
                paired[arm][horizon] = {}
                for name in METRICS:
                    ours = values[horizon]["metrics"][name]
                    other = result["GRU"][horizon]["metrics"][name]
                    if ours["mean"] is None or other["mean"] is None:
                        continue
                    a, b = ours["episode_values_seed_mean"], other["episode_values_seed_mean"]
                    episodes = sorted(a)
                    if set(a) != set(b):
                        raise ValueError("paired comparison episode mismatch")
                    delta = np.asarray([a[e] - b[e] for e in episodes])
                    sampled = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(1)
                    paired[arm][horizon][name] = {
                        "difference_arm_minus_GRU": float(delta.mean()),
                        "episode_differences": dict(zip(episodes, delta.tolist())),
                        "episode_bootstrap_95_descriptive": np.quantile(sampled, [.025, .975]).tolist(),
                    }
    return result, paired


def plot_energy(rows, out, pool):
    group = [r for r in rows if r["pool_horizon"] == pool and not r["zero_true_contrast"]]
    order = ["NATIVE", "GRU", "DIRECT", "SPECTRAL", "SPECTRAL_FT", "NONLINEAR_STATE"]
    arms = [a for a in order if any(r["arm"] == a for r in group)]
    if not group:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(.5, .5, "All true action contrasts are exactly zero.\nDirection is undefined; see spurious contrast errors.",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(out / f"energy_plot_h{pool}.png", dpi=170)
        plt.close(fig)
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False, layout="constrained")
    axes = axes.ravel()
    global_x = max(r["true_contrast_rms"] for r in group)
    global_ymin = min(0.0, min(r["directed_prediction_rms"] for r in group))
    global_ymax = max(global_x, max(r["directed_prediction_rms"] for r in group))
    margin = max(global_ymax - global_ymin, global_x, 1e-12) * .04
    max_orth = max(np.sqrt(r["orthogonal_mse"]) for r in group)
    color_norm = Normalize(vmin=0.0, vmax=max_orth if max_orth > 0 else 1.0)
    for ax, arm in zip(axes, arms):
        subset = [r for r in group if r["arm"] == arm]
        x = np.asarray([r["true_contrast_rms"] for r in subset])
        y = np.asarray([r["directed_prediction_rms"] for r in subset])
        # All seeds shown individually; no ensemble is used for prediction/score.
        orth = np.sqrt([r["orthogonal_mse"] for r in subset])
        scatter = ax.scatter(x, y, c=orth, norm=color_norm, cmap="viridis", s=8, alpha=.3, rasterized=True)
        ax.plot([0, global_x], [0, global_x], "k--", lw=1, label="correct parallel component")
        ax.axhline(0, color="0.5", lw=.7)
        ax.set_xlim(-.025 * global_x, global_x * 1.04)
        ax.set_ylim(global_ymin - margin, global_ymax + margin)
        ax.set_title(arm)
        ax.set_xlabel("True action contrast RMS (384 visual features)")
        ax.set_ylabel("Prediction projected along true contrast")
        ax.grid(alpha=.2)
    for ax in axes[len(arms):]:
        ax.axis("off")
    fig.colorbar(scatter, ax=list(axes[:len(arms)]), shrink=.75, label="Orthogonal contrast RMS")
    fig.suptitle(f"H{pool} response plot: endpoint; all pairs and selected training seeds\n"
                 "Pairs are repeated measurements; vertical diagonal agreement is insufficient without orthogonal/full error")
    fig.savefig(out / f"energy_plot_h{pool}.png", dpi=170)
    plt.close(fig)


def plot_curves(summary, out):
    specs = [("prefix_full_mse", "Complete visual MSE (prefix average)"),
             ("prefix_contrast_mse", "Action-contrast MSE (prefix average)"),
             ("directional_gain", "Directional gain at endpoint"),
             ("orthogonal_mse", "Orthogonal contrast MSE at endpoint")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, label) in zip(axes.ravel(), specs):
        for arm, horizons in summary.items():
            values = [horizons[str(h)]["metrics"][name]["mean"] for h in (1, 2, 3, 5)]
            if any(v is None for v in values):
                continue
            line, = ax.plot([1, 2], values[:2], "o-", label=arm)
            ax.plot([3, 5], values[2:], "s--", color=line.get_color())
        if name == "directional_gain":
            ax.axhline(1, color="0.5", lw=.8)
            ax.axhline(0, color="0.5", lw=.8)
        ax.set_title(label)
        ax.set_xlabel("Future macro steps; H1/H2 pool and H3/H5 pool differ")
        ax.set_xticks([1, 2, 3, 5])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Episode-balanced, then seed-averaged metrics; frozen 384-dimensional visual representation")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / "response_metric_curves.png", dpi=170)
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-sealed", action="store_true")
    args = parser.parse_args()
    if not args.score_sealed:
        raise SystemExit("explicit --score-sealed required after all forecasts are sealed")
    predictions = ROOT / "predictions"
    model_seal = json.loads((ROOT / "FINAL_MODEL_PREDICTIONS_SEALED.json").read_text())
    native_seal = json.loads((predictions / "NATIVE_SEALED.json").read_text())
    for name, digest in model_seal["hashes"].items():
        if sha(predictions / name) != digest:
            raise ValueError(f"model prediction changed: {name}")
    for name, digest in native_seal["outputs"].items():
        if sha(name) != digest:
            raise ValueError(f"native prediction changed: {name}")
    selected = json.loads((ROOT / "FINAL_SELECTED_BY_VALIDATION.json").read_text())
    if selected != model_seal["selected"]:
        raise ValueError("validation selection changed after seal")
    out = ROOT / "scored"
    out.mkdir(exist_ok=False)
    figures = out / "figures"
    figures.mkdir()
    root_rows, pair_rows, metric_checks, hashes = [], [], {}, {}
    for pool in (2, 5):
        truth_path = ROOT / f"test_h{pool}.npz"
        hashes[str(truth_path)] = sha(truth_path)
        collection = json.loads((ROOT / f"test_h{pool}_COLLECTION.json").read_text())
        if hashes[str(truth_path)] != collection["dataset_sha256"]:
            raise ValueError("test outcome file changed after collection seal")
        public_path = ROOT / f"test_public_h{pool}.npz"
        if sha(public_path) != native_seal["inputs"][str(public_path)]:
            raise ValueError("public deployment inputs changed after native seal")
        with np.load(public_path, allow_pickle=False) as public:
            public_words = public["words"].copy()
        with np.load(truth_path, allow_pickle=False) as f:
            truth = f["future_visual"].astype(np.float64)
            goal = f["goal_visual"].astype(np.float64)
            root_ids, episodes, words = f["root_ids"], f["episode_ids"], f["words"]
        if not np.array_equal(words, public_words):
            raise ValueError("test truth word order differs from public deployment")
        models = [(arm, name) for arm, choice in selected.items() for name in choice["models"]]
        models.append(("NATIVE", f"NATIVE_h{pool}"))
        for arm, name in models:
            path = predictions / f"{name}.npz"
            hashes[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as f:
                if not np.array_equal(f["root_ids"], root_ids) or not np.array_equal(f["episode_ids"], episodes):
                    raise ValueError(f"root binding mismatch: {name}")
                if arm == "NATIVE" and not np.array_equal(f["words"], words):
                    raise ValueError("native word binding mismatch")
                pred = f["visual" if arm == "NATIVE" else f"visual_h{pool}"].astype(np.float64)
            seed = "frozen" if arm == "NATIVE" else name.rsplit("_s", 1)[1].split("_")[0]
            metric = evaluate_response(pred, truth)
            metric_checks[f"{name}_h{pool}"] = metric["contract"]
            for horizon in (1, 2) if pool == 2 else (3, 5):
                for r, root_id in enumerate(root_ids):
                    root_rows.append({"arm": arm, "model": name, "seed": seed,
                                      "pool_horizon": pool, "horizon": horizon,
                                      "root": str(root_id), "episode": str(episodes[r]),
                                      **root_step_metrics(pred, truth, goal, metric, horizon, r)})
            for row in metric["pair_rows"]:
                if row["step"] == pool:
                    pair_rows.append({"arm": arm, "model": name, "seed": seed,
                                      "pool_horizon": pool, "episode": str(episodes[row["root"]]),
                                      "root_id": str(root_ids[row["root"]]), **row})
    summary, paired = summarize(root_rows)
    write_csv(out / "ROOT_METRICS.csv", root_rows)
    write_csv(out / "PAIR_CONTRASTS_TERMINAL.csv", pair_rows)
    write_json(out / "METRICS.json", {
        "contract": {"primary": "complete 384-dimensional frozen visual tokens",
                     "candidate_pairs_independent": False, "inference_unit": "episode",
                     "seed_ensemble": False, "bootstrap_replicates": 2000,
                     "bootstrap_scope": "descriptive only; only four test episodes",
                     "selection_objective": "visual-only terminal goal MSE; not original visual+proprio planner cost",
                     "visual_tokens_do_not_identify_object_response": True},
        "summary": summary, "paired_arm_minus_GRU": paired,
        "metric_checks": metric_checks, "synthetic_checks": verify_contract()})
    write_json(out / "EVALUATION_INPUT_HASHES.json", hashes)
    plot_energy(pair_rows, figures, 2)
    plot_energy(pair_rows, figures, 5)
    plot_curves(summary, figures)
    print(json.dumps({"status": "COMPLETE", "root_rows": len(root_rows), "terminal_pair_rows": len(pair_rows),
                      "scored_directory": str(out)}, indent=2))


if __name__ == "__main__":
    run()
