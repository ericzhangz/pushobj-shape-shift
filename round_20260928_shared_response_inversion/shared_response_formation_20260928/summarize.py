"""Summarize frozen predictions; never tunes or updates a response model."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf8"))


def main():
    result = read("scored_verified/RESULTS.json")
    original = read("scored/RESULTS.json")
    geometry_fields = {"object_centroid_error_px", "terminal_object_centroid_error_px"}
    geometry_drift = []
    for old, new in zip(original["summary"], result["summary"]):
        for key in old:
            if key in geometry_fields:
                geometry_drift.append(abs(old[key] - new[key]))
            else:
                assert old[key] == new[key], (old["case"], old["arm"], key)
    assert result["geometry_counts"] == original["geometry_counts"]
    means = result["arm_means"]
    arms = list(means)
    fits = read("FIT_INFO.json")
    training = read("TRAINING.json")
    checks = read("OPERATOR_CHECKS.json")
    cases = [v["case"] for v in checks["checks"]]
    relative = {arm: {metric: {
        reference: 100 * (means[arm][metric] / means[reference][metric] - 1)
        for reference in ("NATIVE", "PRIOR")}
        for metric in ("visual_mse", "contrast_mse", "pair_gap_mae", "regret")}
        for arm in arms}
    support = {r["case"]: r["support"] for r in checks["checks"]}
    source_gaps = {}
    for case in cases:
        for arm in ("SOURCE_MU1", "SOURCE_MU10"):
            last = training[f"{case}_{arm}"][-1]
            source_gaps[f"{case}_{arm}"] = {
                "auxiliary_mse": last["auxiliary_mse"],
                "deployed_mse": last["deployed_mse"],
                "deployment_over_auxiliary": last["deployed_mse"] / last["auxiliary_mse"],
                "consistency_mse": last["consistency_mse"]}
    predictions = torch.load(OUT / "PREDICTIONS.pt", map_location="cpu", weights_only=False)
    targets = torch.load(OUT / "scored_verified/TARGETS.pt", map_location="cpu", weights_only=False)
    decomposition = []
    for case, bundle in predictions.items():
        candidate_ids = list(bundle["plans"])
        for arm in arms:
            errors = torch.stack([
                bundle["plans"][cid]["arms"][arm]["z"]["visual"].double()[0, 1:, 0] -
                targets[case][cid]["truth_z"]["visual"].double()[0, 1:, 0]
                for cid in candidate_ids])
            for scope, e in (("all_future", errors), ("terminal", errors[:, -1:])):
                shared = e.mean(0)
                common_mse = float(shared.square().mean())
                differential_mse = float((e - shared).square().mean())
                total_mse = float(e.square().mean())
                assert abs(total_mse - common_mse - differential_mse) < 1e-12
                if scope == "terminal":
                    scored = next(r for r in result["summary"] if r["case"] == case and r["arm"] == arm)
                    assert abs(scored["contrast_mse"] - 16 / 7 * differential_mse) < 1e-12
                decomposition.append({"case": case, "arm": arm, "scope": scope,
                                      "total_mse": total_mse, "common_error_mse": common_mse,
                                      "differential_error_mse": differential_mse})
    decomposition_means = {
        arm: {scope: {field: float(np.mean([r[field] for r in decomposition
                        if r["arm"] == arm and r["scope"] == scope]))
                      for field in ("total_mse", "common_error_mse", "differential_error_mse")}
              for scope in ("all_future", "terminal")} for arm in arms}
    counts = {}
    for phase, filename in (("prepare", "PREPARE.json"), ("fit_and_predict", "SEALED.json"),
                            ("post_operator_checks", "OPERATOR_CHECKS.json")):
        rows = read(filename)["counts"].values()
        counts[phase] = {k: sum(r[k] for r in rows) for k in
                        ("forward_calls", "batch_transitions", "vjp_seed_vectors", "backward_calls")}
    counts["total_predictor_forward_calls"] = sum(
        counts[p]["forward_calls"] for p in ("prepare", "fit_and_predict", "post_operator_checks"))
    table = {arm: {"seconds_mean": float(np.mean([
        fits[f"{case}_{arm}"]["seconds"] for case in cases]))}
        for arm in ("DIRECT", "SOURCE_MU1", "SOURCE_MU10")}
    payload = {"arm_means": means, "relative_percent": relative, "support": support,
               "source_gaps": source_gaps, "counts": counts, "time": table,
               "scoring_recheck_core_metrics_identical": True,
               "max_centroid_summary_recheck_drift_px": max(geometry_drift),
               "error_decomposition": decomposition,
               "error_decomposition_means": decomposition_means,
               "geometry_counts": result["geometry_counts"]}
    (OUT / "FINAL_METRICS.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["Native", "Prior", "Step ridge", "Direct", "Source 1", "Source 10"]
    colors = ["#777777", "#3973ac", "#77aadd", "#228833", "#aa3377", "#cc99bb"]
    for ax, metric, title in zip(axes[0],
            ("visual_mse", "contrast_mse"),
            ("Held actions: full visual error", "Held actions: terminal consequence-difference error")):
        values = [means[arm][metric] for arm in arms]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Latent MSE (lower is better)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    ax = axes[1, 0]
    rows = training["L1_SOURCE_MU1"]
    ax.plot(range(1, 9), [r["auxiliary_mse"] for r in rows], "o-", color=colors[4], label="Source 1: auxiliary")
    ax.plot(range(1, 9), [r["deployed_mse"] for r in rows], "s-", color="#e69f00", label="Source 1: deployed")
    ax.axhline(support["L1"]["DIRECT"]["mse"], color=colors[3], ls="--", label="Direct: deployed")
    ax.set_title("L1 support: auxiliary fit versus deployment")
    ax.set_xlabel("Source outer iteration")
    ax.set_ylabel("Support visual MSE")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    ax = axes[1, 1]
    values_by_case = {(r["case"], r["arm"]): r for r in result["summary"]}
    x = np.arange(4)
    for j, arm in enumerate(("DIRECT", "SOURCE_MU1", "SOURCE_MU10")):
        vals = [100 * (values_by_case[(case, arm)]["visual_mse"] /
                       values_by_case[(case, "PRIOR")]["visual_mse"] - 1) for case in cases]
        ax.bar(x + (j - 1) * .24, vals, .24, label=labels[arms.index(arm)], color=colors[arms.index(arm)])
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, cases)
    ax.set_ylabel("Visual error change versus prior (%)")
    ax.set_title("Online support does not transfer uniformly")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("Formation test: four development cases, 32 held action chains", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "FORMATION_RESULTS.png", dpi=180)
    fig.savefig(OUT / "FORMATION_RESULTS.pdf")
    print(json.dumps({"means": means, "time": table, "counts": counts,
                      "support": support, "source_gaps": source_gaps}, indent=2))


if __name__ == "__main__":
    main()
