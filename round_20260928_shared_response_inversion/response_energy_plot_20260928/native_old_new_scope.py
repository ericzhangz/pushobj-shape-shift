"""Post-hoc contextual re-expression of old FULL and new NATIVE evidence only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from metric_contract import evaluate_response

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "interaction_validation_20260927"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    oldp = OLD / "native/PREDICTIONS.pt"
    oldt = OLD / "evaluation_scored/EVALUATION_TARGETS.pt"
    newp = ROOT / "predictions/NATIVE_h5.npz"
    newt = ROOT / "test_h5.npz"
    paths = [oldp, oldt, newp, newt]
    hashes = {str(p): sha(p) for p in paths}
    old_seal = json.loads((OLD / "native/SEALED.json").read_text())
    assert hashes[str(oldp)] == old_seal["prediction_sha256"]
    pred = torch.load(oldp, map_location="cpu", weights_only=False)
    truth = torch.load(oldt, map_location="cpu", weights_only=False)
    cases = ["T0", "T1", "L0", "L1"]
    candidates = list(pred[cases[0]]["plans"])
    assert len(candidates) == 8
    pp, tt = [], []
    for case in cases:
        assert list(pred[case]["plans"]) == candidates
        assert set(truth[case]) == set(candidates)
        pr, tr = [], []
        for candidate in candidates:
            p = pred[case]["plans"][candidate]["arms"]["FULL"]["z"]["visual"]
            t = truth[case][candidate]["truth_z"]["visual"]
            assert tuple(p.shape) == tuple(t.shape) == (1, 6, 1, 384)
            assert torch.max(torch.abs(p[:, :1] - t[:, :1])).item() < 2e-5
            pr.append(p[0, 1:, 0].numpy())
            tr.append(t[0, 1:, 0].numpy())
        pp.append(pr)
        tt.append(tr)
    old_metric = evaluate_response(np.asarray(pp), np.asarray(tt))
    with np.load(newp, allow_pickle=False) as p, np.load(newt, allow_pickle=False) as t:
        assert np.array_equal(p["root_ids"], t["root_ids"])
        assert np.array_equal(p["words"], t["words"])
        new_roots = t["root_ids"].tolist()
        new_metric = evaluate_response(p["visual"], t["future_visual"])
    old_rows = [dict(case=cases[r["root"]], left_name=candidates[r["left"]],
                     right_name=candidates[r["right"]], **r)
                for r in old_metric["pair_rows"] if r["step"] == 5]
    new_rows = [dict(root_id=new_roots[r["root"]], **r)
                for r in new_metric["pair_rows"] if r["step"] == 5]
    assert len(old_rows) == 4 * 28 and len(new_rows) == 8 * 190
    focus = next(r for r in old_rows if r["case"] == "L1"
                 and {r["left_name"], r["right_name"]} == {"g99_after", "fact_matched_return"})
    valid = [r for r in old_rows + new_rows if not r["zero_true_contrast"]]
    xmax = max(r["true_contrast_rms"] for r in valid)
    ymin = min(0., min(r["directed_prediction_rms"] for r in valid))
    ymax = max(xmax, max(r["directed_prediction_rms"] for r in valid))
    margin = .04 * max(ymax - ymin, xmax)
    maxorth = max(np.sqrt(r["orthogonal_mse"]) for r in valid)
    norm = Normalize(0, maxorth)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, rows, title, marker in zip(axes, (old_rows, new_rows),
            ("Old FULL: all 4 cases / 8 candidates", "New NATIVE: all 8 roots / 20 candidates"), ("s", "o")):
        nonzero = [r for r in rows if not r["zero_true_contrast"]]
        scatter = ax.scatter([r["true_contrast_rms"] for r in nonzero],
                             [r["directed_prediction_rms"] for r in nonzero],
                             c=[np.sqrt(r["orthogonal_mse"]) for r in nonzero],
                             norm=norm, cmap="viridis", marker=marker,
                             s=24 if marker == "s" else 10, alpha=.6)
        ax.plot([0, xmax], [0, xmax], "k--", lw=1)
        ax.axhline(0, color="0.5", lw=.7)
        ax.set_xlim(-.025 * xmax, 1.04 * xmax)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(title)
        ax.set_xlabel("True endpoint action contrast RMS (384 features)")
        ax.set_ylabel("Prediction projected along true contrast")
        ax.grid(alpha=.2)
    x, y = focus["true_contrast_rms"], focus["directed_prediction_rms"]
    axes[0].scatter([x], [y], s=80, facecolors="none", edgecolors="red", linewidths=1.2)
    axes[0].annotate("L1: fact / g99", (x, y), xytext=(15, 22), textcoords="offset points",
                     arrowprops={"arrowstyle": "->", "color": "red"}, color="red", fontsize=9)
    fig.colorbar(scatter, ax=list(axes), shrink=.78, label="Orthogonal contrast RMS")
    fig.suptitle("Post-hoc context: old failure evidence and new action pool, identical axes\n"
                 "Unmatched histories/actions; no pooled statistics; all pairs retained")
    fig.savefig(ROOT / "scored/figures/native_old_new_scope.png", dpi=180)
    plt.close(fig)
    output = {
        "purpose": "post-hoc contextual re-expression after new pool did not support general native collapse",
        "not_matched_experiment": True, "do_not_pool_statistics": True,
        "no_change_to_frozen_comparison": True, "new_model_calls": 0, "new_training": 0,
        "old_original_visual_shape": [1, 6, 1, 384], "old_scored_slice": "[0,1:,0]",
        "old_root_frame_excluded": True, "old_cases": cases, "old_candidates": candidates,
        "input_sha256": hashes,
        "old_full": {k: v for k, v in old_metric.items() if k != "pair_rows"},
        "new_native": {k: v for k, v in new_metric.items() if k != "pair_rows"},
        "old_all_terminal_pairs": old_rows, "new_all_terminal_pairs": new_rows,
        "previously_discussed_L1_pair": focus,
    }
    with (ROOT / "scored/NATIVE_OLD_NEW_CONTEXT.json").open("x", encoding="utf8") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    assert all(sha(p) == hashes[str(p)] for p in paths)
    print(json.dumps({"old": old_metric["aggregate"], "new": new_metric["aggregate"],
                      "L1_pair": focus}, indent=2))


if __name__ == "__main__":
    main()
