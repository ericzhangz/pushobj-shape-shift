"""Independent full/contrast recomputation using centered errors, not pair loops."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    selected = json.loads((ROOT / "FINAL_SELECTED_BY_VALIDATION.json").read_text())
    expected = json.loads((ROOT / "scored/METRICS.json").read_text())["summary"]
    rows = []
    for pool, horizons in ((2, (1, 2)), (5, (3, 5))):
        with np.load(ROOT / f"test_h{pool}.npz", allow_pickle=False) as f:
            truth = f["future_visual"].astype(np.float64)
            episodes = f["episode_ids"]
        groups = dict(selected)
        groups["NATIVE"] = {"models": [f"NATIVE_h{pool}"]}
        for arm, spec in groups.items():
            measurements = {h: [] for h in horizons}
            for name in spec["models"]:
                with np.load(ROOT / "predictions" / f"{name}.npz", allow_pickle=False) as f:
                    pred = f["visual" if arm == "NATIVE" else f"visual_h{pool}"].astype(np.float64)
                error = pred - truth
                centered = error - error.mean(axis=1, keepdims=True)
                k = truth.shape[1]
                full = (error ** 2).mean(axis=(1, 3))
                contrast = 2 * k / (k - 1) * (centered ** 2).mean(axis=(1, 3))
                for h in horizons:
                    ep = []
                    for eid in np.unique(episodes):
                        mask = episodes == eid
                        ep.append([full[mask, :h].mean(), contrast[mask, :h].mean()])
                    measurements[h].append(np.mean(ep, axis=0))
            for h in horizons:
                values = np.mean(measurements[h], axis=0)
                for key, value in zip(("prefix_full_mse", "prefix_contrast_mse"), values):
                    ref = expected[arm][str(h)]["metrics"][key]["mean"]
                    assert abs(value - ref) < 1e-12
                    rows.append(dict(arm=arm, horizon=h, metric=key, recomputed=float(value),
                                     difference=float(value - ref)))
    (ROOT / "INDEPENDENT_RESULT_RECOMPUTATION.json").write_text(json.dumps(
        dict(status="PASS", checks=len(rows), rows=rows), indent=2), encoding="utf8")
    print("PASS", len(rows), "independent aggregate checks")


if __name__ == "__main__":
    main()
