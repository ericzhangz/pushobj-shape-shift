"""Pre-score optimization amendment, reusing the same model and training path.

This is an additional optimizer stage from each validation-selected checkpoint,
not a claim that an Adam run was resumed with its original optimizer state.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
import fit_models as f


def main():
    root = f.ROOT
    assert (root / "MODEL_PREDICTIONS_SEALED.json").exists()
    assert (root / "PRE_SCORE_AMENDMENT.md").exists()
    if (root / "EXTENSION_LOG.json").exists():
        raise FileExistsError("Refusing to overwrite optimization extension")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    tensor = lambda x: torch.as_tensor(x, device=device)
    train, val = [f.load_data(root / name) for name in ("train.npz", "validation.npz")]
    public = [f.load_data(root / f"test_public_h{h}.npz") for h in (2, 5)]
    assert not any(k.startswith("future_") for d in public for k in d)
    assert np.array_equal(public[0]["root_ids"], public[1]["root_ids"])
    old_log = json.loads((root / "TRAINING_LOG.json").read_text())
    runs = [r for r in old_log["runs"] if r["arm"] == "SPECTRAL"]
    began = time.perf_counter()
    for row in old_log["runs"]:
        if row["arm"] == "SPECTRAL":
            continue
        arm, rank, seed = row["arm"], row["rank"], row["seed"]
        torch.manual_seed(seed)
        state = torch.load(root / "checkpoints" / f"{row['name']}.pt", map_location="cpu", weights_only=False)
        mu, scale, pm, ps = [state[k] for k in ("mu", "scale", "pm", "ps")]
        tx, vx, qx = [tensor(f.history(d, mu, scale, pm, ps)) for d in (train, val, public[0])]
        ty, vy = [tensor(((d["future_visual"] - mu) / scale).astype(np.float32)) for d in (train, val)]
        ta = tensor(((train["history_visual"][:, -1] - mu) / scale).astype(np.float32))
        tw, vw = [tensor(d["words"].astype(np.int64)) for d in (train, val)]
        model = f.Reference(arm, rank).to(device)
        model.load_state_dict(state["state"])
        if arm == "NONLINEAR_STATE":
            model.read.requires_grad_(False)
        details = f.optimize(model, tx, ty, ta, tw, vx, vy, vw)
        name = row["name"] + "_EXTENDED"
        details.update(optimizer_steps_this_run=f.CONFIG["steps"],
            restarted_from=row["name"], optimizer_state_restored=False,
            pretrained_from=row["pretrained_from"],
            total_optimizer_steps_budget=row["optimizer_steps_including_initializer"] + f.CONFIG["steps"])
        state["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, root / "checkpoints" / f"{name}.pt")
        arrays = {"root_ids": public[0]["root_ids"], "episode_ids": public[0]["episode_ids"]}
        with torch.no_grad():
            for h, d in zip((2, 5), public):
                _, yp = model(qx, tensor(d["words"].astype(np.int64)))
                arrays[f"visual_h{h}"] = yp.cpu().numpy() * scale + mu
        assert all(np.isfinite(v).all() for k, v in arrays.items() if k.startswith("visual_"))
        np.savez_compressed(root / "predictions" / f"{name}.npz", **arrays)
        runs.append(dict(name=name, arm=arm, rank=rank, seed=seed, **details))
        f.save_json(root / "EXTENSION_LOG.json", dict(runs=runs, test_outcomes_read=False,
                    additional_seconds=time.perf_counter() - began))
        print(json.dumps({k: runs[-1][k] for k in ("name", "best_validation_mse", "best_step", "seconds")}), flush=True)
    selected = {}
    for arm in f.CONFIG["arms"]:
        rows = [r for r in runs if r["arm"] == arm]
        ranks = sorted(set(r["rank"] for r in rows))
        rank = min(ranks, key=lambda k: np.mean([r["best_validation_mse"] for r in rows if r["rank"] == k]))
        selected[arm] = dict(rank=rank, models=[r["name"] for r in rows if r["rank"] == rank])
    f.save_json(root / "FINAL_SELECTED_BY_VALIDATION.json", selected)
    f.save_json(root / "FINAL_MODEL_PREDICTIONS_SEALED.json", dict(selected=selected,
        hashes={p.name: f.sha(p) for p in sorted((root / "predictions").glob("*.npz"))},
        test_outcomes_read=False, code_hash=f.sha(__file__), base_code_hash=f.sha(f.__file__),
        amendment_sha256=f.sha(root / "PRE_SCORE_AMENDMENT.md"),
        initial_seal_sha256=f.sha(root / "MODEL_PREDICTIONS_SEALED.json")))


if __name__ == "__main__":
    main()
