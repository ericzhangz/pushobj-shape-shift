"""Classical reference models; training never opens test outcomes.

All models consume identical visible histories and complete length-two branches.
The spectral estimator is standard affine response realization, not a new method.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CONFIG = dict(seeds=[928, 929, 930], ranks=[4, 8], gru_dims=[8, 32],
              steps=1500, check_every=50, lr=0.001, weight_decay=0.00001,
              clip=5.0, history_hidden=64, pinv_rcond=1e-8,
              selection="validation length-two full visual MSE only",
              arms=["SPECTRAL", "DIRECT", "SPECTRAL_FT", "GRU", "NONLINEAR_STATE"],
              loss="mean(current-anchor visual MSE, future-step-1 MSE, future-step-2 MSE)",
              spectral_observer_loss="MSE to training-only whitened response coordinates",
              nonlinear_role="diagnostic: fixed spectral readout; jointly learned legal initializer and nonlinear state transition")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def load_data(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def history(data, mu, scale, prop_mean, prop_scale):
    v = (data["history_visual"] - mu) / scale
    p = (data["history_proprio"] - prop_mean) / prop_scale
    a = data["history_actions"]
    result = np.concatenate([v.reshape(len(v), -1), p.reshape(len(v), -1),
                             a.reshape(len(v), -1)], axis=1)
    assert result.shape[1] == 1184 and np.isfinite(result).all()
    return result.astype(np.float32)


def spectral(train_y, anchor_y, words, rank):
    # y_a is taken from the same (a,b) records; all repeated prefixes must agree.
    q, n, d = 5, len(anchor_y), anchor_y.shape[-1]
    lookup = {tuple(w): i for i, w in enumerate(words.tolist())}
    assert len(lookup) == q * q
    first = []
    for a in range(q):
        ys = np.stack([train_y[:, lookup[a, b], 0] for b in range(q)])
        assert np.max(np.abs(ys - ys[:1])) < 2e-5, "inconsistent common prefix"
        first.append(ys[0])
    H = np.concatenate([anchor_y] + first, axis=1).T.astype(np.float64)
    mean = H.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(H - mean, full_matrices=False)
    assert rank < n - 1 and S[rank - 1] > S[0] * 1e-8
    O = U[:, :rank] * (S[:rank] / np.sqrt(n))[None]
    X = np.sqrt(n) * Vt[:rank]
    Xa = np.concatenate([np.ones((1, n)), X])
    A, diagnostics = [], []
    for a in range(q):
        Ha = np.concatenate([first[a]] + [train_y[:, lookup[a, b], 1]
                                           for b in range(q)], axis=1).T
        B = np.linalg.pinv(O, rcond=CONFIG["pinv_rcond"]) @ (Ha - mean) @ np.linalg.pinv(Xa)
        A.append(B)
        diagnostics.append(dict(action=a, shifted_fit_mse=float(np.mean((mean + O @ B @ Xa - Ha) ** 2))))
    info = dict(rank=rank, columns=n, singular_values=S.tolist(),
                table_mse=float(np.mean((mean + O @ X - H) ** 2)),
                captured_variance=float(np.sum(S[:rank] ** 2) / np.sum(S ** 2)),
                shifted=diagnostics)
    return X.T.astype(np.float32), np.stack(A).astype(np.float32), O[:d].astype(np.float32), mean[:d, 0].astype(np.float32), info


class Reference(nn.Module):
    def __init__(self, arm, rank, dim_in=1184):
        super().__init__()
        self.arm, self.rank = arm, rank
        self.init = nn.Sequential(nn.Linear(dim_in, 64), nn.Tanh(), nn.Linear(64, rank))
        self.read = nn.Linear(rank, 384)
        if arm == "GRU":
            self.transition = nn.GRUCell(5, rank)
        elif arm == "NONLINEAR_STATE":
            self.transition = nn.Sequential(nn.Linear(rank + 5, 32), nn.Tanh(), nn.Linear(32, rank))
        else:
            self.A = nn.Parameter(torch.cat([torch.zeros(5, rank, 1),
                .98 * torch.eye(rank).repeat(5, 1, 1)], dim=2))
            with torch.no_grad():
                self.A.add_(.01 * torch.randn_like(self.A))

    def install(self, A, C, bias):
        with torch.no_grad():
            if hasattr(self, "A"):
                self.A.copy_(torch.as_tensor(A, device=self.A.device))
            self.read.weight.copy_(torch.as_tensor(C, device=self.read.weight.device))
            self.read.bias.copy_(torch.as_tensor(bias, device=self.read.bias.device))

    def forward(self, history_input, words):
        initial = self.init(history_input)
        anchor = self.read(initial)
        n, k, h = len(initial), len(words), words.shape[1]
        s = initial[:, None].expand(n, k, self.rank).reshape(n * k, self.rank)
        outputs = []
        for t in range(h):
            a = words[:, t][None].expand(n, k).reshape(-1)
            if self.arm == "GRU":
                s = self.transition(torch.nn.functional.one_hot(a, 5).float(), s)
            elif self.arm == "NONLINEAR_STATE":
                s = s + self.transition(torch.cat([s, torch.nn.functional.one_hot(a, 5).float()], dim=1))
            else:
                sx = torch.cat([torch.ones_like(s[:, :1]), s], dim=1)
                s = torch.bmm(self.A[a], sx[:, :, None])[:, :, 0]
            outputs.append(self.read(s).reshape(n, k, 384))
        return anchor, torch.stack(outputs, dim=2)


def optimize(model, tx, ty, ta, tw, vx, vy, vw, target_x=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    best, best_loss, best_step, curve = None, float("inf"), None, []
    began = time.perf_counter()
    for step in range(CONFIG["steps"] + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            if target_x is None:
                anchor, pred = model(tx, tw)
                loss = ((anchor - ta).square().mean() + 2 * (pred - ty).square().mean()) / 3
            else:
                loss = (model.init(tx) - target_x).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training: {model.arm}, {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, CONFIG["clip"])
            optimizer.step()
        if step % CONFIG["check_every"] == 0:
            with torch.no_grad():
                anchor, tp = model(tx, tw)
                _, vp = model(vx, vw)
                train_loss = float((tp - ty).square().mean())
                val_loss = float((vp - vy).square().mean())
                init_loss = None if target_x is None else float((model.init(tx) - target_x).square().mean())
            curve.append(dict(step=step, train_future_mse=train_loss, validation_future_mse=val_loss,
                              spectral_coordinate_mse=init_loss))
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best is not None
    model.load_state_dict(best)
    return dict(best_validation_mse=best_loss, best_step=best_step, curve=curve,
                seconds=time.perf_counter() - began,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in parameters))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT)
    args = parser.parse_args()
    config_path = ROOT / "FROZEN_MODEL_CONFIG.json"
    if args.freeze:
        assert not config_path.exists()
        save_json(config_path, CONFIG)
        return
    assert json.loads(config_path.read_text()) == CONFIG
    if (ROOT / "TRAINING_LOG.json").exists():
        raise FileExistsError("Refusing to overwrite an existing training run")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    train, val = load_data(args.data / "train.npz"), load_data(args.data / "validation.npz")
    # Public deployment inputs only. This file must contain no future labels.
    public = load_data(args.data / "test_public_h2.npz")
    public_h5 = load_data(args.data / "test_public_h5.npz")
    assert np.array_equal(public["root_ids"], public_h5["root_ids"])
    for key in ("history_visual", "history_proprio", "history_actions"):
        assert np.array_equal(public[key], public_h5[key])
    public["words_h2"], public["words_h5"] = public["words"], public_h5["words"]
    assert not any(k.startswith("future") for k in public)
    train_episodes, val_episodes, test_episodes = [set(d["episode_ids"].tolist()) for d in (train, val, public)]
    assert not (train_episodes & val_episodes or train_episodes & test_episodes or val_episodes & test_episodes)
    all_y = train["future_visual"].reshape(-1, 384)
    mu = all_y.mean(0)
    scale = float(np.sqrt(np.mean((all_y - mu) ** 2)))
    props = train["history_proprio"].reshape(-1, 4)
    pm, ps = props.mean(0), np.maximum(props.std(0), 1.0)
    tensor = lambda x: torch.as_tensor(x, device=device)
    tx, vx, qx = [tensor(history(d, mu, scale, pm, ps)) for d in (train, val, public)]
    ty, vy = [tensor(((d["future_visual"] - mu) / scale).astype(np.float32)) for d in (train, val)]
    ta = tensor(((train["history_visual"][:, -1] - mu) / scale).astype(np.float32))
    tw, vw = [tensor(d["words"].astype(np.int64)) for d in (train, val)]
    outputs = ROOT / "predictions"
    weights = ROOT / "checkpoints"
    outputs.mkdir(exist_ok=True)
    weights.mkdir(exist_ok=True)
    save_json(ROOT / "TRAINING_INPUT_HASHES.json", {p.name: sha(p) for p in
               [args.data / "train.npz", args.data / "validation.npz", args.data / "test_public_h2.npz",
                args.data / "test_public_h5.npz", config_path, Path(__file__)]})
    result = dict(normalization_scale=scale, episodes=dict(train=sorted(train_episodes),
                  validation=sorted(val_episodes), test=sorted(test_episodes)), runs=[])
    specs = {}
    for rank in CONFIG["ranks"]:
        specs[rank] = spectral(ty.cpu().numpy(), ta.cpu().numpy(), train["words"], rank)
    save_json(ROOT / "SPECTRAL_TABLE_DIAGNOSTICS.json", {r: s[-1] for r, s in specs.items()})
    models = {}
    # Baselines are independent reference arms, not modules combined in a proposal.
    for arm in CONFIG["arms"]:
        for rank in CONFIG["gru_dims"] if arm == "GRU" else CONFIG["ranks"]:
            for seed in CONFIG["seeds"]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = Reference(arm, rank).to(device)
                target_x = None
                if arm in ("SPECTRAL", "SPECTRAL_FT", "NONLINEAR_STATE"):
                    X, A, C, bias, _ = specs[rank]
                    model.install(A, C, bias)
                    if arm == "SPECTRAL":
                        model.A.requires_grad_(False)
                        model.read.requires_grad_(False)
                        target_x = tensor(X)
                    elif arm == "SPECTRAL_FT":
                        model.load_state_dict(models["SPECTRAL", rank, seed])
                    else:
                        spectral_state = models["SPECTRAL", rank, seed]
                        model.init.load_state_dict({k[5:]: v for k, v in spectral_state.items() if k.startswith("init.")})
                        model.read.requires_grad_(False)
                details = optimize(model, tx, ty, ta, tw, vx, vy, vw, target_x)
                details["optimizer_steps_this_run"] = CONFIG["steps"]
                details["pretrained_from"] = (f"SPECTRAL_r{rank}_s{seed}"
                    if arm in ("SPECTRAL_FT", "NONLINEAR_STATE") else None)
                details["optimizer_steps_including_initializer"] = CONFIG["steps"] * (2 if details["pretrained_from"] else 1)
                name = f"{arm}_r{rank}_s{seed}"
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                models[arm, rank, seed] = state
                torch.save(dict(state=state, arm=arm, rank=rank, mu=mu, scale=scale, pm=pm, ps=ps), weights / f"{name}.pt")
                # All ranks/seeds sealed before outcome access; selection is val-only below.
                arrays = {"root_ids": public["root_ids"], "episode_ids": public["episode_ids"]}
                with torch.no_grad():
                    for horizon in (2, 5):
                        qw = tensor(public[f"words_h{horizon}"].astype(np.int64))
                        _, yp = model(qx, qw)
                        arrays[f"visual_h{horizon}"] = yp.cpu().numpy() * scale + mu
                assert all(np.isfinite(v).all() for k, v in arrays.items() if k.startswith("visual"))
                np.savez_compressed(outputs / f"{name}.npz", **arrays)
                result["runs"].append(dict(name=name, arm=arm, rank=rank, seed=seed, **details))
                save_json(ROOT / "TRAINING_LOG.json", result)
                print(json.dumps({k: result["runs"][-1][k] for k in ("name", "best_validation_mse", "best_step", "seconds")}), flush=True)
    selected = {}
    for arm in CONFIG["arms"]:
        rows = [r for r in result["runs"] if r["arm"] == arm]
        ranks = sorted(set(r["rank"] for r in rows))
        rank = min(ranks, key=lambda k: np.mean([r["best_validation_mse"] for r in rows if r["rank"] == k]))
        selected[arm] = dict(rank=rank, models=[r["name"] for r in rows if r["rank"] == rank])
    save_json(ROOT / "SELECTED_BY_VALIDATION.json", selected)
    save_json(ROOT / "MODEL_PREDICTIONS_SEALED.json", dict(selected=selected,
              hashes={p.name: sha(p) for p in sorted(outputs.glob("*.npz"))},
              test_outcomes_read=False, code_hash=sha(__file__)))


if __name__ == "__main__":
    main()
