"""Frozen native reference; only public history, actions, and identifiers are read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FORMATION = ROOT.parent / "shared_response_formation_20260928"
sys.path.insert(0, str(FORMATION))
import formation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def run(data_dir):
    out = ROOT / "predictions"
    out.mkdir(exist_ok=True)
    targets = [out / f"NATIVE_h{h}.npz" for h in (2, 5)]
    seal_path = out / "NATIVE_SEALED.json"
    if seal_path.exists() or any(path.exists() for path in targets):
        raise FileExistsError("native predictions/seal already exist")
    started = time.perf_counter()
    model, _ = formation.runtime()
    counts = {"predictor_forward_calls": 0, "predicted_sample_transitions": 0}

    def count(module, inputs):
        counts["predictor_forward_calls"] += 1
        counts["predicted_sample_transitions"] += int(inputs[0].shape[0])

    hook = model.predictor.register_forward_pre_hook(count)
    input_hashes, output_hashes, prefix_checks = {}, {}, {}
    with torch.no_grad():
        for horizon, target in zip((2, 5), targets):
            source = data_dir / f"test_public_h{horizon}.npz"
            input_hashes[str(source)] = sha(source)
            with np.load(source, allow_pickle=False) as archive:
                if any(key.startswith("future_") for key in archive.files):
                    raise ValueError("public input archive contains future labels")
                keys = ("history_visual", "history_proprio_latent", "history_actions",
                        "words", "macro_actions", "root_ids", "episode_ids")
                data = {key: archive[key].copy() for key in keys}
            words = data["words"]
            n, candidates = len(data["root_ids"]), len(words)
            if words.shape != (candidates, horizon):
                raise ValueError("word horizon mismatch")
            if data["history_visual"].shape != (n, 3, 384):
                raise ValueError("three-frame visual history required")
            if data["history_proprio_latent"].shape != (n, 3, 10):
                raise ValueError("checkpoint proprio embeddings required")
            if data["history_actions"].shape != (n, 2, 10):
                raise ValueError("two aligned historical action blocks required")
            visual = np.empty((n, candidates, horizon, 384), np.float32)
            proprio = np.empty((n, candidates, horizon, 10), np.float32)
            future_actions = torch.as_tensor(data["macro_actions"][words], device="cuda", dtype=torch.float32)
            max_prefix_error = 0.0
            for r in range(n):
                initial = {
                    "visual": torch.as_tensor(data["history_visual"][r], device="cuda", dtype=torch.float32)
                        [None, :, None, :].expand(candidates, -1, -1, -1),
                    "proprio": torch.as_tensor(data["history_proprio_latent"][r], device="cuda", dtype=torch.float32)
                        [None].expand(candidates, -1, -1),
                }
                history = torch.as_tensor(data["history_actions"][r], device="cuda", dtype=torch.float32)
                actions = torch.cat((history[None].expand(candidates, -1, -1), future_actions), dim=1)
                prediction = formation.cached_rollout(model, initial, actions)
                visual[r] = prediction["visual"][:, 3:, 0].cpu().numpy()
                proprio[r] = prediction["proprio"][:, 3:].cpu().numpy()
                for depth in range(1, horizon):
                    for i in range(candidates):
                        same = np.all(words == words[i], axis=1) if depth == horizon else np.all(words[:, :depth] == words[i, :depth], axis=1)
                        delta = np.max(np.abs(visual[r, same, :depth] - visual[r, i, :depth]))
                        max_prefix_error = max(max_prefix_error, float(delta))
                print(f"native H{horizon} {data['root_ids'][r]}", flush=True)
            if max_prefix_error > 2e-5:
                raise AssertionError(f"shared-prefix mismatch {max_prefix_error}")
            if not (np.isfinite(visual).all() and np.isfinite(proprio).all()):
                raise FloatingPointError("native predictions contain nonfinite values")
            np.savez_compressed(target, visual=visual, proprio_latent=proprio,
                                root_ids=data["root_ids"], episode_ids=data["episode_ids"], words=words)
            output_hashes[str(target)] = sha(target)
            prefix_checks[f"h{horizon}"] = max_prefix_error
    hook.remove()
    manifest = {"status": "SEALED", "truth_read": False, "goal_read": False,
                "training_steps": 0, "environment_calls": 0,
                "inputs": input_hashes, "outputs": output_hashes,
                "common_prefix_max_abs": prefix_checks, "counts": counts,
                "code": {str(Path(__file__)): sha(__file__),
                         str(FORMATION / "formation.py"): sha(FORMATION / "formation.py")},
                "elapsed_seconds": time.perf_counter() - started}
    with seal_path.open("x", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    run(parser.parse_args().data_dir)
