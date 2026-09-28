"""Native rollout/decoder diagnosis. No training or event intervention."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = Path('D:/EV-TTT/adajepa_official_51d8665')
DATA = REPO/'artifacts/round_6_shared_revision/results'
CHECKPOINT = Path('D:/EV-TTT/pushobj_shape_shift')
CASES = [('T', 0), ('T', 1), ('L', 0), ('L', 1)]
sys.path.insert(0, str(REPO))
os.environ['TORCH_HOME'] = 'D:/EV-TTT/adajepa_runtime/torch'
from research.reframe_v3.round6_reference import _local_hub_loader, _sealed_candidates
from research.reframe_v3.shadow_selection_audit import _load_runtime, _load_anchor_observations
from research.reframe_v3.rgb_geometry_probe import load_completed
from research.contrast_probe import native_objective_breakdown


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def runtime():
    torch.manual_seed(260927)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with patch.object(torch.hub, 'load', _local_hub_loader()):
        model, prep, cfg = _load_runtime(CHECKPOINT, torch.device('cuda:0'))
    assert model.num_hist == 3 and cfg.frameskip == 5
    assert type(model.encoder).__name__ == 'SmallResNetGeM'
    assert model.decoder is not None
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    return model, prep, cfg


def to_device(obs):
    return {k: v.cuda() for k, v in obs.items()}


def detached(obs):
    return {k: v.detach().cpu() for k, v in obs.items()}


@torch.no_grad()
def reconstruct(model, prep, images, proprio):
    zs, decoded = [], []
    for start in range(0, len(images), 8):
        obs = to_device(prep.transform_obs({'visual': images[None, start:start+8],
                                           'proprio': proprio[None, start:start+8]}))
        z = model.encode_obs(obs)
        d, _ = model.decode_obs(z)
        assert torch.isfinite(d['visual']).all()
        zs.append(detached(z))
        decoded.append(d['visual'].cpu())
    return {k: torch.cat([z[k] for z in zs], 1) for k in zs[0]}, torch.cat(decoded, 1)


def predict():
    out = ROOT/'native'
    out.mkdir(exist_ok=False)
    plans, protected = {}, [CHECKPOINT/'checkpoints/model_latest.pth', CHECKPOINT/'hydra.yaml',
        REPO/'models/visual_world_model.py', REPO/'models/vqvae.py',
        REPO/'research/reframe_v3/rgb_rigid_observer.py', ROOT/'RUN_PROTOCOL.md', Path(__file__)]
    for shape, sample in CASES:
        donor = DATA/f'donor_{shape}_seed101_n12_capture'
        source = DATA/f'reference_{shape}_seed101_s{sample}_m2_eight'
        key = f'{shape}{sample}'
        plans[key] = _sealed_candidates(source, SimpleNamespace(split=shape, sample=sample, mpc=2, donor=donor))
        protected += [Path(p[2]) for p in plans[key]]
        protected += [donor/'real_evidence'/f's{sample}_executed_mpc{i}.npz' for i in (0, 1)]
        protected += [donor/'real_evidence'/f's{sample}_history_mpc2.pt',
                      donor/'real_evidence'/f'sample_{sample:03d}_initial.npz', source/'PREDICTIONS.csv']
    hashes = {str(p): digest(p) for p in protected}
    save_json(out/'INPUT_MANIFEST.json', hashes)
    model, prep, cfg = runtime()
    versions = [(v, v._version) for v in list(model.parameters())+list(model.buffers())]
    counters = {'predictor_calls': 0, 'encoder_calls': 0, 'encoder_frames': 0,
                'decoder_calls': 0, 'decoder_frames': 0}
    def pred_hook(module, inputs):
        counters['predictor_calls'] += 1
        if counters['predictor_calls'] > 320:
            raise RuntimeError('native predictor budget exceeded')
    def enc_hook(module, inputs):
        counters['encoder_calls'] += 1
        counters['encoder_frames'] += inputs[0].shape[0]
    def dec_hook(module, inputs):
        counters['decoder_calls'] += 1
        counters['decoder_frames'] += inputs[0].shape[0]*inputs[0].shape[1]
    hooks = [model.predictor.register_forward_pre_hook(pred_hook),
             model.encoder.register_forward_pre_hook(enc_hook), model.decoder.register_forward_pre_hook(dec_hook)]
    started = time.perf_counter()
    bundle = {}
    with torch.no_grad():
        for shape, sample in CASES:
            key = f'{shape}{sample}'
            donor = DATA/f'donor_{shape}_seed101_n12_capture'
            images, props, historical_actions = load_completed(donor, sample, count=2)
            assert len(images) == 11
            current_raw, goal_raw = _load_anchor_observations(donor, sample, 2)
            assert np.array_equal(current_raw['visual'][0,-1], images[-1])
            assert np.array_equal(current_raw['proprio'][0,-1], props[-1])
            source_z, source_decoded = reconstruct(model, prep, images, props)
            goal_z = model.encode_obs(to_device(prep.transform_obs(goal_raw)))
            full = to_device(prep.transform_obs({'visual': images[None, ::5], 'proprio': props[None, ::5]}))
            single = {k: v[:, -1:] for k, v in full.items()}
            history = torch.as_tensor(historical_actions.reshape(1, 2, 10)).cuda()
            case = {'source_rgb': images, 'source_proprio': props, 'source_z': source_z,
                    'source_decoded': source_decoded, 'goal_z': detached(goal_z), 'plans': {}}
            for candidate, action, path, tensor_key in plans[key]:
                entry = {'actions': action, 'tensor_path': path, 'tensor_key': tensor_key, 'arms': {}}
                for arm, obs, start in [('SINGLE', single, 0), ('FULL', full, 2)]:
                    aligned = action.cuda() if start == 0 else torch.cat([history, action.cuda()], 1)
                    before = counters['predictor_calls']
                    z, joined = model.rollout(obs_0=obs, act=aligned)
                    assert counters['predictor_calls']-before == 5
                    z = {k: v[:, start:] for k, v in z.items()}
                    assert z['visual'].shape == (1, 6, 1, 384)
                    assert all(torch.isfinite(v).all() for v in z.values())
                    decoded, _ = model.decode_obs(z)
                    assert torch.isfinite(decoded['visual']).all()
                    score = native_objective_breakdown(z, goal_z, 2, alpha=1., base=2.)
                    assert score['stage'] == 'terminal'
                    entry['arms'][arm] = {'z': detached(z), 'decoded': decoded['visual'].cpu(),
                        'cost': {k: float(score[k].item()) for k in ('visual', 'proprio', 'total')}}
                case['plans'][candidate] = entry
            bundle[key] = case
            print(f'predicted {key}: {len(case["plans"])} candidates; {counters}', flush=True)
    for hook in hooks:
        hook.remove()
    assert counters['predictor_calls'] == 320
    assert all(v._version == old for v, old in versions)
    assert all(digest(p) == h for p, h in hashes.items())
    torch.save(bundle, out/'PREDICTIONS.pt')
    save_json(out/'SEALED.json', {**counters, 'prediction_sha256': digest(out/'PREDICTIONS.pt'),
        'wall_seconds_after_load': time.perf_counter()-started, 'decoder': type(model.decoder).__name__,
        'parameters': sum(v.numel() for v in model.parameters()), 'training_steps': 0,
        'environment_calls': 0, 'future_truth_read_by_runner': False, 'input_hashes_unchanged': True,
        'parameter_buffer_versions_unchanged': True, 'tf32': False})
    print('SEALED native predictions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['predict'])
    args = parser.parse_args()
    globals()[args.phase]()
