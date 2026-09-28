"""Frozen crossed-response collection through the existing full-prefix replay oracle.

Only visible RGB/proprio/action arrays enter the public NPZ datasets. Simulator
state is read solely inside the exact replay gate and is never exported there.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'artifacts/round_6_shared_revision/results'
CHECKPOINT = Path(os.environ['PUSHOBJ_CHECKPOINT_DIR'])
sys.path.insert(0, str(REPO))
from research.replay_oracle import PushObjReplayOracle
from research.reframe_v3.round6_reference import _local_hub_loader
from research.reframe_v3.shadow_selection_audit import _load_runtime
from research.reframe_v3.rgb_geometry_probe import load_completed


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def freeze():
    path = OUT / 'FROZEN_DATA_CONFIG.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    splits = {'train': list(range(2, 8)), 'validation': [8, 9], 'test': [10, 11]}
    anchors, missing = [], []
    for split, samples in splits.items():
        for shape in ('T', 'L'):
            donor = DATA / f'donor_{shape}_seed101_n12_capture'
            metadata = json.loads((donor / 'run_metadata.json').read_text())
            seeds = {int(r['sample_id']): int(r['seed']) for r in
                     map(json.loads, (donor / 'oracle/query_budget.jsonl').read_text().splitlines())}
            for sample in samples:
                for mpc in ([2, 3, 4] if split == 'train' else [2, 4]):
                    history = donor / 'real_evidence' / f's{sample}_history_mpc{mpc}.pt'
                    row = {'root_id': f'{shape}{sample}_m{mpc}',
                           'episode_id': f'{shape}{sample}', 'shape': shape,
                           'sample': sample, 'mpc': mpc, 'split': split,
                           'seed': seeds[sample], 'donor': str(donor)}
                    if not history.exists():
                        missing.append(row)
                        continue
                    assert metadata['shapes'][sample] == shape
                    row['source_hashes'] = {str(p): sha(p) for p in [history,
                        donor / 'real_evidence' / f'sample_{sample:03d}_initial.npz']}
                    anchors.append(row)
    q = 5
    words5 = [[a] * 5 for a in range(q)]
    words5 += [[(a + k) % q for k in range(5)] for a in range(q)]
    words5 += [[(a - k) % q for k in range(5)] for a in range(q)]
    words5 += [[a, (a + 1) % q, a, (a + 1) % q, a] for a in range(q)]
    assert len(set(map(tuple, words5))) == 20
    config = {'schema': 'crossed_visible_response_v1', 'seed': 260928,
              'frameskip': 5, 'shapes': ['T', 'L'], 'episode_splits': splits,
              'shape_label_given_to_model': False, 'anchors': anchors,
              'missing_histories_excluded_before_collection': missing,
              'action_vectors_env': [[0., 0.], [.12, 0.], [-.12, 0.], [0., .12], [0., -.12]],
              'words_h2': list(map(list, itertools.product(range(q), repeat=2))),
              'words_h5': words5, 'test_h3': 'first three observations of test_h5',
              'validation_selection_horizon': 2,
              'test_usage': 'Do not use test futures for selection or model formation.',
              'checkpoint_dir': str(CHECKPOINT),
              'source_hashes': {str(p): sha(p) for p in [Path(__file__),
                  REPO / 'research/replay_oracle.py', REPO / 'preprocessor.py',
                  CHECKPOINT / 'checkpoints/model_latest.pth', CHECKPOINT / 'hydra.yaml']}}
    dump(path, config)
    return config


class Collector:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(4)
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with patch.object(torch.hub, 'load', _local_hub_loader()):
            self.model, self.prep, cfg = _load_runtime(CHECKPOINT, torch.device('cuda:0'))
        assert cfg.frameskip == 5 and self.model.num_hist == 3
        assert not self.model.training and all(not p.requires_grad for p in self.model.parameters())
        micro = torch.tensor(config['action_vectors_env'], dtype=torch.float32)[:, None, :].repeat(1, 5, 1)
        self.macro = self.prep.normalize_actions(micro).reshape(5, 10).contiguous()
        reconstructed = self.prep.denormalize_actions(self.macro.reshape(5, 5, 2))
        assert torch.allclose(reconstructed, micro, atol=1e-8, rtol=0)
        self.counts = {'environment_rollouts': 0, 'environment_micro_steps': 0,
                       'encoder_calls': 0, 'encoded_frames': 0, 'predictor_calls': 0,
                       'training_calls': 0}

    @torch.no_grad()
    def encode(self, images, props):
        vis, pro = [], []
        for start in range(0, len(images), 32):
            obs = self.prep.transform_obs({'visual': images[None, start:start+32],
                                          'proprio': props[None, start:start+32]})
            z = self.model.encode_obs({k: v.cuda() for k, v in obs.items()})
            vis.append(z['visual'][0, :, 0].cpu().numpy())
            pro.append(z['proprio'][0].cpu().numpy())
            self.counts['encoder_calls'] += 1
            self.counts['encoded_frames'] += len(vis[-1])
        return np.concatenate(vis), np.concatenate(pro)

    def assets(self, row):
        donor, sample, mpc = Path(row['donor']), row['sample'], row['mpc']
        for path, expected in row['source_hashes'].items():
            if sha(path) != expected:
                raise RuntimeError(f'Changed frozen input: {path}')
        history = torch.load(donor / 'real_evidence' / f's{sample}_history_mpc{mpc}.pt',
                             map_location='cpu')
        prefix = history['executed_prefix']
        if tuple(prefix.shape) != (1, mpc, 10):
            raise ValueError('Factual prefix does not match frozen anchor')
        images, props, actions = load_completed(donor, sample, count=mpc)
        if not np.array_equal(actions.reshape(1, mpc, 10), prefix.numpy()):
            raise ValueError('Factual action sources disagree')
        with np.load(donor / 'real_evidence' / f'sample_{sample:03d}_initial.npz', allow_pickle=False) as a:
            initial, goal = a['initial_state'].copy(), a['goal_state'].copy()
            goal_rgb, goal_prop = a['goal_visual'].copy(), a['goal_proprio'].copy()
        oracle = PushObjReplayOracle(initial, goal, row['seed'], {'shape': row['shape']},
                 {'with_velocity': True, 'with_target': True}, self.prep, 5, 1., 2.,
                 OUT / 'controller')
        return history, prefix, images, props, goal_rgb, goal_prop, oracle

    def replay(self, oracle, actions):
        env, obs, states, elapsed = oracle._replay(actions)
        try:
            self.counts['environment_rollouts'] += 1
            self.counts['environment_micro_steps'] += actions.shape[1] * 5
            return obs, states, elapsed
        finally:
            env.close()

    def anchor(self, row, words, dataset_name):
        history, prefix, images, props, goal_rgb, goal_prop, oracle = self.assets(row)
        obs, states, gate_s = self.replay(oracle, prefix)
        current = history['current_real_observation']
        passed = all(np.array_equal(obs[k][-1], np.asarray(current[k])[0, -1])
                     for k in ('visual', 'proprio'))
        passed &= np.array_equal(states[-1], np.asarray(history['current_public_state']).reshape(-1, 7)[0])
        if not passed:
            raise RuntimeError(f'Exact factual replay failed: {row["root_id"]}')
        branch_rgb, branch_prop, branch_secs = [], [], []
        seen = {}
        for word in words:
            candidate = self.macro[word][None]
            all_obs, _, elapsed = self.replay(oracle, torch.cat([prefix, candidate], dim=1))
            rgb = all_obs['visual'][row['mpc'] * 5::5]
            prop = all_obs['proprio'][row['mpc'] * 5::5]
            assert len(rgb) == len(word) + 1
            if not np.array_equal(rgb[0], images[-1]) or not np.array_equal(prop[0], props[-1]):
                raise RuntimeError('Branch anchor differs from factual history')
            for step in range(1, len(word)+1):
                key = tuple(word[:step])
                if key in seen:
                    if not np.array_equal(seen[key][0], rgb[step]) or not np.array_equal(seen[key][1], prop[step]):
                        raise RuntimeError('Repeated action prefix has inconsistent response')
                else:
                    seen[key] = (rgb[step], prop[step])
            branch_rgb.append(rgb)
            branch_prop.append(prop)
            branch_secs.append(elapsed)
        # Repeat a complete branch; exact RGB and proprio check, no hidden state exported.
        candidate = self.macro[words[0]][None]
        repeated, _, repeated_s = self.replay(oracle, torch.cat([prefix, candidate], dim=1))
        repeated_rgb = repeated['visual'][row['mpc'] * 5::5]
        repeated_prop = repeated['proprio'][row['mpc'] * 5::5]
        if not np.array_equal(branch_rgb[0], repeated_rgb) or not np.array_equal(branch_prop[0], repeated_prop):
            raise RuntimeError('Repeated complete response is not exact')
        branch_rgb, branch_prop = np.stack(branch_rgb), np.stack(branch_prop)
        indices = np.array([row['mpc'] - 2, row['mpc'] - 1, row['mpc']]) * 5
        hv, hp = self.encode(images[indices], props[indices])
        # Goal archive follows [batch,time,...]; normalize to a single frame.
        goal_rgb = goal_rgb.reshape(-1, *goal_rgb.shape[-3:])[0]
        goal_prop = goal_prop.reshape(-1, 4)[0]
        gv, gp = self.encode(goal_rgb[None], goal_prop[None])
        K, H = len(words), len(words[0])
        fv, fp = self.encode(branch_rgb[:, 1:].reshape(-1, 224, 224, 3),
                            branch_prop[:, 1:].reshape(-1, 4))
        rawdir = OUT / 'raw_observations' / dataset_name
        rawdir.mkdir(parents=True, exist_ok=True)
        rawpath = rawdir / f'{row["root_id"]}.npz'
        np.savez_compressed(rawpath, history_visual=images[indices], history_proprio=props[indices],
                            future_visual=branch_rgb[:, 1:], future_proprio=branch_prop[:, 1:],
                            words=np.asarray(words, dtype=np.int64))
        gate = {'root_id': row['root_id'], 'dataset': dataset_name, 'anchor_exact': bool(passed),
                'complete_repeat_exact': True, 'shared_word_prefix_exact': True,
                'gate_wall_s': gate_s, 'branch_wall_s_sum': sum(branch_secs),
                'branch_wall_s_mean': float(np.mean(branch_secs)), 'repeat_wall_s': repeated_s,
                'raw_observation_sha256': sha(rawpath),
                'rollouts': oracle.rollout_count, 'micro_steps': oracle.environment_steps}
        public = {'history_visual': hv, 'history_proprio': props[indices],
                  'history_proprio_latent': hp, 'history_actions': prefix[0, -2:].numpy(),
                  'future_visual': fv.reshape(K, H, 384), 'future_proprio': branch_prop[:, 1:],
                  'future_proprio_latent': fp.reshape(K, H, -1),
                  'goal_visual': gv[0], 'goal_proprio': goal_prop, 'goal_proprio_latent': gp[0],
                  'root_ids': row['root_id'], 'episode_ids': row['episode_id']}
        return public, gate

    def collect(self, name):
        target = OUT / f'{name}.npz'
        if target.exists():
            raise FileExistsError(f'Refusing overwrite: {target}')
        split = 'test' if name.startswith('test_') else name
        words = self.config['words_h5' if name == 'test_h5' else 'words_h2']
        rows = [row for row in self.config['anchors'] if row['split'] == split]
        values, gates = [], []
        start = time.perf_counter()
        for n, row in enumerate(rows):
            public, gate = self.anchor(row, words, name)
            values.append(public)
            gates.append(gate)
            dump(OUT / f'{name}_PROGRESS.json', {'completed': n+1, 'planned': len(rows),
                                               'last_root': row['root_id'], 'counts': self.counts})
            print(name, n+1, '/', len(rows), row['root_id'], 'branch_s', gate['branch_wall_s_mean'], flush=True)
        result = {key: np.stack([v[key] for v in values]) for key in values[0]}
        result.update(words=np.asarray(words, dtype=np.int64), macro_actions=self.macro.numpy(),
                      action_vectors_env=np.asarray(self.config['action_vectors_env'], dtype=np.float32),
                      proprio_mean=self.prep.proprio_mean.numpy(), proprio_std=self.prep.proprio_std.numpy())
        np.savez_compressed(target, **result)
        if name.startswith('test_'):
            public_name = name.replace('test_', 'test_public_')
            np.savez_compressed(OUT / f'{public_name}.npz',
                                **{k: v for k, v in result.items() if not k.startswith('future_')})
        dump(OUT / f'{name}_COLLECTION.json', {'dataset_sha256': sha(target),
             'config_sha256': sha(OUT / 'FROZEN_DATA_CONFIG.json'), 'gates': gates,
             'dataset_shapes': {k: list(v.shape) for k, v in result.items()},
             'wall_s': time.perf_counter()-start, 'cumulative_counts': self.counts,
             'hidden_state_exported': False, 'native_predictor_used': False})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=['train', 'validation', 'test_h2', 'test_h5'],
                        default=['train', 'validation', 'test_h2', 'test_h5'])
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    config = freeze()
    if args.freeze_only:
        print('Frozen', {s: sum(r['split'] == s for r in config['anchors'])
                         for s in config['episode_splits']}, flush=True)
        return
    collector = Collector(config)
    started = time.perf_counter()
    for name in args.datasets:
        collector.collect(name)
    dump(OUT / 'COLLECTION_COMPLETE.json', {'datasets': args.datasets,
         'counts': collector.counts, 'wall_s': time.perf_counter()-started,
         'status': 'COMPLETE', 'code_sha256': sha(Path(__file__))})


if __name__ == '__main__':
    main()
