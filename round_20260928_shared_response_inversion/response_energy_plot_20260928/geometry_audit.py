"""Secondary RGB centroid audit; never used to select or train a world model."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from collect_responses import OUT, Collector, freeze, dump, sha
from research.reframe_v3.rgb_rigid_observer import observe_rgb


class Readout:
    def __init__(self):
        self.model = Collector(freeze()).model
        self.cache = {}
        self.decoder_calls = 0
        self.decoder_frames = 0

    def image(self, rgb):
        key = hashlib.sha256(rgb.tobytes()).digest()
        if key not in self.cache:
            try:
                obs = observe_rgb(rgb)
                self.cache[key] = (True, obs['object_centroid'], '')
            except ValueError as exc:
                self.cache[key] = (False, np.array([np.nan, np.nan]), str(exc))
        return self.cache[key]

    def raw(self, images):
        values = [self.image(rgb) for rgb in images.reshape(-1, 224, 224, 3)]
        return np.array([x[0] for x in values]), np.stack([x[1] for x in values]), [x[2] for x in values]

    @torch.no_grad()
    def predicted(self, codes):
        codes = codes.reshape(-1, 384)
        valids, centers, errors, out_of_range = [], [], [], []
        for start in range(0, len(codes), 16):
            v = torch.as_tensor(codes[start:start+16], device='cuda', dtype=torch.float32)[None, :, None]
            image, _ = self.model.decoder(v)
            image = image.cpu().numpy().transpose(0, 2, 3, 1)
            if not np.isfinite(image).all():
                raise FloatingPointError('Nonfinite decoder output')
            out_of_range.extend(np.mean(np.abs(image) > 1, axis=(1, 2, 3)).tolist())
            rgb = np.clip(np.rint((image + 1) * 127.5), 0, 255).astype(np.uint8)
            valid, center, error = self.raw(rgb)
            valids.extend(valid.tolist()); centers.extend(center); errors.extend(error)
            self.decoder_calls += 1
            self.decoder_frames += len(image)
        return np.asarray(valids), np.stack(centers), errors, np.asarray(out_of_range)


def summary(valid, errors):
    return {'valid': int(valid.sum()), 'total': len(valid), 'valid_fraction': float(valid.mean()),
            'failure_counts': dict(Counter(e for e in errors if e))}


def error_stats(a, b, mask):
    if not np.any(mask):
        return {'n': 0, 'mae_px': None, 'rmse_px': None, 'max_px': None}
    distance = np.linalg.norm(a[mask] - b[mask], axis=-1)
    return {'n': int(mask.sum()), 'mae_px': float(distance.mean()),
            'rmse_px': float(np.sqrt(np.mean(distance ** 2))), 'max_px': float(distance.max())}


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def train_calibration():
    destination = OUT / 'GEOMETRY_TRAIN_CALIBRATION.json'
    if destination.exists():
        raise FileExistsError(destination)
    started = time.perf_counter()
    reader = Readout()
    with np.load(OUT / 'train.npz', allow_pickle=False) as a:
        roots, episodes, codes = a['root_ids'], a['episode_ids'], a['future_visual']
    real_valid, rec_valid, real_centers, rec_centers, real_errors, rec_errors = [], [], [], [], [], []
    range_fractions, rows = [], []
    for i, root in enumerate(roots):
        with np.load(OUT / 'raw_observations/train' / f'{root}.npz', allow_pickle=False) as a:
            rgb = a['future_visual']
        rv, rc, re = reader.raw(rgb)
        pv, pc, pe, oor = reader.predicted(codes[i])
        real_valid.extend(rv); rec_valid.extend(pv)
        real_centers.extend(rc); rec_centers.extend(pc)
        real_errors.extend(re); rec_errors.extend(pe); range_fractions.extend(oor)
        for j in range(len(rv)):
            rows.append({'root_id': str(root), 'episode_id': str(episodes[i]),
                         'candidate': j // 2, 'step': j % 2 + 1,
                         'raw_valid': bool(rv[j]), 'reconstruction_valid': bool(pv[j]),
                         'raw_error': re[j], 'reconstruction_error': pe[j],
                         'centroid_error_px': float(np.linalg.norm(rc[j]-pc[j])) if rv[j] and pv[j] else '',
                         'decoder_out_of_range_fraction': float(oor[j])})
        print('geometry train', i+1, '/', len(roots), root, flush=True)
    rv, pv = np.asarray(real_valid), np.asarray(rec_valid)
    rc, pc = np.stack(real_centers), np.stack(rec_centers)
    result = {'scope': 'Training responses only; secondary readout calibration, no parameter selection.',
              'raw': summary(rv, real_errors), 'reconstruction': summary(pv, rec_errors),
              'centroid_error_only_valid': error_stats(rc, pc, rv & pv),
              'decoder_out_of_range_fraction_mean': float(np.mean(range_fractions)),
              'decoder_calls': reader.decoder_calls, 'decoder_frames': reader.decoder_frames,
              'geometry_unique_rgb_evaluations': len(reader.cache), 'test_files_read': False,
              'model_training_calls': 0, 'environment_calls': 0,
              'interpretation': 'Palette-derived visible silhouette centroid, not rigid-body ground truth or contact state. Readability does not certify an off-manifold prediction.',
              'wall_s': time.perf_counter()-started, 'code_sha256': sha(Path(__file__))}
    write_rows(OUT / 'GEOMETRY_TRAIN_CALIBRATION.csv', rows)
    dump(destination, result)


def test_audit():
    destination = OUT / 'GEOMETRY_TEST_SUMMARY.json'
    if destination.exists():
        raise FileExistsError(destination)
    final = json.loads((OUT / 'FINAL_MODEL_PREDICTIONS_SEALED.json').read_text())
    native = json.loads((OUT / 'predictions/NATIVE_SEALED.json').read_text())
    if final['test_outcomes_read'] or native['truth_read']:
        raise ValueError('Prediction process violated truth isolation')
    # Verify selected model files and native predictions before any truth access.
    selected = final['selected']
    selected_names = [name for spec in selected.values() for name in spec['models']]
    for name in selected_names + ['NATIVE_h5']:
        if sha(OUT / 'predictions' / f'{name}.npz') != final['hashes'][f'{name}.npz']:
            raise ValueError(f'Changed sealed predictions: {name}')
    for path, digest in native['outputs'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed native predictions: {path}')
    started = time.perf_counter()
    reader = Readout()
    with np.load(OUT / 'test_h5.npz', allow_pickle=False) as a:
        roots, episodes, codes = a['root_ids'], a['episode_ids'], a['future_visual']
    n, k, horizon = codes.shape[:3]
    real_v, real_c, real_e = [], [], []
    for root in roots:
        with np.load(OUT / 'raw_observations/test_h5' / f'{root}.npz', allow_pickle=False) as a:
            v, c, e = reader.raw(a['future_visual'])
        real_v.extend(v); real_c.extend(c); real_e.extend(e)
    real_v, real_c = np.asarray(real_v), np.stack(real_c)
    rv, rc, re, oor = reader.predicted(codes)
    readouts = {'RECON_REAL': (rv, rc, re)}
    ranges = {'RECON_REAL': float(oor.mean())}
    for name in ['NATIVE'] + selected_names:
        filename, key = ('NATIVE_h5.npz', 'visual') if name == 'NATIVE' else (name + '.npz', 'visual_h5')
        with np.load(OUT / 'predictions' / filename, allow_pickle=False) as a:
            if not np.array_equal(roots, a['root_ids']):
                raise ValueError('Prediction root order mismatch')
            prediction = a[key]
        if prediction.shape != codes.shape:
            raise ValueError('Prediction shape mismatch')
        valid, center, errors, oor = reader.predicted(prediction)
        readouts[name] = (valid, center, errors)
        ranges[name] = float(oor.mean())
        print('geometry test', name, 'valid', int(valid.sum()), '/', len(valid), flush=True)
    common = real_v.copy()
    for valid, _, _ in readouts.values():
        common &= valid
    # One common support across every selected seed, every arm, native and reconstruction.
    point_episode = np.repeat(episodes, k * horizon)
    point_root = np.repeat(roots, k * horizon)
    endpoint = np.arange(len(real_v)) % horizon == horizon - 1
    arm_for = {name: arm for arm, spec in selected.items() for name in spec['models']}
    arm_for.update(NATIVE='NATIVE', RECON_REAL='RECON_REAL')
    metrics, details = [], []
    for name, (valid, center, errors) in readouts.items():
        for episode in sorted(set(episodes.tolist())):
            mask_ep = point_episode == episode
            for window, time_mask in [('all_five_steps', np.ones(len(valid), bool)), ('endpoint', endpoint)]:
                own = mask_ep & time_mask & real_v & valid
                shared = mask_ep & time_mask & common
                metrics.append({'model': name, 'arm': arm_for[name], 'episode': episode,
                                'window': window, 'possible_points': int((mask_ep & time_mask).sum()),
                                'own_valid_n': int(own.sum()), 'common_valid_n': int(shared.sum()),
                                'own_valid_mae_px': error_stats(real_c, center, own)['mae_px'],
                                'common_valid_mae_px': error_stats(real_c, center, shared)['mae_px']})
        for j in range(len(valid)):
            details.append({'model': name, 'root_id': str(point_root[j]),
                            'episode_id': str(point_episode[j]), 'candidate': (j // horizon) % k,
                            'step': j % horizon + 1, 'raw_valid': bool(real_v[j]),
                            'predicted_valid': bool(valid[j]), 'common_valid': bool(common[j]),
                            'raw_error': real_e[j], 'predicted_error': errors[j],
                            'centroid_error_px': float(np.linalg.norm(center[j]-real_c[j]))
                                if valid[j] and real_v[j] else ''})
    models = {}
    for name, (valid, center, errors) in readouts.items():
        windows = {}
        for window in ('all_five_steps', 'endpoint'):
            rows = [r for r in metrics if r['model'] == name and r['window'] == window]
            own = [r['own_valid_mae_px'] for r in rows if r['own_valid_mae_px'] is not None]
            shared = [r['common_valid_mae_px'] for r in rows if r['common_valid_mae_px'] is not None]
            windows[window] = {'episode_macro_own_valid_mae_px': float(np.mean(own)) if own else None,
                              'episode_macro_common_valid_mae_px': float(np.mean(shared)) if shared else None,
                              'episodes_with_own_support': len(own), 'episodes_with_common_support': len(shared)}
        models[name] = {'arm': arm_for[name], **summary(valid, errors), 'windows': windows,
                        'decoder_out_of_range_fraction_mean': ranges[name]}
    dump(destination, {'scope': 'H5 complete pool; all validation-selected seeds; secondary diagnostic.',
        'raw': summary(real_v, real_e), 'models': models, 'common_valid': int(common.sum()),
        'total': len(common), 'common_endpoint_valid': int((common & endpoint).sum()),
        'endpoint_total': int(endpoint.sum()), 'decoder_calls': reader.decoder_calls,
        'decoder_frames': reader.decoder_frames, 'unique_rgb_readouts': len(reader.cache),
        'training_calls': 0, 'environment_calls': 0, 'wall_s': time.perf_counter()-started,
        'interpretation': 'Errors only on explicitly reported readable support. Common mask includes every selected seed, native, real reconstruction and raw RGB. Visible silhouette centroids are not rigid-body state or proof of contact dynamics.',
        'final_seal_sha256': sha(OUT / 'FINAL_MODEL_PREDICTIONS_SEALED.json'),
        'native_seal_sha256': sha(OUT / 'predictions/NATIVE_SEALED.json'), 'code_sha256': sha(Path(__file__))})
    write_rows(OUT / 'GEOMETRY_TEST_EPISODES.csv', metrics)
    write_rows(OUT / 'GEOMETRY_TEST_POINTS.csv', details)
    np.savez_compressed(OUT / 'GEOMETRY_FOR_PLOTTING.npz', root_ids=roots, episode_ids=episodes,
                        true_centroid=real_c.reshape(n, k, horizon, 2),
                        raw_valid=real_v.reshape(n, k, horizon), common_valid=common.reshape(n, k, horizon))


def raw_motion():
    # Uses completed readout values, no new decoder calls or model decisions.
    if not (OUT / 'FINAL_MODEL_PREDICTIONS_SEALED.json').exists():
        raise FileNotFoundError('Final prediction seal required')
    with np.load(OUT / 'GEOMETRY_FOR_PLOTTING.npz', allow_pickle=False) as a:
        roots, episodes = a['root_ids'], a['episode_ids']
        centers, valids = a['true_centroid'], a['raw_valid']
    rows = []
    for index, root in enumerate(roots):
        with np.load(OUT / 'raw_observations/test_h5' / f'{root}.npz', allow_pickle=False) as a:
            image = a['history_visual'][-1]
        try:
            anchor = observe_rgb(image)['object_centroid']
            anchor_valid, anchor_error = True, ''
        except ValueError as exc:
            anchor, anchor_valid, anchor_error = np.array([np.nan, np.nan]), False, str(exc)
        distances = np.linalg.norm(centers[index] - anchor, axis=-1)
        mask = valids[index] & anchor_valid
        endpoint = distances[:, -1][mask[:, -1]]
        left, right = np.triu_indices(centers.shape[1], 1)
        pairmask = valids[index, left, -1] & valids[index, right, -1]
        pairs = np.linalg.norm(centers[index, left, -1] - centers[index, right, -1], axis=-1)[pairmask]
        measure = lambda array, op: float(op(array)) if len(array) else None
        rows.append({'root_id': str(root), 'episode_id': str(episodes[index]),
            'anchor_valid': anchor_valid, 'anchor_error': anchor_error,
            'valid_future_points': int(mask.sum()), 'total_future_points': mask.size,
            'allstep_displacement_mean_px': measure(distances[mask], np.mean),
            'allstep_displacement_median_px': measure(distances[mask], np.median),
            'allstep_displacement_max_px': measure(distances[mask], np.max),
            'endpoint_displacement_mean_px': measure(endpoint, np.mean),
            'endpoint_displacement_median_px': measure(endpoint, np.median),
            'endpoint_displacement_max_px': measure(endpoint, np.max),
            'valid_endpoint_pairs': int(pairmask.sum()), 'total_endpoint_pairs': len(pairmask),
            'endpoint_pair_gap_mean_px': measure(pairs, np.mean),
            'endpoint_pair_gap_median_px': measure(pairs, np.median),
            'endpoint_pair_gap_max_px': measure(pairs, np.max)})
    write_rows(OUT / 'GEOMETRY_TRUE_MOTION_BY_ROOT.csv', rows)
    path = OUT / 'GEOMETRY_TEST_SUMMARY.json'
    report = json.loads(path.read_text())
    report['true_object_motion_by_root'] = rows
    report['true_motion_note'] = 'All eight roots retained. Displacement is visible silhouette-centroid motion, so occlusion and rotation can move the centroid without equal rigid-body translation. No movement threshold used to select cases.'
    dump(path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'test', 'motion'], default='train')
    args = parser.parse_args()
    {'train': train_calibration, 'test': test_audit, 'motion': raw_motion}[args.phase]()


if __name__ == '__main__':
    main()
