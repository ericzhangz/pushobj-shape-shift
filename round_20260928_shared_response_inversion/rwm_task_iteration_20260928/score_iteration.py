"""Evaluate sealed forecasts only; no fitting, selection of methods, or oracle use."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
BASE = Path(__file__).resolve().parents[1] / 'shared_response_formation_20260928'
sys.path.insert(0, str(BASE))
import score_formation as scoring

LEGACY = ('NATIVE', 'PRIOR', 'STEP_RIDGE', 'DIRECT')
REQUIRED = {*LEGACY, 'SHARED_PRIOR', 'SHARED_RIDGE_FIXED',
            'SHARED_RIDGE_SCALE', 'SHARED_RIDGE_METRIC',
            'SHARED_RIDGE_METRIC_SCALE_ONLY'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, result):
    with (ROOT / name).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def load():
    # Check the prediction seal before opening any query truth or old metrics.
    seal = json.loads((ROOT / 'SEALED.json').read_text(encoding='utf8'))
    assert sha(ROOT / 'PREDICTIONS.pt') == seal['prediction_sha256']
    original = json.loads((BASE / 'scored_verified/RESULTS.json').read_text(encoding='utf8'))
    target_path = BASE / 'scored_verified/TARGETS.pt'
    assert sha(target_path) == original['targets_sha256']
    assert sha(BASE / 'PREDICTIONS.pt') == original['prediction_sha256']
    bundle = torch.load(ROOT / 'PREDICTIONS.pt', map_location='cpu', weights_only=False)
    targets = torch.load(target_path, map_location='cpu', weights_only=False)
    legacy = torch.load(BASE / 'PREDICTIONS.pt', map_location='cpu', weights_only=False)
    assert set(bundle) == set(targets) == {'T0', 'T1', 'L0', 'L1'}
    all_arms = None
    for case, content in bundle.items():
        assert len(content['plans']) == 8
        assert set(content['plans']) == set(targets[case]) == set(legacy[case]['plans'])
        for key in content['goal_z']:
            assert torch.equal(content['goal_z'][key], legacy[case]['goal_z'][key])
        for cid, plan in content['plans'].items():
            assert REQUIRED <= set(plan['arms'])
            if all_arms is None:
                all_arms = set(plan['arms'])
            assert set(plan['arms']) == all_arms
            assert torch.equal(plan['actions'], legacy[case]['plans'][cid]['actions'])
            for arm in LEGACY:
                for key, value in plan['arms'][arm]['z'].items():
                    assert torch.equal(value, legacy[case]['plans'][cid]['arms'][arm]['z'][key])
    return bundle, targets, original


def moments(error):
    common = error.mean(0)
    total = float(error.square().mean())
    shared = float(common.square().mean())
    centered = float((error - common).square().mean())
    assert abs(total - shared - centered) < 1e-12
    n = error.shape[0]
    return {'mse': total, 'common_mse': shared, 'centered_mse': centered,
            'all_pairs_mse': 2 * n / (n - 1) * centered}


def direction(error, delta):
    cross = float((error * delta).mean())
    energy = float(delta.square().mean())
    change = float((error + delta).square().mean() - error.square().mean())
    assert abs(change - 2 * cross - energy) < 1e-12
    denominator = float(error.norm() * delta.norm())
    return {'twice_cross_term': 2 * cross, 'update_energy': energy,
            'mse_change': change,
            'oracle_alpha_diagnostic_only': -cross / energy if energy else None,
            'error_update_cosine': float((error * delta).sum()) / denominator
            if denominator else None}


def vector_comparison(target, reference):
    target_sq = float(target.square().sum())
    reference_sq = float(reference.square().sum())
    cross = float((target * reference).sum())
    scale = cross / reference_sq if reference_sq else None
    residual = float((target - scale * reference).norm()) if scale is not None else None
    return {'target_update_rms': float(target.square().mean()) ** .5,
            'reference_update_rms': float(reference.square().mean()) ** .5,
            'cosine': cross / (target_sq * reference_sq) ** .5
            if target_sq and reference_sq else None,
            'norm_ratio': (target_sq / reference_sq) ** .5 if reference_sq else None,
            'best_scalar_diagnostic_only': scale,
            'relative_proportional_residual': residual / target_sq ** .5
            if residual is not None and target_sq else None}


def numerical():
    if (ROOT / 'RESULTS.json').exists():
        raise FileExistsError('RESULTS.json already exists')
    started = time.perf_counter()
    bundle, targets, original = load()
    summaries, steps, candidates, pairs, directions, comparisons, replays = [], [], [], [], [], [], []
    truth_replays = []
    for case, content in bundle.items():
        ids = list(content['plans'])
        truth = torch.stack([targets[case][cid]['truth_z']['visual'][0, 1:, 0].double() for cid in ids])
        true_cost = {cid: targets[case][cid]['recorded_truth_cost'] for cid in ids}
        best = min(true_cost, key=true_cost.get)
        arms = list(content['plans'][ids[0]]['arms'])
        first = {}
        for cid in ids:
            drift = abs(targets[case][cid]['truth_cost']['total'] - true_cost[cid])
            truth_replays.append({'case': case, 'candidate': cid, 'cost_replay_abs': drift,
                                  'passes_original_2e_5': drift <= 2e-5})
        for arm in arms:
            predicted = torch.stack([content['plans'][cid]['arms'][arm]['z']['visual'][0, 1:, 0].double() for cid in ids])
            error = predicted - truth
            first[arm] = predicted[:, 0]
            costs = {cid: content['plans'][cid]['arms'][arm]['cost']['total'] for cid in ids}
            selected = min(costs, key=costs.get)
            arm_pairs = []
            for left, right in itertools.combinations(ids, 2):
                i, j = ids.index(left), ids.index(right)
                predicted_gap, true_gap = costs[left] - costs[right], true_cost[left] - true_cost[right]
                row = {'case': case, 'arm': arm, 'left': left, 'right': right,
                       'predicted_gap': predicted_gap, 'true_gap': true_gap,
                       'abs_gap_error': abs(predicted_gap - true_gap),
                       'reversal': bool(predicted_gap * true_gap < 0),
                       'terminal_contrast_mse': float((error[i, -1] - error[j, -1]).square().mean())}
                arm_pairs.append(row)
            pairs.extend(arm_pairs)
            info = {'case': case, 'arm': arm, 'visual_mse': float(error.square().mean()),
                    'first_step_mse': float(error[:, 0].square().mean()),
                    'first_two_mse': float(error[:, :2].square().mean()),
                    'last_three_mse': float(error[:, 2:].square().mean()),
                    'terminal_contrast_mse': sum(r['terminal_contrast_mse'] for r in arm_pairs) / len(arm_pairs),
                    'pair_gap_mae': sum(r['abs_gap_error'] for r in arm_pairs) / len(arm_pairs),
                    'pair_reversals': sum(r['reversal'] for r in arm_pairs),
                    'selected': selected, 'best': best,
                    'regret': true_cost[selected] - true_cost[best]}
            assert abs(info['terminal_contrast_mse'] - moments(error[:, -1])['all_pairs_mse']) < 1e-12
            summaries.append(info)
            for step in range(5):
                steps.append({'case': case, 'arm': arm, 'step': step + 1, **moments(error[:, step])})
            if arm in LEGACY:
                old = next(r for r in original['summary'] if r['case'] == case and r['arm'] == arm)
                for key, old_key in [('visual_mse', 'visual_mse'), ('pair_gap_mae', 'pair_gap_mae'),
                                     ('regret', 'regret'), ('terminal_contrast_mse', 'contrast_mse')]:
                    assert abs(info[key] - old[old_key]) < 1e-12, (case, arm, key)
                assert info['selected'] == old['selected'] and info['best'] == old['best']
            for i, cid in enumerate(ids):
                entry = content['plans'][cid]['arms'][arm]
                z = entry['z']
                scoring.check_z(z)
                actual = targets[case][cid]['truth_z']
                anchor = max(float((z[k][:, :1] - actual[k][:, :1]).abs().max()) for k in z)
                assert anchor < 2e-5, (case, cid, arm, anchor)
                cost = scoring.native_objective_breakdown(z, content['goal_z'], 4, alpha=1., base=2.)
                assert cost['stage'] == 'terminal'
                drift = max(abs(float(cost[k]) - entry['cost'][k]) for k in ('visual', 'proprio', 'total'))
                assert drift < 2e-5, (case, cid, arm, drift)
                replays.append({'case': case, 'candidate': cid, 'arm': arm,
                                'anchor_max_abs': anchor, 'cost_max_abs': drift})
                candidates.append({'case': case, 'candidate': cid, 'arm': arm,
                                   'visual_mse': float(error[i].square().mean()),
                                   'per_step_visual_mse': [float(e.square().mean()) for e in error[i]],
                                   'proprio_mse': float((z['proprio'][:, 1:].double() - actual['proprio'][:, 1:].double()).square().mean()),
                                   'predicted_cost': costs[cid], 'true_cost': true_cost[cid]})
        baseline_error = first['SHARED_PRIOR'] - truth[:, 0]
        deltas = {arm: value - first['SHARED_PRIOR'] for arm, value in first.items() if arm.startswith('SHARED_RIDGE')}
        for arm, delta in deltas.items():
            for label, d, e in [('ALL_CANDIDATES', delta, baseline_error)] + [
                    (cid, delta[i], baseline_error[i]) for i, cid in enumerate(ids)]:
                directions.append({'case': case, 'arm': arm, 'candidate': label,
                                   'baseline': 'SHARED_PRIOR', **direction(e, d)})
        for target_arm, reference_arm in [
                ('SHARED_RIDGE_METRIC', 'SHARED_RIDGE_SCALE'),
                ('SHARED_RIDGE_METRIC', 'SHARED_RIDGE_METRIC_SCALE_ONLY'),
                ('SHARED_RIDGE_METRIC', 'SHARED_RIDGE_FIXED'),
                ('SHARED_RIDGE_SCALE', 'SHARED_RIDGE_FIXED')]:
            target, reference = deltas[target_arm], deltas[reference_arm]
            for label, t, r in [('ALL_CANDIDATES', target, reference)] + [
                    (cid, target[i], reference[i]) for i, cid in enumerate(ids)]:
                comparisons.append({'case': case, 'target_arm': target_arm,
                                    'reference_arm': reference_arm, 'candidate': label,
                                    'baseline': 'SHARED_PRIOR', **vector_comparison(t, r)})
    inherited_failures = sum(not r['passes_original_2e_5'] for r in truth_replays)
    assert inherited_failures == 4, inherited_failures
    means = {}
    first_case = next(iter(bundle.values()))
    first_plan = next(iter(first_case['plans'].values()))
    for arm in first_plan['arms']:
        rows = [r for r in summaries if r['arm'] == arm]
        means[arm] = {key: sum(r[key] for r in rows) / len(rows) for key in
                      ('visual_mse', 'first_step_mse', 'first_two_mse', 'last_three_mse',
                       'terminal_contrast_mse', 'pair_gap_mae', 'regret')}
    output = {'summary': summaries, 'arm_means': means, 'per_step': steps,
              'per_candidate': candidates, 'pairs': pairs,
              'first_step_direction_diagnostic': directions,
              'first_step_update_geometry': comparisons, 'replay': replays,
              'truth_cost_replay': truth_replays,
              'prediction_sha256': sha(ROOT / 'PREDICTIONS.pt'),
              'targets_sha256': sha(BASE / 'scored_verified/TARGETS.pt'),
              'scorer_sha256': sha(Path(__file__)),
              'inherited_truth_cost_replay_failures': inherited_failures,
              'oracle_scalars_used_to_fit_or_deploy': False,
              'new_predictor_calls': 0, 'new_encoder_calls': 0, 'new_environment_calls': 0,
              'wall_seconds': time.perf_counter() - started}
    write('RESULTS.json', output)
    print(json.dumps({'arm_means': means, 'truth_replay_failures': inherited_failures}, indent=2), flush=True)


@torch.no_grad()
def geometry():
    if (ROOT / 'GEOMETRY.json').exists():
        raise FileExistsError('GEOMETRY.json already exists')
    started = time.perf_counter()
    bundle, targets, _ = load()
    model, _, _ = scoring.runtime()
    versions = [(t, t._version) for t in list(model.parameters()) + list(model.buffers())]
    count = {'predictor_calls': 0, 'encoder_calls': 0, 'decoder_calls': 0, 'decoder_frames': 0}

    def no_prediction(*args, **kwargs):
        count['predictor_calls'] += 1
        raise RuntimeError('Decoder-only evaluation must not predict.')

    def no_encoding(*args, **kwargs):
        count['encoder_calls'] += 1
        raise RuntimeError('Decoder-only evaluation must not encode.')

    def decoded(module, inputs):
        count['decoder_calls'] += 1
        count['decoder_frames'] += int(inputs[0].shape[0] * inputs[0].shape[1])

    hooks = [model.predictor.register_forward_pre_hook(no_prediction),
             model.decoder.register_forward_pre_hook(decoded)]
    rows, summaries, true_rows = [], [], []
    expected_calls = sum(len(p['arms']) for c in bundle.values() for p in c['plans'].values())
    expected_frames = sum(int(item['z']['visual'].shape[0] * item['z']['visual'].shape[1])
                          for c in bundle.values() for p in c['plans'].values() for item in p['arms'].values())
    try:
        with patch.object(model, 'encode_obs', side_effect=no_encoding), \
             patch.object(model.encoder, 'forward', side_effect=no_encoding), \
             patch.object(model, 'predict', side_effect=no_prediction):
            for case, content in bundle.items():
                start_rows = len(rows)
                for cid, plan in content['plans'].items():
                    true_geometry = [scoring.geometry(frame) for frame in targets[case][cid]['rgb']]
                    for step, g in enumerate(true_geometry):
                        true_rows.append({'case': case, 'candidate': cid, 'step': step, **g})
                    for arm, item in plan['arms'].items():
                        decoded_obs, _ = model.decode_obs({k: v.cuda() for k, v in item['z'].items()})
                        frames = scoring.rgb(decoded_obs['visual'])
                        for step in range(1, 6):
                            g = scoring.geometry(frames[step])
                            rows.append({'case': case, 'candidate': cid, 'arm': arm, 'step': step,
                                         'valid': g['valid'], 'readout_error': g['error'],
                                         'true_valid': true_geometry[step]['valid'],
                                         'true_readout_error': true_geometry[step]['error'],
                                         'object_x': g['object_x'], 'object_y': g['object_y'],
                                         'centroid_error_px': scoring.centroid_error(g, true_geometry[step])})
                case_rows = rows[start_rows:]
                for arm in next(iter(content['plans'].values()))['arms']:
                    subset = [r for r in case_rows if r['arm'] == arm]
                    valid = [r['centroid_error_px'] for r in subset if r['centroid_error_px'] is not None]
                    summaries.append({'case': case, 'arm': arm, 'attempts': len(subset),
                                      'predicted_readout_failures': sum(not r['valid'] for r in subset),
                                      'matched_valid_count': len(valid),
                                      'matched_missing_count': len(subset) - len(valid),
                                      'centroid_error_px': sum(valid) / len(valid) if valid else None})
                print('decoded', case, flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert all(t._version == v for t, v in versions)
    assert count['predictor_calls'] == count['encoder_calls'] == 0
    assert count['decoder_calls'] == expected_calls and count['decoder_frames'] == expected_frames
    assert len(rows) == expected_calls * 5
    true_failures = {case: sum(r['case'] == case and r['step'] > 0 and not r['valid'] for r in true_rows)
                     for case in bundle}
    write('GEOMETRY.json', {'summary': summaries, 'rows': rows, 'true_rows': true_rows,
                            'true_future_readout_failures': true_failures, 'counts': count,
                            'expected_decoder_calls': expected_calls, 'expected_decoder_frames': expected_frames,
                            'new_environment_calls': 0, 'backbone_unchanged': True,
                            'prediction_sha256': sha(ROOT / 'PREDICTIONS.pt'),
                            'targets_sha256': sha(BASE / 'scored_verified/TARGETS.pt'),
                            'scorer_sha256': sha(Path(__file__)),
                            'wall_seconds': time.perf_counter() - started})
    print(json.dumps({'counts': count, 'true_failures': true_failures}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('numerical', 'geometry'))
    args = parser.parse_args()
    globals()[args.phase]()
