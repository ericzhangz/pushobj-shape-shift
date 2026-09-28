"""Score only sealed m4 forecasts; outcome arrays are opened inside score()."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROBE = ROOT.parent / 'interaction_validation_20260927'
sys.path.insert(0, str(PROBE))
# Do not import the probe's ROOT: all new artifacts belong to this round.
from native_interaction_probe import REPO, DATA, CHECKPOINT, digest, runtime, reconstruct
from research.contrast_probe import native_objective_breakdown
from research.reframe_v3.rgb_rigid_observer import observe_rgb


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def mse(left, right):
    # Convert before subtraction, including contrast computations below.
    assert left.shape == right.shape
    return float((left.double() - right.double()).square().mean())


def rgb(decoded):
    values = decoded.detach().cpu().numpy()[0].transpose(0, 2, 3, 1)
    assert values.shape == (6, 224, 224, 3) and np.isfinite(values).all()
    return np.clip(np.rint((values + 1) * 127.5), 0, 255).astype(np.uint8)


def geometry(frame):
    try:
        observation = observe_rgb(frame)
    except ValueError as error:
        return dict(valid=False, error=str(error), object_x=None, object_y=None)
    center = observation['object_centroid']
    return dict(valid=True, error='', object_x=float(center[0]),
                object_y=float(center[1]))


def centroid_error(predicted, actual):
    if not (predicted['valid'] and actual['valid']):
        return None
    return float(np.hypot(predicted['object_x'] - actual['object_x'],
                          predicted['object_y'] - actual['object_y']))


def mean_valid(rows, column):
    values = [row[column] for row in rows if row[column] is not None]
    return float(np.mean(values)) if values else None


def check_z(z):
    assert set(z) == {'visual', 'proprio'}
    assert z['visual'].shape == (1, 6, 1, 384)
    assert z['proprio'].shape == (1, 6, 10)
    assert all(value.device.type == 'cpu' and torch.isfinite(value).all()
               for value in z.values())


@torch.no_grad()
def score(output_name='scored'):
    prediction_path = ROOT / 'PREDICTIONS.pt'
    seal_path = ROOT / 'SEALED.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    prediction_sha = digest(prediction_path)
    assert prediction_sha == seal['prediction_sha256'], 'prediction seal mismatch'
    bundle = torch.load(prediction_path, map_location='cpu', weights_only=False)
    assert set(bundle) == {'T0', 'T1', 'L0', 'L1'}
    # No truth file or runtime is opened before the prediction seal is checked.
    if Path(output_name).name != output_name:
        raise ValueError('output must be one directory name')
    out = ROOT / output_name
    out.mkdir(exist_ok=False)
    protected = [prediction_path, seal_path, Path(__file__),
                 PROBE / 'native_interaction_probe.py',
                 CHECKPOINT / 'checkpoints/model_latest.pth', CHECKPOINT / 'hydra.yaml',
                 REPO / 'models/visual_world_model.py',
                 REPO / 'research/contrast_probe.py',
                 REPO / 'research/reframe_v3/rgb_rigid_observer.py']
    provenance = {str(path): digest(path) for path in protected}
    model, prep, cfg = runtime()
    versions = [(value, value._version) for value in
                list(model.parameters()) + list(model.buffers())]
    counters = dict(predictor_calls=0, encoder_calls=0, encoder_frames=0,
                    decoder_calls=0, decoder_frames=0)

    def predictor_hook(module, inputs):
        counters['predictor_calls'] += 1
        raise RuntimeError('scoring must not call the predictor')

    def decoder_hook(module, inputs):
        counters['decoder_calls'] += 1
        counters['decoder_frames'] += int(inputs[0].shape[0] * inputs[0].shape[1])

    hooks = [model.predictor.register_forward_pre_hook(predictor_hook),
             model.decoder.register_forward_pre_hook(decoder_hook)]
    started = time.perf_counter()
    targets, errors, pairs, selections, frames, replays, summaries = {}, [], [], [], [], [], []
    all_arms = None
    for key, case in bundle.items():
        shape, sample = key[0], int(key[1:])
        oracle_path = DATA / f'evaluate_{shape}_seed101_s{sample}_m4_eight/ORACLE_COSTS.csv'
        provenance[str(oracle_path)] = digest(oracle_path)
        with oracle_path.open(newline='', encoding='utf-8') as stream:
            records = list(csv.DictReader(stream))
        truths = {record['candidate_id']: record for record in records}
        assert len(records) == len(truths) == len(case['plans']) == 8
        assert set(truths) == set(case['plans'])
        arms = tuple(next(iter(case['plans'].values()))['arms'])
        assert arms and all(set(entry['arms']) == set(arms)
                            for entry in case['plans'].values())
        if all_arms is None:
            all_arms = arms
        assert set(arms) == set(all_arms), 'all cases must evaluate the same arms'
        case_targets = {}
        for candidate, entry in case['plans'].items():
            record = truths[candidate]
            action_path = Path(record['action_tensor'])
            provenance[str(action_path)] = digest(action_path)
            bound_action = torch.load(action_path, map_location='cpu',
                                      weights_only=False)[record['tensor_key']]
            assert torch.equal(bound_action, entry['actions']), (key, candidate, 'action mismatch')
            assert entry['actions'].shape == (1, 5, 10)
            sidecar = Path(record['outcome_sidecar'])
            provenance[str(sidecar)] = digest(sidecar)
            with np.load(sidecar, allow_pickle=False) as data:
                # Deliberately never access simulator state, even if present.
                images = data['visual'].copy()
                props = data['proprio'].copy()
            # These outcome sidecars already contain the six macro-step frames.
            assert images.shape == (6, 224, 224, 3) and images.dtype == np.uint8
            assert props.shape == (6, 4) and np.isfinite(props).all()
            # encode_obs invokes encoder.forward directly, bypassing module hooks.
            # reconstruct uses batches of eight; these six-frame inputs use one.
            counters['encoder_calls'] += 1
            counters['encoder_frames'] += len(images)
            truth_z, reconstruction = reconstruct(model, prep, images, props)
            check_z(truth_z)
            truth_score = native_objective_breakdown(truth_z, case['goal_z'], 4,
                                                       alpha=1., base=2.)
            assert truth_score['stage'] == 'terminal'
            truth_cost = {name: float(truth_score[name].item())
                          for name in ('visual', 'proprio', 'total')}
            actual_cost = float(record['c_env_ref'])
            assert np.isfinite([actual_cost, *truth_cost.values()]).all()
            case_targets[candidate] = dict(truth_z=truth_z, rgb=images, proprio=props,
                reconstructed=reconstruction, truth_cost=truth_cost,
                recorded_truth_cost=actual_cost, outcome_sidecar=str(sidecar))
            replays.append(dict(case=key, candidate=candidate, arm='TRUTH',
                cost_replay_abs=abs(truth_cost['total'] - actual_cost), anchor_max_abs=0.))
            real_geometry = [geometry(image) for image in images]
            reconstruction_geometry = [geometry(image) for image in rgb(reconstruction)]
            for arm_name, series in [('REAL', real_geometry),
                                     ('RECON_REAL', reconstruction_geometry)]:
                for step, measurement in enumerate(series):
                    frames.append(dict(case=key, candidate=candidate, arm=arm_name,
                        step=step, time_seconds=step * .5, **measurement,
                        object_centroid_error_px=centroid_error(measurement, real_geometry[step])))
            for arm, arm_entry in entry['arms'].items():
                z = arm_entry['z']
                check_z(z)
                anchor_error = max(float((z[name][:, :1] - truth_z[name][:, :1]).abs().max())
                                   for name in z)
                assert anchor_error <= 2e-5, (key, candidate, arm, 'anchor mismatch', anchor_error)
                replay = native_objective_breakdown(z, case['goal_z'], 4, alpha=1., base=2.)
                assert replay['stage'] == 'terminal'
                replay_error = max(abs(float(replay[name].item()) - float(arm_entry['cost'][name]))
                                   for name in ('visual', 'proprio', 'total'))
                assert replay_error <= 2e-5, (key, candidate, arm, 'stored cost mismatch', replay_error)
                replays.append(dict(case=key, candidate=candidate, arm=arm,
                    cost_replay_abs=replay_error, anchor_max_abs=anchor_error))
                decoded, _ = model.decode_obs({name: value.cuda() for name, value in z.items()})
                predicted_geometry = [geometry(image) for image in rgb(decoded['visual'])]
                for step, measurement in enumerate(predicted_geometry):
                    centroid = centroid_error(measurement, real_geometry[step])
                    frames.append(dict(case=key, candidate=candidate, arm=arm, step=step,
                        time_seconds=step * .5, **measurement, object_centroid_error_px=centroid))
                    if step:
                        errors.append(dict(case=key, candidate=candidate, arm=arm, step=step,
                            visual_mse=mse(z['visual'][:, step], truth_z['visual'][:, step]),
                            proprio_mse=mse(z['proprio'][:, step], truth_z['proprio'][:, step]),
                            object_centroid_error_px=centroid))
            print(f'scored {key}/{candidate}; {len(arms)} arms', flush=True)
        actual = {candidate: item['recorded_truth_cost'] for candidate, item in case_targets.items()}
        for arm in arms:
            predicted = {candidate: float(entry['arms'][arm]['cost']['total'])
                         for candidate, entry in case['plans'].items()}
            assert np.isfinite(list(predicted.values())).all()
            selected, best = min(predicted, key=predicted.get), min(actual, key=actual.get)
            selection = dict(case=key, arm=arm, selected=selected, best=best,
                regret=actual[selected] - actual[best], selected_cost=actual[selected],
                best_cost=actual[best], selected_recorded_success=truths[selected].get('success', ''),
                predicted_exact_ties='|'.join(c for c in predicted if predicted[c] == predicted[selected]))
            selections.append(selection)
            for left, right in itertools.combinations(case['plans'], 2):
                predicted_gap, true_gap = predicted[left] - predicted[right], actual[left] - actual[right]
                pleft = case['plans'][left]['arms'][arm]['z']['visual'][:, -1].double()
                pright = case['plans'][right]['arms'][arm]['z']['visual'][:, -1].double()
                tleft = case_targets[left]['truth_z']['visual'][:, -1].double()
                tright = case_targets[right]['truth_z']['visual'][:, -1].double()
                pairs.append(dict(case=key, arm=arm, left=left, right=right,
                    predicted_gap=predicted_gap, true_gap=true_gap,
                    abs_gap_error=abs(predicted_gap - true_gap),
                    squared_gap_error=(predicted_gap - true_gap) ** 2,
                    reversal=bool(predicted_gap * true_gap < 0),
                    contrast_mse=mse(pleft - pright, tleft - tright)))
            erows = [row for row in errors if row['case'] == key and row['arm'] == arm]
            prows = [row for row in pairs if row['case'] == key and row['arm'] == arm]
            grows = [row for row in frames if row['case'] == key and row['arm'] == arm and row['step'] > 0]
            summaries.append(dict(**selection,
                visual_mse=mean_valid(erows, 'visual_mse'), proprio_mse=mean_valid(erows, 'proprio_mse'),
                terminal_visual_mse=mean_valid([row for row in erows if row['step'] == 5], 'visual_mse'),
                contrast_mse=mean_valid(prows, 'contrast_mse'), pair_gap_mae=mean_valid(prows, 'abs_gap_error'),
                pair_gap_mse=mean_valid(prows, 'squared_gap_error'),
                pair_reversals=sum(row['reversal'] for row in prows),
                object_centroid_error_px=mean_valid(erows, 'object_centroid_error_px'),
                terminal_object_centroid_error_px=mean_valid([row for row in erows if row['step'] == 5], 'object_centroid_error_px'),
                centroid_error_count=sum(row['object_centroid_error_px'] is not None for row in erows),
                centroid_error_missing=sum(row['object_centroid_error_px'] is None for row in erows),
                predicted_geometry_failures=sum(not row['valid'] for row in grows)))
        targets[key] = case_targets
    for hook in hooks:
        hook.remove()
    assert all(value._version == version for value, version in versions)
    assert all(digest(path) == value for path, value in provenance.items())
    assert digest(prediction_path) == prediction_sha
    assert len(errors) == 4 * 8 * len(all_arms) * 5
    assert len(pairs) == 4 * 28 * len(all_arms)
    assert len(frames) == 4 * 8 * (len(all_arms) + 2) * 6
    torch.save(targets, out / 'TARGETS.pt')
    for name, rows in [('ERRORS', errors), ('PAIRS', pairs), ('SELECTION', selections),
                       ('GEOMETRY', frames), ('REPLAY', replays), ('SUMMARY', summaries)]:
        write_csv(out / f'{name}.csv', rows)
    means = {}
    metrics = ('visual_mse', 'proprio_mse', 'terminal_visual_mse', 'contrast_mse',
               'pair_gap_mae', 'pair_gap_mse', 'regret', 'object_centroid_error_px',
               'terminal_object_centroid_error_px')
    for arm in all_arms:
        subset = [row for row in summaries if row['arm'] == arm]
        means[arm] = {metric: mean_valid(subset, metric) for metric in metrics}
    geometry_counts = {}
    for arm in ('REAL', 'RECON_REAL', *all_arms):
        subset = [row for row in frames if row['arm'] == arm]
        geometry_counts[arm] = dict(attempts=len(subset),
            failures=sum(not row['valid'] for row in subset),
            future_attempts=sum(row['step'] > 0 for row in subset),
            future_failures=sum(not row['valid'] and row['step'] > 0 for row in subset))
    save_json(out / 'PROVENANCE.json', provenance)
    save_json(out / 'RESULTS.json', dict(status='COMPLETE', cases=4, candidates=32,
        arms=list(all_arms), summary=summaries, arm_means=means, geometry_counts=geometry_counts,
        prediction_sha256=prediction_sha, targets_sha256=digest(out / 'TARGETS.pt'),
        max_recorded_cost_replay_abs=max(row['cost_replay_abs'] for row in replays if row['arm'] == 'TRUTH'),
        max_prediction_cost_replay_abs=max(row['cost_replay_abs'] for row in replays if row['arm'] != 'TRUTH'),
        **counters, hidden_state_read=False, training_steps=0, environment_calls=0,
        parameter_buffer_versions_unchanged=True, all_bound_actions_equal=True,
        all_latent_anchors_equal_within_2e_5=True, evaluation_inputs_unchanged=True,
        outcome_sampling='six stored macro-step frames, no additional subsampling',
        visual_error='latent MSE over future steps 1..5, float64 subtraction',
        proprio_error='encoded proprio MSE; no physical inverse',
        contrast='all 28 unordered candidate pairs, terminal visual vector differences',
        centroid='visible RGB object centroid; failed readouts excluded with explicit counts',
        aggregate_weighting='equal case weight; centroid means conditional on valid readouts',
        wall_seconds_after_load=time.perf_counter() - started))
    print(json.dumps(means, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='scored')
    score(parser.parse_args().out)
