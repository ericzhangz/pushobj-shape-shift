"""F0 coarse-boundary lift gate, with no environment or candidate truth access.

The deployable calculation is VWorldModel.rollout(observation_transition=...).
Teacher cache walking here constructs supervised F0 queries, not a second
deployment rollout. Passing this gate is not evidence of D-only transfer.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import time
from unittest.mock import patch

import torch

from research.reframe_v3.controlled_generator import ControlledGenerator
from research.reframe_v3.completed_phase_evidence import merge_completed_lowlevel, phase_chains
from research.reframe_v3.matched_feedback_forecast import _join_encoded_observation_action
from research.reframe_v3.round6_reference import _local_hub_loader, _write_csv, _write_json
from research.reframe_v3.shadow_selection_audit import (
    _load_anchor_observations, _load_runtime, _load_segments, _seed_all, _transform_anchor_obs,
)


OBS_DIM = 394
VIS_DIM = 384
ANCHORS = (('T', 0), ('T', 1), ('L', 0), ('L', 1))


def observation_vector(observed):
    return torch.cat((observed['visual'], observed['proprio'].unsqueeze(2)), -1)


def native_loss(prediction, target):
    difference = prediction - target
    return (difference[..., :VIS_DIM].square().mean()
            + difference[..., VIS_DIM:].square().mean())


def commands(count, horizon, low_actions, random):
    flat = low_actions.detach().cpu().reshape(-1, 2)
    mean, std = flat.mean(0), flat.std(0, unbiased=False)
    noise = torch.randn(count, horizon, 5, 2, generator=random)
    return (mean + noise * std).reshape(count, horizon, 10).to(low_actions)


def base_histories(model, encoded, actions):
    histories = []
    for chain in phase_chains(encoded, actions):
        observed = chain['observed']
        for index in range(chain['actions'].shape[1]):
            first = max(0, index-model.num_hist+1)
            source = {key: value[:, first:index+1] for key, value in observed.items()}
            past = chain['actions'][:, first:index]
            padded = torch.cat((past, torch.zeros_like(chain['actions'][:, :1])), 1)
            histories.append(_join_encoded_observation_action(model, source, padded).detach())
    if not histories:
        raise ValueError('no legal completed native windows')
    return histories


@torch.no_grad()
def teacher_single(model, histories, query_actions, random):
    selected = torch.randint(len(histories), (query_actions.shape[0],), generator=random).tolist()
    pools = defaultdict(list)
    for index, history_id in enumerate(selected):
        pools[histories[history_id].shape[1]].append((histories[history_id], query_actions[index:index+1]))
    result = {}
    for length, items in pools.items():
        history = torch.cat([item[0] for item in items], 0)
        action = torch.cat([item[1] for item in items], 0)
        history = history.clone()
        history[:, -1:] = model.replace_actions_from_z(history[:, -1:], action)
        target = model.predict(history)[:, -1:, ..., :OBS_DIM]
        result[length] = (history.detach(), action.detach(), target.detach())
    return result


@torch.no_grad()
def teacher_chains(model, initial_encoded, plans):
    count = plans.shape[0]
    initial = {key: value.repeat(count, *([1]*(value.ndim-1))) for key, value in initial_encoded.items()}
    history = _join_encoded_observation_action(model, initial, plans[:, :1])
    result = defaultdict(list)
    for index in range(plans.shape[1]):
        action = plans[:, index:index+1]
        history = history.clone()
        history[:, -1:] = model.replace_actions_from_z(history[:, -1:], action)
        target = model.predict(history[:, -model.num_hist:])[:, -1:, ..., :OBS_DIM]
        used = history[:, -model.num_hist:]
        result[used.shape[1]].append((used.detach(), action.detach(), target.detach()))
        next_token = torch.cat((target, torch.zeros_like(history[:, -1:, ..., -10:])), -1)
        history = torch.cat((history, next_token), 1)[:, -model.num_hist:]
    return {length: tuple(torch.cat([item[i] for item in items], 0) for i in range(3))
            for length, items in result.items()}


def merge_queries(left, right):
    return {length: tuple(torch.cat([source[length][i] for source in (left, right)
                                    if length in source], 0) for i in range(3))
            for length in sorted(set(left) | set(right))}


@torch.no_grad()
def factual_queries(model, encoded, actions):
    """Only complete stride-five targets in the supplied evidence are labels."""
    grouped, times = defaultdict(list), []
    for chain in phase_chains(encoded, actions):
        for index, start in enumerate(chain['lowlevel_starts']):
            first = max(0, index-model.num_hist+1)
            observed = {key: value[:, first:index+1] for key, value in chain['observed'].items()}
            packed = chain['actions'][:, first:index+1]
            history = _join_encoded_observation_action(model, observed, packed).detach()
            target = observation_vector({key: value[:, index+1:index+2]
                                         for key, value in chain['observed'].items()})
            grouped[history.shape[1]].append((history, packed[:, -1:], target.detach()))
            times.append({'phase': chain['phase'], 'start': start,
                          'end': chain['lowlevel_ends'][index]})
    if not grouped:
        raise ValueError('no complete factual coarse transition')
    return {length: tuple(torch.cat([item[i] for item in items], 0) for i in range(3))
            for length, items in grouped.items()}, times


def adapt_completed(student, factual, teacher, seed, steps=100):
    """Equal factual/teacher objectives, no held labels, no step selection."""
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    random = torch.Generator().manual_seed(seed)
    rows = []
    student.train()
    for step in range(steps):
        history, action, target = sample_batch(factual, random)
        teacher_history, teacher_action, teacher_target = sample_batch(teacher, random)
        optimizer.zero_grad(set_to_none=True)
        factual_loss = native_loss(student(history, action), target)
        preservation_loss = native_loss(student(teacher_history, teacher_action), teacher_target)
        loss = .5*(factual_loss+preservation_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite legal D objective')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in student.parameters()):
            raise FloatingPointError('invalid legal D gradient')
        optimizer.step()
        rows.append({'kind': student.kind, 'step': step+1,
                     'factual_loss': float(factual_loss.detach()),
                     'teacher_loss': float(preservation_loss.detach())})
    student.eval()
    return rows


@torch.no_grad()
def factual_metrics(model, student, dataset, arm):
    errors, visual, proprio = [], [], []
    for history, action, target in dataset.values():
        predicted = (model.predict(history)[:, -1:, ..., :OBS_DIM] if student is None
                     else student(history, action))
        for index in range(history.shape[0]):
            difference = predicted[index:index+1] - target[index:index+1]
            v = float(difference[..., :VIS_DIM].square().mean())
            p = float(difference[..., VIS_DIM:].square().mean())
            visual.append(v)
            proprio.append(p)
            errors.append(v+p)
    return {'arm': arm, 'queries': len(errors), 'rms': (sum(errors)/len(errors))**.5,
            'visual_mse': sum(visual)/len(visual), 'proprio_mse': sum(proprio)/len(proprio)}


def save_student(student, path):
    torch.save({'kind': student.kind, 'parameters': student.parameter_count(),
                'state_dict': {key: value.cpu() for key, value in student.state_dict().items()}}, path)


def sample_batch(dataset, random, batch=64):
    lengths = sorted(dataset)
    counts = torch.tensor([dataset[length][0].shape[0] for length in lengths], dtype=torch.float64)
    group = int(torch.multinomial(counts, 1, generator=random))
    pool = dataset[lengths[group]]
    indices = torch.randint(pool[0].shape[0], (batch,), generator=random).to(pool[0].device)
    return tuple(value[indices] for value in pool)


def train_lift(student, dataset, seed, steps=600):
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    random = torch.Generator().manual_seed(seed)
    rows = []
    student.train()
    for step in range(steps):
        history, action, target = sample_batch(dataset, random)
        optimizer.zero_grad(set_to_none=True)
        loss = native_loss(student(history, action), target)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite coarse lift training objective')
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in student.parameters()):
            raise FloatingPointError('invalid shared-field gradient')
        optimizer.step()
        rows.append({'kind': student.kind, 'step': step+1, 'loss': float(loss.detach())})
        if (step+1) % 100 == 0:
            print(student.kind, 'coarse-match step', step+1, 'loss', rows[-1]['loss'], flush=True)
    student.eval()
    return rows


@torch.no_grad()
def single_metrics(student, dataset, substeps=1):
    errors, changes = [], []
    for history, action, target in dataset.values():
        prediction = student(history, action, substeps=substeps)
        current = history[:, -1:, ..., :OBS_DIM]
        for index in range(history.shape[0]):
            errors.append(float(native_loss(prediction[index:index+1], target[index:index+1])))
            changes.append(float(native_loss(target[index:index+1], current[index:index+1])))
    mse, change = sum(errors)/len(errors), sum(changes)/len(changes)
    if change <= 1e-12:
        raise ValueError('teacher change too small to define the predeclared relative gate')
    return {'kind': student.kind, 'scope': 'single', 'substeps': substeps,
            'queries': len(errors), 'rms': mse**.5, 'teacher_change_rms': change**.5,
            'relative_rms': (mse/change)**.5}


@torch.no_grad()
def native_paths(model, current, plans, student=None, substeps=1):
    count = plans.shape[0]
    observed = {key: value.repeat(count, *([1]*(value.ndim-1))) for key, value in current.items()}
    options = {} if student is None else {
        'observation_transition': lambda history, action: student(history, action, substeps=substeps)}
    path, _ = model.rollout(observed, plans, **options)
    return observation_vector(path)


def path_metrics(student, prediction, target, substeps=1):
    rows = []
    for endpoint in range(1, target.shape[1]):
        mse = float(native_loss(prediction[:, endpoint:endpoint+1], target[:, endpoint:endpoint+1]))
        change = float(native_loss(target[:, endpoint:endpoint+1], target[:, :1]))
        if change <= 1e-12:
            raise ValueError('teacher path change too small to define the predeclared gate')
        rows.append({'kind': student.kind, 'scope': f'H{endpoint}', 'substeps': substeps,
                     'queries': target.shape[0], 'rms': mse**.5,
                     'teacher_change_rms': change**.5, 'relative_rms': (mse/change)**.5})
    return rows


def run(args):
    if args.out.exists():
        raise FileExistsError('refusing to overwrite formation output')
    args.out.mkdir(parents=True)
    _seed_all(260926)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    started = time.perf_counter()
    with patch.object(torch.hub, 'load', _local_hub_loader()):
        model, preprocessor, config = _load_runtime(args.checkpoint_dir, device)
    if (int(config.frameskip) != 5 or model.num_hist != 3 or model.concat_dim != 1
            or model.proprio_dim != 10 or model.action_dim != 10):
        raise ValueError('unverified native model layout')
    model_tensors = list(model.parameters()) + list(model.buffers())
    versions = [value._version for value in model_tensors]
    _write_json(args.out/'RUN_CONTRACT.json', {
        'stage': 'F0_COARSE_BOUNDARY_GATE', 'anchors': ANCHORS, 'mpc': 2,
        'checkpoint_dir': str(args.checkpoint_dir), 'seed': 260926,
        'teacher_single_train': 128, 'teacher_H5_train': 32,
        'teacher_single_validation': 64, 'teacher_H5_validation': 16,
        'training_steps': 600, 'batch_size': 64, 'lr': 1e-3,
        'gate_single_relative_rms': .25, 'gate_H5_relative_rms': .50,
        'validation_used_for_fit_or_selection': False, 'candidate_outcomes_read': False,
        'environment_branches': 0, 'information': 'completed D inputs and independent F0 queries only',
        'no_D_label_updates_at_this_gate': True,
        'teacher_cache_walk': 'training query construction only; held H5 uses native rollout',
        'matched_teacher_data_and_optimization_steps': True,
        'matched_compute_or_parameter_count': False,
        'flow_network_calls_per_chunk': 10, 'packed_network_calls_per_chunk': 1})
    metric_rows, gates = [], []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for anchor_index, (split, sample) in enumerate(ANCHORS):
        case_start = time.perf_counter()
        case = args.out/f'{split}_s{sample}_m2'
        case.mkdir()
        donor = args.donor_root/f'donor_{split}_seed101_n12_capture'
        segments = _load_segments(donor, sample, 2, 5, preprocessor, full_resolution=True)
        fine, low_actions = merge_completed_lowlevel(segments)
        fine = {key: value.to(device) for key, value in fine.items()}
        low_actions = low_actions.to(device)
        with torch.no_grad():
            encoded_fine = model.encode_obs(fine)
        current_raw, goal_raw = _load_anchor_observations(donor, sample, 2)
        current, _ = _transform_anchor_obs(preprocessor, current_raw, goal_raw, device)
        with torch.no_grad():
            initial_encoded = model.encode_obs(current)
            histories = base_histories(model, encoded_fine, low_actions)
        random = torch.Generator().manual_seed(260926+100*anchor_index)
        train_single = teacher_single(model, histories, commands(128, 1, low_actions, random), random)
        train_chains = teacher_chains(model, initial_encoded, commands(32, 5, low_actions, random))
        train = merge_queries(train_single, train_chains)
        # A separate fixed stream, consumed only after the training queries exist.
        held_random = torch.Generator().manual_seed(360926+100*anchor_index)
        held = teacher_single(model, histories, commands(64, 1, low_actions, held_random), held_random)
        held_plans = commands(16, 5, low_actions, held_random)
        teacher_path = native_paths(model, current, held_plans)
        starts = torch.cat([group[0][:, -1, 0, :OBS_DIM] for group in train.values()], 0)
        center, scale = starts.mean(0), starts.std(0, unbiased=False)
        torch.save({'train': {k: tuple(t.cpu() for t in v) for k, v in train.items()},
                    'held_single': {k: tuple(t.cpu() for t in v) for k, v in held.items()},
                    'held_plans': held_plans.cpu(), 'teacher_path': teacher_path.cpu(),
                    'center': center.cpu(), 'scale': scale.cpu()}, case/'TEACHER_QUERIES.pt')
        traces = []
        case_metrics = []
        for kind in ('flow', 'packed'):
            _seed_all(260926+anchor_index)
            student = ControlledGenerator(kind=kind).to(device)
            student.configure_normalization(center, scale)
            traces.extend(train_lift(student, train, 460926+anchor_index))
            case_metrics.append(single_metrics(student, held))
            predicted = native_paths(model, current, held_plans, student)
            case_metrics.extend(path_metrics(student, predicted, teacher_path))
            torch.save({'kind': kind, 'parameters': student.parameter_count(),
                        'state_dict': {key: value.cpu() for key, value in student.state_dict().items()}},
                       case/f'{kind.upper()}0.pt')
            torch.save(predicted.cpu(), case/f'{kind.upper()}0_HELD_H5.pt')
            if kind == 'flow':
                case_metrics.append(single_metrics(student, held, substeps=2))
                refined = native_paths(model, current, held_plans, student, substeps=2)
                case_metrics.extend(path_metrics(student, refined, teacher_path, substeps=2))
                refinement = float(native_loss(refined[:, -1:], predicted[:, -1:]))**.5
                flow_single = next(row for row in case_metrics if row['scope'] == 'single' and row['substeps'] == 1)
                flow_h5 = next(row for row in case_metrics if row['scope'] == 'H5' and row['substeps'] == 1)
                passed = flow_single['relative_rms'] <= .25 and flow_h5['relative_rms'] <= .50
                gates.append({'split': split, 'sample': sample, 'passed': passed,
                              'single_relative_rms': flow_single['relative_rms'],
                              'H5_relative_rms': flow_h5['relative_rms'],
                              'H5_refinement_rms': refinement,
                              'decision': 'ELIGIBLE_FOR_D_FORMATION' if passed else 'STOP_AT_F0_LIFT_GATE'})
            del student
        for row in case_metrics:
            row.update(split=split, sample=sample)
        metric_rows.extend(case_metrics)
        _write_csv(case/'TRAINING.csv', traces)
        _write_csv(case/'HELD_FIDELITY.csv', case_metrics)
        _write_json(case/'SUMMARY.json', {'gate': gates[-1], 'seconds': time.perf_counter()-case_start})
        print('FINISHED', split, sample, gates[-1], flush=True)
    if versions != [value._version for value in model_tensors] or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('frozen host tensors changed')
    _write_csv(args.out/'HELD_FIDELITY.csv', metric_rows)
    _write_csv(args.out/'GATES.csv', gates)
    _write_json(args.out/'RUN_SUMMARY.json', {
        'status': 'F0_LIFT_GATE_COMPLETE_NO_CANDIDATE_TRUTH_READ',
        'anchors': len(gates), 'flow_gates_passed': sum(row['passed'] for row in gates),
        'environment_branches': 0, 'D_label_updates': 0,
        'host_tensor_versions_unchanged': True, 'host_gradients_absent': True,
        'seconds': time.perf_counter()-started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        'interpretation': 'representation and numerical fidelity gate, not online transfer or originality'})


def run_prefix(args):
    """Time-split formation; full-D lift is explicitly NOT reused here."""
    if args.out.exists():
        raise FileExistsError('refusing to overwrite prefix formation output')
    args.out.mkdir(parents=True)
    _seed_all(260926)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    with patch.object(torch.hub, 'load', _local_hub_loader()):
        model, preprocessor, config = _load_runtime(args.checkpoint_dir, device)
    if (int(config.frameskip) != 5 or model.num_hist != 3 or model.concat_dim != 1
            or model.proprio_dim != 10 or model.action_dim != 10):
        raise ValueError('unverified native model layout')
    model_tensors = list(model.parameters()) + list(model.buffers())
    versions = [value._version for value in model_tensors]
    started = time.perf_counter()
    _write_json(args.out/'RUN_CONTRACT.json', {
        'stage': 'PREFIX_D_FORMATION', 'anchors': ANCHORS, 'fit_cut_lowlevel': 5,
        'D_train_targets': [5], 'D_hold_targets': [6, 7, 8, 9, 10],
        'prefix_teacher_real_frames': list(range(6)),
        'prefix_teacher_start_for_H5': 5,
        'full_D_teacher_or_normalization_reused': False,
        'teacher_train_single': 128, 'teacher_train_H5': 32,
        'teacher_held_single': 64, 'teacher_held_H5': 16,
        'lift_steps': 600, 'lift_lr': 1e-3, 'D_steps': 100, 'D_lr': 1e-4,
        'batch': 64, 'factual_weight': .5, 'teacher_weight': .5,
        'seed': 260926, 'step_or_hyperparameter_selection': False,
        'candidate_outcomes_read': False, 'environment_branches': 0,
        'matched_data_and_updates_not_compute': True,
        'D_gate': 'support, pure second-chunk teacher-forced and native H2 all strictly improve vs PREFIX_FLOW0; held TF and H2 also beat F0',
        'gate_is_development_screen_not_certificate': True})
    all_metrics, gates, timings = [], [], []
    for anchor_index, (split, sample) in enumerate(ANCHORS):
        case = args.out/f'{split}_s{sample}_m2'
        case.mkdir()
        donor = args.donor_root/f'donor_{split}_seed101_n12_capture'
        # Never encode the withheld second chunk before prefix lift / update.
        prefix_segments = _load_segments(donor, sample, 1, 5, preprocessor, full_resolution=True)
        prefix_fine, prefix_actions = merge_completed_lowlevel(prefix_segments)
        prefix_fine = {key: value.to(device) for key, value in prefix_fine.items()}
        prefix_actions = prefix_actions.to(device)
        prefix_current = {key: value[:, -1:] for key, value in prefix_fine.items()}
        with torch.no_grad():
            prefix_encoded = model.encode_obs(prefix_fine)
        histories = base_histories(model, prefix_encoded, prefix_actions)
        initial_encoded = {key: value[:, -1:] for key, value in prefix_encoded.items()}
        factual, factual_times = factual_queries(model, prefix_encoded, prefix_actions)
        if factual_times != [{'phase': 0, 'start': 0, 'end': 5}]:
            raise AssertionError('prefix supervision crossed cut')
        random = torch.Generator().manual_seed(560926+100*anchor_index)
        train = merge_queries(teacher_single(model, histories, commands(128, 1, prefix_actions, random), random),
                              teacher_chains(model, initial_encoded, commands(32, 5, prefix_actions, random)))
        held_random = torch.Generator().manual_seed(660926+100*anchor_index)
        held = teacher_single(model, histories, commands(64, 1, prefix_actions, held_random), held_random)
        held_plans = commands(16, 5, prefix_actions, held_random)
        teacher_path = native_paths(model, prefix_current, held_plans)
        starts = torch.cat([group[0][:, -1, 0, :OBS_DIM] for group in train.values()], 0)
        center, scale = starts.mean(0), starts.std(0, unbiased=False)
        torch.save({'train': {k: tuple(t.cpu() for t in v) for k, v in train.items()},
                    'held_single': {k: tuple(t.cpu() for t in v) for k, v in held.items()},
                    'held_plans': held_plans.cpu(), 'teacher_path': teacher_path.cpu(),
                    'center': center.cpu(), 'scale': scale.cpu(),
                    'D_train_times': factual_times}, case/'PREFIX_TEACHER_QUERIES.pt')
        students, lift_metrics, traces = {}, [], []
        for kind in ('flow', 'packed'):
            arm_start = time.perf_counter()
            _seed_all(260926+anchor_index)
            student = ControlledGenerator(kind=kind).to(device).configure_normalization(center, scale)
            traces.extend(train_lift(student, train, 760926+anchor_index))
            lift_metrics.append(single_metrics(student, held))
            prediction = native_paths(model, prefix_current, held_plans, student)
            lift_metrics.extend(path_metrics(student, prediction, teacher_path))
            save_student(student, case/f'PREFIX_{kind.upper()}0.pt')
            students[kind+'0'] = student
            timings.append({'split': split, 'sample': sample, 'arm': kind+'0',
                            'seconds': time.perf_counter()-arm_start, 'parameters': student.parameter_count()})
        one = next(row for row in lift_metrics if row['kind'] == 'flow' and row['scope'] == 'single')
        h5 = next(row for row in lift_metrics if row['kind'] == 'flow' and row['scope'] == 'H5')
        lift_pass = one['relative_rms'] <= .25 and h5['relative_rms'] <= .5
        _write_csv(case/'PREFIX_LIFT_HELD.csv', lift_metrics)
        _write_csv(case/'PREFIX_LIFT_TRAINING.csv', traces)
        if not lift_pass:
            gates.append({'split': split, 'sample': sample, 'prefix_lift_pass': False,
                          'D_gate_pass': False, 'decision': 'STOP_AT_PREFIX_F0_LIFT'})
            print('PREFIX STOP', split, sample, 'lift gate', flush=True)
            continue
        update_traces = []
        # Deep copies preserve V0; no label or input from the held interval yet.
        import copy
        for kind in ('flow', 'packed'):
            arm_start = time.perf_counter()
            revised = copy.deepcopy(students[kind+'0'])
            update_traces.extend(adapt_completed(revised, factual, train, 860926+anchor_index))
            save_student(revised, case/f'PREFIX_{kind.upper()}_D.pt')
            students[kind+'D'] = revised
            timings.append({'split': split, 'sample': sample, 'arm': kind+'D',
                            'seconds': time.perf_counter()-arm_start, 'parameters': revised.parameter_count()})
        _write_csv(case/'D_TRAINING.csv', update_traces)
        _write_json(case/'PREFIX_MODELS_SEALED.json', {
            'status': 'SEALED_BEFORE_D_HOLD_READ', 'held_inputs_encoded': False,
            'D_train_times': factual_times, 'future_candidate_truth_read': False})
        # Labels 6..10 are opened only after both prefix models are frozen/saved.
        segments = _load_segments(donor, sample, 2, 5, preprocessor, full_resolution=True)
        fine, actions = merge_completed_lowlevel(segments)
        fine = {key: value.to(device) for key, value in fine.items()}
        actions = actions.to(device)
        with torch.no_grad():
            encoded = model.encode_obs(fine)
        full, times = factual_queries(model, encoded, actions)
        # The sole T=2 example is phase0 5->10 with actual preceding 0->5 cache.
        pure_hold = {2: full[2]}
        hold_overlap = {1: tuple(value[1:] for value in full[1])}
        if full[2][0].shape[0] != 1 or full[1][0].shape[0] != 5:
            raise AssertionError('unexpected two-chunk held native layout')
        initial = {key: value[:, :1] for key, value in fine.items()}
        plan = actions.reshape(1, 2, 10)
        real_endpoint = observation_vector({key: value[:, -1:] for key, value in encoded.items()})
        case_rows, paths = [], {}
        for arm in ('F0', 'flow0', 'flowD', 'packed0', 'packedD'):
            student = None if arm == 'F0' else students[arm]
            for scope, dataset in (('support_0_5', factual), ('hold_TF_5_10', pure_hold),
                                   ('hold_overlapping_1_6_to_4_9', hold_overlap)):
                row = factual_metrics(model, student, dataset, arm)
                row.update(split=split, sample=sample, scope=scope)
                case_rows.append(row)
            predicted = native_paths(model, initial, plan, student)
            paths[arm] = predicted.cpu()
            difference = predicted[:, -1:] - real_endpoint
            v = float(difference[..., :VIS_DIM].square().mean())
            p = float(difference[..., VIS_DIM:].square().mean())
            case_rows.append({'arm': arm, 'queries': 1, 'rms': (v+p)**.5,
                              'visual_mse': v, 'proprio_mse': p,
                              'split': split, 'sample': sample, 'scope': 'hold_self_history_H2_0_10'})
        def get(arm, scope):
            return next(row['rms'] for row in case_rows if row['arm'] == arm and row['scope'] == scope)
        d_pass = all(get('flowD', scope) < get('flow0', scope) for scope in (
            'support_0_5', 'hold_TF_5_10', 'hold_self_history_H2_0_10')) and all(
                get('flowD', scope) < get('F0', scope) for scope in ('hold_TF_5_10', 'hold_self_history_H2_0_10'))
        gates.append({'split': split, 'sample': sample, 'prefix_lift_pass': True,
                      'D_gate_pass': d_pass,
                      'decision': 'ELIGIBLE_FOR_FULL_D_AND_UNSEEN_WORD_TEST' if d_pass else 'STOP_CURRENT_D_RECOVERY_RULE'})
        _write_csv(case/'FACTUAL_TRANSFER.csv', case_rows)
        torch.save(paths, case/'FACTUAL_PATHS.pt')
        all_metrics.extend(case_rows)
        _write_json(case/'SUMMARY.json', {'gate': gates[-1], 'supervised_times': factual_times,
                                         'evaluation_times': times})
        print('PREFIX FINISHED', split, sample, gates[-1], flush=True)
    if versions != [value._version for value in model_tensors] or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('frozen host tensors changed')
    if all_metrics:
        _write_csv(args.out/'FACTUAL_TRANSFER.csv', all_metrics)
    _write_csv(args.out/'GATES.csv', gates)
    _write_csv(args.out/'TIMINGS.csv', timings)
    _write_json(args.out/'RUN_SUMMARY.json', {
        'status': 'PREFIX_FORMATION_COMPLETE_NO_CANDIDATE_TRUTH_READ',
        'prefix_lifts_passed': sum(row['prefix_lift_pass'] for row in gates),
        'D_gates_passed': sum(row['D_gate_pass'] for row in gates),
        'factual_metrics_written': bool(all_metrics),
        'environment_branches': 0, 'seconds': time.perf_counter()-started,
        'host_tensor_versions_unchanged': True, 'host_gradients_absent': True})


@torch.no_grad()
def run_attribution(args):
    """Post-seal evaluation, not another optimization or a changed pass gate."""
    if args.source is None:
        raise ValueError('attribution requires sealed prefix --source')
    summary = json.loads((args.source/'RUN_SUMMARY.json').read_text(encoding='utf-8'))
    if summary['status'] != 'PREFIX_FORMATION_COMPLETE_NO_CANDIDATE_TRUTH_READ':
        raise ValueError('incomplete/unverified prefix source')
    if args.out.exists():
        raise FileExistsError('refusing to overwrite attribution output')
    args.out.mkdir(parents=True)
    _seed_all(260926)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    with patch.object(torch.hub, 'load', _local_hub_loader()):
        model, preprocessor, _ = _load_runtime(args.checkpoint_dir, device)
    tensors = list(model.parameters()) + list(model.buffers())
    versions = [value._version for value in tensors]
    rows = []
    for split, sample in ANCHORS:
        source = args.source/f'{split}_s{sample}_m2'
        if not (source/'PREFIX_MODELS_SEALED.json').is_file():
            continue
        seal = json.loads((source/'PREFIX_MODELS_SEALED.json').read_text(encoding='utf-8'))
        if seal['status'] != 'SEALED_BEFORE_D_HOLD_READ':
            raise ValueError('prefix models not sealed before held opening')
        donor = args.donor_root/f'donor_{split}_seed101_n12_capture'
        segments = _load_segments(donor, sample, 2, 5, preprocessor, full_resolution=True)
        fine, actions = merge_completed_lowlevel(segments)
        fine = {key: value.to(device) for key, value in fine.items()}
        actions = actions.to(device)
        encoded = model.encode_obs(fine)
        factual, _ = factual_queries(model, encoded, actions)
        history, action, real = factual[2]
        f0_tf = model.predict(history)[:, -1:, ..., :OBS_DIM]
        initial = {key: value[:, :1] for key, value in fine.items()}
        plans = actions.reshape(1, 2, 10)
        f0_h2 = native_paths(model, initial, plans)
        saved_h2 = torch.load(source/'FACTUAL_PATHS.pt', map_location=device)
        def rms(left, right):
            return float(native_loss(left, right))**.5
        students = {}
        for tag in ('0', 'D'):
            weights = torch.load(source/f'PREFIX_FLOW{tag if tag == "0" else "_D"}.pt', map_location=device)
            student = ControlledGenerator(kind='flow').to(device)
            student.load_state_dict(weights['state_dict'], strict=True)
            student.eval()
            students[tag] = student
        before_tf = students['0'](history, action)
        before_h2 = native_paths(model, initial, plans, students['0'])
        after_tf = students['D'](history, action)
        after_h2 = native_paths(model, initial, plans, students['D'])
        for tag, tf, h2 in (('0', before_tf, before_h2), ('D', after_tf, after_h2)):
            replay_rms = rms(h2, saved_h2['flow'+tag])
            if replay_rms > 1e-6:
                raise AssertionError('sealed model replay differs from original predictions')
            refined_tf = students[tag](history, action, substeps=2)
            refined_h2 = native_paths(model, initial, plans, students[tag], substeps=2)
            for scope, predicted, refined, teacher, target, previous in (
                ('hold_TF_5_10', tf, refined_tf, f0_tf, real, before_tf),
                ('hold_self_history_H2_0_10', h2[:, -1:], refined_h2[:, -1:],
                 f0_h2[:, -1:], real, before_h2[:, -1:])):
                rows.append({'split': split, 'sample': sample, 'arm': 'flow'+tag, 'scope': scope,
                             'true_outcome_rms': rms(predicted, target),
                             'F0_reference_rms': rms(predicted, teacher),
                             'D_change_from_flow0_rms': rms(predicted, previous),
                             'refinement_delta_rms': rms(predicted, refined),
                             'refined_true_outcome_rms': rms(refined, target),
                             'sealed_path_replay_rms': replay_rms})
        print('POST-SEAL ATTRIBUTION', split, sample, flush=True)
    if not rows:
        raise ValueError('no sealed prefix models for attribution')
    if versions != [value._version for value in tensors] or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('post-seal evaluation changed frozen host')
    _write_csv(args.out/'REPLACEMENT_UPDATE_NUMERICS.csv', rows)
    _write_json(args.out/'RUN_SUMMARY.json', {
        'status': 'POST_SEAL_ATTRIBUTION_COMPLETE', 'source': str(args.source),
        'rows': len(rows), 'training_steps': 0, 'environment_branches': 0,
        'candidate_outcomes_read': False, 'gate_decisions_changed': False,
        'definition_TF': 'same actual held native history/action; F0 is a teacher, not environment truth',
        'definition_H2': 'each operator uses its own native predicted cache from same real initial frame',
        'refinement': 'same saved fields; RK2 substeps 1 versus 2, no fitting',
        'host_tensor_versions_unchanged': True, 'host_gradients_absent': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--donor-root', type=Path, default=Path('artifacts/round_6_shared_revision/results'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('D:/EV-TTT/pushobj_shape_shift'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('lift', 'prefix', 'attribution'), default='lift')
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    {'lift': run, 'prefix': run_prefix, 'attribution': run_attribution}[args.stage](args)
