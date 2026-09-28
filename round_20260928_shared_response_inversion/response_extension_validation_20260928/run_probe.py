"""Audit existing native response extension; never opens query outcome payloads."""
from pathlib import Path
import json
import math
import sys
import time

import torch

BASE = Path(__file__).resolve().parents[1] / 'shared_response_formation_20260928'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import formation as f

TOL = 2e-5
ETAS = (.01, .1, 1.)


def write(name, value):
    with (OUT / name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def canonical(vector):
    vector = vector / vector.norm()
    return vector if vector[vector.abs().argmax()] >= 0 else -vector


def max_error(left, right):
    return max(float((left[k].double() - right[k].double()).abs().max()) for k in left)


def support_stats(result, target, theta, prior, lam):
    data = float((result['visual'][0, 3:, 0].double() - target.double()).square().sum())
    penalty = lam * float((theta.double() - prior.double()).square().sum())
    return {'data_sum': data, 'mse': data / (2 * f.Q), 'penalty': penalty,
            'hard_objective': data + penalty}


def package(result, goal):
    trimmed = {k: v[:, 2:] for k, v in result.items()}
    score = f.native_objective_breakdown(trimmed, goal, 4, alpha=1., base=2.)
    return {'z': f.cpu(trimmed),
            'cost': {k: float(score[k]) for k in ('visual', 'proprio', 'total')}}


def feature_matrix(trace):
    return torch.cat([row['feature'].cpu().double() for row in trace], 0).T


def offline_directions(model, data, counter):
    directions, records = {}, {}
    for shape in ('T', 'L'):
        counter.phase = 'offline_direction_' + shape
        residuals = []
        for sample in range(2, 12):
            episode = data[f'{shape}{sample}']
            for start in range(episode['frames'] - 15):
                initial = f.episode_view(episode, [start, start + 5, start + 10])
                action = episode['actions'][start:start + 15].reshape(1, 3, 10).cuda()
                pred = f.cached_rollout(model, initial, action)['visual'][0, -1, 0].cpu()
                residuals.append(episode['z']['visual'][0, start + 15, 0] - pred)
        W = torch.stack(residuals, 1).double()
        left, singular, _ = torch.linalg.svd(W, full_matrices=False)
        directions[shape] = canonical(left[:, 0])
        records[shape] = {'windows': W.shape[1], 'leading_singular_value': float(singular[0]),
                          'leading_energy_fraction': float(singular[0].square() / W.square().sum())}
    return directions, records


@torch.no_grad()
def main():
    if (OUT / 'PREDICTIONS.pt').exists() or (OUT / 'AUDIT.json').exists():
        raise FileExistsError('This bounded run already has outputs.')
    started = time.perf_counter()
    inputs = [BASE / 'PREPARED.pt', BASE / 'PREDICTIONS.pt', BASE / 'formation.py',
              OUT / 'RUN_PROTOCOL.md', Path(__file__), BASE / 'PRIORS.json']
    old_seal = json.loads((BASE / 'SEALED.json').read_text())
    assert f.sha(BASE / 'PREPARED.pt') == old_seal['prepared_sha256']
    assert f.sha(BASE / 'PREDICTIONS.pt') == old_seal['prediction_sha256']
    assert f.sha(BASE / 'formation.py') == old_seal['runner_sha256']
    preparation = json.loads((BASE / 'PREPARE.json').read_text())
    for name, digest in preparation['input_hashes'].items():
        if name.replace('\\', '/').endswith(('models/visual_world_model.py', 'model_latest.pth', 'hydra.yaml')):
            assert f.sha(name) == digest, name
            inputs.append(Path(name))
    data = torch.load(BASE / 'PREPARED.pt', map_location='cpu', weights_only=False)
    saved = torch.load(BASE / 'PREDICTIONS.pt', map_location='cpu', weights_only=False)
    priors_info = json.loads((BASE / 'PRIORS.json').read_text())
    model, _ = f.runtime()
    versions = [(tensor, tensor._version) for tensor in list(model.parameters()) + list(model.buffers())]
    counter = f.Counter(model)
    output_vectors, offline = offline_directions(model, data, counter)
    results, audit, traces, recipes, checks = {}, {}, {}, {}, []
    for shape, sample in f.CASES:
        case = f'{shape}{sample}'
        print('CASE', case, flush=True)
        prior_path, direct_path = BASE / f'PRIOR_{shape}.pt', BASE / f'{case}_DIRECT.pt'
        inputs += [prior_path, direct_path]
        prior = torch.load(prior_path, map_location='cpu', weights_only=False).cuda()
        theta = torch.load(direct_path, map_location='cpu', weights_only=False).cuda()
        lam = float(priors_info[shape]['alpha'])
        assert lam == 10.
        initial, actions, target = f.support_view(data[case])
        counter.phase = case + '_support'
        trace = []
        support = f.cached_rollout(model, initial, actions, theta=theta, trace=trace)
        V = feature_matrix(trace)
        basis, singular, _ = torch.linalg.svd(V, full_matrices=False)
        rank = int((singular > singular[0] * 1e-12).sum())
        basis = basis[:, :rank]
        delta = theta.cpu().double() - prior.cpu().double()
        projected_delta = (delta @ basis) @ basis.T
        center = (prior.cpu().double() + projected_delta).float().cuda()
        center_trace = []
        center_support = f.cached_rollout(model, initial, actions, theta=center, trace=center_trace)
        support_difference = max_error(support, center_support)
        assert support_difference < TOL, (case, 'projection support', support_difference)
        actual = support_stats(support, target, theta, prior, lam)
        centered = support_stats(center_support, target, center, prior, lam)
        discarded = delta - projected_delta
        audit[case] = {'support_feature_rank': rank, 'support_singular_values': singular.tolist(),
                       'support_feature_condition': float(singular[0] / singular[-1]),
                       'projection_relative_residual': float(discarded.norm() / delta.norm()),
                       'projection_predicted_objective_reduction': lam * float(discarded.square().sum()),
                       'projection_support_max_abs': support_difference,
                       'direct_support': actual, 'projected_support': centered,
                       'query_direct_replay_max_abs': 0.}
        torch.save(center.cpu(), OUT / f'{case}_PROJECTED.pt')
        query_initial = f.episode_view(data[case], [10, 15, 20])
        previous_actions = data[case]['actions'][10:20].reshape(1, 2, 10).cuda()
        goal = f.dev(saved[case]['goal_z'])
        results[case] = {'goal_z': saved[case]['goal_z'], 'plans': {}}
        traces[case] = {'support_features': V, 'support_basis': basis,
                        'support_direct': f.cpu(support), 'support_projected': f.cpu(center_support),
                        'queries': {}}
        first_features = []
        counter.phase = case + '_queries'
        for cid, old in saved[case]['plans'].items():
            aligned = torch.cat([previous_actions, old['actions'].cuda()], 1)
            direct_trace = []
            direct_result = f.cached_rollout(model, query_initial, aligned, theta=theta, trace=direct_trace)
            direct_pack = package(direct_result, goal)
            drift = max_error(direct_pack['z'], old['arms']['DIRECT']['z'])
            assert drift < TOL, (case, cid, 'DIRECT replay', drift)
            audit[case]['query_direct_replay_max_abs'] = max(audit[case]['query_direct_replay_max_abs'], drift)
            projected_result = f.cached_rollout(model, query_initial, aligned, theta=center)
            results[case]['plans'][cid] = {
                'actions': old['actions'],
                'arms': {**{a: old['arms'][a] for a in ('NATIVE', 'PRIOR', 'STEP_RIDGE')},
                         'DIRECT': direct_pack, 'PROJECTED': package(projected_result, goal)}}
            features = feature_matrix(direct_trace)
            first_features.append(features[:, 0])
            effects = delta @ features
            supported = projected_delta @ features
            block_edges = (0, 1, 1213, 1223, 13343)
            blocks = torch.stack([delta[:, block_edges[i]:block_edges[i+1]] @
                                   features[block_edges[i]:block_edges[i+1]] for i in range(4)])
            direct_y = direct_pack['z']['visual'][0, 1:, 0].double().T
            prior_y = old['arms']['PRIOR']['z']['visual'][0, 1:, 0].double().T
            native_y = torch.cat([row['native'].cpu() for row in direct_trace], 1)[0, :, 0, :f.Q].double().T
            # Same-state update plus all change induced through the generated history.
            feedback = direct_y - prior_y - effects
            traces[case]['queries'][cid] = {
                'features': features, 'native': native_y, 'update_effect': effects,
                'support_projected_effect': supported, 'orthogonal_effect': effects-supported,
                'feature_block_effects': blocks, 'history_feedback_effect': feedback,
                'support_coordinates': torch.linalg.pinv(V) @ features,
                'feature_kernel_with_support': V.T @ features}
            assert float((blocks.sum(0)-effects).abs().max()) < 1e-10
        Q = torch.stack(first_features, 1)
        NQ = Q - basis @ (basis.T @ Q)
        mean_null = NQ.mean(1)
        centered_null = NQ - mean_null[:, None]
        left_null, singular_null, _ = torch.linalg.svd(centered_null, full_matrices=False)
        audit[case]['first_query_feature_norms'] = Q.norm(dim=0).tolist()
        audit[case]['first_query_null_norms'] = NQ.norm(dim=0).tolist()
        audit[case]['first_query_null_fractions'] = (NQ.norm(dim=0)/Q.norm(dim=0)).tolist()
        audit[case]['common_null_norm'] = float(mean_null.norm())
        audit[case]['centered_null_singular_values'] = singular_null.tolist()
        audit[case]['exact_first_step_bounds'] = []
        for eta in (0., *ETAS):
            rho = math.sqrt(eta * centered['hard_objective'] / lam)
            audit[case]['exact_first_step_bounds'].append({
                'eta': eta, 'rho': rho,
                'per_candidate_pair_rms_max': (2*rho*NQ.norm(dim=0)/math.sqrt(f.Q)).tolist(),
                'common_pair_rms_max': 2*rho*float(mean_null.norm())/math.sqrt(f.Q),
                'centered_pair_rms_max': 2*rho*float(singular_null[0])/math.sqrt(8*f.Q)})
        # Pure posterior averaging cannot alter the mean of a fixed first-step affine law.
        assert float(traces[case]['queries'][next(iter(saved[case]['plans']))]['history_feedback_effect'][:, 0].abs().max()) < TOL
        if case not in ('T0', 'L1'):
            print('DONE', case, 'projection', audit[case]['projection_relative_residual'], flush=True)
            continue
        assert float(mean_null.norm()) > 1e-12 and float(singular_null[0]) > 1e-12
        directions = {'COMMON': canonical(mean_null), 'DIFFERENTIAL': canonical(left_null[:, 0])}
        recipes[case] = {'output_direction': output_vectors[shape], 'input_directions': directions,
                         'center_hard_objective': centered['hard_objective'], 'lambda': lam, 'etas': ETAS}
        counter.phase = case + '_perturbations'
        for kind, v in directions.items():
            assert float((v @ V).abs().max()) < 1e-10
            u = output_vectors[shape]
            for eta in ETAS:
                rho = math.sqrt(eta * centered['hard_objective'] / lam)
                name = f'{kind}_ETA{eta:g}'.replace('.', 'p')
                for sign, suffix in ((1., 'PLUS'), (-1., 'MINUS')):
                    arm = name + '_' + suffix
                    varied = center + (sign * rho * torch.outer(u, v)).float().cuda()
                    varied_support = f.cached_rollout(model, initial, actions, theta=varied)
                    difference = max_error(varied_support, center_support)
                    assert difference < TOL, (case, arm, 'support invariance', difference)
                    stats = support_stats(varied_support, target, varied, prior, lam)
                    stats.update({'case': case, 'arm': arm, 'eta': eta, 'rho': rho,
                                  'support_max_abs': difference,
                                  'expected_objective_increment': lam*rho*rho,
                                  'actual_objective_increment': stats['hard_objective']-centered['hard_objective']})
                    checks.append(stats)
                    for cid, old in saved[case]['plans'].items():
                        aligned = torch.cat([previous_actions, old['actions'].cuda()], 1)
                        varied_result = f.cached_rollout(model, query_initial, aligned, theta=varied)
                        results[case]['plans'][cid]['arms'][arm] = package(varied_result, goal)
                plus = torch.stack([entry['arms'][name+'_PLUS']['z']['visual'][0,1,0].double()
                                    for entry in results[case]['plans'].values()])
                minus = torch.stack([entry['arms'][name+'_MINUS']['z']['visual'][0,1,0].double()
                                     for entry in results[case]['plans'].values()])
                expected = 2*rho*torch.outer(Q.T @ v, u)
                agreement = float((plus-minus-expected).abs().max())
                assert agreement < TOL, (case, name, 'first affine pair', agreement)
                audit[case].setdefault('first_pair_formula_checks', []).append({'arm_pair':name,'max_abs':agreement})
        print('DONE', case, 'projection', audit[case]['projection_relative_residual'], flush=True)
    assert all(tensor._version == version for tensor, version in versions)
    count = sum(row['batch_transitions'] for row in counter.rows.values())
    assert count == 1549 and count <= 1600, (count, counter.rows)
    torch.save(results, OUT/'PREDICTIONS.pt')
    torch.save(traces, OUT/'TRACES.pt')
    torch.save(recipes, OUT/'RECIPES.pt')
    write('AUDIT.json', {'cases':audit,'offline_output_directions':offline,'perturbations':checks})
    write('SEALED.json', {'prediction_sha256':f.sha(OUT/'PREDICTIONS.pt'),
                          'traces_sha256':f.sha(OUT/'TRACES.pt'), 'recipes_sha256':f.sha(OUT/'RECIPES.pt'),
                          'input_sha256':{str(path):f.sha(path) for path in inputs},
                          'counts':counter.rows,'sample_transitions':count,
                          'wall_seconds':time.perf_counter()-started,
                          'query_outcomes_read':False,'environment_calls':0,'training_steps':0,
                          'backbone_parameters_buffers_unchanged':True})
    print('SEALED', count, 'sample transitions', flush=True)


if __name__ == '__main__':
    main()
