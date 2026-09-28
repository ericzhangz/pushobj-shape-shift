"""Score sealed extension witnesses; never selects or modifies response parameters."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
BASE = Path(__file__).resolve().parents[1] / 'shared_response_formation_20260928'
sys.path.insert(0, str(BASE))
import score_formation as scoring


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, result):
    with (ROOT/name).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def load():
    seal = json.loads((ROOT/'SEALED.json').read_text())
    assert sha(ROOT/'PREDICTIONS.pt') == seal['prediction_sha256']
    old = json.loads((BASE/'scored_verified/RESULTS.json').read_text())
    target_path = BASE/'scored_verified/TARGETS.pt'
    assert sha(target_path) == old['targets_sha256']
    return (torch.load(ROOT/'PREDICTIONS.pt', map_location='cpu', weights_only=False),
            torch.load(target_path, map_location='cpu', weights_only=False), old)


def moments(error):
    common = error.mean(0)
    total = float(error.square().mean())
    shared = float(common.square().mean())
    differential = float((error-common).square().mean())
    assert abs(total-shared-differential) < 1e-12
    return {'mse': total, 'common_mse': shared, 'centered_mse': differential,
            'all_pairs_mse': 16/7*differential}


def numerical():
    started = time.perf_counter()
    bundle, targets, original = load()
    summaries, steps, pairs, directions, replays = [], [], [], [], []
    for case, content in bundle.items():
        ids = list(content['plans'])
        truth = torch.stack([targets[case][cid]['truth_z']['visual'][0,1:,0].double() for cid in ids])
        true_cost = {cid: targets[case][cid]['recorded_truth_cost'] for cid in ids}
        best = min(true_cost, key=true_cost.get)
        arms = list(content['plans'][ids[0]]['arms'])
        for arm in arms:
            predicted = torch.stack([content['plans'][cid]['arms'][arm]['z']['visual'][0,1:,0].double() for cid in ids])
            error = predicted-truth
            costs = {cid:content['plans'][cid]['arms'][arm]['cost']['total'] for cid in ids}
            selected = min(costs, key=costs.get)
            gaps = [abs(costs[left]-costs[right]-true_cost[left]+true_cost[right])
                    for left,right in itertools.combinations(ids,2)]
            info = {'case':case,'arm':arm,'visual_mse':float(error.square().mean()),
                    'first_step_mse':float(error[:,0].square().mean()),
                    'first_two_mse':float(error[:,:2].square().mean()),
                    'last_three_mse':float(error[:,2:].square().mean()),
                    'terminal_contrast_mse':moments(error[:,-1])['all_pairs_mse'],
                    'pair_gap_mae':sum(gaps)/len(gaps), 'selected':selected,'best':best,
                    'regret':true_cost[selected]-true_cost[best]}
            summaries.append(info)
            for step in range(5):
                steps.append({'case':case,'arm':arm,'step':step+1,**moments(error[:,step])})
            if arm in ('NATIVE','PRIOR','STEP_RIDGE','DIRECT'):
                old = next(r for r in original['summary'] if r['case']==case and r['arm']==arm)
                for key in ('visual_mse','pair_gap_mae','regret'):
                    assert abs(info[key]-old[key]) < 1e-12, (case,arm,key)
            for cid in ids:
                entry = content['plans'][cid]['arms'][arm]
                z = entry['z']
                scoring.check_z(z)
                actual = targets[case][cid]['truth_z']
                anchor = max(float((z[k][:,:1]-actual[k][:,:1]).abs().max()) for k in z)
                assert anchor < 2e-5
                cost = scoring.native_objective_breakdown(z,content['goal_z'],4,alpha=1.,base=2.)
                assert cost['stage']=='terminal'
                drift = max(abs(float(cost[k])-entry['cost'][k]) for k in ('visual','proprio','total'))
                assert drift < 2e-5
                replays.append({'case':case,'candidate':cid,'arm':arm,'anchor_max_abs':anchor,'cost_max_abs':drift})
        prior = torch.stack([content['plans'][cid]['arms']['PRIOR']['z']['visual'][0,1,0].double() for cid in ids])
        for arm in ('STEP_RIDGE','DIRECT'):
            revised = torch.stack([content['plans'][cid]['arms'][arm]['z']['visual'][0,1,0].double() for cid in ids])
            delta, baseline_error = revised-prior, prior-truth[:,0]
            for label, d, e in [('ALL_CANDIDATES',delta,baseline_error)]+[
                    (cid,delta[i],baseline_error[i]) for i,cid in enumerate(ids)]:
                cross = float((e*d).mean())
                energy = float(d.square().mean())
                change = float((e+d).square().mean()-e.square().mean())
                assert abs(change-2*cross-energy)<1e-12
                directions.append({'case':case,'arm':arm,'candidate':label,
                                   'twice_cross_term':2*cross,'update_energy':energy,'mse_change':change,
                                   'oracle_alpha_diagnostic_only':-cross/energy,
                                   'error_update_cosine':float((e*d).sum()/(e.norm()*d.norm()))})
        if case in ('T0','L1'):
            for kind in ('COMMON','DIFFERENTIAL'):
                for eta in ('0p01','0p1','1'):
                    prefix = kind+'_ETA'+eta
                    plus = torch.stack([content['plans'][cid]['arms'][prefix+'_PLUS']['z']['visual'][0,1:,0].double() for cid in ids])
                    minus = torch.stack([content['plans'][cid]['arms'][prefix+'_MINUS']['z']['visual'][0,1:,0].double() for cid in ids])
                    for step in range(5):
                        diff = moments(plus[:,step]-minus[:,step])
                        pairs.append({'case':case,'kind':kind,'eta':float(eta.replace('p','.')),
                                      'step':step+1,'pair_rms':diff['mse']**.5,
                                      'common_rms':diff['common_mse']**.5,
                                      'centered_rms':diff['centered_mse']**.5})
    output = {'summary':summaries,'per_step':steps,'witness_pair_disagreement':pairs,
              'first_step_direction_diagnostic':directions,'replay':replays,
              'prediction_sha256':sha(ROOT/'PREDICTIONS.pt'),
              'targets_sha256':sha(BASE/'scored_verified/TARGETS.pt'),
              'scorer_sha256':sha(Path(__file__)),
              'inherited_truth_cost_replay_failures':4,
              'oracle_alpha_used_to_fit_or_deploy':False,
              'new_predictor_calls':0,'new_environment_calls':0,
              'wall_seconds':time.perf_counter()-started}
    write('RESULTS.json',output)
    print(json.dumps({'case_summaries':summaries,
                      'direction_checks':[r for r in directions if r['candidate']=='ALL_CANDIDATES']},indent=2),flush=True)


@torch.no_grad()
def geometry():
    bundle, targets, _ = load()
    model, _, _ = scoring.runtime()
    versions = [(t,t._version) for t in list(model.parameters())+list(model.buffers())]
    count = {'predictor_calls':0,'encoder_calls':0,'decoder_calls':0,'decoder_frames':0}
    def forbidden(module, inputs):
        count['predictor_calls'] += 1
        raise RuntimeError('Evaluation must not predict or encode.')
    def decoded(module, inputs):
        count['decoder_calls'] += 1
        count['decoder_frames'] += int(inputs[0].shape[0]*inputs[0].shape[1])
    hooks = [model.predictor.register_forward_pre_hook(forbidden),
             model.decoder.register_forward_pre_hook(decoded)]
    rows, summaries, true_failures = [], [], {}
    for case, content in bundle.items():
        start_rows = len(rows)
        true_failures[case] = 0
        for cid, entry in content['plans'].items():
            true_geometry = [scoring.geometry(frame) for frame in targets[case][cid]['rgb']]
            true_failures[case] += sum(not g['valid'] for g in true_geometry[1:])
            for arm, item in entry['arms'].items():
                decoded_obs, _ = model.decode_obs({k:v.cuda() for k,v in item['z'].items()})
                frames = scoring.rgb(decoded_obs['visual'])
                for step in range(1,6):
                    g = scoring.geometry(frames[step])
                    rows.append({'case':case,'candidate':cid,'arm':arm,'step':step,
                                 'valid':g['valid'],'readout_error':g['error'],
                                 'centroid_error_px':scoring.centroid_error(g,true_geometry[step])})
        case_rows = rows[start_rows:]
        for arm in next(iter(content['plans'].values()))['arms']:
            subset = [r for r in case_rows if r['arm']==arm]
            valid = [r['centroid_error_px'] for r in subset if r['centroid_error_px'] is not None]
            summaries.append({'case':case,'arm':arm,'predicted_readout_failures':sum(not r['valid'] for r in subset),
                              'matched_valid_count':len(valid),'matched_missing_count':len(subset)-len(valid),
                              'centroid_error_px':sum(valid)/len(valid) if valid else None})
        print('decoded',case,flush=True)
    for hook in hooks:
        hook.remove()
    assert all(t._version==v for t,v in versions)
    assert count['predictor_calls']==0 and count['decoder_calls']==352 and count['decoder_frames']==2112
    write('GEOMETRY.json',{'summary':summaries,'rows':rows,'true_future_readout_failures':true_failures,
                           'counts':count,'new_environment_calls':0,'backbone_unchanged':True,
                           'prediction_sha256':sha(ROOT/'PREDICTIONS.pt')})
    print(json.dumps({'counts':count,'true_failures':true_failures}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('numerical','geometry'))
    args = parser.parse_args()
    globals()[args.phase]()
