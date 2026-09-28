"""Independently recompute support/extension diagnostics from sealed audit traces."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = json.loads((ROOT/'SEALED.json').read_text())
    assert sha(ROOT/'TRACES.pt') == seal['traces_sha256']
    assert sha(ROOT/'PREDICTIONS.pt') == seal['prediction_sha256']
    traces = torch.load(ROOT/'TRACES.pt', map_location='cpu', weights_only=False)
    predictions = torch.load(ROOT/'PREDICTIONS.pt', map_location='cpu', weights_only=False)
    audit = json.loads((ROOT/'AUDIT.json').read_text())
    output = {}
    for case, entry in traces.items():
        queries = list(entry['queries'].values())
        effects = torch.stack([q['update_effect'][:, 0] for q in queries])
        coordinates = torch.stack([q['support_coordinates'][:, 0] for q in queries])
        common = effects.mean(0)
        total = float(effects.square().mean())
        common_energy = float(common.square().mean())
        orth = torch.stack([q['orthogonal_effect'][:, 0] for q in queries])
        blocks = torch.stack([q['feature_block_effects'][:, :, 0] for q in queries])
        V = entry['support_features']
        mean_checks = []
        if case in ('T0', 'L1'):
            for kind in ('COMMON', 'DIFFERENTIAL'):
                for eta in ('0p01', '0p1', '1'):
                    prefix = f'{kind}_ETA{eta}'
                    pairs = []
                    for candidate in predictions[case]['plans'].values():
                        arms = candidate['arms']
                        mean_pair = (arms[prefix+'_PLUS']['z']['visual'].double()+
                                     arms[prefix+'_MINUS']['z']['visual'].double())/2
                        center = arms['PROJECTED']['z']['visual'].double()
                        pairs.append(float((mean_pair[:, 1] - center[:, 1]).abs().max()))
                    assert max(pairs) < 2e-5
                    mean_checks.append({'pair':prefix,'first_step_mean_vs_center_max_abs':max(pairs)})
        projection_query = max(float((candidate['arms']['DIRECT']['z'][key].double()-
                                      candidate['arms']['PROJECTED']['z'][key].double()).abs().max())
                               for candidate in predictions[case]['plans'].values()
                               for key in ('visual','proprio'))
        output[case] = {
            'first_update_rms': total**.5,
            'first_update_common_energy_fraction': common_energy/total,
            'first_update_centered_rms': float((effects-common).square().mean())**.5,
            'first_update_orthogonal_rms': float(orth.square().mean())**.5,
            'first_update_block_rms_nonadditive': {
                name:float(blocks[:,i].square().mean())**.5
                for i,name in enumerate(('constant','state','action','state_action'))},
            'support_coordinates_mean':coordinates.mean(0).tolist(),
            'support_coordinates_min':coordinates.min(0).values.tolist(),
            'support_coordinates_max':coordinates.max(0).values.tolist(),
            'support_gram':(V.T@V).tolist(),
            'projection_query_max_abs':projection_query,
            'first_step_symmetric_mean_checks':mean_checks,
        }
    checks = audit['perturbations']
    result = {'cases':output,
              'max_perturbed_support_token_difference':max(r['support_max_abs'] for r in checks),
              'max_penalty_budget_formula_error':max(abs(r['expected_objective_increment']-
                  r['actual_objective_increment']) for r in checks),
              'sample_transitions':seal['sample_transitions'],
              'input_hashes':{n:sha(ROOT/n) for n in ('AUDIT.json','TRACES.pt','PREDICTIONS.pt','SEALED.json')},
              'query_truth_read':False}
    with (ROOT/'DIAGNOSTICS.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
