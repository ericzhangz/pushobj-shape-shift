"""Summarize recorded paths and observed query history without model calls."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parent
BASE = Path(__file__).resolve().parents[1] / 'shared_response_formation_20260928'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = json.loads((ROOT/'SEALED.json').read_text())
    assert sha(ROOT/'TRACES.pt') == seal['traces_sha256']
    prepared = BASE/'PREPARED.pt'
    assert sha(prepared) == seal['input_sha256'][str(prepared)]
    data = torch.load(prepared, map_location='cpu', weights_only=False)
    traces = torch.load(ROOT/'TRACES.pt', map_location='cpu', weights_only=False)
    cases = {}
    for case, entry in traces.items():
        # The completed support ends at frame 20. Query starts from observed
        # frames 10, 15, 20, not the generated final support cache.
        cache = {}
        for key in ('visual', 'proprio'):
            generated = entry['support_direct'][key][:, -3:].double()
            observed = data[case]['z'][key][:, [10, 15, 20]].double()
            diff = generated-observed
            cache[key] = {'mse':float(diff.square().mean()),
                          'max_abs':float(diff.abs().max())}
        queries = list(entry['queries'].values())
        direct = torch.stack([q['update_effect'] for q in queries])
        feedback = torch.stack([q['history_feedback_effect'] for q in queries])
        steps = []
        for t in range(5):
            d, h = direct[:, :, t], feedback[:, :, t]
            energy = float(d.square().mean()+h.square().mean()+2*(d*h).mean())
            assert abs(energy-float((d+h).square().mean())) < 1e-12
            steps.append({'step':t+1, 'same_state_update_rms':float(d.square().mean())**.5,
                          'generated_history_effect_rms':float(h.square().mean())**.5,
                          'total_change_rms':float((d+h).square().mean())**.5,
                          'twice_cross_term':float(2*(d*h).mean())})
        assert steps[0]['generated_history_effect_rms'] < 2e-5
        cases[case] = {'support_end_vs_observed_query_cache':cache, 'steps':steps}
    result = {'cases':cases, 'query_future_truth_read':False, 'new_model_calls':0,
              'hashes':{str(p):sha(p) for p in (ROOT/'TRACES.pt',prepared,Path(__file__))}}
    with (ROOT/'PATH_DIAGNOSTICS.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(cases,indent=2))


if __name__ == '__main__':
    main()
