"""Read-only CPU inventory. Query outcome NPY headers only, never payloads."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np

ROOT = Path('D:/EV-TTT/adajepa_official_51d8665/artifacts/round_6_shared_revision/results')
OUT = Path(__file__).parent


def npz_headers(path):
    rows = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.namelist():
            with archive.open(entry) as handle:
                version = np.lib.format.read_magic(handle)
                shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
            rows[entry[:-4]] = {'shape': list(shape), 'dtype': str(dtype), 'fortran': fortran}
    return rows


def main():
    result = {'contract': {'model_calls': 0, 'environment_calls': 0,
                         'training_steps': 0, 'outcome_payload_read': False,
                         'query_anchor': 'mpc4, frame20, 2s',
                         'legal_support': 'sample0/1 executed chunks0..3 only'},
              'episodes': [], 'query_outcome_headers': [], 'duplicates': []}
    for shape in ('T', 'L'):
        donor = ROOT / f'donor_{shape}_seed101_n12_capture'
        for sample in range(12):
            paths = sorted((donor/'real_evidence').glob(f's{sample}_executed_mpc*.npz'),
                           key=lambda p: int(re.search(r'mpc(\d+)', p.name)[1]))
            legal = [p for p in paths if sample >= 2 or int(re.search(r'mpc(\d+)', p.name)[1]) < 4]
            frames, props, actions = [], [], []
            for idx, path in enumerate(legal):
                with np.load(path, allow_pickle=False) as archive:
                    rgb, prop, action = [archive[k].copy() for k in ('visual','proprio','normalized_model_actions')]
                assert rgb.shape == (6,224,224,3) and prop.shape == (6,4) and action.shape == (1,1,10)
                if idx:
                    assert np.array_equal(frames[-1][-1], rgb[0])
                    assert np.array_equal(props[-1][-1], prop[0])
                frames.append(rgb if not idx else rgb[1:])
                props.append(prop if not idx else prop[1:])
                actions.append(action)
            n = len(legal)*5
            rgb = np.concatenate(frames)
            acts = np.concatenate(actions, axis=1)
            row = {'case': shape+str(sample), 'role': 'online_support' if sample<2 else 'offline_prior',
                   'available_chunks': len(paths), 'legal_chunks': len(legal),
                   'legal_paths': [str(p) for p in legal], 'unique_frames': len(rgb),
                   'lowlevel_transitions': n, 'duration_seconds': n/10,
                   'full_history_single_targets_all_phases': max(0,n-14),
                   'phase0_full_history_future_chain': max(0,len(legal)-2),
                   'phase0_anchor_frame': 10,
                   'phase0_future_frames': list(range(15,n+1,5)),
                   'H2_windows_all_phases': max(0,n-19),
                   'H3_windows_all_phases': max(0,n-24),
                   'initial_rgb_sha256': hashlib.sha256(rgb[0].tobytes()).hexdigest(),
                   'action_sha256': hashlib.sha256(acts.tobytes()).hexdigest()}
            result['episodes'].append(row)
            if sample<2:
                base = ROOT / f'evaluate_{shape}_seed101_s{sample}_m4_eight/branch_sidecars'
                outcomes = sorted(base.glob('*_outcome.npz'))
                assert len(outcomes) == 8
                for path in outcomes:
                    heads = npz_headers(path)
                    assert heads['visual']['shape'] == [6,224,224,3]
                    assert heads['proprio']['shape'] == [6,4]
                    result['query_outcome_headers'].append({'case': row['case'], 'path': str(path), 'fields': heads})
    for key in ('initial_rgb_sha256','action_sha256'):
        seen = {}
        for row in result['episodes']:
            if row[key] in seen:
                result['duplicates'].append({'hash_field': key, 'cases': [seen[row[key]],row['case']]})
            seen[row[key]]=row['case']
    result['totals']={}
    for role in ('online_support','offline_prior'):
        rows=[r for r in result['episodes'] if r['role']==role]
        result['totals'][role]={'episodes': len(rows), **{key:sum(r[key] for r in rows) for key in
            ('legal_chunks','unique_frames','lowlevel_transitions','full_history_single_targets_all_phases',
             'phase0_full_history_future_chain','H2_windows_all_phases','H3_windows_all_phases')}}
    result['query_candidate_count']=len(result['query_outcome_headers'])
    path=OUT/'DATA_AUDIT.json'
    with path.open('x',encoding='utf8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps({'totals':result['totals'],'duplicates':result['duplicates'],
                      'query_candidate_count':result['query_candidate_count']},indent=2))


if __name__=='__main__':
    main()
