"""Read-only split/history/manifest audit. Never decodes test future arrays."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    config = json.loads((ROOT / 'FROZEN_DATA_CONFIG.json').read_text())
    train_hashes = json.loads((ROOT / 'TRAINING_INPUT_HASHES.json').read_text())
    for name, expected in train_hashes.items():
        assert digest(ROOT / name) == expected, name
    code = ast.parse((ROOT / 'fit_models.py').read_text())
    loaded_paths = []
    for node in ast.walk(code):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load_data':
            loaded_paths.extend(n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    assert sorted(loaded_paths) == sorted(['train.npz', 'validation.npz', 'test_public_h2.npz', 'test_public_h5.npz'])

    public_keys = ['history_visual', 'history_proprio', 'history_proprio_latent', 'history_actions',
                   'root_ids', 'episode_ids', 'words', 'macro_actions', 'action_vectors_env']
    datasets = {}
    for name in ['train', 'validation', 'test_public_h2', 'test_public_h5']:
        with np.load(ROOT / (name + '.npz'), allow_pickle=False) as f:
            if name.startswith('test_public'):
                assert not any(k.startswith('future') for k in f.files)
            datasets[name] = {k: f[k] for k in public_keys}
            if name in ('train', 'validation'):
                datasets[name]['future_visual'] = f['future_visual']
                datasets[name]['future_proprio'] = f['future_proprio']
    ep_sets = [set(datasets[k]['episode_ids']) for k in ['train', 'validation', 'test_public_h2']]
    assert all(not ep_sets[i] & ep_sets[j] for i in range(3) for j in range(i))
    for key in public_keys:
        if key != 'words':
            assert np.array_equal(datasets['test_public_h2'][key], datasets['test_public_h5'][key]), key
    for name, data in datasets.items():
        assert np.array_equal(data['action_vectors_env'], np.asarray(config['action_vectors_env'], np.float32))
        assert np.array_equal(data['macro_actions'], datasets['train']['macro_actions'])
        expected_words = config['words_h5' if name.endswith('h5') else 'words_h2']
        assert np.array_equal(data['words'], expected_words)

    manifest_results, source_count, original_hash_count = [], 0, 0
    rows_by_id = {r['root_id']: r for r in config['anchors']}
    for row in config['anchors']:
        for path, expected in row['source_hashes'].items():
            assert digest(path) == expected, path
            original_hash_count += 1
    for name in ['train', 'validation', 'test_h2', 'test_h5']:
        manifest = json.loads((ROOT / (name + '_COLLECTION.json')).read_text())
        # Hash bytes of sealed files without decompressing or reading their values.
        assert digest(ROOT / (name + '.npz')) == manifest['dataset_sha256']
        assert digest(ROOT / 'FROZEN_DATA_CONFIG.json') == manifest['config_sha256']
        public_name = name.replace('test_', 'test_public_') if name.startswith('test') else name
        data = datasets[public_name]
        assert [g['root_id'] for g in manifest['gates']] == data['root_ids'].tolist()
        assert all(g[k] is True for g in manifest['gates'] for k in
                   ['anchor_exact', 'complete_repeat_exact', 'shared_word_prefix_exact'])
        expected_roots = [r['root_id'] for r in config['anchors']
                          if r['split'] == ('test' if name.startswith('test') else name)]
        assert data['root_ids'].tolist() == expected_roots
        for i, root in enumerate(data['root_ids']):
            row = rows_by_id[root]
            donor = Path(row['donor'])
            history_path = donor / 'real_evidence' / f"s{row['sample']}_history_mpc{row['mpc']}.pt"
            hist = torch.load(history_path, map_location='cpu', weights_only=False)
            prefix = hist['executed_prefix'].numpy()
            assert prefix.shape == (1, row['mpc'], 10)
            assert np.array_equal(prefix[0, -2:], data['history_actions'][i])
            past_props, past_actions = [], []
            for chunk in range(row['mpc']):
                source = donor / 'real_evidence' / f"s{row['sample']}_executed_mpc{chunk}.npz"
                with np.load(source, allow_pickle=False) as f:
                    prop = f['proprio']
                    action = f['normalized_model_actions']
                if chunk:
                    assert np.array_equal(past_props[-1][-1], prop[0])
                past_props.append(prop if chunk == 0 else prop[1:])
                past_actions.append(action)
            past_props = np.concatenate(past_props)
            completed = np.concatenate(past_actions, axis=1)
            assert np.array_equal(prefix, completed)
            indices = np.asarray([row['mpc'] - 2, row['mpc'] - 1, row['mpc']]) * 5
            assert np.array_equal(past_props[indices], data['history_proprio'][i])
            assert np.array_equal(past_props[-1], np.asarray(hist['current_real_observation']['proprio'])[0, -1])
            source_count += 1
        manifest_results.append({'dataset': name, 'roots': len(expected_roots),
                                 'episodes': len(set(data['episode_ids'])), 'all_manifest_gates_pass': True,
                                 'actual_npz_and_config_hashes_match': True})

    prefix_errors = {}
    for name in ['train', 'validation']:
        d = datasets[name]
        lookup = {tuple(w): j for j, w in enumerate(d['words'])}
        vm, pm = 0.0, 0.0
        for a in range(5):
            v = np.stack([d['future_visual'][:, lookup[a, b], 0] for b in range(5)])
            p = np.stack([d['future_proprio'][:, lookup[a, b], 0] for b in range(5)])
            vm = max(vm, float(np.max(np.abs(v-v[:1]))))
            pm = max(pm, float(np.max(np.abs(p-p[:1]))))
        assert vm < 2e-5 and pm == 0
        prefix_errors[name] = {'visual_encoded_max_error': vm, 'raw_proprio_max_error': pm}
    summary = {'status': 'PASS', 'test_future_arrays_opened': False,
               'test_npz_hash_bytes_read_only': True, 'native_model_calls': 0, 'environment_calls': 0,
               'training_load_data_targets': sorted(loaded_paths),
               'training_input_hashes_match': True, 'episode_splits_disjoint': True,
               'dataset_manifests': manifest_results, 'factual_history_checks': source_count,
               'original_frozen_source_hash_checks': original_hash_count,
               'test_h2_h5_public_history_identical': True, 'dictionary_words_match_frozen_config': True,
               'train_validation_prefix_checks': prefix_errors,
               'test_common_prefix_verification': 'manifest and source-code gate only; no future values read',
               'caveat': 'Test has 4 episode clusters, not 8 independent roots. Old episodes were development assets.'}
    (ROOT / 'INDEPENDENT_DATA_AUDIT.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
