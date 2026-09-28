"""CPU-only independent count and chronology audit of the prepared cache."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).parent
prepared = json.loads((ROOT/'PREPARE.json').read_text(encoding='utf8'))
path = ROOT/'PREPARED.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest() == prepared['sha256']
episodes = torch.load(path,map_location='cpu',weights_only=False)
audit = json.loads((ROOT/'DATA_AUDIT.json').read_text(encoding='utf8'))
expected = {r['case']:r for r in audit['episodes']}
assert set(expected)==set(episodes)
rows=[]
for key, ep in episodes.items():
    n=ep['frames']
    assert n==expected[key]['unique_frames']
    assert ep['actions'].shape==(n-1,2)
    assert ep['z']['visual'].shape==(1,n,1,384)
    assert ep['z']['proprio'].shape==(1,n,10)
    assert all(v.device.type=='cpu' and torch.isfinite(v).all() for v in ep['z'].values())
    assert torch.isfinite(ep['actions']).all()
    starts=list(range(n-15))
    for start in starts:
        assert start+15<n
        assert ep['actions'][start:start+15].numel()==30
    rows.append({'case':key,'frames':n,'full_history_one_step_windows':len(starts),
                 'maximum_target_frame':max((s+15 for s in starts),default=None),
                 'sample':ep['sample']})
priors=json.loads((ROOT/'PRIORS.json').read_text(encoding='utf8'))
counts={shape:sum(r['full_history_one_step_windows'] for r in rows if r['case'][0]==shape and r['sample']>=2)
        for shape in ('T','L')}
for shape,count in counts.items():
    assert priors[shape]['windows']==count
    for choice in priors[shape]['cv']:
        assert len(choice['episode_mse'])==10
    best=min(priors[shape]['cv'],key=lambda r:r['mean_episode_mse'])['alpha']
    assert best==priors[shape]['alpha']
out={'prepared_digest_matches':True,'cache_counts_match_independent_inventory':True,
     'offline_windows':counts,'offline_windows_total':sum(counts.values()),
     'loo_folds_per_shape':10,'loo_choice_matches_reported':True,'episode_rows':rows,
     'query_outcomes_opened':False,'gpu_calls':0,'model_calls':0}
with (ROOT/'INTEGRITY_COUNTS.json').open('x',encoding='utf8') as handle:
    json.dump(out,handle,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='episode_rows'},indent=2))
