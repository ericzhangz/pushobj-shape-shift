"""Recompute saved score aggregates and selection identities from CSV only."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import numpy as np

ROOT=Path(__file__).parent
OUT=ROOT/'scored'
def rows(name):
    with (OUT/f'{name}.csv').open(newline='',encoding='utf8') as h:
        return list(csv.DictReader(h))
def mean(rs,k):
    vals=[float(r[k]) for r in rs if r[k]!='']
    return float(np.mean(vals)) if vals else None
errors,pairs,selection,summary,geometry,replay=[rows(n) for n in
    ('ERRORS','PAIRS','SELECTION','SUMMARY','GEOMETRY','REPLAY')]
result=json.loads((OUT/'RESULTS.json').read_text(encoding='utf8'))
checks=[]
def equal(a,b,what):
    if a is None or b is None:
        assert a is b,(what,a,b)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert math.isclose(a,float(b),rel_tol=2e-12,abs_tol=2e-12),(what,a,b)
    else:
        assert a==b,(what,a,b)
    checks.append(what)
arms=result['arms']
equal(len(errors),4*8*len(arms)*5,'errors count')
equal(len(pairs),4*28*len(arms),'pairs count')
equal(len(geometry),4*8*(len(arms)+2)*6,'geometry count')
equal(len(selection),4*len(arms),'selection count')
recomputed=[]
for row in summary:
    case,arm=row['case'],row['arm']
    er=[r for r in errors if r['case']==case and r['arm']==arm]
    pr=[r for r in pairs if r['case']==case and r['arm']==arm]
    gr=[r for r in geometry if r['case']==case and r['arm']==arm and int(r['step'])>0]
    assert {int(r['step']) for r in er}==set(range(1,6))
    assert len({r['candidate'] for r in er})==8
    got={'visual_mse':mean(er,'visual_mse'),'proprio_mse':mean(er,'proprio_mse'),
         'terminal_visual_mse':mean([r for r in er if r['step']=='5'],'visual_mse'),
         'contrast_mse':mean(pr,'contrast_mse'),'pair_gap_mae':mean(pr,'abs_gap_error'),
         'pair_gap_mse':mean(pr,'squared_gap_error'),
         'pair_reversals':sum(r['reversal']=='True' for r in pr),
         'object_centroid_error_px':mean(er,'object_centroid_error_px'),
         'terminal_object_centroid_error_px':mean([r for r in er if r['step']=='5'],'object_centroid_error_px'),
         'centroid_error_count':sum(r['object_centroid_error_px']!='' for r in er),
         'centroid_error_missing':sum(r['object_centroid_error_px']=='' for r in er),
         'predicted_geometry_failures':sum(r['valid']=='False' for r in gr),
         'regret':float(row['selected_cost'])-float(row['best_cost'])}
    sel=next(r for r in selection if r['case']==case and r['arm']==arm)
    js=next(r for r in result['summary'] if r['case']==case and r['arm']==arm)
    for k,v in got.items():
        equal(v,None if row[k]=='' else float(row[k]),f'{case}/{arm}/CSV/{k}')
        equal(v,js[k],f'{case}/{arm}/JSON/{k}')
    equal(float(sel['regret']),got['regret'],f'{case}/{arm}/selection regret')
    # Independently reconstruct all relative candidate costs from pair differences.
    first=pr[0]['left']
    pred={first:0.}
    truth={first:0.}
    for p in pr:
        if p['left']==first:
            pred[p['right']]=-float(p['predicted_gap'])
            truth[p['right']]=-float(p['true_gap'])
        equal(abs(float(p['predicted_gap'])-float(p['true_gap'])),float(p['abs_gap_error']),
              f'{case}/{arm}/{p["left"]}/{p["right"]}/gap')
        equal((float(p['predicted_gap'])-float(p['true_gap']))**2,float(p['squared_gap_error']),
              f'{case}/{arm}/{p["left"]}/{p["right"]}/squaredgap')
    assert len(pred)==len(truth)==8
    equal(pred[sel['selected']],min(pred.values()),f'{case}/{arm}/argmin prediction')
    equal(truth[sel['best']],min(truth.values()),f'{case}/{arm}/argmin truth')
    equal(truth[sel['selected']]-min(truth.values()),got['regret'],f'{case}/{arm}/relative regret')
    for p in pr:
        equal(pred[p['left']]-pred[p['right']],float(p['predicted_gap']),f'{case}/{arm}/pred pair identity')
        equal(truth[p['left']]-truth[p['right']],float(p['true_gap']),f'{case}/{arm}/true pair identity')
    for e in er:
        g=next(r for r in gr if r['candidate']==e['candidate'] and r['step']==e['step'])
        equal(e['object_centroid_error_px'],g['object_centroid_error_px'],f'{case}/{arm}/centroid same frame')
    recomputed.append({'case':case,'arm':arm,**got})
for arm in arms:
    for metric,written in result['arm_means'][arm].items():
        vals=[r[metric] for r in recomputed if r['arm']==arm and r[metric] is not None]
        equal(float(np.mean(vals)) if vals else None,written,f'{arm}/global/{metric}')
for arm in ('REAL','RECON_REAL',*arms):
    gr=[r for r in geometry if r['arm']==arm]
    computed={'attempts':len(gr),'failures':sum(r['valid']=='False' for r in gr),
              'future_attempts':sum(int(r['step'])>0 for r in gr),
              'future_failures':sum(r['valid']=='False' and int(r['step'])>0 for r in gr)}
    for k,v in computed.items():
        equal(v,result['geometry_counts'][arm][k],f'{arm}/geometry/{k}')
tr=[r for r in replay if r['arm']=='TRUTH']
pp=[r for r in replay if r['arm']!='TRUTH']
assert len(tr)==32 and len(pp)==32*len(arms)
truth_max=max(float(r['cost_replay_abs']) for r in tr)
prediction_max=max(float(r['cost_replay_abs']) for r in pp)
anchor_max=max(float(r['anchor_max_abs']) for r in pp)
assert prediction_max<=2e-5 and anchor_max<=2e-5
truth_gate_passed=truth_max<=2e-5
equal(truth_max,result['max_recorded_cost_replay_abs'],'truth replay max')
equal(prediction_max,result['max_prediction_cost_replay_abs'],'prediction replay max')
# Both stage values enter the same terminal branch for the enforced six-frame layout.
assert 2<6-1 and 4<6-1
report={'status':'PASS' if truth_gate_passed else 'AGGREGATES_PASS_REFERENCE_COST_DRIFT',
        'truth_replay_gate_2e_5_passed':truth_gate_passed,
        'truth_replay_rows_exceeding_2e_5':sum(float(r['cost_replay_abs'])>2e-5 for r in tr),
        'scalar_checks':len(checks),'cases':4,'arms':arms,
        'row_counts':dict(errors=len(errors),pairs=len(pairs),geometry=len(geometry),replay=len(replay)),
        'max_truth_cost_replay_abs':truth_max,'max_prediction_cost_replay_abs':prediction_max,
        'max_latent_anchor_abs':anchor_max,'step2_and_step4_terminal_equivalent_at_horizon6':True,
        'summary_and_selection_recomputed':True,'outcome_npz_opened':False,'gpu_calls':0,
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [*(OUT.glob('*.csv')),OUT/'RESULTS.json']}}
with (ROOT/'INTEGRITY_SCORE_CHECKS.json').open('x',encoding='utf8') as h:
    json.dump(report,h,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))
