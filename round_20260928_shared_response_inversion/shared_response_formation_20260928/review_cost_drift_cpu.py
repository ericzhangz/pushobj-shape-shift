"""CPU sensitivity to old recorded versus newly re-encoded truth costs."""
from pathlib import Path
import itertools
import json
import sys
import torch

ROOT=Path(__file__).parent
REPO=Path('D:/EV-TTT/adajepa_official_51d8665')
sys.path.insert(0,str(REPO))
from research.contrast_probe import native_objective_breakdown
targets=torch.load(ROOT/'scored/TARGETS.pt',map_location='cpu',weights_only=False)
predictions=torch.load(ROOT/'PREDICTIONS.pt',map_location='cpu',weights_only=False)
result={'contract':{'device':'cpu','outcome_npz_opened':False,'new_encoder_calls':0,
                    'target_cost_source':'already scored TARGETS.pt; no re-encoding',
                    'old_truth':'historical ORACLE_COSTS c_env_ref',
                    'new_truth':'current frozen encoder/goal terminal cost'},
        'cases':[],'arm_rows':[],'pair_rows':[],'candidate_rows':[]}
stage_max=0.
for case in targets:
    plans=predictions[case]['plans']
    values=targets[case]
    old={c:values[c]['recorded_truth_cost'] for c in plans}
    new={c:values[c]['truth_cost']['total'] for c in plans}
    best_old=min(old,key=old.get)
    best_new=min(new,key=new.get)
    new_drift=[]
    for c in plans:
        z=values[c]['truth_z']
        goal=predictions[case]['goal_z']
        a=native_objective_breakdown(z,goal,2,alpha=1.,base=2.)
        b=native_objective_breakdown(z,goal,4,alpha=1.,base=2.)
        assert a['stage']==b['stage']=='terminal'
        difference=max(abs(float(a[k])-float(b[k])) for k in ('visual','proprio','total'))
        stage_max=max(stage_max,difference)
        assert difference==0.
        assert float(b['total'])==new[c]
        delta=new[c]-old[c]
        new_drift.append(delta)
        result['candidate_rows'].append({'case':case,'candidate':c,'old':old[c],'new':new[c],
                                        'new_minus_old':delta,'exceeds_2e_5':abs(delta)>2e-5})
    changed=[]
    for left,right in itertools.combinations(plans,2):
        oldgap=old[left]-old[right]
        newgap=new[left]-new[right]
        opposite=oldgap*newgap<0
        sign=lambda x:1 if x>0 else -1 if x<0 else 0
        row={'case':case,'left':left,'right':right,'old_true_gap':oldgap,'new_true_gap':newgap,
             'gap_drift':newgap-oldgap,'opposite_sign':opposite,
             'sign_changed_including_exact_tie':sign(oldgap)!=sign(newgap),
             'opposite_sign_both_gaps_above_1e_6':opposite and abs(oldgap)>1e-6 and abs(newgap)>1e-6}
        if row['sign_changed_including_exact_tie']:
            changed.append(row)
        result['pair_rows'].append(row)
    arms=list(next(iter(plans.values()))['arms'])
    for arm in arms:
        pred={c:plans[c]['arms'][arm]['cost']['total'] for c in plans}
        selected=min(pred,key=pred.get)
        olderrors=[]
        newerrors=[]
        for l,r in itertools.combinations(plans,2):
            predgap=pred[l]-pred[r]
            olderrors.append(abs(predgap-(old[l]-old[r])))
            newerrors.append(abs(predgap-(new[l]-new[r])))
        row={'case':case,'arm':arm,'selected':selected,'old_regret':old[selected]-old[best_old],
             'new_regret':new[selected]-new[best_new],
             'old_pair_gap_mae':sum(olderrors)/len(olderrors),
             'new_pair_gap_mae':sum(newerrors)/len(newerrors)}
        row['regret_change']=row['new_regret']-row['old_regret']
        row['pair_gap_mae_change']=row['new_pair_gap_mae']-row['old_pair_gap_mae']
        result['arm_rows'].append(row)
    result['cases'].append({'case':case,'old_best':best_old,'new_best':best_new,
                            'best_id_changed':best_old!=best_new,
                            'max_abs_candidate_drift':max(abs(x) for x in new_drift),
                            'pair_sign_changes':changed})
result['arm_means']={}
for arm in arms:
    rs=[r for r in result['arm_rows'] if r['arm']==arm]
    result['arm_means'][arm]={k:sum(r[k] for r in rs)/len(rs) for k in
        ('old_regret','new_regret','regret_change','old_pair_gap_mae','new_pair_gap_mae','pair_gap_mae_change')}
result['summary']={'stage2_stage4_max_abs_difference':stage_max,
    'all_truth_stages_terminal':True,'candidate_rows':len(result['candidate_rows']),
    'candidates_exceeding_2e_5':sum(r['exceeds_2e_5'] for r in result['candidate_rows']),
    'max_abs_candidate_drift':max(abs(r['new_minus_old']) for r in result['candidate_rows']),
    'pair_rows':len(result['pair_rows']),
    'pair_opposite_signs':sum(r['opposite_sign'] for r in result['pair_rows']),
    'pair_sign_changes_including_ties':sum(r['sign_changed_including_exact_tie'] for r in result['pair_rows']),
    'pair_opposite_signs_both_gaps_above_1e_6':sum(r['opposite_sign_both_gaps_above_1e_6'] for r in result['pair_rows']),
    'best_candidate_changes':sum(r['best_id_changed'] for r in result['cases']),
    'max_abs_arm_regret_change':max(abs(r['regret_change']) for r in result['arm_rows']),
    'max_abs_arm_gap_mae_change':max(abs(r['pair_gap_mae_change']) for r in result['arm_rows'])}
with (ROOT/'COST_DRIFT_SENSITIVITY.json').open('x',encoding='utf8') as h:
    json.dump(result,h,indent=2)
print(json.dumps({'summary':result['summary'],'cases':result['cases'],'arm_means':result['arm_means']},indent=2))
