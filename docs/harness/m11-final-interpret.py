"""Pure reductions of prospectively frozen, already measured assessment receipts."""
import json
import math
import os
from pathlib import Path

import torch

from ralup.meta.es import estimate_es
from ralup.policy.checks import _ref,_write
from ralup.records import sha256_file

base=Path(os.environ['M11_ATTEMPT_DIR']);store=base.parent
source=store/'native-final-calibration-v1'
report=json.loads((source/'report.json').read_bytes())
assert report['status']=='final_normalizer_assessment_measured_pending_budget'
for ref in report['runs']+[report['pilot']]:assert sha256_file(ref['path'])==ref['sha256']
g=json.loads(Path(report['pilot']['path']).read_bytes())
phi=torch.tensor(g['phi_before']);epsilon=torch.tensor(g['perturbations'])
plus=torch.tensor([p['plus']['objective'] for p in g['pairs']]);minus=torch.tensor([p['minus']['objective'] for p in g['pairs']])

def grad(a,b,eps=epsilon):return estimate_es(phi,eps,a,b,g['sigma'])
def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return None if denominator==0 else float(torch.dot(a,b))/denominator

full=grad(plus,minus);first=grad(plus[:4],minus[:4],epsilon[:4]);second=grad(plus[4:],minus[4:],epsilon[4:])
panels={}
for label in ('A','B'):
    a=torch.tensor([p[label]['objective'] for p in g['panels'][::2]])
    b=torch.tensor([p[label]['objective'] for p in g['panels'][1::2]])
    ar=torch.tensor([p[label]['delta'] for p in g['panels'][::2]])
    br=torch.tensor([p[label]['delta'] for p in g['panels'][1::2]])
    panels[label]={'differences':(a-b).tolist(),'gradient':grad(a,b).tolist(),
                   'raw_delta_gradient':grad(ar,br).tolist(),'gradient_norm':float(grad(a,b).norm()),
                   'union_cosine':cosine(grad(a,b),full)}
rawplus=torch.tensor([p['plus']['final_query']-p['plus']['initial_query'] for p in g['pairs']])
rawminus=torch.tensor([p['minus']['final_query']-p['minus']['initial_query'] for p in g['pairs']])
runs=[json.loads(Path(ref['path']).read_bytes()) for ref in report['runs']]
all_results=[p[s] for p in g['pairs'] for s in ('plus','minus')]+[r['result'] for r in runs]
clipping={}
for r in runs:
    result=r['result'];counts=[0]*14;scalars=0
    for step in result['work']['steps']:
        feature=step['feature_work'];scalars+=feature['scalars']
        counts=[a+b for a,b in zip(counts,feature['clipped_by_feature'])]
    clipping[r['name']]={'by_feature':counts,'scalars':scalars,'total':sum(counts),'fraction':sum(counts)/scalars}
ranks={str(h):sorted([{'name':r['name'],'objective':r['result']['objective'],'initial':r['result']['initial_query'],
                       'final':r['result']['final_query'],'raw_delta':r['result']['final_query']-r['result']['initial_query'],
                       'elapsed_seconds':r['result']['work']['elapsed_seconds'],'panels':r['panels']} for r in runs if r['name'] in [f'{v}-h{h}' for v in ('initialized','outer_lr003','outer_lr01')]],key=lambda x:x['final']) for h in (16,32,64,496)}
comparisons={}
for horizon in (16,32,64,496):
    initial=next(r for r in runs if r['name']==f'initialized-h{horizon}')
    comparisons[str(horizon)]={}
    for label in ('outer_lr003','outer_lr01'):
        candidate=next(r for r in runs if r['name']==f'{label}-h{horizon}')
        assert candidate['result']['work']['input_sha256']==initial['result']['work']['input_sha256']
        assert candidate['result']['initial_query']==initial['result']['initial_query']
        panels_delta={}
        for panel in ('A','B'):
            a=initial['panels'][0][panel];b=candidate['panels'][0][panel]
            assert a['initial']==b['initial']
            panels_delta[panel]={'initial_query':a['initial'],'initialized_final':a['final'],'candidate_final':b['final'],
                                 'candidate_minus_initialized_final':b['final']-a['final'],
                                 'initialized_raw_change':a['delta'],'candidate_raw_change':b['delta']}
        comparisons[str(horizon)][label]={'combined_final_delta':candidate['result']['final_query']-initial['result']['final_query'],
                                          'paired_panel_endpoint_differences':panels_delta}
summary={'status':'retained_assessment_arithmetic_completed','source_report':_ref(source/'report.json'),'pilot':report['pilot'],
 'observed_rollouts':len(all_results),'optimizer_updates':sum(r['work']['optimizer_steps'] for r in all_results),
 'training_targets':sum(r['work']['training_targets'] for r in all_results),'query_targets':sum(r['work']['query_targets'] for r in all_results),
 'query_evaluations':sum(r['work']['query_evaluations'] for r in all_results),
 'summed_rollout_seconds':sum(r['work']['elapsed_seconds'] for r in all_results),
 'signal':{'sigma':g['sigma'],'pairs':len(g['pairs']),'paired_differences':(plus-minus).tolist(),
           'gradient_norm':float(full.norm()),'independent_halves_cosine':cosine(first,second),'nested_m4_m8_cosine':cosine(first,full),
           'panel_gradients':panels,'panel_A_B_cosine':cosine(torch.tensor(panels['A']['gradient']),torch.tensor(panels['B']['gradient'])),
           'unnormalized_union_cosine':cosine(grad(rawplus,rawminus),full)},
 'rankings':ranks,'comparisons_vs_initialized':comparisons,'clipping_by_feature_and_age':clipping,
 'cost':{'reset_seconds':sum(r['work']['reset_seconds'] for r in all_results),
         'query_seconds':sum(r['work']['query_seconds'] for r in all_results),
         'snapshot_seconds':sum(r['work']['snapshot_seconds'] for r in all_results)},
 'interpretation_contract':'Pure archived arithmetic, zero additional model scoring. Nested estimates are dependent, query panels share inner trajectories, one8pair draw is not convergence. No pilot vector is final-selection eligible; no full-source clone claim transferred from old producer.'}
_write(base/'report.json',summary)
print(json.dumps({'report':_ref(base/'report.json'),'signal':summary['signal'],'rankings':{k:[(r['name'],r['final']) for r in v] for k,v in ranks.items()}}),flush=True)
