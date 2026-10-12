"""S01–S06 paired endpoint inference; frozen-policy engineering scope is explicit."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import binomtest, rankdata, t

from ..config import ConfirmationManifest
from ..records import ArtifactRef, read_artifact


@dataclass(frozen=True)
class PairedResults:
    manifest_sha256: str
    outcomes: list[dict]
    quality_cost_curves: list[dict] | None = None
    project_costs: dict | None = None

    @classmethod
    def from_artifact(cls, reference: ArtifactRef):
        value=read_artifact(reference)
        if value.get('schema')!='ralup.synthetic_paired_results.v1' or value.get('synthetic') is not True:
            raise ValueError('Authenticated synthetic paired result required')
        return cls(value['manifest_sha256'],value['outcomes'],value['quality_cost_curves'],value['project_costs'])


def _interval(values):
    n=len(values)
    mean=float(np.mean(values)) if n else None
    if n<2:
        return {'n':n,'mean_delta':mean,'interval':None,'ratio':None if mean is None else 2**mean,
                'ratio_interval':None,'sample_sd':None,'t_critical':None,'degenerate_variance':False}
    sd=float(np.std(values,ddof=1));critical=float(t.ppf(.9875,n-1))
    radius=critical*sd/math.sqrt(n);interval=[mean-radius,mean+radius]
    return {'n':n,'mean_delta':mean,'interval':interval,'ratio':2**mean,
            'ratio_interval':[2**x for x in interval],'sample_sd':sd,'t_critical':critical,
            'degenerate_variance':sd==0}


def _rank(values, rng):
    nonzero=np.asarray([v for v in values if v!=0],dtype=float)
    if not len(nonzero):return {'p_two_sided':1.,'nonzero_n':0,'zeros':len(values),'method':'exact tied midranks'}
    ranks=np.rint(2*rankdata(abs(nonzero),method='average')).astype(int)
    observed=int(np.dot(ranks,np.sign(nonzero)));total=int(ranks.sum())
    if len(ranks)<=50:
        counts={0:1}
        for rank in ranks:
            next_counts={}
            for key,count in counts.items():
                for new in (key-int(rank),key+int(rank)):
                    next_counts[new]=next_counts.get(new,0)+count
            counts=next_counts
        p=sum(count for key,count in counts.items() if abs(key)>=abs(observed))/2**len(ranks)
        method='exact sign enumeration including tied midranks'
    else:
        hits=0
        for _ in range(100):
            signs=rng.integers(0,2,size=(1000,len(ranks)))*2-1
            hits+=int(np.count_nonzero(abs(signs@ranks)>=abs(observed)))
        p=(hits+1)/100001;method='seeded sign permutation 100000 draws, plus-one correction'
    return {'p_two_sided':float(p),'nonzero_n':len(nonzero),'zeros':len(values)-len(nonzero),
            'statistic_signed_doubled_rank':observed,'total_doubled_rank':total,'method':method}


def _sensitivities(joint, declaration):
    bootstrap=declaration['bootstrap'];seed=bootstrap['seed']
    if type(seed) is not int or seed<0 or bootstrap['replicates']!=10000 or bootstrap['interval_confidence']!=.975:
        raise ValueError('Predeclared seeded 10000 paired bootstrap at 97.5% required')
    rng=np.random.default_rng(seed);n=len(joint)
    means=[]
    if n:
        for _ in range(100):
            indices=rng.integers(0,n,size=(100,n))
            means.extend(np.mean(joint[indices],axis=1).tolist())
    result={'role':'predeclared sensitivity only','joint_complete_n':n,
            'limitation':'small-n percentile bootstrap/sign/rank do not replace primary paired t intervals'}
    for index,arm in enumerate(('Bstar','LR')):
        values=joint[:,index].tolist() if n else []
        nonzero=[v for v in values if v!=0];wins=sum(v<0 for v in nonzero)
        result[arm]={'bootstrap':{'interval':np.quantile(np.asarray(means)[:,index],[.0125,.9875]).tolist() if n else None,
             'replicates':10000,'seed':seed,'resampling_unit':'paired complete training seed'},
             'sign':{'p_two_sided':float(binomtest(wins,len(nonzero),.5).pvalue) if nonzero else 1.,
                     'wins':wins,'nonzero_n':len(nonzero),'zeros':len(values)-len(nonzero)},
             'rank':_rank(values,rng)}
    return result


def analyze_confirmation(results: PairedResults, manifest: ConfirmationManifest) -> dict:
    """Report both exact co-primary intervals, retaining every planned seed.

    n always names the frozen independent target seed inventory. Missing,
    nonfinite, divergent or invalidated arms deny all-run finite-BPB inference.
    Conditional intervals are separately named and cannot rescue that claim.
    """
    from ..evaluate import _validate_contract
    from ..config import canonical_bytes
    import json
    payload=json.loads(canonical_bytes(manifest.payload));_validate_contract(payload)
    if results.manifest_sha256!=manifest.sha256:raise ValueError('Result/manifest identity mismatch')
    analysis=read_artifact(ArtifactRef(**payload['analysis']))
    from .power import _protocol
    expected=_protocol(analysis.get('sensitivities',{}).get('bootstrap',{}).get('seed'))['sensitivities']
    if analysis.get('sensitivities')!=expected or analysis.get('ordinal')!='divergence_worse_than_finite;both_diverged_tie':
        raise ValueError('Frozen analysis must predeclare bootstrap/sign/rank and failure ordinal sensitivity')
    seeds=payload['seeds'];arms=payload['arms'];by_key={};units={}
    for original in results.outcomes:
        row=dict(original);key=(row['seed'],row['arm'])
        if key in by_key:raise ValueError('Duplicate seed/arm: chunks and timing repeats cannot inflate n')
        if key[0] not in seeds or key[1] not in arms:raise ValueError('Unplanned seed/arm: no optional additions')
        if row['status'] not in ('completed','diverged','invalidated','missing','nonfinite'):
            raise ValueError('Known retained outcome disposition required')
        if not isinstance(row['unit'],str) or not row['unit']:raise ValueError('Fresh complete independent unit required')
        if key[0] in units and units[key[0]]!=row['unit']:raise ValueError('Within-seed arms must share independent unit')
        units[key[0]]=row['unit']
        if row['status']=='completed' and (type(row['bpb']) not in (int,float) or not math.isfinite(row['bpb']) or row['bpb']<0):
            row['status']='nonfinite';row['bpb']=None
        if row['status']!='completed':row['bpb']=None
        by_key[key]=row
    if len(set(units.values()))!=len(units):raise ValueError('Fresh independent units cannot reuse a parent')
    per_seed=[];failures={};marginals={}
    for seed in seeds:
        outcomes={arm:by_key.get((seed,arm),{'seed':seed,'arm':arm,'unit':units.get(seed),'status':'missing','bpb':None}) for arm in arms}
        per_seed.append({'seed':seed,'outcomes':outcomes,
            'delta_vs_B':None if any(outcomes[a]['status']!='completed' for a in ('C','Bstar')) else outcomes['C']['bpb']-outcomes['Bstar']['bpb'],
            'delta_vs_LR':None if any(outcomes[a]['status']!='completed' for a in ('C','LR')) else outcomes['C']['bpb']-outcomes['LR']['bpb']})
    for arm in arms:
        rows=[row['outcomes'][arm] for row in per_seed]
        counts={status:sum(r['status']==status for r in rows) for status in ('completed','diverged','invalidated','missing','nonfinite')}
        failures[arm]={**counts,'denominator_planned_seeds':len(seeds),'divergence_probability':counts['diverged']/len(seeds),
            'unresolved_probability':sum(counts[s] for s in ('invalidated','missing','nonfinite'))/len(seeds)}
        finite=[row['bpb'] for row in rows if row['status']=='completed']
        marginals[arm]={'successful_n':len(finite),'success_conditioned_mean_bpb':float(np.mean(finite)) if finite else None}
    contrasts={};conditional={};ordinal={}
    complete=all(row['status']=='completed' for seed in per_seed for row in seed['outcomes'].values())
    for arm,key in (('Bstar','delta_vs_B'),('LR','delta_vs_LR')):
        values=[row[key] for row in per_seed if row[key] is not None];conditional[arm]=_interval(values)
        contrasts[arm]=_interval(values) if complete else {**_interval([]),'n':len(seeds),'withheld':'all planned arms must complete finitely'}
        signs=[-1 if value<0 else 1 if value>0 else 0 for value in values]
        contrasts[arm]['finite_per_seed_signs']=signs
        ordered=[]
        for row in per_seed:
            c,b=row['outcomes']['C'],row['outcomes'][arm]
            if c['status']=='completed' and b['status']=='completed':ordered.append(-1 if c['bpb']<b['bpb'] else 1 if c['bpb']>b['bpb'] else 0)
            elif c['status']=='diverged' and b['status']=='diverged':ordered.append(0)
            elif c['status']=='diverged' and b['status']=='completed':ordered.append(1)
            elif c['status']=='completed' and b['status']=='diverged':ordered.append(-1)
        nonzero=[v for v in ordered if v]
        ordinal[arm]={'wins':ordered.count(-1),'losses':ordered.count(1),'ties':ordered.count(0),
            'unresolved':len(seeds)-len(ordered),'p_two_sided':float(binomtest(ordered.count(-1),len(nonzero),.5).pvalue) if nonzero else 1.,
            'rule':analysis['ordinal'],'role':'predeclared sensitivity only'}
    supported=complete and all(contrasts[a]['interval'] is not None and contrasts[a]['interval'][1]<0 for a in ('Bstar','LR'))
    status='supported' if supported else 'inconclusive'
    if any(failures[a]['diverged'] for a in arms):status='substantive_failure'
    elif not complete:status='incomplete'
    elif any(contrasts[a]['interval'] is not None and contrasts[a]['interval'][0]>0 for a in ('Bstar','LR')):status='negative'
    joint=np.asarray([[r['delta_vs_B'],r['delta_vs_LR']] for r in per_seed if r['delta_vs_B'] is not None and r['delta_vs_LR'] is not None],dtype=float).reshape(-1,2)
    means={}
    for arm in arms:
        costs=[r for r in results.outcomes if r['arm']==arm and 'deployment_costs' in r]
        means[arm]={'n_planned':len(seeds),'n_observed_costs':len(costs),
            'offline_search_meta_costs':None,
            'boundary':'mean complete deployment per fresh full-budget run; attributable warm start and online/evaluation/checkpoint work; offline search/meta separate'}
        for name in ('stream_targets','total_targets','known_total_targets','unique_targets','seconds','physical_events'):
            values=[r['deployment_costs'].get(name) for r in costs]
            means[arm][name]=math.fsum(values)/len(seeds) if len(costs)==len(seeds) and all(v is not None for v in values) else None
        means[arm]['complete_full_budget_mean_available']=len(costs)==len(seeds) and all(r['status']=='completed' for r in costs)
    return {'schema':'ralup.synthetic_confirmation_analysis.v1','manifest_sha256':manifest.sha256,
        'scope':'synthetic-engineering','n':len(seeds),'independent_unit':'fresh complete target-training seed',
        'status':status,'H1_H2_supported':bool(supported),'all_run_bpb_claim_established':bool(supported),
        'scientific_confirmation_established':False,'contrasts':contrasts,'ratio_vs_B':contrasts['Bstar']['ratio'],
        'ratio_vs_LR':contrasts['LR']['ratio'],'per_seed':per_seed,'marginal_bpb':marginals,'failures':failures,
        'success_conditioned_quality':conditional,'ordinal_sensitivity':ordinal,
        'sensitivities':_sensitivities(joint,analysis['sensitivities']),
        'quality_cost_curves':results.quality_cost_curves or [],'project_costs':results.project_costs,
        'mean_deployment_costs':means,
        'conclusion':'Frozen synthetic policy and byte fixture only; WP08 scientific confirmation not attempted. H3/H4/H5 and meta-training recipe reliability are unestablished.'}
