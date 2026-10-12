"""S37 paired attribution and S38 evidence-triggered extension contracts.

Current scientific producers have no finalized learned policies/learning budget.
All positive executable artifacts here are explicitly data-only engineering
fixtures. A real campaign needs integration of genuine E01/WP06 producers, not
new labels on SGD fixtures. This module neither trains nor reads protected data.
"""
import hashlib
import json
import math
from pathlib import Path

from ..config import canonical_bytes
from ..records import (ArtifactRef, GateResult, IncompleteConfirmation,
                       read_artifact, write_immutable_json)
from ..search import TrialSpec, source_identity
from .cost import _totals

FEATURES={'features':14,'history':True,'ema':True,'mask_after_scan':True,'normalizer_fit':'META-only'}
INPUTS={'stream','model','budget','scorer','warm_start','continuation','normalizer','learning'}
EXTENSIONS={'larger_shape','independent_meta_clusters','from_scratch','carry_state','fp8','mxfp4','w4a4','new_corpus'}


def _ref(reference):
    return {'path':str(reference.path),'sha256':reference.sha256}


def _reference(value):
    if type(value) is not dict or set(value)!={'path','sha256'}:
        raise ValueError('Exact path/SHA-256 artifact reference required')
    return ArtifactRef(**value)


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _seeds(values):
    if (not isinstance(values,list) or not values or any(type(x) is not int or x<0 for x in values)
            or len(set(values))!=len(values)):
        raise ValueError('Distinct explicit independent training seeds required')


def _parent(reference):
    value=read_artifact(reference)
    if value.get('schema')!='ralup.attribution_parent.v1':
        raise IncompleteConfirmation('No authenticated finalized attribution parent/learning producer')
    if value.get('scope')!='synthetic-engineering':
        raise IncompleteConfirmation('Genuine scientific attribution producer integration is absent; SGD fixtures are not C/LR/Bstar')
    if value['source_identity']!=source_identity():
        raise ValueError('Exact current whole-source identity required; old full states cannot restore current')
    if set(value['inputs'])!=INPUTS:raise ValueError('Complete paired parent/input/learning authority required')
    for item in value['inputs'].values():
        document=read_artifact(_reference(item))
        if document.get('scope')!='synthetic-engineering':
            raise ValueError('Engineering parent inputs must retain explicit synthetic scope; no protected/scientific payload')
    warm=read_artifact(_reference(value['inputs']['warm_start']))
    if 'cost' not in warm:raise ValueError('Authenticated complete or explicitly unknown common warm-start cost required')
    _totals([warm['cost']])
    _seeds(value['seeds'])
    if type(value['budget']) is not int or value['budget']<=0:raise ValueError('Finite positive logical budget required')
    if canonical_bytes(value['feature_contract'])!=canonical_bytes(FEATURES):raise ValueError('Full retained feature/history/EMA scans and META-only fitting required')
    if type(value['selected_after_protected']) is not bool:raise ValueError('Explicit selection timing required')
    return value


def _permutation(parent):
    permutation=parent['history_permutation'];known={}
    for group in parent['history_groups']:
        if (not isinstance(group['shape'],list) or not group['shape']
                or any(type(x) is not int or x<=0 for x in group['shape'])):
            raise ValueError('Explicit history tensor shapes required')
        for name in group['names']:
            if not isinstance(name,str) or not name or name in known:raise ValueError('Unique history tensor names required')
            known[name]=group['shape']
    if permutation.get('predeclared') is not True or permutation.get('retrain') is not True:
        raise ValueError('History permutation must be predeclared and retrained')
    groups=permutation['groups'];indices=permutation['indices'];seen=set()
    if not groups or len(groups)!=len(indices):raise ValueError('Explicit paired permutation lists required')
    for names,order in zip(groups,indices):
        if (not names or len(set(names))!=len(names) or set(names)&seen or any(n not in known for n in names)
                or any(known[n]!=known[names[0]] for n in names)
                or any(type(x) is not int for x in order) or sorted(order)!=list(range(len(names)))):
            raise ValueError('History shuffle restricted to a predeclared matched-shape bijection')
        seen.update(names)
    if seen!=set(known):raise ValueError('Permutation must account for every declared history tensor')
    return permutation


def make_ablation_trials(frozen_parent: ArtifactRef, contrast: str, phase: str, retune: bool) -> list[TrialSpec]:
    """Freeze paired proposals and unfulfilled learning obligations, never results.

    retune=False is an original-setting intervention. Information removals still
    require new training. Inference scrambling is a separately named diagnostic.
    Augmented ordinary baselines match the base logical budget and explicitly
    declare additional exposures instead of hiding them in a matched-cost label.
    """
    from ..baselines.registry import attribution_registry
    entries=attribution_registry()
    if contrast not in entries:raise ValueError('Unknown prioritized attribution contrast')
    if phase not in ('explore','confirm') or type(retune) is not bool:raise ValueError('Explicit phase and retuning decision required')
    parent=_parent(frozen_parent)
    if phase=='confirm' and parent['selected_after_protected']:
        raise ValueError('Post-test selection remains exploratory; fresh experimental design required')
    entry=entries[contrast]
    if contrast in ('history_permuted','history_scramble_inference'):_permutation(parent)
    trials=[]
    for seed in parent['seeds']:
        pairing={key:parent['inputs'][key]['sha256'] for key in sorted(INPUTS)}
        pairing.update(seed=seed,parent=frozen_parent.sha256,logical_budget=parent['budget'])
        for candidate in (True,False):
            current={'variant':'affine45','retrain':False,'privileged':False,'raw_ordinary':False,'settings':{'parameters':45,'h_range':[.25,4]}} if candidate else entry
            retrain=(retune or current['retrain']) and not (not candidate and contrast in ('initialized','history_scramble_inference','norm_matched'))
            contract={'schema':'ralup.attribution_trial.v1','scope':parent['scope'],
                'source_identity':parent['source_identity'],'source_revision':parent['source_revision'],
                'parent':_ref(frozen_parent),'contrast':contrast,'arm':'candidate' if candidate else 'contrast',
                'variant':current['variant'],'variant_settings':current['settings'],'phase':phase,'seed':seed,
                'budget':parent['budget'],'pairing':pairing,'retune':retune,
                'comparison_kind':'retuned_competition' if retune else 'original_setting_intervention',
                'retraining_status':'required' if retrain else 'not_retrained',
                'retuning_status':'required' if retune else 'not_retuned',
                'backbone_lr_tuning':'independent' if retune else 'parent_setting',
                'feature_cost_retained':not current['raw_ordinary'],
                'compute_information':FEATURES if not current['raw_ordinary'] else None,
                'telemetry_required':True,'is_privileged_diagnostic':current['privileged'],
                'claim_status':'exploratory' if phase=='explore' else 'confirmatory-engineering',
                'representation':'E2M1-full-plus-minus-6','history_permutation':None,
                'assisted_fraction':None,'extra_targets':0,'extra_targets_kind':None}
            if not candidate and contrast in ('history_permuted','history_scramble_inference'):
                contract['history_permutation']=_permutation(parent)
            if not candidate and contrast=='random_assistance':
                rate=parent['assisted_fraction']
                if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<=rate<=1:
                    raise ValueError('Measured frozen candidate assisted fraction required')
                contract['assisted_fraction']=rate
            if not candidate and current['raw_ordinary']:
                key=current['settings']['augmentation'];extra=parent['budget_augmentation'][key]
                if type(extra) is not int or extra<=0:raise ValueError('Finite predeclared augmentation required')
                contract.update(extra_targets=extra,extra_targets_kind=key)
            content=canonical_bytes(contract)
            trials.append(TrialSpec(_digest(contract),contract['variant'],seed,parent['budget'],
                                    parent['source_revision'],content.decode()))
    return trials


def _interval(values,confidence):
    # Same complete-seed paired t inference as E04, with family correction.
    import numpy as np
    from scipy.stats import t
    if len(values)<2:return None
    sd=float(np.std(values,ddof=1));mean=float(np.mean(values))
    if sd==0:return None  # Constant engineering differences cannot certify variance.
    radius=float(t.ppf((1+confidence)/2,len(values)-1))*sd/math.sqrt(len(values))
    return [mean-radius,mean+radius]


def _execution(run,trial,design,parent):
    if run.get('schema')!='ralup.attribution_run.v1' or run.get('scope')!='synthetic-engineering':
        raise ValueError('Explicit engineering run schema required; genuine science producer absent')
    if canonical_bytes(run['trial'])!=canonical_bytes(trial) or run['design_sha256']!=design.sha256:
        raise ValueError('Run must match regenerated exact prospective trial and family declaration')
    receipt=read_artifact(_reference(run['execution']))
    if (receipt.get('schema')!='ralup.attribution_execution.v1' or receipt.get('scope')!=run['scope']
            or receipt['trial_hash']!=_digest(trial) or receipt['design_sha256']!=design.sha256
            or receipt['parent_sha256']!=trial['parent']['sha256']
            or receipt['learning_design_sha256']!=parent['inputs']['learning']['sha256']):
        raise ValueError('Execution must bind exact trial/design/parent/learning authority')
    if trial['retraining_status']=='required' and receipt['training_status']!='retrained':
        raise ValueError('Inference-only intervention cannot claim retrained information ablation')
    if trial['retuning_status']=='required' and (receipt['tuning_status']!='independently_retuned' or receipt['backbone_lr_tuned'] is not True):
        raise ValueError('Retuned competitiveness requires independently tuned backbone LR evidence')
    if trial['retuning_status']=='not_retuned' and receipt['tuning_status']!='not_retuned':raise ValueError('Intervention cannot be labeled independently retuned')
    if trial['retraining_status']=='not_retrained' and receipt['training_status']!='not_retrained':raise ValueError('Frozen/inference diagnostic cannot be relabeled retrained')
    status=run['status'];score=run['development_bpb']
    if trial['feature_cost_retained']:
        for key in ('feature_scans','history_updates','ema_updates','meta_feature_scans'):
            if type(receipt[key]) is not int or receipt[key]<(1 if status=='completed' else 0):raise ValueError('Retained feature/history/EMA/META computations must be charged')
        if receipt['features_per_scan']!=14:raise ValueError('All fourteen features must be computed before masking')
    if status not in ('completed','failed','diverged','blocked') or run['cost']['status']!=status:
        raise ValueError('Explicit matching terminal and cost status required')
    if status=='completed':
        if (type(score) not in (int,float) or not math.isfinite(score) or score<0
                or receipt['complete_target_training_seed'] is not True):raise ValueError('Complete finite independent training seed required')
    elif score is not None:raise ValueError('Failed/diverged/nonterminal arms retain null endpoint BPB')
    if receipt.get('operational_telemetry') is not True:
        raise ValueError('Mandatory operational telemetry applies in every mode, including raw ordinary baselines')
    if receipt.get('cost_sha256')!=_digest(run['cost']):
        raise ValueError('Execution must authenticate exact raw physical cost receipt')
    warm=read_artifact(_reference(parent['inputs']['warm_start']))['cost']
    if canonical_bytes({'parents':run['cost']['parents']}) != canonical_bytes({'parents':[warm]}):
        raise ValueError('Every deployment must retain authenticated common warm-start ancestry and cost')
    _totals([run['cost']])
    # The shared cost union permits duplicate inherited references, but a root
    # execution receipt must name each new physical pass once and never borrow
    # a pass already charged anywhere in its authenticated warm ancestry.
    physical_ids=[event['event_id'] for event in run['cost']['events']]
    inherited_ids=set(_totals([warm])['raw_classifications'])
    if len(set(physical_ids))!=len(physical_ids) or set(physical_ids)&inherited_ids:
        raise ValueError('Root execution requires distinct new physical event IDs disjoint from warm ancestry')
    if status=='completed':
        target_stream=sum(e['stream_targets'] for e in run['cost']['events'])
        expected_stream=trial['budget']+(trial['extra_targets'] if trial['extra_targets_kind']=='extra_data_targets' else 0)
        if target_stream!=expected_stream:raise ValueError('Charged target stream must match finite base budget and explicit augmentation')
        if trial['extra_targets_kind']=='extra_compute_targets':
            processed=sum(e['targets'] for e in run['cost']['events'])
            if processed<trial['budget']+trial['extra_targets']:raise ValueError('Actual additional compute augmentation must be charged')
    return receipt


def _regressions(pairs,covariates):
    import numpy as np
    if not isinstance(covariates,list) or set(covariates)-{'entropy','length'}:raise ValueError('Predeclared entropy/length covariates required')
    rows=[]
    for candidate,control in pairs:
        if candidate['status']!='completed' or control['status']!='completed':continue
        left={a['document']:a for a in candidate['articles']};right={a['document']:a for a in control['articles']}
        if len(left)!=len(candidate['articles']) or len(right)!=len(control['articles']) or left.keys()!=right.keys():raise ValueError('Paired article identities must match uniquely')
        for name,a in left.items():
            b=right[name]
            if any(a[x]!=b[x] for x in covariates):raise ValueError('Paired article covariates differ')
            numbers=[a['development_bpb'],b['development_bpb'],*[a[x] for x in covariates]]
            if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in numbers):raise ValueError('Finite article BPB/covariates required')
            rows.append({'contrast':candidate['trial']['contrast'],'seed':candidate['trial']['seed'],'document':name,
                         'delta_bpb':a['development_bpb']-b['development_bpb'],**{x:a[x] for x in covariates}})
    by_contrast={}
    for contrast in sorted(set(r['contrast'] for r in rows)):
        slopes={};contrast_rows=[r for r in rows if r['contrast']==contrast]
        for covariate in covariates:
            # Retain within-panel article variation. Each complete training seed
            # supplies one descriptive OLS slope (with intercept); seeds have
            # equal weight in the aggregate, regardless of article count.
            seeds=sorted(set(r['seed'] for r in contrast_rows));per_seed=[]
            for seed in seeds:
                group=[r for r in contrast_rows if r['seed']==seed]
                x=np.asarray([r[covariate] for r in group],dtype=float)
                y=np.asarray([r['delta_bpb'] for r in group],dtype=float)
                centered_x=x-x.mean();denom=float(np.sum(centered_x**2))
                slope=float(np.sum(centered_x*(y-y.mean()))/denom) if denom>0 else None
                per_seed.append({'seed':seed,'n_articles':len(group),'slope':slope})
            estimable=[row['slope'] for row in per_seed if row['slope'] is not None]
            slopes[covariate]={'n_seeds':len(seeds),'n_estimable_seeds':len(estimable),
                'per_seed':per_seed,'slope':float(np.mean(estimable)) if estimable else None,
                'aggregation':'equal_weight_mean_over_estimable_complete_training_seeds'}
        by_contrast[contrast]=slopes
    return {'schema':'ralup.attribution_article_regressions.v2',
            'method':'within_seed_article_ols_intercept_then_equal_seed_mean',
            'role':'descriptive exploratory; seed remains independent unit','per_article':rows,
            'by_contrast':by_contrast,'causal_inference':False}



def analyze_attribution(runs: list[ArtifactRef], declared_family: dict) -> dict:
    """Authenticate exact proposals/receipts, retain failures, adjust family CIs."""
    declaration=_reference(declared_family['declaration']);family=read_artifact(declaration)
    if family.get('schema')!='ralup.attribution_family.v1' or family.get('scope')!='synthetic-engineering':
        raise ValueError('Explicit authenticated engineering mechanism family required')
    _seeds(family['seeds']);phase=family['phase'];contrasts=family['contrasts']
    if (phase not in ('explore','confirm') or not isinstance(contrasts,list) or not contrasts
            or len(set(contrasts))!=len(contrasts) or type(family['retune']) is not bool):raise ValueError('Finite declared family and explicit retuning decision required')
    if type(family['alpha']) not in (int,float) or not 0<family['alpha']<1:raise ValueError('Familywise alpha required')
    if family['multiplicity']!='bonferroni':raise ValueError('Declared Bonferroni multiplicity rule required')
    if phase=='confirm' and any(family[k] is not True for k in ('predeclared','fresh_design','selection_before_outcomes')):
        raise ValueError('Mechanism confirmation requires a fresh predeclared design before outcomes')
    parent_ref=_reference(family['parent']);parent=_parent(parent_ref)
    if family['seeds']!=parent['seeds']:raise ValueError('Family must bind exact prospective independent seed inventory')
    expected={}
    for contrast in contrasts:
        for trial in make_ablation_trials(parent_ref,contrast,phase,family['retune']):
            value=json.loads(trial.attribution_contract);expected[(contrast,value['seed'],value['arm'])]=value
    observed={};costs=[];receipts={}
    for reference in runs:
        run=read_artifact(reference);trial=run['trial'];key=(trial['contrast'],trial['seed'],trial['arm'])
        if key not in expected or key in observed:raise ValueError('Unexpected/duplicate contrast training unit')
        receipt=_execution(run,expected[key],declaration,parent)
        observed[key]=run;receipts[key]=receipt;costs.append(run['cost'])
    report={};all_pairs=[];confidence=1-family['alpha']/len(contrasts)
    for contrast in contrasts:
        pairs=[];deltas=[];missing=[];failures=[]
        for seed in family['seeds']:
            keys=[(contrast,seed,arm) for arm in ('candidate','contrast')]
            if any(k not in observed for k in keys):missing.append(seed);continue
            left,right=(observed[k] for k in keys);pairs.append((left,right))
            if right['trial']['feature_cost_retained'] and all(r['status']=='completed' for r in (left,right)):
                for field in ('feature_scans','features_per_scan','history_updates','ema_updates','meta_feature_scans'):
                    if receipts[keys[0]][field]!=receipts[keys[1]][field]:raise ValueError('Policy/control information computation costs must match')
            if any(r['status']!='completed' for r in (left,right)):failures.append(seed);continue
            deltas.append(left['development_bpb']-right['development_bpb'])
        interval=_interval(deltas,confidence)
        outcome=('incomplete' if missing else 'failed' if failures else 'inconclusive' if interval is None
                 else 'favorable' if interval[1]<0 else 'unfavorable' if interval[0]>0 else 'inconclusive')
        report[contrast]={'outcome':outcome,'planned_n':len(family['seeds']),'complete_n':len(deltas),
            'ordered_complete_deltas':deltas,'mean_delta':sum(deltas)/len(deltas) if deltas else None,
            'interval':interval if not missing and not failures else None,'conditional_interval':interval if missing or failures else None,
            'missing_seeds':missing,'failed_seeds':failures,'comparison_kind':expected[(contrast,family['seeds'][0],'contrast')]['comparison_kind'],
            'is_privileged_diagnostic':expected[(contrast,family['seeds'][0],'contrast')]['is_privileged_diagnostic'],
            'mechanism_claim_eligible':contrast not in ('norm_matched','history_scramble_inference')}
        all_pairs.extend(pairs)
    deployments=[_totals([r['cost']]) for r in observed.values()]
    metrics=('stream_targets','total_targets','unique_targets','seconds','known_total_targets','known_interval_seconds')
    means={key:sum(d[key] for d in deployments)/len(deployments)
           if deployments and all(d[key] is not None for d in deployments) else None for key in metrics}
    fresh_needed=(parent['selected_after_protected'] or any(family[k] is not True for k in ('predeclared','fresh_design','selection_before_outcomes')))
    return {'schema':'ralup.attribution_analysis.v1','scope':'synthetic-engineering',
        'claim_status':'exploratory' if phase=='explore' else 'confirmatory-engineering',
        'scientific_mechanism_established':False,'fresh_design_required':fresh_needed,'family_declaration':_ref(declaration),
        'multiplicity':{'rule':'bonferroni','familywise_alpha':family['alpha'],'comparisons':len(contrasts),'interval_confidence':confidence},
        'contrasts':report,'project_cost':_totals(costs),
        'deployment_cost':{'mean_warm_inclusive':means,'cost_available':all(means[x] is not None for x in ('total_targets','unique_targets','seconds')),
            'unit':'all supplied deployment attempts, failures retained','offline_META_search_project_cost_is_separate':True,
            'common_engineering_seconds':None},'regressions':_regressions(all_pairs,family['regression_covariates']),
        'uses_protected_scores':False,'independent_unit':'fresh complete target-training seed',
        'limitations':['Engineering receipts are not finalized trained C/LR/Bstar or genuine retuned attribution.',
                       'Unknown common engineering and partial ancestor costs remain unknown.']}


def check_extension_eligibility(evidence: dict, extension: str) -> GateResult:
    """Separate exact branch contracts; unsupported/unknown evidence blocks.

    Passing synthetic declarations verifies gate mechanics only. It cannot
    authorize native training or scientific representation/corpus transfer.
    """
    if extension not in EXTENSIONS:return GateResult('blocked',(),('Unknown extension; new specification required',))
    refs=[]
    try:
        observation_ref=_reference(evidence['observation']);observation=read_artifact(observation_ref);refs.append(observation_ref)
        contract_ref=_reference(evidence['contract']);contract=read_artifact(contract_ref);refs.append(contract_ref)
        if observation.get('schema')!='ralup.extension_observation.v1' or contract.get('schema')!='ralup.extension_contract.v1':
            raise ValueError('Explicit observation and own extension specification required')
        if contract['extension']!=extension or contract['scope']!=observation['scope'] or contract['scope']!='synthetic-engineering':
            raise ValueError('Exact engineering extension scope required; genuine science producer absent')
        if contract.get('observation_sha256')!=observation_ref.sha256:
            raise ValueError('Own prospective extension contract must bind exact observed evidence')
        if contract['source_identity']!=source_identity():raise ValueError('Current exact source numerical admission required')
        if observation['selected_after_protected']:raise ValueError('Post-test selection remains exploratory; fresh extension declaration required')
        diagnosis=observation.get('diagnosis')
        supported_diagnosis=(isinstance(diagnosis,dict) and diagnosis.get('predeclared') is True
            and bool(diagnosis.get('hypothesis')) and bool(diagnosis.get('falsification')))
        if observation['outcome'] in ('unfavorable','negative') and not supported_diagnosis:
            return GateResult('negative',tuple(refs),('Supported unfavorable local evidence does not justify blind extension',))
        if observation['outcome']!='favorable' and not supported_diagnosis:
            return GateResult('inconclusive',tuple(refs),('No supported positive evidence or falsifiable missing-condition diagnosis',))
        if observation['supported_local_evidence'] is not True:raise ValueError('Supported local evidence required')
        for key in ('numerical_contract','provenance','activation_gradient_contract','packing_contract'):
            if not isinstance(contract[key],dict) or not contract[key]:raise ValueError('Own numerical/provenance/activation-gradient/packing contract required')
        if contract['learning_contract']!={'finalized':True,'own_tuning':True}:raise ValueError('Own finalized independently tuned learning contract required')
        budget=contract['budget']
        if budget.get('includes_full_inherited_cost') is not True:
            raise ValueError('Measured extension budget must include complete inherited warm/learning/engineering costs')
        for key in ('measured_seconds','measured_bytes','reserved_seconds','reserved_bytes'):
            if type(budget[key]) not in (int,float) or not math.isfinite(budget[key]) or budget[key]<=0:raise ValueError('Measured finite budget and actual reserved resources required')
        if budget['measured_seconds']>budget['reserved_seconds'] or budget['measured_bytes']>budget['reserved_bytes']:raise ValueError('Extension exceeds declared reserved budget')
        branch=contract['branch_contract']
        if extension=='larger_shape' and branch['parameters']!=19300000:raise ValueError('Own 19.3M numerical/learning/packing shape specification required')
        if extension=='independent_meta_clusters':
            _seeds(branch['meta_seeds'])
            if len(branch['meta_seeds'])<2:raise ValueError('Independent meta-training seed clusters required')
        if extension=='from_scratch' and branch['reset']!='fresh_all_states':raise ValueError('From-scratch branch requires explicit fresh state contract')
        if extension=='carry_state' and branch['carry']!='complete_current_state':raise ValueError('Carry branch requires complete current-source state contract')
        representation={'fp8':'E4M3-FP8','mxfp4':'E2M1-E8M0-MXFP4','w4a4':'W4A4'}.get(extension)
        if representation and (branch['representation']!=representation or contract['numerical_contract']['representation']!=representation):
            raise ValueError('Own exact representation specification required; cannot inherit FP4 label')
        if extension=='mxfp4' and branch['block_size']!=32:raise ValueError('MXFP4 requires block-32 E8M0 scale specification')
        if extension=='new_corpus' and (not branch['corpus'] or branch['corpus'].lower() in ('wikitext-2','wikitext-103')):
            raise ValueError('New independent corpus specification required; WikiText-2 is not transfer')
        return GateResult('pass',tuple(refs),())
    except (KeyError,TypeError,ValueError,OSError) as error:
        return GateResult('blocked',tuple(refs),(str(error) or 'Incomplete own extension contract',))


def _scientific_prerequisite(config):
    """Reauthenticate original E02 producer without manufacturing new policy arms."""
    from ..evaluate import _refuse_design
    reference=_reference(config['prerequisite_design'])
    try:_refuse_design(reference)
    except IncompleteConfirmation as error:
        design=read_artifact(reference)
        root=Path(__file__).resolve().parents[3]
        index=json.loads((root/'docs/harness/e02-evidence.json').read_bytes())
        attempt=next(r for r in index['attempts'] if r['attempt']=='plan-confirmation-v1')
        finish_ref=_reference(attempt['files']['finish.json']);finish=read_artifact(finish_ref)
        return {'schema':'ralup.attribution_prerequisites.v1','status':'blocked','scope':'scientific-development',
            'claim_status':'exploratory','scientific_mechanism_established':False,
            'authentic_producer':_ref(reference),'blockers':[str(error),
            'Finalized trained C/LR/Bstar, final learning budget and independently retuned attribution authority absent; integrate E01/reopened WP06.'],
            'known_cost':{'producer_finish':_ref(finish_ref),'producer_elapsed_seconds_lower_bound':finish['elapsed_seconds'],
                'scope':'original E02 administrative producer only; no numerical attribution attempt',
                'current_attempt_cost':'separate immutable launcher finish receipt',
                'scientific_deployment_cost':None,'complete_project_cost':None},
            'inherited_scientific_cost_unknowns':design['costs'],'unmeasured_common_engineering_cost':None,
            'current_prerequisites':design['scientific_prerequisites'],
            'requested_contrasts':config.get('requested_contrasts',[]),'retune':config.get('retune'),
            'contrasts_measured':[],'protected_scoring_attempted':False,'wp08':'not_attempted'}
    raise IncompleteConfirmation('Authentic prerequisite producer advanced; integrate genuine finalized attribution parent before execution')


def run_attribution_suite(config_path: Path) -> dict:
    """Analyze supplied authentic engineering outcomes or publish honest refusal."""
    from ..records import sha256_file
    configuration=ArtifactRef(config_path,sha256_file(config_path));config=read_artifact(configuration)
    if config.get('schema')!='ralup.attribution_suite.v1':raise ValueError('Explicit attribution suite config required')
    if config.get('scope')=='scientific-development':
        if 'prerequisite_design' not in config:raise IncompleteConfirmation('Authentic scientific prerequisite producer required; finalized trained parent absent')
        result=_scientific_prerequisite(config)
        result['config']=_ref(configuration);result['source_identity']=source_identity()
        if config.get('output'):write_immutable_json(config['output'],result)
        return result
    if config.get('scope')!='synthetic-engineering':raise ValueError('Explicit engineering or scientific scope required')
    result=analyze_attribution([_reference(r) for r in config['runs']],config['declared_family'])
    result['config']=_ref(configuration);result['source_identity']=source_identity()
    write_immutable_json(config['output'],result)
    return result
