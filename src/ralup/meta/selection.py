"""Fail-closed exploratory selection from eligible training and endpoint receipts.

These records authorize only development nomination. They cannot establish B0,
production or confirmation eligibility. Calibration/campaign outputs alone are
not eligibility records; no current M11 scientific artifact meets this contract.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import torch
from ..config import canonical_bytes
from ..records import ArtifactRef
from ..policy.controllers import ControllerArtifact,ControllerState,authenticate
from ..search import source_identity

ROOT=Path(__file__).resolve().parents[3]
DATA_INDEX=ROOT/'docs/data/h02-evidence.json'
RECIPE_INDEX=ROOT/'docs/harness/m07-closing-evidence.json'
FAMILIES={'affine45','learned_lr49','static3'}


def _require(condition,message):
    if not condition:raise ValueError(message)


def _read(ref):return authenticate(ArtifactRef(**ref))
def _ref(ref):return {'path':str(ref.path.resolve()),'sha256':ref.sha256}
def _digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
def _positive(value):return type(value) is int and value>0

def _time(value):
    result=datetime.fromisoformat(value)
    _require(result.tzinfo is not None,'Explicit UTC/timezone evidence required')
    return result


def _completed(record,schema):
    _require(record.get('schema')==schema and record.get('status')=='completed' and record.get('synthetic') is False,'Completed genuine authority required; calibration/synthetic/unresolved evidence is ineligible')


def _producer(ref,design_ref,source):
    producer=_read(ref);design=_read(design_ref)
    _require({k:v['sha256'] for k,v in producer['source'].items() if k.startswith('src/')}==source,'Producer source differs from declared complete runtime')
    _require(design_ref in producer['inputs'] and _time(design['created_utc'])<=_time(producer['utc']),'Design must be frozen and bound by producer before endpoint/training work')
    return producer


def _eligible(artifact):
    value=artifact.to_json();record=_read(value['training'])
    _require(record.get('schema')=='ralup.meta_selection_eligibility.v1' and record.get('status')=='eligible_exploratory' and record.get('synthetic') is False and record.get('confirmation_eligible') is False and record.get('B0') is None,'Completed matched-family eligibility required; calibration/synthetic/unresolved controllers are ineligible')
    _require(set(record['families'])==FAMILIES,'Matched candidate/LR/static family authority required')
    design=_read(record['design']);source=source_identity()
    _require(design.get('schema')=='ralup.matched_meta_training.v1' and design.get('purpose')=='final-matched-training' and design.get('synthetic') is False,'Resolved final matched-training design required')
    _require(design['source_identity']==source,'Eligibility source is not current runtime')
    for field,artifact_field in (('normalization','normalization'),('backbone','B0'),('data','data')):
        _require(design[field]==value[artifact_field],'Controller normalization/recipe/data differs from eligibility')
    indexed=json.loads(RECIPE_INDEX.read_bytes())['exploratory_backbone']
    _require(design['backbone']==indexed,'Indexed explicitly exploratory recipe required; no B0 acceptance')
    recipe=_read(indexed)
    _require((recipe['schema'],recipe['status'],recipe['b0'],recipe['selected_b0'],recipe['confirmation_eligible'])==('ralup.exploratory_backbone.v1','EXPLORATORY',None,False,False),'Exploratory recipe scope mismatch')
    accepted=json.loads(DATA_INDEX.read_bytes())
    _require(design['data']['sha256']==accepted['manifest']['sha256'],'Accepted real partition manifest required')
    corpus=_read(design['data']);_require(corpus['synthetic'] is False,'Synthetic data cannot authorize selection')
    norm=_read(design['normalization']);provenance=json.loads(norm['source_manifest'])
    _require(provenance['schema']=='ralup.frozen_normalizer_sources.v1' and provenance['partition']=='meta_train' and _positive(provenance['rows']) and provenance['trajectories'],'META-only normalizer authority required')
    for key in ('mean','std'):
        _require(len(norm[key])==14 and all(type(v) in (int,float) and math.isfinite(v) and (key!='std' or v>=1e-6) for v in norm[key]),'Finite fourteen-feature normalizer required')
    for ref in provenance['trajectories']:
        trajectory=_read(ref)
        # The corpus above is already authenticated. Compare its resolved file
        # and digest, preserving original relative trajectory-reference bytes.
        _require(trajectory['schema']=='ralup.feature_trajectory.v1' and trajectory['partition']=='meta_train' and trajectory['role']=='reference' and trajectory['synthetic'] is False and _ref(ArtifactRef(**trajectory['data_manifest']))==design['data'],'Normalizer reference/data scope mismatch')
    opportunities=design['opportunities']
    _require(type(opportunities) is list and {o['origin'] for o in opportunities}=={'baseline','own-policy'},'Baseline and own-current-policy opportunities required')
    for op in opportunities:
        _require(set(op)=={'origin','age','horizon','pairs','query_targets','features','history','refresh_steps'} and all(_positive(op[k]) for k in ('age','horizon','pairs','query_targets')) and op['features']==14 and op['history'] is True,'Finite matched full-feature/history opportunities required')
        _require((_positive(op['refresh_steps']) and op['refresh_steps']>=op['age']) if op['origin']=='own-policy' else op['refresh_steps']==0,'Own-policy refresh generation must be charged')
    inventories=[]
    for family,ref in record['families'].items():
        campaign=_read(ref);_completed(campaign,'ralup.matched_family_completion.v1')
        _require(campaign['family']==family and campaign['design']==record['design'] and campaign['source_identity']==source,'Matched family training/source/design mismatch')
        previous=ControllerState(family,torch.tensor(campaign['initial_parameters'],dtype=torch.float32)).parameters.tolist()
        _require(len(campaign['generations'])==len(opportunities),'Incomplete matched family opportunities')
        inventory=[]
        for i,(ref,op) in enumerate(zip(campaign['generations'],opportunities)):
            completion=_read(ref);_completed(completion,'ralup.meta_generation_completion.v1')
            _require(completion['source_identity']==source and completion['design']==record['design'] and completion['opportunity_index']==i,'Generation opportunity/source mismatch')
            _producer(completion['producer'],record['design'],source)
            if op['origin']=='own-policy':
                refresh=_read(completion['refresh']);_completed(refresh,'ralup.own_policy_refresh.v1')
                _require(refresh['family']==family and refresh['acting_parameters']==previous and refresh['source_identity']==source and refresh['data']==design['data'] and refresh['design']==record['design'],'Refresh must use this family own current unperturbed policy')
                _require(refresh['optimizer_steps']==op['refresh_steps'] and op['age'] in refresh['ages'] and refresh['features']==14 and refresh['history'] is True,'Incomplete own-policy refresh opportunity/cost')
                _producer(refresh['producer'],record['design'],source)
            else:_require(completion['refresh'] is None,'Baseline cannot substitute policy refresh')
            g=_read(completion['generation'])
            _require(g['family']==family and g['outer_step']==i+1 and g['phi_before']==previous and g['horizon']==op['horizon'] and len(g['pairs'])==op['pairs'],'Generation vector/horizon/pair lineage mismatch')
            previous=ControllerState(family,torch.tensor(g['phi_after'],dtype=torch.float32)).parameters.tolist()
            inputs=set();query_count=0
            for pair in g['pairs']:
                a,b=pair['plus'],pair['minus']
                _require(a['work']['input_sha256']==b['work']['input_sha256'] and a['work']['initial_state_hash']==b['work']['initial_state_hash'] and a['initial_query']==b['initial_query'],'Matched antithetic state/input required')
                for result in (a,b):
                    _require(result['work']['optimizer_steps']==op['horizon'] and result['work']['query_targets']==2*op['query_targets'],'Incomplete generation/query opportunity')
                    _require(all(type(result[k]) in (int,float) and math.isfinite(result[k]) for k in ('initial_query','final_query')),'Finite measured generation endpoints required')
                    query_count+=result['work']['query_targets'];inputs.add(result['work']['input_sha256'])
            _require(len(inputs)==1 and g['optimizer_steps']==2*op['pairs']*op['horizon'] and g['query_targets']==query_count,'Generation cost/input authority mismatch')
            inventory.append(next(iter(inputs)))
        _require(campaign['parameters']==previous,'Final campaign vector differs from completed lineage')
        if family==artifact.family:_require(value['parameters']==previous,'Selected vector is not eligible trained vector')
        inventories.append(inventory)
    _require(all(v==inventories[0] for v in inventories),'Unequal family training/query opportunities')
    _require(artifact.family in FAMILIES,'Unsupported selection family')
    return value,source,corpus,accepted


def _query_count(ref,partition,corpus,accepted):
    """Authenticate membership and actual source bytes without a model forward."""
    from ..data.blocks import TargetBlock,make_block
    docs=corpus['audit']['protected_documents'] if partition=='development' else corpus['audit']['retained_documents']
    docs={accepted['frozen_hashes']['source_sha256']+':'+d['id']:d for d in docs if (d['split']=='validation' if partition=='development' else d['partition']==partition and not d['protected'])}
    seen=set()
    for row in _read(ref)['blocks']:
        block=TargetBlock(**{k:tuple(v) if isinstance(v,list) else v for k,v in row.items()});doc=docs.get(block.document_id)
        _require(doc is not None and doc['content_sha256']==block.article_hash and 0<=block.target_start<block.target_end<=doc['bytes'],'Endpoint documents outside authenticated selection partition')
        start=max(0,block.target_start-128)
        with Path(doc['content_path']).open('rb') as stream:
            stream.seek(doc['content_start']);content=stream.read(doc['bytes'])
        _require(hashlib.sha256(content).hexdigest()==doc['content_sha256'],'Endpoint source bytes changed')
        expected=make_block(block.article_hash,block.target_start,block.target_end,content[start:block.target_start],content[block.target_start:block.target_end],document_id=block.document_id)
        _require((block.inputs,block.labels,block.positions)==(expected.inputs,expected.labels,expected.positions) and len(block.score_mask)==len(expected.score_mask) and all(type(x) is bool and (not x or y) for x,y in zip(block.score_mask,expected.score_mask)),'Endpoint query bytes/masks mismatch')
        offsets={(block.document_id,i) for i in block.scored_offsets()}
        _require(not offsets&seen,'Duplicate endpoint targets');seen.update(offsets)
    _require(bool(seen),'Nonempty authenticated endpoint required')
    if partition=='development':_require(len(seen)==sum(d['bytes'] for d in docs.values()),'Selection requires full official development coverage')
    return len(seen)


def _endpoint(row,partition,authority):
    _require(row.get('partition')==partition,'Selection partition forbidden')
    _require(set(row)=={'partition','controller','endpoint'},'Authenticated endpoint receipt required; caller-labelled loss rows forbidden')
    value,source,corpus,accepted=authority;receipt=_read(row['endpoint']);_completed(receipt,'ralup.selection_endpoint.v1')
    design=_read(receipt['design'])
    _require(design['schema']=='ralup.selection_endpoint_design.v1' and design['source_identity']==source and receipt['source_identity']==source,'Endpoint source/design mismatch')
    producer=_producer(receipt['producer'],receipt['design'],source)
    _require(_time(producer['utc'])<=_time(receipt['completed_utc']),'Endpoint completion precedes producer startup')
    _require(design['controller_sha256']==receipt['controller_sha256']==_digest(value) and design['eligibility']==value['training'],'Endpoint controller/eligibility mismatch')
    _require(design['partition']==receipt['partition']==partition and design['data']==value['data'],'Endpoint partition/data mismatch')
    _require(type(design['seed']) is int and design['seed']>=0 and design['seed']==receipt['seed'],'Endpoint seed mismatch')
    _require(_positive(design['full_horizon']) and design['start_step']==receipt['start_step']==0 and receipt['end_step']==receipt['full_horizon']==design['full_horizon'],'Full continuation selection required')
    count=_query_count(design['queries'],partition,corpus,accepted)
    _require(receipt['target_count']==count and receipt['input_sha256']==design['queries']['sha256'],'Endpoint input/target count mismatch')
    _require(all(type(receipt[k]) in (int,float) and math.isfinite(receipt[k]) and receipt[k]>=0 for k in ('nll_sum','elapsed_seconds')),'Finite measured endpoint loss/cost required')
    if partition=='meta-development':_require(design['nomination'] is None,'META selection cannot consume official nomination')
    return {'controller':row['controller'],'loss':receipt['nll_sum']/count,'seconds':receipt['elapsed_seconds'],'inventory':(receipt['seed'],receipt['input_sha256'],receipt['full_horizon']),'nomination':design['nomination'],'started':_time(producer['utc']),'design_created':_time(design['created_utc']),'completed':_time(receipt['completed_utc'])}


def select_controller(meta_dev: list[dict],official_dev: list[dict]) -> ControllerArtifact:
    if not meta_dev or not official_dev:raise ValueError('Both development selection stages required')
    try:
        stages=[];eligibility=set();cache={}
        for rows,partition in ((meta_dev,'meta-development'),(official_dev,'development')):
            groups=defaultdict(list)
            for row in rows:
                _require(row.get('partition')==partition,'Selection partition forbidden')
                artifact=row['controller'];_require(type(artifact) is ControllerArtifact,'Frozen controller artifact required')
                key=_digest(artifact.to_json())
                if key not in cache:cache[key]=_eligible(artifact)
                eligibility.add(artifact.training.sha256)
                groups[key].append(_endpoint(row,partition,cache[key]))
            _require({r[0]['controller'].family for r in groups.values()}==FAMILIES,'All three eligible families must receive development opportunities')
            inventories=[]
            for rows in groups.values():
                inventory={r['inventory'] for r in rows};_require(len(inventory)==len(rows),'Duplicate selection seed/opportunity');inventories.append(inventory)
            _require(all(i==inventories[0] for i in inventories),'Unequal development opportunities')
            stages.append(groups)
        _require(len(eligibility)==1,'One shared matched-training authority required')
        def rank(rows):return (sum(r['loss'] for r in rows)/len(rows),sum(r['seconds'] for r in rows))
        nominees={}
        for key,rows in stages[0].items():
            family=rows[0]['controller'].family
            if family not in nominees or (rank(rows),key)<(rank(stages[0][nominees[family]]),nominees[family]):nominees[family]=key
        _require(set(stages[1])==set(nominees.values()),'Official stage differs from meta-development nominees')
        nomination_refs=set()
        for rows in stages[1].values():
            for row in rows:
                nomination=_read(row['nomination']);nomination_refs.add(row['nomination']['sha256'])
                _require(nomination['schema']=='ralup.meta_nomination.v1' and nomination['nominees']==sorted(nominees.values()),'Frozen stage-two nominees mismatch')
                _require(sorted(r['sha256'] for r in nomination['meta_endpoints'])==sorted(r['endpoint']['sha256'] for r in meta_dev),'Nomination does not bind complete META-development endpoint inventory')
                for ref in nomination['meta_endpoints']:_read(ref)
                _require(max(r['completed'] for rows in stages[0].values() for r in rows)<=_time(nomination['created_utc']),'Nomination precedes completed META endpoints')
                _require(_time(nomination['created_utc'])<=row['design_created']<=row['started'],'Nominees must be frozen before official endpoint work')
        _require(len(nomination_refs)==1,'One prospective official nomination required')
        winner=min(stages[1],key=lambda k:(rank(stages[1][k]),k))
        return stages[1][winner][0]['controller']
    except (KeyError,TypeError,OSError,OverflowError) as error:
        raise ValueError('Incomplete or malformed selection authority') from error
