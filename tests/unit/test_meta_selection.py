"""Finite authority fixtures test admission only; they are not measured science."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef,sha256_file
from ralup.policy.controllers import ControllerArtifact,ControllerState,Affine45,LearnedLR49,Static3,freeze_controller
from ralup.meta.selection import select_controller


def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()


class Evidence:
    def __init__(self,root):self.root=root;self.serial=0
    def write(self,value):
        self.serial+=1;p=self.root/f'evidence-{self.serial}.json';p.write_bytes(canonical_bytes(value));return {'path':str(p),'sha256':sha256_file(p)}
    def read(self,ref):return json.loads(Path(ref['path']).read_bytes())
    def change(self,ref,**changes):return self.write({**self.read(ref),**changes})


def selection_fixture(tmp_path,monkeypatch):
    from ralup.meta import selection
    from ralup.search import source_identity
    e=Evidence(tmp_path);source=source_identity();prefix='a'*64
    documents=[]
    for partition in ('meta-train','meta-development','development'):
        raw=(partition+' bytes').encode();path=tmp_path/(partition+'.txt');path.write_bytes(raw)
        documents.append({'id':partition,'partition':partition,'split':'validation' if partition=='development' else 'train','protected':partition=='development','content_path':str(path),'content_start':0,'bytes':len(raw),'content_sha256':hashlib.sha256(raw).hexdigest()})
    data=e.write({'synthetic':False,'audit':{'retained_documents':documents[:2],'protected_documents':documents[2:]},'partitions':{'meta-train':{},'meta-development':{}}})
    data_index=tmp_path/'data-index.json';data_index.write_bytes(canonical_bytes({'manifest':data,'frozen_hashes':{'source_sha256':prefix}}))
    monkeypatch.setattr(selection,'DATA_INDEX',data_index,raising=False)
    recipe=e.write({'schema':'ralup.exploratory_backbone.v1','status':'EXPLORATORY','b0':None,'selected_b0':False,'confirmation_eligible':False,'recipe':{'matrix_lr':.02}})
    recipe_index=tmp_path/'recipe-index.json';recipe_index.write_bytes(canonical_bytes({'exploratory_backbone':recipe}));monkeypatch.setattr(selection,'RECIPE_INDEX',recipe_index,raising=False)
    trajectory=e.write({'schema':'ralup.feature_trajectory.v1','partition':'meta_train','role':'reference','synthetic':False,'data_manifest':data})
    normalizer=e.write({'mean':[0.]*14,'std':[1.]*14,'source_manifest':json.dumps({'schema':'ralup.frozen_normalizer_sources.v1','partition':'meta_train','rows':2,'trajectories':[trajectory]})})
    opportunities=[{'origin':origin,'age':1,'horizon':2,'pairs':1,'query_targets':4,'features':14,'history':True,'refresh_steps':4 if origin=='own-policy' else 0} for origin in ('baseline','own-policy')]
    design=e.write({'schema':'ralup.matched_meta_training.v1','purpose':'final-matched-training','synthetic':False,'created_utc':'2026-10-11T00:00:00+00:00','source_identity':source,'backbone':recipe,'normalization':normalizer,'data':data,'opportunities':opportunities})
    def producer(design,utc='2026-10-11T01:00:00+00:00'):return e.write({'utc':utc,'source':{k:{'sha256':v} for k,v in source.items()},'inputs':[design]})
    campaigns={};states={}
    for controller in (Affine45(),LearnedLR49(),Static3()):
        state=freeze_controller(controller);phi=state.parameters.tolist();states[state.family]=phi;records=[]
        for i,op in enumerate(opportunities):
            refresh=None
            if i:
                refresh=e.write({'schema':'ralup.own_policy_refresh.v1','status':'completed','synthetic':False,'family':state.family,'acting_parameters':phi,'source_identity':source,'data':data,'design':design,'optimizer_steps':4,'features':14,'history':True,'ages':[1],'producer':producer(design)})
            work={'optimizer_steps':2,'query_targets':8,'input_sha256':digest({'i':i}),'initial_state_hash':digest({'state':state.family if i else 'baseline'}),'initial_query':2.}
            result={'initial_query':2.,'final_query':1.9,'work':work}
            generation=e.write({'family':state.family,'outer_step':i+1,'phi_before':phi,'phi_after':phi,'horizon':2,'pairs':[{'plus':result,'minus':result}],'optimizer_steps':4,'query_targets':16})
            records.append(e.write({'schema':'ralup.meta_generation_completion.v1','status':'completed','synthetic':False,'design':design,'source_identity':source,'producer':producer(design),'opportunity_index':i,'generation':generation,'refresh':refresh}))
        campaigns[state.family]=e.write({'schema':'ralup.matched_family_completion.v1','status':'completed','synthetic':False,'family':state.family,'design':design,'source_identity':source,'initial_parameters':states[state.family],'parameters':phi,'generations':records})
    eligibility=e.write({'schema':'ralup.meta_selection_eligibility.v1','status':'eligible_exploratory','synthetic':False,'confirmation_eligible':False,'B0':None,'design':design,'families':campaigns})
    artifacts={family:ControllerArtifact(family,torch.tensor(phi),ArtifactRef(**normalizer),ArtifactRef(**recipe),ArtifactRef(**data),ArtifactRef(**eligibility)) for family,phi in states.items()}
    from ralup.data.blocks import make_block
    from dataclasses import asdict
    def endpoint(artifact,partition,loss,nomination=None):
        doc=next(d for d in documents if d['partition']==partition);raw=Path(doc['content_path']).read_bytes()
        block=asdict(make_block(doc['content_sha256'],0,len(raw),b'',raw,document_id=prefix+':'+doc['id']));block={k:list(v) if isinstance(v,tuple) else v for k,v in block.items()}
        query=e.write({'blocks':[block]})
        frozen=e.write({'schema':'ralup.selection_endpoint_design.v1','created_utc':'2026-10-11T05:00:00+00:00' if partition=='development' else '2026-10-11T01:30:00+00:00','source_identity':source,'controller_sha256':digest(artifact.to_json()),'eligibility':eligibility,'partition':partition,'data':data,'queries':query,'seed':19,'full_horizon':4,'start_step':0,'nomination':nomination})
        receipt=e.write({'schema':'ralup.selection_endpoint.v1','status':'completed','synthetic':False,'design':frozen,'producer':producer(frozen,'2026-10-11T06:00:00+00:00' if partition=='development' else '2026-10-11T02:00:00+00:00'),'completed_utc':'2026-10-11T07:00:00+00:00' if partition=='development' else '2026-10-11T03:00:00+00:00','source_identity':source,'controller_sha256':digest(artifact.to_json()),'partition':partition,'seed':19,'start_step':0,'end_step':4,'full_horizon':4,'nll_sum':loss*len(raw),'target_count':len(raw),'elapsed_seconds':4.,'input_sha256':query['sha256']})
        return {'partition':partition,'controller':artifact,'endpoint':receipt}
    meta=[endpoint(a,'meta-development',1.+i) for i,a in enumerate(artifacts.values())]
    nomination=e.write({'schema':'ralup.meta_nomination.v1','created_utc':'2026-10-11T04:00:00+00:00','meta_endpoints':[r['endpoint'] for r in meta],'nominees':sorted(digest(a.to_json()) for a in artifacts.values())})
    official=[endpoint(a,'development',3.-i,nomination) for i,a in enumerate(artifacts.values())]
    return e,artifacts,meta,official,endpoint


def test_complete_authority_fixture_selects_only_frozen_nominees(tmp_path,monkeypatch):
    e,a,meta,official,_=selection_fixture(tmp_path,monkeypatch)
    assert select_controller(meta,official) is a['static3']


@pytest.mark.parametrize('damage',['relabel','missing_family','synthetic','calibration','vector','source','own_policy','opportunity','full_horizon','loss','nominees','nomination_time','endpoint_input'])
def test_selection_rejects_broken_authority(tmp_path,monkeypatch,damage):
    e,a,meta,official,endpoint=selection_fixture(tmp_path,monkeypatch)
    if damage=='relabel':meta[0]=dict(endpoint(a['affine45'],'meta-train',.01),partition='meta-development')
    elif damage in ('full_horizon','loss','endpoint_input'):
        change={'end_step':2} if damage=='full_horizon' else {'nll_sum':float('nan')} if damage=='loss' else {'input_sha256':'f'*64}
        # Nonfinite bytes can occur outside the canonical writer; hash authentication alone must not admit them.
        if damage=='loss':
            p=tmp_path/'nonfinite.json';p.write_text(json.dumps({**e.read(official[0]['endpoint']),**change}));official[0]['endpoint']={'path':str(p),'sha256':sha256_file(p)}
        else:official[0]['endpoint']=e.change(official[0]['endpoint'],**change)
    elif damage in ('nominees','nomination_time'):
        receipt=e.read(official[0]['endpoint']);d=e.read(receipt['design']);n=e.read(d['nomination'])
        n['nominees']=[] if damage=='nominees' else n['nominees'];n['created_utc']='2026-10-12T00:00:00+00:00' if damage=='nomination_time' else n['created_utc']
        newdesign=e.change(receipt['design'],nomination=e.write(n));official[0]['endpoint']=e.change(official[0]['endpoint'],design=newdesign)
    else:
        original=a['affine45'];record=e.read({'path':str(original.training.path),'sha256':original.training.sha256})
        if damage=='missing_family':record['families'].pop('learned_lr49')
        elif damage=='synthetic':record['synthetic']=True
        elif damage=='calibration':record['status']='calibration_only'
        elif damage=='vector':
            c=e.read(record['families']['affine45']);c['parameters'][0]=.1;record['families']['affine45']=e.write(c)
        elif damage=='source':record['design']=e.change(record['design'],source_identity={'wrong':'f'*64})
        else:
            c=e.read(record['families']['affine45']);r=e.read(c['generations'][1])
            if damage=='own_policy':r['refresh']=e.change(r['refresh'],family='static3')
            else:r['opportunity_index']=0
            c['generations'][1]=e.write(r);record['families']['affine45']=e.write(c)
        replacement=ControllerArtifact(original.family,original.parameters,original.normalization,original.B0,original.data,ArtifactRef(**e.write(record)))
        meta[0]['controller']=replacement
    with pytest.raises(ValueError):select_controller(meta,official)


def test_actual_calibration_pilot_cannot_be_selected():
    store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m11-20261011')
    if not (store/'native-final-calibration-v1/pilot-generation.json').exists():pytest.skip('Local retained genuine pilot absent')
    ref=lambda p:ArtifactRef(p,sha256_file(p))
    p=store/'native-final-calibration-v1/pilot-generation.json';g=json.loads(p.read_bytes())
    a=ControllerArtifact('affine45',torch.tensor(g['phi_after']),ref(store/'native-reference-v1/final-reference-normalizer.json'),ref(Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011/closing-disposition-v2/exploratory-backbone.json')),ref(store/'freeze-final-calibration-v1/query-two-panels.json'),ref(p))
    def row(partition):return dict(partition=partition,controller=a,seed=4112,terminal_loss=1.,elapsed_seconds=1.,horizon=496,full_horizon=496,input_sha256=g['pairs'][0]['plus']['work']['input_sha256'],opportunity_sha256='candidate-only-calibration')
    with pytest.raises(ValueError):select_controller([row('meta-development')],[row('development')])


def replace_endpoint_design(e,row,design,**receipt_changes):
    receipt=e.read(row['endpoint']);newdesign=e.write(design)
    producer=e.change(receipt['producer'],inputs=[newdesign])
    row['endpoint']=e.change(row['endpoint'],design=newdesign,producer=producer,**receipt_changes)


def test_official_subset_cannot_claim_full_development(tmp_path,monkeypatch):
    e,a,meta,official,_=selection_fixture(tmp_path,monkeypatch)
    for row in official:
        receipt=e.read(row['endpoint']);design=e.read(receipt['design']);query=e.read(design['queries'])
        query['blocks'][0]['score_mask'][-1]=False;design['queries']=e.write(query)
        replace_endpoint_design(e,row,design,input_sha256=design['queries']['sha256'],target_count=receipt['target_count']-1)
    with pytest.raises(ValueError,match='full official'):select_controller(meta,official)


def test_nomination_cannot_predate_completed_meta_endpoints(tmp_path,monkeypatch):
    e,a,meta,official,_=selection_fixture(tmp_path,monkeypatch)
    old=e.read(official[0]['endpoint']);design=e.read(old['design'])
    nomination=e.change(design['nomination'],created_utc='2026-10-11T02:30:00+00:00')
    for row in official:
        design=e.read(e.read(row['endpoint'])['design']);design['nomination']=nomination
        replace_endpoint_design(e,row,design)
    with pytest.raises(ValueError,match='completed META'):select_controller(meta,official)
