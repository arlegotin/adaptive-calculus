"""Risk-ordered actual core geometry scopes; short curves never choose B0."""
import copy,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity,create_trial_state
from ralup.records import ArtifactRef,sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write,_offsets,_Sampler,_limits,memory_report,_timed
from ralup.state import capture_snapshot,restore_snapshot,save_snapshot,load_snapshot,exact_cpu_state_hash
from ralup.train import train_step,select_logical_batch
from ralup.optim.base import SchedulePoint
from ralup.baselines.muon_checks import check_muon
from ralup.baselines.soap_madam_checks import check_soap_madam
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-ordinary-geometry-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Ordinary geometry input hash mismatch')

if sys.argv[1]=='freeze':
    template=json.loads((root/'docs/harness/m07-derived-parent-design-v1.json').read_bytes())['fresh_config']
    parent=ref(base/'native-derived-parent/derived-warmstart.json')
    variants=[('muon-plus-none','muon_plus_none_fp32',.02,{}),('muon-plus-row-col','muon_plus_row_col_fp32',.02,{}),
        ('soap-refresh10','soap_pinned_fp32',.001,{'precondition_frequency':10}),('madam-default','madam_clean_fp32',.01,{})]
    cases=[]
    for label,method,lr,settings in variants:
        c=copy.deepcopy(template);c.update(label=label,source_identity=source_identity(),source_revision='737233e7ff8d8d2246aaa02598a270ded9068b4d',
            parent=parent,parent_kind='derived_model_only',budget=16*8192,role='continuation',partition='target_train')
        c['recipe'].update(method=method,learning_rate=lr,method_config=settings,quantizer='row_dynamic_max',rotation='identity',backward='ste')
        cases.append(c)
    m=template['resolved']['data']['manifest'];cursor=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=8192)
    inputs=next_logical_batch(cursor,8192)
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),'helper':ref(Path(__file__)),
        'supervisor':ref(root/'docs/harness/m07-supervise.py'),'supervisor_checks':ref(base/'green-external-deadline/finish.json'),
        'purpose':'current primitive parity/operation and actual core shape/capacity/strict replay for risk-ordered available strong ordinary geometry',
        'cases':cases,'steps':16,'paired_arms':2,'comparisons':[1,8,11,16],'strict_restore_step':8,'maximum_new_steps':128,
        'finite_forecast_seconds':600,'child_deadline_seconds':600,'complete_attempt_forecast_seconds':610,'forecast_bytes':4*1024**3,
        'forecast_basis':'core identity train .5-.9s/step plus128 transactions, four authenticated model-only imports, small current primitive parity and declared snapshot/restore allowance; actual SOAP eig/QR timing unresolved and measured here',
        'state_forecast':{'observed_core_quantized_muon_generation_bytes':110074919,'observed_core_fp32_muon_generation_bytes':58664057,
            'internal_matrices':24,'SOAP_GG_bytes':44040192,'SOAP_Q_bytes':44040192,'SOAP_matrix_moment_bytes':25165824,
            'retained_generations_per_case':6,'pool_staging_receipt_allowance_bytes':4*1024**3},
        'source_routes':{'Muon+':'independent published arXiv2602.21545v3 equations LOCAL, no author software copied/grant inferred; EMA/noNesterov and normalization none vs row_col; all other named orders retained untested competitively',
            'SOAP':'pinned permitted MIT nikhilvyas/SOAP a1e553530fde97d0e6b307d7c82ac6d38b072340; exact CPU eig/QR and timed transfers, source epsilon/bias/decay; refresh1/4 remain unsettled alternatives',
            'Madam':'independent arXiv2006.14560v2 equations LOCAL, no CC BY-NC-SA author code copied; bound3RMS, beta.999, clip10, initialLR.01; no grant/release/legal assurance'},
        'ordinary_contract':'all shared OrderedEmbedding; original-W parameter/momentum convention or SOAP documented original-W moment mapping; independent auxiliary AdamW .001/.9/.99/1e-8; no adapter eligibility filter',
        'retention':'arm0 milestone8 plus completed15/16; arm1 strict8 plus completed15/16; initial arm0 state retained for one shared same-W query; separate I/O audit, raw every-step records, failures preserved',
        'query':{'partition':'development','seed':1707,'target_bytes':8192,'offsets':_offsets(inputs)},
        'prospective_queries':'one shared row_dynamic initial and each qualified arm0 step8/16, same exact8192 list; old262144 same-W scores never substituted',
        'limits':'16-step measured bootstrap only, no LR bracket, competitive defeat, full-budget selection, independent seed replication or B0; all unsettled families/variants advance',
        'cost_contract':'complete primitive/setup/import/replay/checkpoint/failure costs charged; same old mixed parent ancestry once globally,128 actual new physical steps maximum; replica arms not independent seeds'}
    _write(designpath,design);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity();auth(d['helper']);auth(d['supervisor'])
    out=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();runs=[];scopes=[];failure=None
    parity={}
    for name,function in (('muon',check_muon),('soap-madam',check_soap_madam)):
        t=time.perf_counter();r=function('mps',out/(name+'-primitive-parity.json'));result=json.loads(r.path.read_bytes())
        parity[name]={'report':ref(r.path),'status':result['status'],'complete_seconds':time.perf_counter()-t}
    _write(out/'primitive-results.json',parity)
    for config in d['cases']:
        case=out/config['label'];case.mkdir();run={'config':config,'arms':[]};state=None
        try:
            requirement='muon' if config['recipe']['method'].startswith('muon') else 'soap-madam'
            if parity[requirement]['status']!='pass':raise RuntimeError('Current primitive parity/operation blocked; not a defeated method')
            with _Sampler('mps',_limits(config['resolved'])) as sampler:
                def cap():
                    sampler.sample()
                    if sampler.exceeded or time.perf_counter()-started>d['finite_forecast_seconds']:raise RuntimeError('Frozen geometry time/memory envelope exhausted')
                def save(label):
                    snapshot,a=_timed(state.model,lambda:capture_snapshot(state));saved,b=_timed(state.model,lambda:save_snapshot(snapshot,case/label))
                    _write(case/(label+'-io.json'),{'checkpoint':ref(saved.path),'capture_seconds':a,'write_seconds':b,
                        'scope':'included in complete attempt; separate from paired scientific history'})
                    return ref(saved.path)
                cap();state=create_trial_state(config);initial=capture_snapshot(state);expected={}
                run['initial']=save('initial');run['initial_state_sha256']=exact_cpu_state_hash(state)
                for arm in range(2):
                    cap();state=restore_snapshot(initial,'mps');armstart=time.perf_counter();saved=[]
                    with (case/f'arm-{arm}.jsonl').open('x') as raw:
                        for index in range(16):
                            cap()
                            if arm==1 and index==8:
                                r=save('arm1-strict8');saved.append({'step':8,**r});state=restore_snapshot(load_snapshot(ArtifactRef(**r)),'mps')
                            state.schedule['point']=SchedulePoint(config['recipe']['learning_rate'],0.,index/15)
                            logical=select_logical_batch(state,8192);offsets=_offsets(logical);events=len(state.ledger.events);status='failed'
                            try:train_step(state,logical);status='completed'
                            finally:
                                raw.write(canonical_bytes({'execution_id':f'{out.name}:{config["label"]}:{arm}:{index}',
                                    'step':index+1,'inputs':offsets,'events':state.ledger.events[events:],
                                    'attempt_status':status,'unknown_partial_work':status=='failed'}).decode()+'\n');raw.flush()
                            cap()
                            if index+1 in d['comparisons']:
                                value=exact_cpu_state_hash(state);_write(case/f'comparison-{arm}-{index+1}.json',{'step':index+1,'state_sha256':value})
                                if arm==0:expected[index+1]=value
                                elif expected[index+1]!=value:raise RuntimeError(f'Exact complete owners differ at step{index+1}')
                            if (arm==0 and index+1==8) or index>=14:saved.append({'step':index+1,**save(f'arm{arm}-completed-{index+1}')})
                    row={'arm':arm,'saved':saved,'complete_seconds':time.perf_counter()-armstart,
                        'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed}
                    run['arms'].append(row);_write(case/f'arm-{arm}-completed.json',row)
                run['memory']=memory_report(sampler.samples)
            scopes.append(['core',config['recipe']['method'],'row_dynamic_max','identity','ste',32])
        except Exception as error:
            failure={'label':config['label'],'error':str(error),'completed_step':None if state is None else state.schedule['step']};run['failure']=failure
            if state is not None:
                try:run['stopped_state']=save('stopped-state')
                except Exception as e:run['retention_error']=str(e)
        finally:
            runs.append(run);_write(case/'scope-result.json',run)
        if failure:break
    report={'status':'blocked' if failure else 'pass','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'scopes':scopes,'maximum_steps':16,'runs':runs,'primitive_parity':parity,'complete_seconds':time.perf_counter()-started,
        'source_routes':d['source_routes'],'limits':d['limits'],'cost_contract':d['cost_contract']}
    _write(out/'admission.json',report);print(json.dumps({'report':ref(out/'admission.json'),'status':report['status'],'failure':failure}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run required')
