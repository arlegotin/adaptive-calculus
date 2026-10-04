"""Read-only N01 owner/alias/source/cost evidence publication; no GPU work."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import time

ROOT=Path('artifacts/calibration/n01-v1')


def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def auth(value):
    path=Path(value['path']);assert ref(path)=={k:value[k] for k in ('path','sha256')};return path


def read(path):return json.loads(path.read_bytes())


def resolve_science(value):
    """Return an authentic retained canonical file; disposed IDs stay historical."""
    path=Path(value['path'])
    if path.exists():return auth(value)
    for audit_path in ROOT.glob('*/canonical-alias-audit.json'):
        audit=read(audit_path)
        for alias in audit['explicit_aliases_and_discard']:
            old=alias['original_historical_compressed_ref']
            if old=={k:value[k] for k in ('path','sha256')}:
                assert alias['independent_complete_payload_sha_and_length_equal']
                assert alias['original_payload']==alias['canonical_payload']
                assert alias['original_payload']['sha256']==value['uncompressed_sha256']
                return auth(alias['retained_canonical_ref'])
    raise ValueError('Missing scientific file has no authenticated canonical alias')


def run():
    start=time.perf_counter();design=read(Path('docs/harness/n01-design.json'))
    assert all(ref(Path(name))['sha256']==digest for name,digest in design['producer'].items())
    wave=read(ROOT/'wave.json');assert all(r['accepted'] for r in wave['results'])
    parity=read(ROOT/'parity-supplement/report.json');assert parity['accepted']
    capacity=read(ROOT/'capacity-recovery/report.json');assert capacity['new_actual_optimizer_steps']==7
    assert len(capacity['arms'])==6 and all(a['completed_steps']==1 for a in capacity['arms'])
    assert all(a['cost']['stream']['updates']==1 for a in capacity['arms'])
    for entry in capacity['journal']:auth(entry)
    aliases=[];study=[];canonical_checked=set()
    for recipe in design['recipes']:
        result=read(ROOT/recipe/'result.json');assert result['accepted'] and not result['failures']
        audit=read(ROOT/recipe/'canonical-alias-audit.json');done=read(ROOT/recipe/'canonicalization-completed.json');auth(done['audit'])
        aliases.extend(audit['explicit_aliases_and_discard'])
        for arm,file in [('U0','arm.json'),('U1','arm.json'),('R','combined.json')]:
            report=read(ROOT/recipe/arm/file)
            assert report['source']==design['producer'] and report['design']==ref(Path('docs/harness/n01-design.json'))
            for step,value in report['states'].items():
                canonical=resolve_science(value)
                if canonical not in canonical_checked:
                    digest=hashlib.sha256();size=0
                    with gzip.open(canonical,'rb') as stream:
                        for block in iter(lambda:stream.read(2**20),b''):digest.update(block);size+=len(block)
                    assert digest.hexdigest()==value['uncompressed_sha256'] and size==value['uncompressed_bytes']
                    canonical_checked.add(canonical)
            for checkpoint in report['checkpoints'].values():auth(checkpoint)
            timings=[r['full_step_seconds'] for r in report['steps']]
            study.append({'recipe':recipe,'arm':arm,'steps':len(timings),'actual_transaction_block_seconds':sum(timings),
                          'mean_step_seconds':statistics.mean(timings),'min':min(timings),'max':max(timings),
                          'sampled_memory':report['memory'],'scope':'observer-interleaved/cold-startup replay observation, not sustained/thermal throughput'})
    assert len(canonical_checked)==24 and len(aliases)==48
    # Reconstruct ONLY scored-byte offsets from authenticated parent cursor.
    # This does not invoke a model or consume another training/query pass.
    from ralup.state import load_snapshot,_validated
    from ralup.records import ArtifactRef
    from ralup.data.cursor import BatchCursor
    from ralup.data.batches import next_logical_batch
    from ralup.calibrate import _offsets
    from ralup.config import canonical_bytes
    parent=read(auth(design['parent']));saved=_validated(load_snapshot(ArtifactRef(**parent['checkpoint'])))
    cursor=BatchCursor({**saved['cursor'],'run_target_budget':saved['cursor']['cumulative_target_count']+65536})
    arm=read(ROOT/'row_fixed_max/U0/arm.json');intervals={}
    def add(blocks):
        for block in blocks:
            values=sorted(set(block['offsets']));spans=[[v,v+1] for v in values]
            merged=[]
            for first,last in sorted(intervals.get(block['document_id'],[])+spans):
                if merged and first<=merged[-1][1]:merged[-1][1]=max(last,merged[-1][1])
                else:merged.append([first,last])
            intervals[block['document_id']]=merged
    for step in arm['steps']:
        batch=next_logical_batch(cursor,8192)
        digest=hashlib.sha256(canonical_bytes({'offsets':_offsets(batch)}))
        for block in batch.block_slices:digest.update(canonical_bytes({'inputs':list(block.inputs),'positions':list(block.positions),'labels':list(block.labels),'mask':list(block.score_mask)}))
        assert digest.hexdigest()==step['input_sha256'];add(_offsets(batch))
    for arm in capacity['arms']+[capacity['warmup']]:add(arm['input_offsets'])
    add(capacity['query']['input_offsets'])
    query_inputs=read(auth(parity['inputs']))
    from ralup.data.blocks import TargetBlock
    add([{'document_id':b['document_id'],'offsets':list(TargetBlock(**b).scored_offsets())} for b in query_inputs['blocks']])
    n01_unique=sum(b-a for spans in intervals.values() for a,b in spans)
    assert n01_unique==65536
    n01_intervals=json.loads(json.dumps(intervals))
    prior=read(Path('docs/harness/h08-project-coverage.json'))
    old_unique=sum(b-a for spans in prior['intervals'].values() for a,b in spans)
    for key,spans in prior['intervals'].items():
        merged=[]
        for first,last in sorted(intervals.get(key,[])+spans):
            if merged and first<=merged[-1][1]:merged[-1][1]=max(last,merged[-1][1])
            else:merged.append([first,last])
        intervals[key]=merged
    total_unique=sum(b-a for spans in intervals.values() for a,b in spans)
    coverage={'basis':'project-global union of raw source-byte offsets; all72 input hashes verified against authenticated parent/cursor; queries/capacity actual explicit offsets merged; priorH08 union retained unchanged',
              'prior':ref(Path('docs/harness/h08-project-coverage.json')),'n01_scored_source_unique':n01_unique,
              'n01_incremental_novel_unique_after_h08':total_unique-old_unique,'project_union_unique':total_unique,'intervals':intervals,'n01_intervals':n01_intervals}
    Path('docs/harness/n01-project-coverage.json').write_text(json.dumps(coverage,sort_keys=True,indent=2)+'\n')
    memory_paths=[ROOT/'components.json',ROOT/'parity-supplement/report.json',ROOT/'capacity-recovery/report.json']
    milestones=['docs/harness/n01-design.json','docs/harness/n01-parity-supplement.json','docs/harness/n01-storage-supplement.json',
                'docs/harness/n01-administrative-recovery.json','docs/harness/n01-administrative-recovery-v1.json','docs/harness/n01-administrative-recovery-v2.json','docs/harness/n01-administrative-recovery-v3.json','docs/harness/n01-capacity-recovery.json']
    records=list(ROOT.glob('*.json'))+list(ROOT.glob('*/result.json'))+list(ROOT.glob('*/canonical*-*.json'))+memory_paths
    # Publication stdout is still open here; do not record its unfinished hash.
    logs=[p for p in Path('artifacts/checks').glob('n01-*.txt') if not p.name.startswith('n01-consolidate')]
    evidence={'schema_version':1,'task':'N01','status':'implemented_pending_independent_review','product_commit':'ffcd4bb','preimplementation_amendment_commit':'46bb509',
              'primitive':design['primitive'],'producer':design['producer'],'designs':[ref(Path(p)) for p in milestones],
              'gates':{'covering_actual_mps_suite':{'tests':559,'passed':559,'seconds':62.90,'log':ref(Path('artifacts/checks/n01-covering-suite.txt'))},
                       'wrong_derivative_control':'actual wrongautograd derivative rejected by finite-difference oracle; subsequentfocused1passed0.99s after fullsuite, no productchange',
                       'components':ref(ROOT/'components.json'),'replay':ref(ROOT/'wave.json'),'whole_model_composition_parity':ref(ROOT/'parity-supplement/report.json'),
                       'capacity_direct_revised':ref(ROOT/'capacity-recovery/report.json')},
              'capacity':{'original_attempt':'FAILED; searchreport/selection/exactcounts unavailable; no searchsuccess relabel',
                          'lost_completed_optimizer_step_bounds':[1,55],'lost_partial_attempt_bounds':[0,7],'partial_target_upper_each':8192,'failed_process_real_seconds':102.71,
                          'new_actual_complete_optimizer_steps':7,'completed_total_upper':62,'revised_total_full_step_equivalent_upper_including_lost_partial_attempts':69,
                          'new_query_bytes':8192,'old_known_query_bytes':8192,'provisional_packing':64,'packing_selection':'NONE;64validatedreplay provisional,1referenceonly',
                          'direct_observations':[{'recipe':a['recipe'],'packing':a['size'],'steps':1,'actual_block_seconds':a['full_step_seconds'],'memory':a['memory']} for a in capacity['arms']],
                          'full_process_real_seconds':40.26,'limits':'single-step/no sustained or thermalsteady/no optimum confidence; lost-cost uncertainty invalidates precise forecasts/budget decisions until M07 remeasures'},
              'retained_64_study_timing_observations':study,'cost':{'replay_new_completed_steps':72,'replay_new_scored_bytes':589824,
                       'parity_distinct_forward_backward_query_passes':2,'parity_scored_query_bytes':16384,'parity_full_process_real_seconds':16.49,
                       'direct_capacity_actual_optimizer_steps':7,'capacity_actual_scored_step_bytes':57344,'known_capacity_query_bytes_across_attempts':16384,
                       'lost_step_partial_costs':'upperbounded expenditure, not actual counters; classifications query/additional_loss/loss_targets describe same querypass and are not summed as separate passes',
                       'parent_common_cost':design['parent'],'parent_expenditure':'full immutable warmstartreport/cursor/RNG/sourceprovenance preserved; charge commonancestry once, not once per duplicate arm',
                       'replay_chronological_union_seconds':wave['full_project_union_seconds_original_start_through_recovery'],
                       'replay_original_2400s_envelope_exceeded':wave['total_envelope_exceeded'],'hold_elapsed_seconds':read(ROOT/'resource-release.json')['full_hold_elapsed_seconds'],
                       'clock_basis':'waveunion includes hold/rulingwait/originalfailedcontroller/recoveryonce; overlapping arm/canonicalization componentclocks remainclassified, not added twice. Independent parity/capacity process clocks include startup/retention/exit.',
                       'engineering_tests':'allCPU/device/TDDfailediterations and complete559suite charged as engineeringwork; numericalfixtures are synthetic context work, not scientific scored corpusbytes'},
              'storage':{'retained_task_artifact_bytes_before_evidence':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),'free_bytes':shutil.disk_usage(ROOT).free,
                         'new_cap_bytes':2*2**30,'free_floor_bytes':2**30,'canonical_full_steps_retained':24,'canonical_aliases':48,
                         'alias_audits':[ref(ROOT/r/'canonical-alias-audit.json') for r in design['recipes']],
                         'meaning':'disposedcompressedidentities remainhistorical; resolve explicitsame-step aliases to authentic retainedU0completevalues, including everylocalquantizer operand. Allcheckpointgens/failure/process/armcostrecords remain; no old/user artifacts removed.',
                         'resolution_helper':ref(Path(__file__))},
              'project_coverage':ref(Path('docs/harness/n01-project-coverage.json')),
              'historical_failures':{'h08':ref(Path('docs/harness/h08-evidence.json')),'h08_cap_fix':ref(Path('docs/harness/h08-fix1-evidence.json')),'all12_native_v4_scopes':'FAILED unchanged; missinghistoricalforensiccontexts remainunavailable'},
              'administrative_failures':'originalcontroller hold-inducedTimeoutExpired retained; QuESTU0 reportedscientificcompletion authenticated, cleanOSexit/stdout unavailable. Recoveryv1/v2 preflightfailures preserved; no completedscientificwork rerun.',
              'downstream':'M01/M02/M07/M08/M10/M11/E01/E02/E03/E07 receive selectedprimitive/runtime/limitedreplay and provisional64 observations. Newmethod/core/longer/otherpacking scopes require risk-based prospective validation. ScientificB0/duration/confirmation budgets unresolved; protectedtestdenied.',
              'records':[ref(p) for p in sorted(set(records))],'logs':[ref(p) for p in sorted(logs)],'read_only_consolidation_seconds_before_publication':time.perf_counter()-start}
    assert evidence['storage']['retained_task_artifact_bytes_before_evidence']<2*2**30 and evidence['storage']['free_bytes']>2**30
    Path('docs/harness/n01-evidence.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'evidence':ref(Path('docs/harness/n01-evidence.json')),'project_coverage':ref(Path('docs/harness/n01-project-coverage.json')),'n01_unique':n01_unique,'incremental_unique':total_unique-old_unique,'seconds':time.perf_counter()-start}))


if __name__=='__main__':run()
