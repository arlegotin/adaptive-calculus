"""Authenticate and summarize existing H08 artifacts without rerunning benchmarks."""
import hashlib
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import time

PROCESS_START=time.perf_counter()

from ralup.calibrate import reconcile_cost_scope, _offsets
from ralup.config import canonical_bytes
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch
from ralup.records import ArtifactRef
from ralup.state import load_snapshot, _validated
from safetensors.torch import load_file


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def producer(value):
    path = Path(value['path'])
    assert ref(path) == value
    manifest = json.loads(path.read_bytes())
    files = {}
    for name, entry in manifest['producer_files'].items():
        archive = ref(path.parent / entry['path'])
        assert archive['sha256'] == entry['sha256']
        files[name] = archive
    return {'checkpoint': value, 'files': files,
            'identity_sha256': hashlib.sha256(json.dumps({k: v['sha256'] for k,v in files.items()},sort_keys=True).encode()).hexdigest()}


def ledger(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    return {'files': [ref(directory/name) for name in ('run.json','metrics.jsonl','status.jsonl')],
            'source_at_startup': json.loads((directory/'run.json').read_bytes())['source'],
            'project_elapsed_observations': [r['project_elapsed_seconds'] for r in rows if 'project_elapsed_seconds' in r]}


def page_delta(data, label):
    def value(point):
        return int(re.search(r'^'+label+r':\s+(\d+)', data[point]['pressure']['stdout'],re.M)[1])
    return value('after')-value('before')


def main():
    start=time.perf_counter()
    names=['synthetic-v1','synthetic-v2','real-proxy-v1','real-sustained-v1','real-search-v1','real-selected-sustained-v1']
    measurements=[];coverage={};coverage_sources=[]
    def add_intervals(incoming):
        for document,spans in incoming.items():
            combined=sorted(coverage.get(document,[])+spans);merged=[]
            for first,last in combined:
                if merged and first<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],last)
                else:merged.append([first,last])
            coverage[document]=merged
    for name in names:
        directory=Path('artifacts/calibration')/('h08-'+name)/'capacity'
        path=directory/'calibration.json'
        if not path.exists(): continue
        raw=json.loads(path.read_bytes())
        scoped=reconcile_cost_scope(raw)
        add_intervals(scoped['global_coverage']['intervals']);coverage_sources.append(ref(path))
        accounted=directory/'calibration-accounted-v2.json'
        data=json.loads(accounted.read_bytes()) if accounted.exists() else raw
        if 'completed_snapshot' not in raw:
            measurements.append({'name':name,'raw':ref(path),'historical_disposition':'preliminary/superseded; no completed-state capacity forecast; all costs retained',
                                 'project_seconds':raw['project_seconds'],'cost':{k:v for k,v in raw['cost'].items() if k not in ('events','warmup_attempts')}})
            continue
        initial,completed=producer(raw['snapshot']),producer(raw['completed_snapshot'])
        assert initial['identity_sha256']==completed['identity_sha256']
        arms=[]
        for arm in data['arms']:
            times=arm['full_step_seconds']
            arms.append({'packing':arm['microbatch_actual'],'recipe':arm['recipe'],'steps':len(times),
                         'block_seconds':sum(times),'step_seconds_interval':[min(times),max(times)],
                         'memory':arm['memory'],'cap_exceeded':arm['memory_cap_exceeded']})
        measurements.append({'name':name,'raw':ref(path),'accounted':ref(accounted) if accounted.exists() else None,
            'historical_disposition':'preliminary/superseded; all attempted costs retained' if name=='synthetic-v1' else 'measured engineering scope only',
            'producer':initial,'ledger':ledger(directory),'cost':{k:v for k,v in data['cost'].items() if k not in ('events','warmup_attempts')},
            'global_coverage_digest':data.get('global_coverage',{}).get('sha256'),
            'project_seconds':data['project_seconds'],'selected_packing':data['selected_packing'],
            'restored_initial_exact':data['original_initial_hash']==data['post_warmup_initial_hash'],
            'arms':arms,'caps':data['caps'],'storage':data['storage'],'fit':data['fit'],
            'cold_load':data['cold_load'],'forecast':data.get('forecast'),
            'paging_deltas':{key:page_delta(data['system'],key) for key in ('Pageouts','Swapouts')} if not data.get('synthetic') else None,
            'overhead_seconds':{key:data[key] for key in ('reset_seconds','query_seconds','checkpoint_seconds')}})
    warmstarts=[]
    for path in sorted(Path('artifacts/calibration/h08-real-proxy-v1').glob('warmstart-*/warmstart.json')):
        data=json.loads(path.read_bytes())
        warmstarts.append({'report':ref(path),'producer':producer(data['checkpoint']),'ledger':ledger(path.parent),
                          **{k:v for k,v in data.items() if k not in ('cost','checkpoint')},
                          'cost':{k:v for k,v in data['cost'].items() if k!='events'}})
    failed=[]
    for path in sorted(Path('artifacts/calibration/h08-real-import-v1').glob('meta_train/arm-*/checkpoint/g-*/manifest.json')):
        failed.append({'producer':producer(ref(path)),'ledger':ledger(path.parents[2]),'disposition':'one-step post-update bitwise check failed; initial imported weights passed'})
    checks=[{'artifact':ref(path),'last_lines':path.read_text().splitlines()[-3:]} for path in sorted(Path('artifacts/checks').glob('h08-*.txt'))]
    diagnostic=Path('artifacts/calibration/h08-embedding-diagnostic-v1/diagnostic.json')
    data=json.loads(diagnostic.read_bytes())
    result={'schema_version':1,'scope':'H08 infrastructure/capacity/import engineering evidence; all twelve prospective v4 numerical scopes FAILED; downstream numerical/scientific acceptance CLOSED; no B0/convergence/protected score',
            'review_base':'b8d2c27','wp00_prerequisite_commit':'1e27cd4',
            'wp00':[ref('docs/data/wp00-local-gate-v2.json'),ref('docs/data/wp00-local-acceptance-v2.json')],
            'current_runtime':[ref(p) for p in sorted(Path('src/ralup').rglob('*.py'))],
            'producer_policy':'Each measurement producer is authenticated by immutable initial/completed H07 archives; identical producer maps verified. No src edit occurred while any measured process was alive. Historical raw reports unchanged; separate later accounting analysis is not attributed to historical producer. Early untracked calibrate.py omitted by git diff is nevertheless present/authenticated in complete source archives.',
            'measurements':measurements,'warmstarts':warmstarts,'failed_paired_import':failed,'preoptimization_setup_failures':ref(Path('artifacts/calibration/h08-contract-setup-failures.json')),
            'isolated_embedding_diagnostic':{'report':ref(diagnostic),'repeat_data':data['repeat_data'],'rows':data['rows'],
               'input_shape':data['input_shape'],'upstream_shape':data['upstream_shape'],'ledger':ledger(diagnostic.parent),
               'project_seconds':data['project_seconds'],'cost_passes':data['cost']['passes']},
            'analysis_script':ref(Path(__file__)),
            'checks':checks,'current_free_storage_bytes':shutil.disk_usage('.').free,
            'limitations':['sampled maxima only; overlapping RSS/Metal/driver cannot be summed','uncontrolled filesystem caches','bounded timed blocks15-43s and 8-step proposed numerical horizon are not final scientific forecasts','SOAP state/storage/time pending actual M02/M07','same-device tiny-fixture exact contract remains historical; observed real-shape post-step failure not silently accepted'],
            'consolidation_seconds':time.perf_counter()-start}
    # Project counts derive from additive newly executed RuntimeLedger rows, never
    # by summing inherited warmstart expenditure in each continuation report.
    totals={key:0 for key in ('stream','backward','query','additional_loss','forward_context')}
    run_ledgers=[];observed_project_seconds=0.
    for path in sorted(Path('artifacts/calibration').glob('h08-*/**/run.json')):
        directory=path.parent
        metrics=directory/'metrics.jsonl'
        if not metrics.exists():continue
        rows=[json.loads(line) for line in metrics.read_text().splitlines()]
        for row in rows:
            for key in totals:totals[key]+=row.get(key) or 0
        observations=[r['project_elapsed_seconds'] for r in rows if 'project_elapsed_seconds' in r]
        observed_project_seconds+=sum(observations)
        run_ledgers.append(ledger(directory))
    # Reconstruct only input/offset identities from authenticated immutable cursors;
    # no model execution, protected score, new training or numerical rerun.
    parents=json.loads(Path('artifacts/checks/h08-real-warmstarts-cli.txt').read_bytes())
    expected_inputs={}
    for partition,parent_ref in parents.items():
        parent=json.loads(Path(parent_ref['path']).read_bytes())
        saved=_validated(load_snapshot(ArtifactRef(**parent['checkpoint'])))
        initial={**saved['cursor'],'epoch':0,'permutation_position':0,'carry':None,'cumulative_target_count':0,'unique_target_count':0,'repeated_target_count':0,'run_target_budget':73728}
        assert saved['cursor']['epoch']==0
        cursor=BatchCursor(initial);traces=[]
        for step in range(9):
            batch=next_logical_batch(cursor,8192);assert batch.target_count==8192
            offsets=_offsets(batch)
            spans={}
            for block in offsets:
                values=block['offsets'];spans.setdefault(block['document_id'],[]).append([min(values),max(values)+1])
            add_intervals(spans)
            identity=hashlib.sha256(canonical_bytes({'offsets':offsets}))
            for block in batch.block_slices:
                for value in (block.inputs,block.labels,block.score_mask):identity.update(canonical_bytes({'tokens':list(value)}))
            if step:traces.append(identity.hexdigest())
        expected_inputs[partition]=traces
        coverage_sources.append(parent_ref)
    for path in list(Path('artifacts/calibration/h08-mps-contract-v4').glob('*/**/arm.json'))+list(Path('artifacts/calibration/h08-mps-contract-v4').glob('*/R/replay-first.json')):
        arm=json.loads(path.read_bytes())
        for row in arm['steps']:assert row['input_sha256']==expected_inputs[arm['case']['partition']][row['step']-1]
    coverage_path=Path('docs/harness/h08-project-coverage.json')
    coverage_path.write_text(json.dumps({'basis':'H08 project-global union across all retained calibration attempts/query offsets and authenticated cursor reconstruction; half-open scored raw source-byte intervals. Immutable parent/source/permutation/RNG inputs match every wave arm digest; fixtures are explicitly distinct source identities.',
       'sources':coverage_sources,'intervals':coverage,'expected_wave_inputs':expected_inputs},sort_keys=True,indent=2)+'\n')
    distinct=sum(last-first for spans in coverage.values() for first,last in spans)
    result['project_accounting']={'stream':{'targets':totals['stream'],'unique':distinct},'unique_scope':'H08 project-global union; fixtures and real source identities explicit; never summed run-local uniqueness','coverage':ref(coverage_path),'newly_executed_row_counts':totals,'count_scope':'all H08 named raw run rows, including preliminary/failed/component/smoke/wave; inherited parent costs are not re-added per child',
       'run_ledgers':run_ledgers,'sum_observed_run_project_seconds':observed_project_seconds,
       'time_scope':'per-run project observations; wave wall includes child interpreter startup and comparison/discard overhead separately; these overlapping clocks must not be added'}
    wave_directory=Path('artifacts/calibration/h08-mps-contract-v4')
    if (wave_directory/'design.json').exists():
        design=json.loads((wave_directory/'design.json').read_bytes())
        cases=[]
        for path in sorted(wave_directory.glob('*/result.json')):
            outcome=json.loads(path.read_bytes())
            envelope_path=Path(outcome['frozen_envelope']['path'])
            assert ref(envelope_path)==outcome['frozen_envelope']
            frozen=json.loads(envelope_path.read_bytes())
            pairs=frozen['all_six_pairs']+[item['comparisons'] for item in outcome['heldout']]
            tensor_rows=scalar_fields=0
            for pair in pairs:
                for step,check in pair.items():
                    witness=check.get('raw_max_error_witnesses')
                    raw=load_file(witness['path']) if witness else {}
                    if witness:assert ref(Path(witness['path']))==witness
                    for owner,row in check['owners'].items():
                        tensor_rows+=owner!='nll_sum'
                        if owner in raw:
                            values=raw[owner]
                            if row['kind']=='floating':assert float(abs(values[0].double()-values[1].double()))==row['max_abs']
                            else:assert values[0]!=values[1]
                            assert 'raw_witness_flat_index' in row
            for failure in outcome['failures']:
                if 'envelope' in failure:
                    item=next(v for v in outcome['heldout'] if v['name']==failure['stage'])['comparisons'][failure['step']]
                    owner=failure['envelope'];assert item['owners'][owner]['max_abs']>failure['frozen']
                    if owner!='nll_sum':assert item['raw_max_error_witnesses'] is not None
            cases.append({'report':ref(path),'case':outcome['case'],'disposition':outcome['disposition'],'failure_count':len(outcome['failures']),
                          'failures_by_kind':dict(Counter('envelope' if 'envelope' in item else 'discrete' if 'discrete' in item else 'ceiling' for item in outcome['failures'])),
                          'failures_by_stage':dict(Counter(item['stage'] for item in outcome['failures'])),
                          'authenticated_tensor_owner_comparison_rows':tensor_rows,'raw_witness_max_errors_recomputed':True,
                          'frozen_envelope':outcome['frozen_envelope'],
                          'raw_witnesses':[ref(p) for p in sorted(path.parent.glob('*witness.safetensors'))],
                          'audited_discard':[ref(p) for p in sorted(path.parent.glob('*discard.json'))],
                          'replay_generations':[producer(ref(p)) for p in sorted((path.parent/'R/checkpoint').glob('g-*/manifest.json'))]})
        observer=wave_directory/'forensic-retention-v1/observer.json'
        supplement=wave_directory/'forensic-retention-v1/supplement.json'
        case03=wave_directory/'03-meta_train-row_dynamic_max-64/case03-available-input-context.json'
        case11=wave_directory/'11-target_train-quest_fp4_row_reference-64/quantization-input-audit.json'
        audit=json.loads(case11.read_bytes())
        assert ref(audit['raw_paired_inputs']['path'])==audit['raw_paired_inputs']
        result['prospective_numerical_wave']={'forensic_retention_supplement':ref(supplement) if supplement.exists() else None,
              'observer_report':ref(observer) if observer.exists() else None,'case03_available_context_and_missing_pair_limit':ref(case03) if case03.exists() else None,
              'case11_saved_input_audit':{'report':ref(case11),'raw_paired_inputs':audit['raw_paired_inputs'],
                 'changed_code_coordinates':audit['changed_code_coordinates_across_all_pairs_and_stages'],
                 'identical_numeric_input_changed_code_count':audit['identical_numeric_input_changed_code_count'],
                 'scope':'Saved context7 active-forward/context8 cached endpoint operands only; differing operands straddle E2M1 midpoints. Earlier context6 W/U/scales were never captured, so no earliest-divergence causal claim. Original failed scope remains FAILED.',
                 'analysis_seconds':audit['audit_seconds']},
              'disposition':'ALL twelve scopes FAILED; no bound enlarged; downstream numerical/scientific acceptance CLOSED',
              'design':ref(wave_directory/'design.json'),'cases':cases,'wave':ref(wave_directory/'wave.json') if (wave_directory/'wave.json').exists() else None,
              'retained_bytes':sum(p.stat().st_size for p in wave_directory.rglob('*') if p.is_file()),
              'scope':'root-authorized new prospective8-step enumerated proxy engineering; original violations/historical tiny exact unchanged; failed comparisons never accepted'}
    result['consolidation_seconds']=time.perf_counter()-PROCESS_START
    out=Path('docs/harness/h08-evidence.json')
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(ref(out)))


if __name__=='__main__': main()
