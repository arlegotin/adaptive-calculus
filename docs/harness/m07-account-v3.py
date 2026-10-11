"""Read completed physical-execution records; never sum inherited trajectory ledgers."""
import collections,json,time
from pathlib import Path
from datetime import datetime,timezone
from ralup.records import sha256_file
from ralup.config import canonical_bytes

started=time.perf_counter();root=Path.cwd()
base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
output=root/'docs/harness/m07-accounting-v3.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
producer={'helper':ref(Path(__file__)),'created_utc':datetime.now(timezone.utc).isoformat()}
sources=[];attempts=[];executions=set();unique=collections.defaultdict(set);groups={};unknown=[]
totals=collections.Counter()
def offsets(values):
    count=0
    for row in values:
        unique[row['document_id']].update(row['offsets']);count+=len(row['offsets'])
    return count
for p in sorted(base.glob('*/finish.json')):
    finish=json.loads(p.read_bytes());attempts.append({'name':p.parent.name,'finish':ref(p),
        'startup':ref(p.parent/'startup.json'),'stdout':ref(p.parent/'stdout.txt'),'stderr':ref(p.parent/'stderr.txt'),
        'exit_code':finish['exit_code'],'complete_attempt_seconds':finish['elapsed_seconds']})
rawfiles=sorted(base.glob('native-*/*/arm-*.jsonl'))+sorted(base.glob('native-meta-parent/arm-*.jsonl'))+sorted(base.glob('native-meta-parent/warm.jsonl'))+sorted(base.glob('science/*/steps.jsonl'))+sorted(base.glob('scientific-lr-tail/*/steps.jsonl'))
completed_names={a['name'] for a in attempts}
rawfiles=[p for p in rawfiles if p.relative_to(base).parts[0] in completed_names or p.relative_to(base).parts[0]=='science']
for p in rawfiles:
    sources.append(ref(p));counter=collections.Counter();group=p.relative_to(base).parts[0]
    for index,line in enumerate(p.open()):
        row=json.loads(line);identity=str(p)+':'+str(index)
        assert identity not in executions;executions.add(identity)
        events=row['events'];completed=[e for e in events if e.get('status')=='completed_step']
        expected=row['step'] if group=='science' else row['step']-1
        if len(completed)!=1 or completed[0]['step']!=expected:
            unknown.append({'physical_row':identity,'reason':'missing or stale completed event','events':events});continue
        e=completed[0];assert e['loss_targets']==offsets(row['inputs'])
        counter['physical_optimizer_steps']+=1
        for key in ('stream','loss_targets','backward','query','additional_loss','forward_context'):
            counter[key]+=e.get(key,0)
    totals.update(counter);groups.setdefault(group,collections.Counter()).update(counter)
for p in sorted(base.glob('science/*/curve-*.json')):
    row=json.loads(p.read_bytes());inputs=p.parent/'query-inputs.json';values=json.loads(inputs.read_bytes())['offsets']
    count=sum(v['target_count'] for v in row['per_article'].values());assert count==offsets(values)
    sources.append(ref(p));sources.append(ref(inputs));totals.update(physical_query_passes=1,query=count,additional_loss=count,loss_targets=count)
    groups.setdefault('science',collections.Counter()).update(physical_query_passes=1,query=count,additional_loss=count,loss_targets=count)
for name,filename in (('query-admission128','query-results.json'),('query-extension256','query-results.json'),('query-conversion','conversion-results.json'),('query-conversion-expanded','conversion-results.json'),('query-strong-conversion','conversion-results.json'),('query-meta-parent','query-results.json'),('query-ordinary-geometry','query-results.json'),('query-prioritized-views-v2','query-results.json'),('query-quantizer-geometry','query-results.json')):
    if name not in completed_names:continue
    p=base/name/filename;sources.append(ref(p));d=json.loads(p.read_bytes());inputs=base/name/'query-inputs.json'
    sources.append(ref(inputs));values=json.loads(inputs.read_bytes())['offsets']
    for row in d['rows']:
        if row['status']!='completed':unknown.append({'query':row.get('exposure_id'),'reason':'incomplete query work'});continue
        assert row['actual_query_targets']==offsets(values)
        c=collections.Counter(physical_query_passes=1,query=row['actual_query_targets'],additional_loss=row['actual_query_targets'],loss_targets=row['actual_query_targets'],forward_context=row['forward_context'])
        totals.update(c);groups.setdefault(name,collections.Counter()).update(c)
# Science query receipts retain the common block list, but curve files omit forward
# context lengths. Recover these diagnostic counts from each new-query ledger event
# after its own model-only reset; inherited parent events are explicitly excluded.
for p in sorted(base.glob('science/*/result.json')):
    d=json.loads(p.read_bytes());own=[e for e in d['cost']['ledger']['events'] if e.get('status')=='development_query']
    # Parent events survive import; own query step0 follows its reset event and the
    # final four queries are exactly the prospective0/8/16/32 list for these trials.
    own=own[-4:];assert [e['step'] for e in own]==[0,8,16,32]
    count=sum(e['forward_context'] for e in own);totals['forward_context']+=count;groups['science']['forward_context']+=count
    sources.append(ref(p))
# New LR-tail query counts are actual completed score rows; recover their separate
# forward-context counter by subtracting inherited and raw new-training counters.
for p in sorted(base.glob('scientific-lr-tail/*/result.json')):
    d=json.loads(p.read_bytes());sources.append(ref(p))
    inputs=p.parent.parent/'query-inputs.json';sources.append(ref(inputs));values=json.loads(inputs.read_bytes())['offsets']
    for row in d['curve']:
        count=row['score']['target_count'];assert count==offsets(values)
        c=collections.Counter(physical_query_passes=1,query=count,additional_loss=count,loss_targets=count)
        totals.update(c);groups['scientific-lr-tail'].update(c)
    training_context=sum(e.get('forward_context',0) for line in (p.parent/'steps.jsonl').open() for e in json.loads(line)['events'])
    query_context=d['passes']['forward_context']-d['extension']['inherited_passes']['forward_context']-training_context
    assert query_context>=0
    totals['forward_context']+=query_context;groups['scientific-lr-tail']['forward_context']+=query_context
# The interrupted original rotation query has no completed block receipt.
# Its actual partial targets remain unknown. It used a fixed input list already
# recorded by prior complete same-input scores, so the unique-byte union may
# nevertheless be exact; verify the union bounds rather than assuming it.
partial_inputs=base/'query-ordinary-views/query-inputs.json'
possible_unique={key:set(values) for key,values in unique.items()}
if partial_inputs.exists():
    sources.append(ref(partial_inputs));rows=json.loads(partial_inputs.read_bytes())['offsets']
    for row in rows:possible_unique.setdefault(row['document_id'],set()).update(row['offsets'])
    unknown.append({'attempt':'query-ordinary-views','completed_target_lower_bound':0,'attempted_target_upper_bound':262144,
        'actual_partial_targets':None,'reason':'interrupted inside first Hadamard forward with no completed block receipt',
        'finish':ref(base/'query-ordinary-views/finish.json')})
supervision=[]
for name in ('query-ordinary-views-cap','query-ordinary-views-cap-v2'):
    p=base/name/'observation.json'
    if p.exists():sources.append(ref(p));supervision.append(ref(p))
wiring_failure={'attempt':'query-prioritized-views','actual_training_targets':0,'actual_query_targets':0,
    'reason':'authenticated configure_quantization(None) rejection precedes score_blocks; all five queries unattempted',
    'report':ref(base/'query-prioritized-views/query-results.json'),
    'source':ref(base/'query-prioritized-views/producer/docs/harness/m07-prioritized-views.py')}
sources.extend([wiring_failure['report'],wiring_failure['source']])
known_union=sum(map(len,unique.values()));possible_union=sum(map(len,possible_unique.values()))
result={'schema_version':1,'producer':producer,'status':'completed-attempt boundary accounting, including meta/ordinary methods/representation/derived parents; any live attempt excluded; B0 unresolved',
    'scope':'real H02-data physical executions only; all completed M07 attempt wall costs including tests/failures/setup counted separately',
    'attempts':attempts,'known_recorded_attempt_seconds':sum(a['complete_attempt_seconds'] for a in attempts),
    'distinct_real_data_counts':dict(totals),'groups':{k:dict(v) for k,v in groups.items()},
    'global_unique_source_byte_union':known_union,'global_unique_documents':len(unique),
    'global_unique_bounds':[known_union,possible_union],'global_unique_exact_despite_unknown_partial':known_union==possible_union,
    'known_completed_counts_are_lower_bounds':bool(unknown),'known_pre_forward_wiring_failure':wiring_failure,
    'supervision_overlap':{'observations':supervision,'enclosing_attempt':'query-ordinary-views','separate_additional_wall_seconds':0,'reason':'supervisor observations occur within already-counted failed child/wrapper interval; no blind sum; outside interactive work remains unknown'},
    'union_identity':'exact document_id plus source byte offset across optimizer and query records; global union, never sum run-local unique',
    'classification':'query and additional_loss classify the same loss-bearing exposure; backward and stream classify optimizer exposures; do not add classifications into extra passes',
    'inherited_parent_policy':'parent raw execution included once at its physical source; inherited ledger copies never summed; query cache reuse executes zero new passes',
    'unknown_partial_real_data_records':unknown,'common_historical_engineering_seconds':None,
    'unrecorded_interactive_editing_reading_tool_seconds':None,'fixture_training_pass_counts':None,
    'complete_project_total_claim':False,'sources':list({r['path']:r for r in sources}.values()),
    'analysis_seconds_before_publication':time.perf_counter()-started}
with output.open('xb') as f:f.write(canonical_bytes(result)+b'\n')
print(json.dumps({'output':ref(output),'attempts':len(attempts),'seconds':result['known_recorded_attempt_seconds'],
    'counts':dict(totals),'global_unique':result['global_unique_source_byte_union'],'unknown':len(unknown)}))
