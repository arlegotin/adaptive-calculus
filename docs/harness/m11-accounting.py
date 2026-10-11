"""Receipt-only work totals and source-offset union; no model restoration/scoring."""
import hashlib
import json
import os
from pathlib import Path

from ralup.data.batches import next_logical_batch
from ralup.data.blocks import TargetBlock
from ralup.data.cursor import BatchCursor
from ralup.meta.tasks import MetaTask,_input_digest
from ralup.policy.checks import _ref,_write
from ralup.records import sha256_file
from ralup.search import source_identity
from ralup.state import _decode

base=Path(os.environ['M11_ATTEMPT_DIR']);store=base.parent
read=lambda p:json.loads(Path(p).read_bytes())

def auth(ref):
    assert sha256_file(ref['path'])==ref['sha256'];return read(ref['path'])

def blocks(values):return [TargetBlock(**{k:tuple(v) if isinstance(v,list) else v for k,v in row.items()}) for row in values]

unions={'training':{},'query':{}}

def add(kind,document,offsets):unions[kind].setdefault(document,set()).update(offsets)
def add_blocks(kind,values):
    for b in values:add(kind,b.document_id,b.scored_offsets())

def add_journal(path):
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            for item in row.get('inputs',[]):add('training',item['document_id'],item['offsets'])

reports={};totals=[]
for name in ('native-bootstrap-v1','native-bootstrap-v2','native-calibration-v1','native-reference-v1','native-final-calibration-v1','native-current-pool-v1'):
    path=store/name/'report.json'
    if not path.exists():continue
    report=read(path);reports[name]=report
    finish=read(path.parent/'finish.json')
    outputs={x['path']:x for x in finish['completed_objects']}
    assert sha256_file(path)==outputs['report.json']['sha256']
    row={'attempt':name,'report':_ref(path),'finish':_ref(path.parent/'finish.json'),'launcher_seconds':finish['elapsed_seconds'],
         'optimizer_updates':0,'training_targets':0,'query_targets':0,'query_evaluations':0}
    if name=='native-bootstrap-v1':
        assert report['status']=='failed' and report['unknown_partial_work'] is True
        row.update(optimizer_updates=sum(len(r['steps']) for r in report['references']),
                   training_targets=sum(s['work']['stream'] for r in report['references'] for s in r['steps']),
                   historical_unknown_partial_work=True,
                   specific_bound='Query-auth validation traceback precedes score_blocks; retained journal contains failure but no query_progress. Completed64referenceupdates and zeroqueryforwards; generic historical unknown_partial_work remains true.')
    elif name=='native-bootstrap-v2':
        results=[o['result'] for o in report['observations']]
        generation=auth(report['generation'])
        results += [p[side] for p in generation['pairs'] for side in ('plus','minus')]
        row['optimizer_updates']=sum(len(r['steps']) for r in report['references'])
        row['training_targets']=sum(s['work']['stream'] for r in report['references'] for s in r['steps'])
        for result in results:
            w=result['work'];row['optimizer_updates']+=w['optimizer_steps'];row['training_targets']+=w['training_targets'];row['query_targets']+=w['query_targets'];row['query_evaluations']+=w['query_evaluations']
        frozen=read(path.parent/'probe-inputs.json');add_blocks('query',blocks(frozen['query']))
        for batch in frozen['inner']:add_blocks('training',blocks(batch))
    elif name=='native-calibration-v1':
        assert report['status']=='calibration_measured_pending_interpretation'
        d=auth(report['design']);snapshot=auth(d['snapshot'])
        # Decode ONLY authenticated cursor metadata. No tensors/model/optimizer
        # are imported, and strict H07 full-state source checks are not bypassed.
        for key in ('src/ralup/data/cursor.py','src/ralup/data/batches.py','src/ralup/data/blocks.py'):
            assert d['source_identity'][key]==source_identity()[key]
        cursor=BatchCursor(_decode(snapshot['metadata']['items']['cursor'],{}))
        batches=[next_logical_batch(cursor,8192) for _ in range(508)]
        cache={}
        for ref in report['runs']:
            result=auth(ref);spec=result['run'];w=result['result']['work']
            qref=spec['query'];q=blocks(auth(qref)['blocks']);key=(qref['sha256'],spec['horizon'])
            if key not in cache:
                task=MetaTask(None,batches,q,{'query_seed':d['query_seed']},{})
                cache[key]=_input_digest(task,spec['horizon'])
                add_blocks('query',q)
            assert cache[key]==w['input_sha256']
            row['optimizer_updates']+=w['optimizer_steps'];row['training_targets']+=w['training_targets'];row['query_targets']+=w['query_targets'];row['query_evaluations']+=w['query_evaluations']
        small=blocks(auth(d['runs'][0]['query'])['blocks'])
        assert _input_digest(MetaTask(None,batches,small,{'query_seed':d['query_seed']},{}),508)==d['task_full_input_sha256']
        for batch in batches:add_blocks('training',batch.block_slices)
        row['offset_reconstruction']='Only data-cursor subtree decoded under unchanged H03 source; every completed rollout input digest and frozen full508digest verified. No old full training-state restore or additional model execution.'
    elif name=='native-reference-v1':
        row.update(optimizer_updates=sum(len(r['steps']) for r in report['origins']),
                   training_targets=sum(s['work']['stream'] for r in report['origins'] for s in r['steps']))
    elif name=='native-current-pool-v1':
        row.update(optimizer_updates=len(report['steps']),training_targets=sum(w['stream'] for w in report['steps']))
    elif name=='native-final-calibration-v1':
        assert report['status']=='final_normalizer_assessment_measured_pending_budget'
        generation=auth(report['pilot']);results=[p[side] for p in generation['pairs'] for side in ('plus','minus')]
        results += [auth(ref)['result'] for ref in report['runs']]
        for result in results:
            w=result['work'];row['optimizer_updates']+=w['optimizer_steps'];row['training_targets']+=w['training_targets'];row['query_targets']+=w['query_targets'];row['query_evaluations']+=w['query_evaluations']
        d=auth(report['design']);add_blocks('query',blocks(auth(d['queries'])['blocks']))
    for journal in path.parent.glob('*.jsonl'):
        assert sha256_file(journal)==outputs[journal.name]['sha256']
        add_journal(journal)
    totals.append(row)


def spans(values):
    values=sorted(values);result=[]
    for value in values:
        if result and result[-1][1]==value:result[-1][1]+=1
        else:result.append([value,value+1])
    return result

combined={document:set(values) for document,values in unions['training'].items()}
for document,values in unions['query'].items():combined.setdefault(document,set()).update(values)
path=base/'scored-source-offset-intervals.json'
_write(path,{'schema':'ralup.source_offset_union.v1','scope':'M11 actual completed training/query source-offset union, excluding inherited parent creation; merge with all other task offsets at E07 before any project-global unique claim.',
             'documents':{document:spans(values) for document,values in sorted(combined.items())}})
finished=[]
for attempt in sorted(store.iterdir()):
    path=attempt/'finish.json'
    if path.exists():
        f=read(path);finished.append({'attempt':attempt.name,'finish':_ref(path),'exit_code':f['exit_code'],'launcher_seconds':f['elapsed_seconds']})
report={'status':'receipt_accounting_completed','native_attempts':totals,
        'physical_work':{key:sum(t[key] for t in totals) for key in ('optimizer_updates','training_targets','query_targets','query_evaluations')},
        'task_unique_training_source_offsets':sum(map(len,unions['training'].values())),
        'task_unique_query_source_offsets':sum(map(len,unions['query'].values())),
        'task_unique_combined_scored_source_offsets':sum(map(len,combined.values())),
        'source_offsets':_ref(base/'scored-source-offset-intervals.json'),
        'recorded_prior_attempt_walls':finished,'sum_prior_launcher_seconds':sum(f['launcher_seconds'] for f in finished),
        'time_scope':'Completed launcher intervals only. Native child/report/subregion walls are enclosed, not added. CPU/tests/editing/read/tools may overlap; their unknown enclosing cost is not zero and this is NOT full-project elapsed or an exclusive benchmark.',
        'inherited_meta_parent':{'seed':707,'updates':288,'training_targets':2359296,'scope':'Shared original typed META MODEL-ONLY parent ancestry; retain original M07 producer receipts and globally deduplicate once, never multiply by resets/rollouts.'},
        'classification':'Query is also additional-loss work; do not add both as disjoint forward work. No backward/query overlap presumed; source-offset union distinct from physical multiplicity. Old action-telemetry absence is not silently repaired.',
        'new_model_executions':0,'new_optimizer_updates':0,'new_query_targets':0}
_write(base/'report.json',report)
print(json.dumps({'report':_ref(base/'report.json'),'physical_work':report['physical_work'],'task_unique_scored':report['task_unique_combined_scored_source_offsets']}),flush=True)
