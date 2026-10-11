"""Executed-pass and wall-interval unions for the three S26 cost views.

An event identifies one physical loss, gradient-only or HVP execution. Query,
additional-loss and backward are overlapping classifications, never new passes.
Parents contain the same run/event identities wherever inherited. Source ranges
are [corpus identity, document identity, byte start, byte end], half open.
Unknown historical work remains unknown; known lower bounds are also reported.
"""
import math


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _union(intervals):
    total=0
    for spans in intervals.values():
        end=None
        for a,b in sorted(spans):
            total += b-a if end is None else max(0,b-max(a,end))
            end=b if end is None else max(end,b)
    return total


def _collect(roots):
    runs={};visiting=set()
    def visit(run):
        key=run['run_id']
        if not isinstance(key,str) or not key:raise ValueError('Nonempty run identity required')
        if key in visiting:raise ValueError('Cyclic cost ancestry')
        if key in runs:
            if runs[key]!=run:raise ValueError('Conflicting inherited run identity')
            return
        if run['status'] not in ('completed','failed','diverged','blocked'):
            raise ValueError('Known terminal run status required')
        visiting.add(key)
        for parent in run['parents']:visit(parent)
        visiting.remove(key);runs[key]=run
    for run in roots:visit(run)
    return list(runs.values())


def _totals(roots):
    runs=_collect(roots);events={};walls={};coverage={};unknown=False;unknown_unique=False;unknown_wall=False;unknown_ancestor=False
    for run in runs:
        unknown |= bool(run.get('unknown_partial_work',False) or run.get('unknown_ancestor_cost',False))
        unknown_ancestor |= bool(run.get('unknown_ancestor_cost',False))
        if run['wall_intervals'] is None:unknown_wall=True
        for span in run['wall_intervals'] or []:
            clock=span['clock'];a=_number(span['start'],'wall start');b=_number(span['end'],'wall end')
            if not isinstance(clock,str) or not clock or b<a:raise ValueError('Invalid wall interval')
            walls.setdefault(clock,[]).append((a,b))
        for event in run['events']:
            key=event['event_id']
            if not isinstance(key,str) or not key:raise ValueError('Nonempty physical event identity required')
            if key in events:
                if events[key]!=event:raise ValueError('Conflicting physical event identity')
                continue
            if event['kind'] not in ('loss','gradient_only','hvp'):raise ValueError('Distinct physical pass kind required')
            count=_integer(event['targets'],'targets');stream=_integer(event['stream_targets'],'stream targets')
            if stream>count or (event['kind']!='loss' and stream):raise ValueError('Stream advances only on ordinary loss targets')
            if not isinstance(event['classifications'],list) or any(not isinstance(x,str) for x in event['classifications']):
                raise ValueError('Raw classifications required')
            events[key]=event
            if event['source_intervals'] is None:unknown_unique=True
            else:
                if sum(span[3]-span[2] for span in event['source_intervals'])!=count:
                    raise ValueError('Executed source ranges must account for every target exposure')
                for source,document,a,b in event['source_intervals']:
                    _integer(a,'source offset');_integer(b,'source offset')
                    if not source or not document or b<a:raise ValueError('Invalid source interval')
                    coverage.setdefault((source,document),[]).append((a,b))
    counts={kind:sum(e['targets'] for e in events.values() if e['kind']==kind) for kind in ('loss','gradient_only','hvp')}
    known_total=sum(counts.values());known_seconds=_union(walls)
    # Independent clocks have no known relative position. Summing them would
    # silently turn process time into wall time.
    unknown_wall |= len(walls)>1 or unknown_ancestor
    stream=sum(e['stream_targets'] for e in events.values())
    return {'stream_targets':None if unknown_ancestor else stream,'known_stream_targets':stream,
            'loss_targets':counts['loss'],'gradient_only_targets':counts['gradient_only'],'hvp_targets':counts['hvp'],
            'total_targets':None if unknown else known_total,'known_total_targets':known_total,
            'unique_targets':None if unknown_unique or unknown else _union(coverage),
            'known_unique_targets':_union(coverage),'seconds':None if unknown_wall else known_seconds,
            'known_interval_seconds':known_seconds,'physical_events':len(events),
            'common_engineering_seconds':None,
            'raw_classifications':{key:event['classifications'] for key,event in events.items()},
            'unknown_partial_work':unknown,'run_ids':[r['run_id'] for r in runs]}


def cost_quality_frontier(runs: list[dict], ledger: str) -> dict:
    """Return best development BPB against cumulative actual search expenditure.

    Input order is execution/availability order, not score order. A failed trial
    charges all known work but contributes no quality point. Matched endpoint
    curves include each run's ancestors; the cumulative view deduplicates them.
    Unobserved work never becomes zero-cost evidence. Raw ordinary runs contain
    their actual executions, without hypothetical unused policy feature scans.
    """
    columns={'stream':'stream_targets','total':'total_targets','seconds':'seconds','unique':'unique_targets'}
    if ledger not in columns:raise ValueError('ledger must be stream, total, seconds or unique')
    from ..search import _development_only
    _development_only(runs)
    frontier=[];matched=[];seen=set();best=None;prefix=[]
    for run in runs:
        if run['run_id'] in seen:raise ValueError('Duplicate root run identity')
        seen.add(run['run_id'])
        points=run.get('checkpoints') or [{'event_ids':list(dict.fromkeys(e['event_id'] for e in run['events'])),
            'development_bpb':run.get('development_bpb'),'wall_intervals':run['wall_intervals']}]
        previous=set();ordered_ids=list(dict.fromkeys(e['event_id'] for e in run['events']));event_ids=set(ordered_ids)
        for point in points:
            ids=set(point['event_ids'])
            if not previous<=ids<=event_ids or point['event_ids']!=ordered_ids[:len(point['event_ids'])]:
                raise ValueError('Curve endpoint must bind an ordered prefix of distinct executed events')
            previous=ids
            partial={**run,'events':[e for e in run['events'] if e['event_id'] in ids],
                     'wall_intervals':point['wall_intervals']}
            totals=_totals([partial]);cumulative=_totals([*prefix,partial]);score=point['development_bpb']
            if score is not None:_number(score,'development BPB')
            matched.append({'run_id':run['run_id'],'status':run['status'],'development_bpb':score,**totals})
            if run['status']=='completed' and score is not None:
                best=score if best is None else min(best,score)
                frontier.append({'run_id':run['run_id'],'cost':cumulative[columns[ledger]],'development_bpb':best})
        prefix.append(run)
    return {'schema':'ralup.cost_quality_frontier.v1','ledger':ledger,'frontier':frontier,
            'matched_curves':matched,'totals':_totals(runs),
            'failed_runs':[r['run_id'] for r in runs if r['status'] in ('failed','diverged')],
            'scope':'supplied executed evidence only; unknown common engineering cost is not zero',
            'uses_protected_scores':False}


def source_intervals(corpus, offsets):
    """Compress actual scored byte offsets without summing run-local uniques."""
    intervals=[]
    for row in offsets:
        start=end=None
        for offset in row['offsets']:
            if end==offset:end+=1
            else:
                if start is not None:intervals.append([corpus,row['document_id'],start,end])
                start=offset;end=offset+1
        if start is not None:intervals.append([corpus,row['document_id'],start,end])
    return intervals


def extra_execution_events(prefix, raw):
    """Collapse roles on one extra loss; gradient-only/HVP executions stay distinct."""
    events={}
    for extra in raw:
        role=extra['kind'];kind='loss' if role in ('extra_loss','extra_loss_backward') else role
        key=f'{prefix}:extra:{kind}:{extra["execution_id"]}'
        if key not in events:
            events[key]={'event_id':key,'kind':kind,'targets':extra['target_bytes'],
                'stream_targets':0,'classifications':[],'source_intervals':None}
        if events[key]['targets']!=extra['target_bytes']:raise ValueError('Conflicting physical extra execution identity')
        if role not in events[key]['classifications']:events[key]['classifications'].append(role)
    return list(events.values())


def trial_cost_record(reference):
    """Authenticate new physical receipts; retain legacy counters as unknown.

    Historical aggregate counts cannot reveal physical pass IDs, relative wall
    intervals or global byte unions. They remain raw evidence for E07, without
    invented event reconstruction. A normalized parent must bind its original
    artifact; otherwise the unmeasured ancestor contribution stays unknown.
    """
    from ..search import _read,_ref
    from ..records import ArtifactRef
    row=_read(reference);cost=row['cost'];parents=[];unknown_parent=False
    if cost.get('parent'):
        original=_read(ArtifactRef(**cost['parent']))
        parent_ref=original.get('cost',{}).get('execution_record')
        if parent_ref:parents.append(trial_cost_record(ArtifactRef(**parent_ref)))
        else:unknown_parent=True
    wall=cost.get('wall_interval')
    checkpoints=[]
    for i,point in enumerate(row['curve']):
        if 'execution_ids' not in point or 'wall_interval' not in point:continue
        final=i==len(row['curve'])-1
        checkpoints.append({'event_ids':[e['event_id'] for e in row['executions']] if final else point['execution_ids'],
            'development_bpb':point['development_bpb'],'wall_intervals':[wall if final else point['wall_interval']]})
    return {'run_id':row['trial_id'],'family':row['family'],'status':row['status'],
        'parents':parents,'events':row.get('executions',[]),'wall_intervals':None if wall is None else [wall],
        'checkpoints':checkpoints,
        'development_bpb':row['curve'][-1]['development_bpb'] if row['curve'] else None,
        'common_engineering_seconds':None,'evidence':_ref(reference),'historical_raw_cost':cost,
        'unknown_ancestor_cost':unknown_parent,
        'unknown_partial_work':('executions' not in row or row.get('unknown_partial_query_work',False)
                                or row.get('unknown_partial_training_work',False))}
