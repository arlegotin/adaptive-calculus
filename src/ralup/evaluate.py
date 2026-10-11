"""Common raw-byte SUM loss and read-only, stable-order evaluation.

The S34 gateway below currently permits explicitly synthetic fixtures only;
actual scientific authority remains unresolved and official test is unopened.
Evaluation constructs numerical contexts from actual weights independently of
training caches/history, and sums per-block FP64 target losses on the host with
math.fsum in (document identity, target interval, score offsets) order.
"""
from dataclasses import dataclass
import math
import random

import torch
from torch.nn import functional as F

from .data.batches import LogicalBatch
from .data.blocks import TARGET_BYTES, TargetBlock
from typing import Iterable
from .quantization.context import prepare_context


@dataclass(frozen=True)
class ScoreResult:
    nll_sum: float
    target_count: int
    bpb: float
    per_article: dict


def validate_batch(batch: LogicalBatch) -> None:
    count = 0
    for block in batch.block_slices:
        if (type(block.target_start) is not int or type(block.target_end) is not int
                or block.target_start < 0 or not 0 < block.target_end-block.target_start <= TARGET_BYTES):
            raise ValueError('Invalid original target interval: require a positive span <=128 bytes')
        length = len(block.inputs)
        if not 0 < length <= 256 or any(len(v) != length for v in (block.labels,block.positions,block.score_mask)):
            raise ValueError('Invalid byte block dimensions')
        if block.positions != tuple(range(length)):
            raise ValueError('Byte block position IDs must restart at zero')
        if any(type(v) is not int or not 0 <= v <= 256 for v in block.inputs):
            raise ValueError('Invalid input byte/boundary token')
        if any(type(v) is not int or not 0 <= v <= 255 for v in block.labels):
            raise ValueError('Targets must be raw bytes only')
        if any(type(v) is not bool for v in block.score_mask):
            raise ValueError('Loss masks must be boolean')
        if any(not block.target_start <= offset < block.target_end for offset in block.scored_offsets()):
            raise ValueError('Selected byte offset lies outside the original target interval')
        count += sum(block.score_mask)
    if type(batch.target_count) is not int or batch.target_count != count:
        raise ValueError('Declared selected-target count differs from masks')
    if count == 0:
        raise ValueError('Cannot optimize or score an empty logical batch')


def _target_losses(model, batch: LogicalBatch, *, contexts=None, parameter_overrides=None):
    validate_batch(batch)
    device = next(model.parameters()).device
    length = max(len(b.inputs) for b in batch.block_slices)
    shape = (len(batch.block_slices),length)
    inputs = torch.zeros(shape,dtype=torch.int64,device=device)
    positions = torch.zeros_like(inputs)
    labels = torch.zeros_like(inputs)
    padding = torch.ones(shape,dtype=torch.bool,device=device)
    mask = torch.zeros_like(padding)
    for row,block in enumerate(batch.block_slices):
        n = len(block.inputs)
        inputs[row,:n] = torch.tensor(block.inputs,device=device)
        positions[row,:n] = torch.tensor(block.positions,device=device)
        labels[row,:n] = torch.tensor(block.labels,device=device)
        padding[row,:n] = False
        mask[row,:n] = torch.tensor(block.score_mask,device=device)
    if parameter_overrides is None:
        logits = model(inputs,positions,padding,contexts=contexts)
    else:
        # M06's identity-storage learned-weight graph uses the identical model,
        # inputs, activation arithmetic and target scorer. Only the six named
        # projection weights differ; fresh graphs preserve original-W leaves.
        if set(parameter_overrides) != {m['name'] for m in model.quantized_matrix_manifest()}:
            raise ValueError('Functional weight views require exact projection manifest')
        logits = torch.func.functional_call(model,parameter_overrides,(inputs,positions,padding),{'contexts':{}})
    return F.cross_entropy(logits[mask],labels[mask],reduction='none')


def summed_nll(model, batch: LogicalBatch, *, parameter_overrides=None):
    """Scalar SUM for fresh training graphs; H06 alone owns normalization."""
    return _target_losses(model,batch,parameter_overrides=parameter_overrides).sum()


def evaluation_contexts(model, evaluation_context: dict):
    if evaluation_context.get('partition') not in ('fixture','development','meta-train','meta-development','target-training'):
        raise ValueError('Official test targets are protected until the E03 gateway')
    recipe = getattr(model,'_recipe',None)
    if recipe is None:
        return {}
    rng = random.Random(evaluation_context.get('seed',0))
    parameters = dict(model.named_parameters())
    # Fixed scales are runtime scientific state, not recomputed evaluation scales.
    runtime = getattr(model,'_runtime',{})
    return {item['name']:prepare_context(parameters[item['name']],
            {**recipe,**{k:v for k,v in runtime.get(item['name'],{}).items()
                        if k in ('previous_scales','previous_zero_groups','warm_start')}},
            evaluation_context.get('step',0),rng)
            for item in model.quantized_matrix_manifest()}


@torch.no_grad()
def score_blocks(model, blocks, *, evaluation_context: dict, progress_callback=None) -> ScoreResult:
    """Score in stable block order, optionally reporting interruptible boundaries.

    A callback may enforce a deadline before preparation, after preparation and
    after each completed block. Reported counts are completed work, including
    the last block when the callback raises; an exception inside a forward can
    still leave unknown partial work. Numeric scoring/reduction is unchanged.
    """
    completed_targets = forward_context = 0
    def progress(phase, completed_blocks, **details):
        if progress_callback is not None:
            progress_callback({'phase':phase, 'completed_blocks':completed_blocks,
                               'completed_targets':completed_targets, 'forward_context':forward_context, **details})
    progress('before_contexts', 0)
    contexts = evaluation_contexts(model,evaluation_context)
    progress('contexts_ready', 0)
    # Cursor-backed callers materialize only the declared development/query list.
    ordered = sorted(blocks,key=lambda b:(b.document_id,b.target_start,b.target_end,b.scored_offsets()))
    if not ordered:
        raise ValueError('Cannot score an empty logical batch')
    values, counts, article_values, article_counts = [], [], {}, {}
    seen = set()
    for block in ordered:
        for offset in block.scored_offsets():
            key = (block.document_id,offset)
            if key in seen:
                raise ValueError('Evaluation target byte selected more than once')
            seen.add(key)
        count = sum(block.score_mask)
        losses = _target_losses(model,LogicalBatch([block],count),contexts=contexts)
        # MPS lacks float64: this is an explicit evaluation-only host reduction.
        value = math.fsum(losses.detach().cpu().double().tolist())
        if not math.isfinite(value):
            raise ValueError('Nonfinite evaluation NLL')
        values.append(value)
        counts.append(count)
        article_values.setdefault(block.document_id,[]).append(value)
        article_counts[block.document_id] = article_counts.get(block.document_id,0)+count
        completed_targets += count
        forward_context += len(block.inputs)
        progress('block_completed',len(values),block_targets=count,block_nll_sum=value,
                 document_id=block.document_id,target_start=block.target_start,target_end=block.target_end)
    nll, count = math.fsum(values),sum(counts)
    per_article = {key:{'nll_sum':math.fsum(article_values[key]),'target_count':article_counts[key],
                        'bpb':math.fsum(article_values[key])/(math.log(2)*article_counts[key])}
                   for key in sorted(article_values)}
    return ScoreResult(nll,count,nll/(math.log(2)*count),per_article)


# S34: procedural local-owner enforcement, separate from H01/H02 data audits.
from pathlib import Path
import hashlib
import json
from .config import ConfirmationManifest, canonical_bytes
from .records import (ArtifactRef, ConfirmationAudit, IntegrityError,
                      IncompleteConfirmation, ProtectedAccessError, read_artifact,
                      sha256_file, write_immutable_json)


def confirmation_source_identity():
    """All current source plus pinned lock/environment; no historical restore waiver."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/'src/ralup').rglob('*.py')) + [root/'requirements.lock', root/'artifacts/environment/environment.json']
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def confirmation_runtime_identity():
    import importlib.metadata
    import platform
    import subprocess
    import sys
    observed = subprocess.run(['sw_vers'], capture_output=True, text=True) if sys.platform == 'darwin' else None
    return {'python':sys.version, 'executable':sys.executable, 'machine':platform.machine(),
        'system':platform.system(), 'torch':importlib.metadata.version('torch'),
        'os_build':None if observed is None else observed.stdout,
        'os_probe_exit':None if observed is None else observed.returncode,
        'fallback':__import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}


_DEPENDENCIES = ('model','quantizer','backbone','policy','normalizer','b_star','learned_lr','analysis')
_MANIFEST_FIELDS = {'schema','scope','claim','dataset',*_DEPENDENCIES,'resolved_configs','seeds','arms',
    'timing_order','endpoint','n','bonferroni','failure_protocol','allowed_timing_variants','source_identity','runtime_identity'}
_BONFERRONI = {'comparisons':2,'interval_confidence':.975,'success':'both_upper_bounds_below_zero'}
_FAILURE = {'divergence':'substantive_failure','replacement':False,'optional_seed_addition':False}


def _reference(value):
    if not isinstance(value, dict) or set(value) != {'path','sha256'}:
        raise IntegrityError('Exact artifact path/SHA-256 reference required')
    return ArtifactRef(**value)


def _validate_contract(payload):
    if set(payload) != _MANIFEST_FIELDS | {'created_utc','research_cutoff','confirmation_eligible'}:
        raise IntegrityError('Incomplete or unknown S34 manifest fields')
    if (payload['schema'] != 'ralup.synthetic_confirmation.v1' or payload['scope'] != 'synthetic-engineering'
            or payload['confirmation_eligible'] is not False):
        raise IncompleteConfirmation('Genuine scientific authority requires E01/WP06 then E02 producer integration; synthetic manifests cannot authorize science')
    if not isinstance(payload['claim'],str) or not payload['claim'].strip():
        raise IntegrityError('Predeclared claim required')
    seeds, arms = payload['seeds'], payload['arms']
    if (not isinstance(seeds,list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or type(payload['n']) is not int or payload['n'] != len(seeds)):
        raise IntegrityError('Unique complete immutable seeds and matching n required')
    if not isinstance(arms,list) or len(arms) != 3 or set(arms) != {'C','Bstar','LR'}:
        raise IntegrityError('Exact candidate, selected comparator and learned-LR arm inventory required')
    orders = payload['timing_order']
    if (not isinstance(orders,list) or len(orders) != len(seeds)
            or any(not isinstance(row,dict) or set(row) != {'seed','arms'} for row in orders)
            or [row['seed'] for row in orders] != seeds
            or any(not isinstance(row['arms'],list) or len(row['arms']) != 3 or set(row['arms']) != set(arms) for row in orders)):
        raise IntegrityError('Complete frozen randomized timing permutations required')
    endpoint = payload['endpoint']
    if (not isinstance(endpoint,dict) or set(endpoint) != {'target_bytes','continuation_steps'}
            or any(type(value) is not int or value <= 0 for value in endpoint.values())):
        raise IntegrityError('Positive finite immutable endpoint/budget required')
    if payload['bonferroni'] != _BONFERRONI or payload['failure_protocol'] != _FAILURE:
        raise IntegrityError('Two co-primary 97.5% intervals and substantive no-replacement divergence required')
    variants=payload['allowed_timing_variants']
    if not isinstance(variants,list) or not variants or len(set(variants)) != len(variants) or not set(variants) <= {'cold','warm'}:
        raise IntegrityError('Frozen supported timing variants required')
    if (payload['source_identity'] != confirmation_source_identity()
            or payload['runtime_identity'] != confirmation_runtime_identity()):
        raise IntegrityError('Current whole-source/lock/environment or runtime identity changed')
    dataset=payload['dataset']
    if not isinstance(dataset,dict) or set(dataset) != {'manifest','payload','partition'} or dataset['partition'] != 'fixture':
        raise ProtectedAccessError('Synthetic engineering manifests permit only fixture partition, never official test')
    protected = _reference(dataset['payload'])
    if protected.path.name != 'synthetic-protected.json':
        raise ProtectedAccessError('Explicit synthetic protected fixture filename required')
    references = [dataset['manifest']]+[payload[key] for key in _DEPENDENCIES]
    configs=payload['resolved_configs']
    if not isinstance(configs,dict) or set(configs) != set(arms):
        raise IntegrityError('All resolved arm config identities required')
    references += list(configs.values())
    for reference in references:
        ref=_reference(reference)
        if ref.path.resolve() == protected.path.resolve():
            raise ProtectedAccessError('Protected payload cannot be a pre-access dependency')
        value=read_artifact(ref)
        if value.get('synthetic') is not True:
            raise IntegrityError('Synthetic contract requires explicitly synthetic dependencies')
    for arm,reference in configs.items():
        config=read_artifact(_reference(reference))
        if config.get('arm') != arm or config.get('endpoint') != endpoint:
            raise IntegrityError('Resolved arm budget/endpoint differs from frozen contract')


def _refuse_design(reference):
    """Authenticate the actual supported E02 refusal; caller flags cannot advance it."""
    design=read_artifact(reference)
    if design.get('purpose') == 'synthetic-engineering' or design.get('status') == 'engineering_fixed_design':
        raise IncompleteConfirmation('Synthetic engineering precision calculations are not a usable scientific locked design')
    from .analysis.power import _current_prerequisites
    root=Path(__file__).resolve().parents[2]
    index=json.loads((root/'docs/harness/e02-evidence.json').read_bytes())
    attempt=next(row for row in index['attempts'] if row['attempt']=='plan-confirmation-v1')
    startup_ref=_reference(attempt['files']['startup.json']); finish_ref=_reference(attempt['files']['finish.json'])
    startup=read_artifact(startup_ref); finish=read_artifact(finish_ref)
    objects={row['path']:row for row in finish['completed_objects']}
    expected=ArtifactRef(finish_ref.path.parent/'confirmation-plan.json',objects['confirmation-plan.json']['sha256'])
    if reference != expected or finish['exit_code'] != 0 or objects['startup.json']['sha256'] != startup_ref.sha256:
        raise IntegrityError('Design must be the indexed original authenticated E02 producer output')
    if design['source_identity'] != {name:row['sha256'] for name,row in startup['source'].items() if name.startswith('src/')}:
        raise IntegrityError('E02 historical producer source mismatch')
    for name,row in startup['source'].items():
        if sha256_file(finish_ref.path.parent/'producer'/name) != row['sha256']:
            raise IntegrityError('E02 producer archive changed')
    measured=_current_prerequisites(_reference(design['assumptions']['selection']))
    if design.get('scientific_prerequisites') != measured:
        raise IntegrityError('E02 plan prerequisite authority changed')
    missing = ['WP04 inconclusive: accepted robust B0/recipe applicability missing',
        'WP06 NEEDS_REVISION: finalized genuinely trained/development-selected C and learned-LR artifacts missing',
        'B* final comparator selection and immutable final arm/config/budget identities missing',
        'four independent fresh paired final-arm development seeds, complete final scores/failure/cost receipts missing',
        'adequate pilot variance/tail review/precision/resource evidence and scientific locked n missing',
        'confirmation seed/arm list, randomized timing order and final checkpoint/status inventory missing']
    raise IncompleteConfirmation('Cannot freeze scientific confirmation: '+ '; '.join(missing)+
        '. Owning integration: E01/reopened WP06, then E02. No confirm_manifest published.')


def freeze_confirmation(inputs: dict) -> ArtifactRef:
    """Publish an immutable S34 contract or list the missing authentic evidence.

    Current scientific production is unresolved. Positive fixture contracts are
    deliberately typed synthetic; no input flag upgrades their authority.
    """
    from datetime import datetime, timezone
    values=json.loads(canonical_bytes(inputs))
    if 'design' in values:
        _refuse_design(_reference(values['design']))
    output=Path(values.pop('output'))
    if output.exists() or output.with_suffix('.access.jsonl').exists():
        raise FileExistsError('Immutable confirmation manifest/history already exists')
    values.update(created_utc=datetime.now(timezone.utc).isoformat(),research_cutoff='2026-10-03',confirmation_eligible=False)
    _validate_contract(values)
    reference=write_immutable_json(output,values)
    ConfirmationAudit(reference).append('frozen',{'scope':values['scope'],'claim_status':'synthetic_engineering'},initialize=True)
    return reference


def _manifest(reference):
    payload=read_artifact(reference)
    manifest=ConfirmationManifest(payload)
    if manifest.sha256 != reference.sha256:
        raise IntegrityError('Manifest must contain exact canonical bytes without alternate encoding')
    return manifest,payload


def validate_confirmation(reference: ArtifactRef, inventory: list[dict]) -> ConfirmationManifest:
    manifest,payload=_manifest(reference)
    _validate_contract(payload)
    ConfirmationAudit(reference).events()
    if not isinstance(inventory,list):
        raise IncompleteConfirmation('Complete terminal inventory list required, never a readiness boolean')
    expected={(seed,arm) for seed in payload['seeds'] for arm in payload['arms']}
    seen=set(); lineages={}; checkpoints=set()
    protected_path=_reference(payload['dataset']['payload']).path.resolve()
    for row in inventory:
        if not isinstance(row,dict) or set(row) != {'seed','arm','terminal'}:
            raise IncompleteConfirmation('Each planned arm requires an exact terminal receipt reference')
        key=(row['seed'],row['arm'])
        if key in seen or key not in expected:
            raise IncompleteConfirmation('Duplicate/unplanned confirmation seed/arm')
        seen.add(key)
        terminal_ref=_reference(row['terminal'])
        if terminal_ref.path.resolve()==protected_path:
            raise ProtectedAccessError('Protected payload cannot be a terminal receipt')
        terminal=read_artifact(terminal_ref)
        fields={'schema','manifest_sha256','seed','arm','status','endpoint','config_sha256','source_identity','runtime_identity','lineage','checkpoint','failure','costs'}
        if (set(terminal) != fields or terminal['schema'] != 'ralup.synthetic_terminal.v1'
                or terminal['manifest_sha256'] != reference.sha256 or (terminal['seed'],terminal['arm']) != key
                or terminal['endpoint'] != payload['endpoint']
                or terminal['config_sha256'] != payload['resolved_configs'][row['arm']]['sha256']
                or terminal['source_identity'] != payload['source_identity'] or terminal['runtime_identity'] != payload['runtime_identity']):
            raise IntegrityError('Terminal receipt identity/endpoint/source/config mismatch')
        lineage=terminal['lineage']
        if (not isinstance(lineage,dict) or set(lineage) != {'unit','shared_parent'}
                or any(not isinstance(value,str) or not value for value in lineage.values())):
            raise IntegrityError('Complete independent-unit and shared-parent lineage required')
        if row['seed'] in lineages and lineages[row['seed']] != lineage:
            raise IntegrityError('Within-seed arms must share the same independent parent')
        lineages[row['seed']]=lineage
        for name in ('costs','checkpoint'):
            if terminal[name] is not None:
                dep=_reference(terminal[name])
                if dep.path.resolve()==protected_path:
                    raise ProtectedAccessError('Protected payload cannot be a checkpoint/cost artifact')
                if read_artifact(dep).get('synthetic') is not True:
                    raise IntegrityError('Synthetic terminal dependencies required')
        if terminal['costs'] is None:
            raise IntegrityError('Retained cost/failure receipt required for every terminal arm')
        if terminal['status']=='completed':
            if terminal['checkpoint'] is None or terminal['failure'] is not None:
                raise IncompleteConfirmation('Completed arm needs final checkpoint and no failure')
            checkpoint_ref=_reference(terminal['checkpoint'])
            checkpoint=read_artifact(checkpoint_ref)
            expected_checkpoint={'schema':'ralup.synthetic_final_checkpoint.v1','synthetic':True,
                'seed':row['seed'],'arm':row['arm'],'endpoint':payload['endpoint'],
                'config_sha256':terminal['config_sha256'],'source_identity':payload['source_identity'],
                'runtime_identity':payload['runtime_identity'],'final':True}
            if checkpoint != expected_checkpoint or checkpoint.get('final') is not True or checkpoint_ref.sha256 in checkpoints:
                raise IntegrityError('Checkpoint does not authenticate this exact final seed/arm/config/endpoint')
            checkpoints.add(checkpoint_ref.sha256)
        elif terminal['status']=='diverged':
            if terminal['checkpoint'] is not None or not isinstance(terminal['failure'],dict) or terminal['failure'].get('disposition') != 'substantive_failure':
                raise IntegrityError('Terminal divergence retains failure, never an invented final checkpoint')
        else:
            raise IncompleteConfirmation('Arm lacks declared valid completed/diverged terminal status')
    if seen != expected:
        raise IncompleteConfirmation('Missing planned confirmation seed/arm terminal receipt')
    if len({value['unit'] for value in lineages.values()}) != len(payload['seeds']) or len({value['shared_parent'] for value in lineages.values()}) != len(payload['seeds']):
        raise IntegrityError('Independent seeds cannot reuse a complete training unit/parent')
    return manifest


def confirmation_access_events(reference):
    _manifest(reference)
    return ConfirmationAudit(reference).events()


def open_protected_targets(*, phase: str, manifest_ref: ArtifactRef, inventory: list[dict]) -> Iterable[TargetBlock]:
    """Reserve one locked scorer BEFORE payload reads; no training/search interface.

    H01/H02 reconstruction audits remain separate and confer no scoring access.
    A failed opening is retained and cannot silently restart/replace the scorer.
    """
    _manifest(manifest_ref)
    audit=ConfirmationAudit(manifest_ref)
    try:
        if phase != 'confirm':
            raise ProtectedAccessError('Protected target opening requires phase=confirm')
        manifest=validate_confirmation(manifest_ref,inventory)
        def guard(rows):
            if any(row['kind'] in ('opened','amendment') for row in rows):
                raise ProtectedAccessError('Locked scorer already opened or manifest amended; no replacement')
        reservation=audit.append('opened',{'inventory_sha256':hashlib.sha256(canonical_bytes({'inventory':inventory})).hexdigest(),
                                         'scope':'synthetic-engineering','claim_status':'synthetic_engineering'},guard=guard)
    except (ProtectedAccessError,IncompleteConfirmation,IntegrityError) as error:
        audit.append('denied',{'reason':str(error)})
        raise
    payload_ref=_reference(dict(manifest.payload['dataset']['payload']))
    try:
        payload=read_artifact(payload_ref)
        if set(payload) != {'schema','articles'} or payload['schema'] != 'ralup.synthetic_protected.v1' or not isinstance(payload['articles'],list) or not payload['articles']:
            raise ProtectedAccessError('Only explicit synthetic protected article fixtures supported')
        from .data.articles import Article
        from .data.blocks import tile_article
        blocks=[]
        for content in payload['articles']:
            if not isinstance(content,str) or not content:
                raise IntegrityError('Synthetic articles must be nonempty strings')
            data=content.encode('utf-8')
            blocks.extend(tile_article(Article(data,[],hashlib.sha256(data).hexdigest())))
        blocks=tuple(blocks)
        def ready_guard(rows):
            reservations=[row for row in rows if row['kind']=='opened']
            if (len(reservations)!=1 or reservations[0]['event_id']!=reservation['event_id']
                    or any(row['kind'] in ('payload_ready','payload_failed','result','amendment') for row in rows)):
                raise ProtectedAccessError('Payload completion requires the original pending, unamended reservation')
        audit.append('payload_ready',{'reservation_event_id':reservation['event_id'],
            'payload_sha256':payload_ref.sha256,'materialized_blocks':len(blocks),
            'target_count':sum(sum(block.score_mask) for block in blocks)},guard=ready_guard)
        return blocks
    except (ValueError,OSError,TypeError) as error:
        audit.append('payload_failed',{'reason':str(error),'failure_disposition':'retained_no_retry'})
        raise


def record_confirmation_amendment(reference, amendment):
    _manifest(reference)
    canonical_bytes(amendment)
    audit=ConfirmationAudit(reference)
    audit.events()  # Validate prior history; classification is repeated under the append lock.
    return audit.append('amendment',{'amendment':amendment})


def _require_successful_payload(rows, payload_sha256):
    reservations=[row for row in rows if row['kind']=='opened']
    ready=[row for row in rows if row['kind']=='payload_ready']
    if (len(reservations)!=1 or len(ready)!=1
            or ready[0].get('reservation_event_id')!=reservations[0]['event_id']
            or ready[0].get('payload_sha256')!=payload_sha256
            or ready[0]['sequence']<=reservations[0]['sequence']
            or any(row['kind'] in ('payload_failed','amendment') for row in rows)):
        raise ProtectedAccessError('No successful authenticated/materialized locked payload; pending, failed or amended opening')
    return ready[0]


def record_protected_result(reference, result):
    _,payload=_manifest(reference); _validate_contract(payload)
    result_ref=_reference(result)
    value=read_artifact(result_ref)
    if value.get('manifest_sha256') != reference.sha256 or value.get('synthetic') is not True:
        raise IntegrityError('Result must bind exact synthetic manifest identity')
    def guard(rows):
        _require_successful_payload(rows,payload['dataset']['payload']['sha256'])
        if any(row['kind']=='result' for row in rows):
            raise ProtectedAccessError('Locked scorer result already recorded; no replacement')
    return ConfirmationAudit(reference).append('result',{'result':result},guard=guard)


def existing_protected_result(reference):
    _,payload=_manifest(reference); _validate_contract(payload)
    verified=[]
    def guard(rows):
        results=[row for row in rows if row['kind']=='result']
        if not any(row['kind']=='opened' for row in rows) and not results:
            return  # No scorer/result has ever been requested.
        ready=_require_successful_payload(rows,payload['dataset']['payload']['sha256'])
        if not results:
            return
        if len(results)!=1 or results[0]['sequence']<=ready['sequence']:
            raise IntegrityError('Result must follow the single successful locked payload completion')
        ref=_reference(results[0]['result']); read_artifact(ref)
        verified.append(ref)
    ConfirmationAudit(reference).events(guard=guard)
    return verified[0] if verified else None
