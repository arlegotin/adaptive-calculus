"""Deterministic raw-evidence reports. Reconstruction passes are not science gates.

No serialized Python, shell hooks, protected scorer or implicit training runs are
accepted. Historical observations and missing costs remain explicitly typed.
"""
import json
import math
from pathlib import Path

from .config import canonical_bytes
from .records import ArtifactRef, GateResult, sha256_file

SECTIONS = ('frozen_scope', 'sources_and_deviations', 'provenance', 'per_seed',
    'development_searches', 'cost_ledgers', 'ablations', 'regressions',
    'export_and_reproduction', 'remaining_blocks', 'decision')
COST_FIELDS = {'stream_targets', 'loss_targets', 'gradient_only_targets', 'hvp_targets',
    'warm_start_seconds', 'deployment_seconds', 'tuning_seconds', 'project_seconds'}


def _read(ref):
    if isinstance(ref, dict): ref=ArtifactRef(ref['path'], ref['sha256'])
    if sha256_file(ref.path) != ref.sha256: raise ValueError('incomplete evidence: source hash mismatch')
    return json.loads(ref.path.read_bytes())


def _number(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0


def research_disposition(evidence: dict) -> dict:
    """Classify a stated outcome; this arithmetic grants no inference authority."""
    outcome=evidence.get('outcome','inconclusive')
    if outcome not in ('negative','inconclusive','blocked','positive'):
        raise ValueError('Unknown research outcome')
    if evidence.get('meta_recipe_claim') and evidence.get('independent_meta_run_clusters',0)<2:
        raise ValueError('Training-recipe claims require independent meta-run clusters')
    baseline=evidence.get('baseline_cost');candidate=evidence.get('candidate_cost')
    meta=evidence.get('meta_cost');tuning=evidence.get('method_tuning_cost')
    comparable=evidence.get('quality_comparable') is True
    saving=baseline-candidate if comparable and _number(baseline) and _number(candidate) else None
    efficient=outcome=='positive' and saving is not None and saving>0
    decision={'negative':'local_non_advantage','inconclusive':'inconclusive','blocked':'blocked',
        'positive':'positive_quality_and_lower_deployment_cost' if efficient else 'positive_quality_but_slower'}[outcome]
    if outcome=='positive' and (not comparable or saving is None): decision='positive_quality_efficiency_unresolved'
    return dict(decision=decision,claim_status='development_only',efficiency_improvement=efficient,
        break_even_reuses=meta/saving if saving is not None and saving>0 and _number(meta) else None,
        full_project_break_even_reuses=(meta+tuning)/saving if saving is not None and saving>0 and _number(meta) and _number(tuning) else None,
        inference_authority='classification of supplied evidence only; confirmation must be independently authenticated')


def _validate(e):
    if e.get('schema')!='ralup.research_evidence.v1': raise ValueError('incomplete evidence: unsupported schema')
    if set(e['sections'])!=set(SECTIONS): raise ValueError('incomplete evidence: S39 sections required')
    sources=e['sources']; known={(r['path'],r['sha256']) for r in sources}
    for r in sources:
        if sha256_file(Path(r['path']))!=r['sha256']: raise ValueError('incomplete evidence: source hash mismatch')
    # Only operational primary ledgers are dereferenced here. Historical
    # locations inside preserved mapping documents remain historical identities.
    section=e['sections']['sources_and_deviations']
    primary=section.get('primary_ledgers',[]) if isinstance(section,dict) else []
    for r in primary:
        try:
            matches=sha256_file(Path(r['path']))==r['sha256']
        except OSError as error:
            raise ValueError('incomplete evidence: primary ledger unavailable') from error
        if not matches:
            raise ValueError('incomplete evidence: primary ledger source hash mismatch')
    rows=_read(e['trials'])
    ids=[r['run_id'] for r in rows]
    if len(set(ids))!=len(ids) or sorted(ids)!=sorted(e['registry']):
        raise ValueError('incomplete evidence: missing/duplicate registered trial, including failures')
    for row in rows:
        if row['status'] not in ('completed','diverged','failed','blocked','interrupted'):
            raise ValueError('incomplete evidence: unknown trial disposition')
        if (row['source']['path'],row['source']['sha256']) not in known:
            raise ValueError('incomplete evidence: trial source hash not authenticated')
        if not COST_FIELDS <= row['cost'].keys():
            raise ValueError('incomplete evidence: required full cost missing; stream-only is insufficient')
        if any(v is not None and not _number(v) for v in row['cost'].values()):
            raise ValueError('incomplete evidence: invalid cost')
        if row.get('development_bpb') is not None and not _number(row['development_bpb']):
            raise ValueError('incomplete evidence: nonfinite development score')
        if row.get('provenance') not in ('supplied_historical','measured_cpu','measured_m3','inferred','simulated','exploratory','confirmatory'):
            raise ValueError('incomplete evidence: provenance label required')
    # Quality-only outcomes preserve unknown project/accounting costs.
    # research_disposition gates deployment efficiency on comparable measured
    # baseline/candidate costs, and each break-even numerator independently.
    return rows


def build_report(artifacts: list[ArtifactRef], destination: Path) -> ArtifactRef:
    """Regenerate S39 tables from authenticated raw ledgers, preserving nulls."""
    if len(artifacts)!=1: raise ValueError('One immutable evidence root binds all report inputs')
    e=_read(artifacts[0]); rows=_validate(e)
    disposition=research_disposition(e.get('claim',{}))
    report=dict(schema='ralup.research_report.v1',scope=e['scope'],
        input_sha256=artifacts[0].sha256,scientific_status=e['scientific_status'],
        disposition=disposition,sections=e['sections'],
        tables=dict(per_seed=rows, failures=[r['run_id'] for r in rows if r['status']!='completed'],
            precision=e.get('precision',[]),cost_quality=[dict(run_id=r['run_id'],
                development_bpb=r.get('development_bpb'),**r['cost']) for r in rows]),
        costs=dict(deployment='includes attributable common warm start and all online work',
            search='includes failed trials and method-specific tuning',
            project='deduplicate shared ancestors and overlapping wall/source intervals; unknown remains null',
            common_engineering_seconds=None),
        protected_scoring_performed=False,
        sources=e['sources'])
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    path=destination/'report.json';path.write_bytes(canonical_bytes(report))
    (destination/'tables.json').write_bytes(canonical_bytes(report['tables']))
    lines=['# RALUP research report', '', f"Disposition: {disposition['decision']}; {disposition['claim_status']}.",
        '', 'This report preserves the scope of each observation. Reconstruction is an engineering check.', '']
    for section in SECTIONS:
        lines.extend(['## '+section.replace('_',' ').title(), '', '```json',
            json.dumps(e['sections'][section],sort_keys=True,indent=2), '```', ''])
    (destination/'report.md').write_text('\n'.join(lines))
    return ArtifactRef(path,sha256_file(path))


def reproduce(manifest: ArtifactRef, environment: dict) -> GateResult:
    """Authenticate inputs and reconstruct raw tables twice; never open test data.

    Numerical replay is a separately declared, source-strict executable operation
    recorded in the evidence bundle. Arbitrary commands in evidence are ignored.
    """
    try:
        e=_read(manifest)
        if e.get('schema')!='ralup.research_evidence.v1':
            raise ValueError('Confirmed-study reproduction gate reopened: a validated result/evidence adapter is required; protected scores are never reopened')
        root=Path(environment['destination'])
        first=build_report([manifest],root/'first')
        second=build_report([manifest],root/'second')
        if first.sha256!=second.sha256: raise ValueError('Reopened gate: deterministic report discrepancy')
        return GateResult('pass',(first,second),())
    except (OSError,ValueError,KeyError,TypeError) as error:
        return GateResult('blocked',(),(str(error),))
