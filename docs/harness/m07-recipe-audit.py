"""Authenticate historical numerical recipe/cache facts without rewriting snapshots."""
import json,os,time
from pathlib import Path
from ralup.records import ArtifactRef,sha256_file
from ralup.search import source_identity
from ralup.state import load_snapshot,_validated
from ralup.calibrate import _write

base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
out=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();rows=[]
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
for name in ('native-ordinary-geometry','native-quantizer-geometry'):
    report=base/name/'admission.json';admission=json.loads(report.read_bytes())
    assert admission['status']=='pass' and admission['source_identity']==source_identity()
    for run in admission['runs']:
        saved=next(r for r in run['arms'][0]['saved'] if r['step']==16)
        checkpoint={k:saved[k] for k in ('path','sha256')}
        payload=_validated(load_snapshot(ArtifactRef(**checkpoint)))
        contexts=payload['model_cache']['_contexts'];assert len(contexts)==24
        grouping={c.grouping for c in contexts.values()};assert len(grouping)==1
        norm=payload['rule'].get('config',{}).get('normalization','none')
        effective={**run['config']['recipe'],'grouping':next(iter(grouping)),
            'normalization':norm,'moment_coordinates':'original_W'}
        rows.append({'label':run['config']['label'],'admission':ref(report),'checkpoint':checkpoint,
            'original_descriptive_recipe':run['config']['recipe'],'resolved_numerical_recipe':effective,
            'actual_model_recipe':payload['model_cache']['_recipe'],'actual_rule':payload['rule'],
            'contexts':[{'matrix':n,'recipe':c.recipe,'grouping':c.grouping,'rotation':c.rotation,
                'backward':c.backward,'scales_shape':list(c.scales.shape),'scales_dtype':str(c.scales.dtype),
                'storage_shape':list(c.storage_values.shape)} for n,c in sorted(contexts.items())],
            'original_configuration_sha256':payload['configuration_sha256'],
            'snapshot_producer':json.loads(Path(checkpoint['path']).read_bytes())['producer_files']})
        if time.perf_counter()-started>90:raise RuntimeError('Finite recipe metadata audit envelope exceeded')
result={'schema':'ralup.recipe_annotation_correction.v1','source_identity':source_identity(),'rows':rows,
    'purpose':'source/cache-evidenced descriptive annotation correction; original configs, sources, checkpoints and numerical results remain immutable',
    'normalization_semantics':'extra named matrix normalization order; original-W moment coordinates are a separate field; intrinsic named-method preconditioning remains in method/config',
    'new_training_targets':0,'new_query_targets':0,'new_numerical_admissions':0,
    'strict_state_contract':'authenticated snapshot bytes/schema/source facts only; no cross-source full-state restore permission',
    'complete_seconds':time.perf_counter()-started}
_write(out/'recipe-correction.json',result);print(json.dumps({'report':ref(out/'recipe-correction.json'),'rows':len(rows),'seconds':result['complete_seconds']}))
