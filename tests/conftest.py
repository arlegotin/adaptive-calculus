"""E03 uses only independently authored synthetic protected bytes."""
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def e03_factory(tmp_path):
    def factory():
        from ralup import evaluate
        from ralup.config import canonical_bytes
        from ralup.records import ArtifactRef
        assert callable(getattr(evaluate, 'freeze_confirmation', None)), 'missing frozen confirmation API'
        root = tmp_path / 'synthetic-confirmation'
        root.mkdir()
        def artifact(name, value):
            path = root / name
            path.write_bytes(canonical_bytes(value))
            return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        payload = artifact('synthetic-protected.json', {'schema': 'ralup.synthetic_protected.v1',
                    'articles': ['Synthetic protected byte fixture.']})
        deps = {key: artifact(key+'.json', {'synthetic': True, 'role': key}) for key in
                ('model','quantizer','backbone','policy','normalizer','b_star','learned_lr','analysis')}
        configs = {arm: artifact(arm+'-config.json', {'synthetic': True, 'arm': arm,
                   'endpoint': {'target_bytes': 64, 'continuation_steps': 1}}) for arm in ('C','Bstar','LR')}
        inputs = {'schema': 'ralup.synthetic_confirmation.v1', 'scope': 'synthetic-engineering',
            'claim': 'Synthetic gateway engineering only; no scientific confirmation',
            'dataset': {'manifest': artifact('data.json', {'synthetic': True, 'partition': 'fixture'}),
                        'payload': payload, 'partition': 'fixture'},
            **deps, 'resolved_configs': configs, 'seeds': [21,22], 'arms': ['C','Bstar','LR'],
            'timing_order': [{'seed':21,'arms':['LR','C','Bstar']},{'seed':22,'arms':['Bstar','LR','C']}],
            'endpoint': {'target_bytes':64,'continuation_steps':1}, 'n':2,
            'bonferroni': {'comparisons':2,'interval_confidence':.975,'success':'both_upper_bounds_below_zero'},
            'failure_protocol': {'divergence':'substantive_failure','replacement':False,'optional_seed_addition':False},
            'allowed_timing_variants':['cold','warm'], 'source_identity': evaluate.confirmation_source_identity(),
            'runtime_identity': evaluate.confirmation_runtime_identity(),
            'output':str(root/'confirm_manifest.json')}
        ref = evaluate.freeze_confirmation(inputs)
        inventory = []
        for seed in (21,22):
            for arm in ('C','Bstar','LR'):
                record = {'schema':'ralup.synthetic_terminal.v1','manifest_sha256':ref.sha256,
                    'seed':seed,'arm':arm,'status':'completed','endpoint':inputs['endpoint'],
                    'config_sha256':configs[arm]['sha256'], 'source_identity':inputs['source_identity'],
                    'runtime_identity':inputs['runtime_identity'],
                    'lineage': {'unit':f'independent-synthetic-{seed}', 'shared_parent':f'parent-{seed}'},
                    'checkpoint':artifact(f'{seed}-{arm}-checkpoint.json',{'schema':'ralup.synthetic_final_checkpoint.v1','synthetic':True,'seed':seed,'arm':arm,
                        'endpoint':inputs['endpoint'],'config_sha256':configs[arm]['sha256'],
                        'source_identity':inputs['source_identity'],'runtime_identity':inputs['runtime_identity'],'final':True}),
                    'failure':None,'costs':artifact(f'{seed}-{arm}-costs.json',{'synthetic':True,'work':'engineering unknown; no science'})}
                receipt=artifact(f'{seed}-{arm}-terminal.json',record)
                inventory.append({'seed':seed,'arm':arm,'terminal':receipt})
        return inputs, ref, inventory, artifact
    return factory
