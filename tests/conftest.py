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


@pytest.fixture
def e04_factory(tmp_path):
    """A fixed synthetic numerical engine contract, never scientific authority."""
    def factory(n=4,steps=2,diverge=None,kind='success'):
        import random
        from ralup import evaluate
        from ralup.config import ConfirmationManifest,canonical_bytes
        from ralup.analysis.power import _protocol
        root=tmp_path/f'synthetic-e04-{len(list(tmp_path.glob("synthetic-e04-*")))}'
        root.mkdir()
        def artifact(name,value):
            path=root/name
            path.write_bytes(canonical_bytes(value))
            return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        seeds=list(range(100,100+n)); endpoint={'target_bytes':64*steps,'continuation_steps':steps}
        deps={key:artifact(key+'.json',{'synthetic':True,'role':key}) for key in
              ('model','quantizer','backbone','policy','normalizer','b_star','learned_lr')}
        analysis={'synthetic':True,'manifest_path':str(root/'confirm_manifest.json'),
                  'output_directory':str(root/'runs'),'engine':'synthetic-byte-unigram-v1',
                  'timing_seed':90210,'ordinal':'divergence_worse_than_finite;both_diverged_tie',
                  **_protocol(113)}
        deps['analysis']=artifact('analysis.json',analysis)
        rates={'success':{'C':1.,'Bstar':0.,'LR':.1},'null':{'C':0.,'Bstar':1.,'LR':1.},
               'inconclusive':{'C':.5,'Bstar':.5,'LR':.5}}[kind]
        configs={arm:artifact(arm+'-config.json',{'synthetic':True,'arm':arm,'endpoint':endpoint,
            'engine':'synthetic-byte-unigram-v1','training_text':'a'*64,'warm_start_bytes':32,
            'learning_rate':rates[arm], 'momentum':.2,'quantization':'uniform4-logit-v1',
            'diverge_seeds':[diverge[0]] if diverge and diverge[1]==arm else []}) for arm in ('C','Bstar','LR')}
        timing_rng=random.Random(90210); orders=[]
        for seed in seeds:
            arms=['C','Bstar','LR'];timing_rng.shuffle(arms);orders.append({'seed':seed,'arms':arms})
        inputs={'schema':'ralup.synthetic_confirmation.v1','scope':'synthetic-engineering',
            'claim':'Numerically executed synthetic byte fixtures; not scientific confirmation',
            'dataset':{'manifest':artifact('data.json',{'synthetic':True,'partition':'fixture'}),
                'payload':artifact('synthetic-protected.json',{'schema':'ralup.synthetic_protected.v1','articles':['a'*37]}),
                'partition':'fixture'},**deps,'resolved_configs':configs,'seeds':seeds,'arms':['C','Bstar','LR'],
            'timing_order':orders,'endpoint':endpoint,'n':n,
            'bonferroni':{'comparisons':2,'interval_confidence':.975,'success':'both_upper_bounds_below_zero'},
            'failure_protocol':{'divergence':'substantive_failure','replacement':False,'optional_seed_addition':False},
            'allowed_timing_variants':['cold','warm'],'source_identity':evaluate.confirmation_source_identity(),
            'runtime_identity':evaluate.confirmation_runtime_identity(),'output':str(root/'confirm_manifest.json')}
        ref=evaluate.freeze_confirmation(inputs)
        return ref,ConfirmationManifest(json.loads(ref.path.read_bytes()))
    return factory
