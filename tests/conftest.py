"""Portable test setup; production source/runtime guards remain unchanged."""
import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption('--local-evidence', action='store_true',
                     help='Run historical tests requiring retained local evidence/data')
    parser.addoption('--cpu-only', action='store_true',
                     help='Disable native execution and native RNG inspection in this test run')


def pytest_configure(config):
    config._ralup_cpu_only = config.getoption('--cpu-only') or os.environ.get('RALUP_CPU_TESTS') == '1'
    if config._ralup_cpu_only:
        # Replay children must see the same CPU RNG scope as their parent.
        config._ralup_cpu_environment = {key: os.environ.get(key) for key in
                                        ('RALUP_CPU_TESTS', 'PYTHONPATH')}
        os.environ['RALUP_CPU_TESTS'] = '1'
        os.environ['PYTHONPATH'] = os.pathsep.join((str(ROOT/'tests/cpu_scope'),
            str(ROOT/'src'), os.environ.get('PYTHONPATH', '')))


def pytest_unconfigure(config):
    if hasattr(config, '_ralup_cpu_environment'):
        for key, value in config._ralup_cpu_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def pytest_report_header(config):
    scope = 'CPU only; native tests and native RNG inspection disabled' if config._ralup_cpu_only else 'native execution follows actual device availability'
    return f'RALUP test scope: {scope}; historical local evidence requires --local-evidence'



def pytest_terminal_summary(terminalreporter, config):
    if config._ralup_cpu_only:
        terminalreporter.write_line('RALUP CPU-only scope: native execution and native RNG inspection disabled; no native validation.')


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker('local_evidence') and not config.getoption('--local-evidence'):
            item.add_marker(pytest.mark.skip(reason='Historical local evidence is opt-in: --local-evidence'))
        device = getattr(item, 'callspec', None)
        if device is not None and device.params.get('device') == 'mps':
            item.add_marker(pytest.mark.native)
        if item.get_closest_marker('native') and config._ralup_cpu_only:
            item.add_marker(pytest.mark.skip(reason='Native execution disabled by explicit CPU-only test scope'))


@pytest.fixture(scope='session', autouse=True)
def engineering_environment():
    """Prepare fresh real discovery only when absent; never borrow an archive."""
    from ralup.config import canonical_bytes
    from ralup.environment import discover_environment
    path = ROOT/'artifacts/environment/environment.json'
    owned = None
    if not path.exists():
        report = discover_environment()
        report['test_preparation'] = {'scope': 'portable engineering tests',
            'scientific_gate': None, 'native_validation': False,
            'historical_reproduction': False}
        content = canonical_bytes(report) + b'\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('xb') as stream:
                stream.write(content)
            owned = content
        except FileExistsError:
            pass  # A concurrently prepared report is never overwritten.
    yield path
    if owned is not None and path.exists() and path.read_bytes() == owned:
        path.unlink()  # Remove only this session's unchanged discovery bytes.


@pytest.fixture(scope='session', autouse=True)
def cpu_scope(request, engineering_environment):
    # Discovery observes the real machine before the explicit CPU RNG scope.
    if not request.config._ralup_cpu_only:
        yield
        return
    import functools
    import torch
    available = torch.backends.mps.is_available
    torch.backends.mps.is_available = functools.lru_cache(maxsize=None)(lambda: False)
    try:
        yield
    finally:
        torch.backends.mps.is_available = available


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
