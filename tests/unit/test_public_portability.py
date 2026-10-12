"""Public fixtures authenticate real bytes; unprepared production stays closed."""
import copy
import json
from pathlib import Path

import pytest

from ralup import evaluate
from ralup.records import sha256_file


def test_fixture_binds_actual_source_lock_and_fresh_environment(e03_factory, engineering_environment):
    inputs, reference, inventory, _ = e03_factory()
    root = Path(evaluate.__file__).resolve().parents[2]
    identity = inputs['source_identity']
    assert identity['src/ralup/evaluate.py'] == sha256_file(root/'src/ralup/evaluate.py')
    assert identity['requirements.lock'] == sha256_file(root/'requirements.lock')
    assert identity['artifacts/environment/environment.json'] == sha256_file(engineering_environment)
    report = json.loads(engineering_environment.read_bytes())
    assert report['python']['version']
    assert report['packages']['torch']
    manifest = evaluate.validate_confirmation(reference, inventory)
    assert manifest.payload['confirmation_eligible'] is False


@pytest.mark.parametrize('identity', ['src/ralup/evaluate.py', 'requirements.lock',
                                    'artifacts/environment/environment.json'])
def test_forged_fixture_source_lock_or_environment_cannot_freeze(e03_factory, identity):
    inputs, _, _, _ = e03_factory()
    forged = copy.deepcopy(inputs)
    forged['output'] = str(Path(inputs['output']).with_name('forged.json'))
    forged['source_identity'][identity] = '0'*64
    with pytest.raises(evaluate.IntegrityError, match='identity'):
        evaluate.freeze_confirmation(forged)
    assert not Path(forged['output']).exists()


def unprepared_tree(tmp_path):
    root = tmp_path/'unprepared'
    module = root/'src/ralup/evaluate.py'
    module.parent.mkdir(parents=True)
    module.write_bytes(b'# independent minimal source fixture\n')
    (root/'requirements.lock').write_bytes(b'# independent lock fixture\n')
    return root, module


def test_production_confirmation_without_environment_fails_closed(tmp_path, monkeypatch):
    root, module = unprepared_tree(tmp_path)
    monkeypatch.setattr(evaluate, '__file__', str(module))
    with pytest.raises(FileNotFoundError, match='environment.json'):
        evaluate.confirmation_source_identity()
    assert not (root/'artifacts').exists()


def test_production_checkpoint_without_environment_fails_closed(tmp_path, monkeypatch):
    from ralup import state
    root, _ = unprepared_tree(tmp_path)
    monkeypatch.setattr(state, 'ROOT', root)
    with pytest.raises(FileNotFoundError, match='environment.json'):
        state._automatic_identities()
    assert not (root/'artifacts').exists()


def test_discovery_fixture_preserves_existing_report_and_cleans_only_own_bytes(tmp_path, monkeypatch):
    import conftest
    monkeypatch.setattr(conftest, 'ROOT', tmp_path)
    path = tmp_path/'artifacts/environment/environment.json'
    path.parent.mkdir(parents=True)
    sentinel = b'preserved independent existing report'
    path.write_bytes(sentinel)
    session = conftest.engineering_environment.__wrapped__()
    assert next(session) == path and path.read_bytes() == sentinel
    with pytest.raises(StopIteration): next(session)
    assert path.read_bytes() == sentinel
    path.unlink()
    session = conftest.engineering_environment.__wrapped__()
    assert next(session) == path
    report = json.loads(path.read_bytes())
    assert report['test_preparation']['scientific_gate'] is None
    assert report['test_preparation']['native_validation'] is False
    assert report['python']['executable']
    with pytest.raises(StopIteration): next(session)
    assert not path.exists()
    session = conftest.engineering_environment.__wrapped__()
    next(session)
    path.write_bytes(sentinel)
    with pytest.raises(StopIteration): next(session)
    assert path.read_bytes() == sentinel
