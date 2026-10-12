"""Actual packed bytes through H07 restore and the H06 common forward/scorer."""
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

from ralup.data.articles import Article
from ralup.data.blocks import tile_article
from ralup.evaluate import evaluation_contexts, score_blocks
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.records import ArtifactRef, sha256_file


def fixture(tmp_path, recipe='row_dynamic_max', rotation='identity', device='cpu', learned=False):
    from ralup.optim.base import MethodAdapter, default_parameter_groups
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.state import capture_snapshot, save_snapshot
    from ralup.train import TrainingState
    with torch.random.fork_rng():
        torch.manual_seed(606)
        model = ByteTransformer(ModelConfig(1, 8, 2)).to(device)
    model.configure_quantization({'name':recipe, 'rotation':rotation, 'backward':'ste'})
    if learned:
        from ralup.baselines.learned_scale import LearnedScaleAdapter, FIXTURE_CONFIG
        method = LearnedScaleAdapter(FIXTURE_CONFIG).bind(dict(model.named_parameters()), default_parameter_groups(model))
    else:
        method = MethodAdapter(dict(model.named_parameters()), default_parameter_groups(model), AdamWUpdateRule())
    state = TrainingState(model, method, configuration={'synthetic':True, 'purpose':'E06 export mechanics'})
    if learned:
        method.prepare_contexts(state, 0)
        for n,s in method.physical_scales.items():
            model._runtime[n] = {'previous_scales':s.clone()}
    elif 'fixed' in recipe:
        contexts = model.prepare_quantized_views(0, None)
        for n,c in contexts.items():
            model._runtime[n] = {'previous_scales':c.scales.clone(), 'previous_zero_groups':c.scale_zero_groups.clone()}
        # This catches accidental final re-ranging of frozen warm-start scales.
        with torch.no_grad():
            model.blocks[0].q.weight.mul_(3)
    raw = b'\x00E06 fixture\xff\n'
    blocks = tile_article(Article(raw, [], hashlib.sha256(raw).hexdigest()))
    ref = save_snapshot(capture_snapshot(state), tmp_path/'checkpoint')
    context = {'source_checkpoint':ref, 'evaluation_context':{'partition':'fixture', 'seed':606, 'step':0}, 'parity_blocks':blocks}
    return state, context


@pytest.mark.parametrize('recipe,rotation,learned', [
    ('row_dynamic_max','identity',False), ('row_dynamic_max','full_hadamard',False),
    ('row_fixed_max','identity',False), ('quest_fp4_row_reference','full_hadamard',False),
    ('tensor_dynamic_max','identity',False), ('row_fixed_max','identity',True)])
def test_final_context_logits_loss_bpb_and_exact_scales(tmp_path, recipe, rotation, learned):
    from ralup.export import export_model, load_export
    state, context = fixture(tmp_path, recipe, rotation, learned=learned)
    before_rng = torch.get_rng_state().clone()
    before_cache = state.model._contexts
    final_contexts = evaluation_contexts(state.model, context['evaluation_context'])
    ref = export_model(state.model, context, tmp_path/'model.fp4')
    decoded = load_export(ref)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert state.model._contexts is before_cache
    for n,c in final_contexts.items():
        assert torch.equal(decoded.contexts[n].codes, c.codes.cpu())
        assert torch.equal(decoded.contexts[n].scales, c.scales.cpu())
    block = context['parity_blocks'][0]
    inputs = torch.tensor([block.inputs]); positions = torch.tensor([block.positions])
    with torch.no_grad():
        expected = state.model(inputs, positions, contexts=final_contexts)
        torch.testing.assert_close(decoded(inputs, positions), expected, atol=0, rtol=0)
    reference = score_blocks(state.model, context['parity_blocks'], evaluation_context=context['evaluation_context'])
    actual = score_blocks(decoded, context['parity_blocks'], evaluation_context=context['evaluation_context'])
    assert actual == reference
    metadata = decoded.metadata
    assert metadata['actual_container_bytes'] == ref.path.stat().st_size
    assert metadata['analytical_payload']['payload_bytes'] < metadata['actual_container_bytes']
    assert metadata['source_checkpoint']['sha256'] == context['source_checkpoint'].sha256
    assert metadata['scientific_final_model'] is False
    assert metadata['parity']['code_mismatches'] == 0
    assert metadata['parity']['target_count'] == len(b'\x00E06 fixture\xff\n')
    assert len(metadata['context_sha256']) == 64
    assert metadata['analytical_payload']['quantized_fraction'] == 768/6976


@pytest.mark.parametrize('damage', ['payload','missing_scales','rotation','shape','schema','float_codes','extra','checksum','ledger','context_hash',
    'parity','source_hash','owner_backend','protected_context'])
def test_corrupt_or_unknown_container_fails_closed(tmp_path, damage):
    from ralup.export import export_model, load_export
    from tests.unit.test_packed_export import rewrite_metadata
    state, context = fixture(tmp_path)
    ref = export_model(state.model, context, tmp_path/'model.fp4')
    if damage == 'payload':
        raw = ref.path.read_bytes(); ref.path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        ref = ArtifactRef(ref.path, sha256_file(ref.path))
    else:
        def mutate(m):
            if damage == 'missing_scales': del m['matrices'][0]['scales']
            elif damage == 'rotation': m['matrices'][0]['rotation'] = 'arbitrary_import'
            elif damage == 'shape': m['matrices'][0]['shape'] = [999, 999]
            elif damage == 'schema': m['schema'] = 'other.v1'
            elif damage == 'float_codes': m['matrices'][0]['codes']['dtype'] = 'float32'
            elif damage == 'extra': m['class'] = 'eval'
            elif damage == 'checksum': m['matrices'][0]['scales']['sha256'] = '0'*64
            elif damage == 'ledger': m['analytical_payload']['code_bytes'] += 1
            elif damage == 'context_hash': m['context_sha256'] = '0'*64
            elif damage == 'parity': m['parity']['code_mismatches'] = 1
            elif damage == 'source_hash': m['source_checkpoint']['sha256'] = 'malformed'
            elif damage == 'owner_backend': m['scale_owner']['backend'] = 'unknown'
            elif damage == 'protected_context': m['evaluation_context']['partition'] = 'test'
        ref = rewrite_metadata(ref, mutate)
    with pytest.raises((ValueError, TypeError)):
        load_export(ref)


def test_source_identity_and_model_mismatch_reject_before_publication(tmp_path):
    from ralup.export import export_model
    state, context = fixture(tmp_path)
    with torch.no_grad(): state.model.output.weight.add_(1)
    with pytest.raises(ValueError, match='source|checkpoint'):
        export_model(state.model, context, tmp_path/'bad.fp4')
    assert not (tmp_path/'bad.fp4').exists()


def test_half_conversion_fails_before_mutating_decoded_model(tmp_path):
    from ralup.export import export_model, load_export
    state, context = fixture(tmp_path)
    decoded = load_export(export_model(state.model, context, tmp_path/'model.fp4'))
    with pytest.raises(ValueError, match='FP32'):
        decoded.half()
    assert all(p.dtype == torch.float32 for p in decoded.parameters())


def test_odd_width_matrix_padding_is_accounted_separately(tmp_path):
    from ralup.export import export_model, load_export
    from ralup.optim.base import MethodAdapter, default_parameter_groups
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.state import capture_snapshot, save_snapshot
    from ralup.train import TrainingState
    model = ByteTransformer(ModelConfig(1,3,1))
    model.configure_quantization({'name':'row_dynamic_max'})
    state = TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()))
    checkpoint = save_snapshot(capture_snapshot(state), tmp_path/'checkpoint')
    raw = b'odd'
    ref = export_model(model, {'source_checkpoint':checkpoint,'evaluation_context':{'partition':'fixture'},
        'parity_blocks':tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))}, tmp_path/'odd.fp4')
    metadata = load_export(ref).metadata
    assert metadata['analytical_payload']['code_bytes'] == 56  # 4*ceil(9/2) + 2*(36/2).
    assert sum(m['padding_count'] for m in metadata['matrices']) == 4


def test_source_strict_restore_is_not_weakened_for_export(tmp_path, monkeypatch):
    from ralup import state as checkpoint
    from ralup.export import export_model
    state, context = fixture(tmp_path)
    identities = checkpoint._automatic_identities()
    first = next(iter(identities))
    identities[first] = ArtifactRef(identities[first].path, '0'*64)
    monkeypatch.setattr(checkpoint, '_automatic_identities', lambda: identities)
    with pytest.raises(ValueError, match='runtime incompatible'):
        export_model(state.model, context, tmp_path/'bad.fp4')
    assert not (tmp_path/'bad.fp4').exists()


def test_real_export_cli_fixture_produces_loadable_packed_file(tmp_path):
    from ralup.export import load_export
    state, context = fixture(tmp_path)
    checkpoint = context['source_checkpoint'].path
    output = tmp_path/'cli.fp4'
    result = subprocess.run([sys.executable, '-m', 'ralup', 'export', '--checkpoint', str(checkpoint), '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    model = load_export(ArtifactRef(output, report['sha256']))
    assert report['actual_container_bytes'] == output.stat().st_size
    assert model.metadata['source_checkpoint']['sha256'] == sha256_file(checkpoint)


@pytest.mark.parametrize('rotation,learned', [('full_hadamard',False), ('identity',True)])
def test_actual_mps_export_composition(tmp_path, rotation, learned):
    from ralup.export import export_model, load_export
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    assert torch.backends.mps.is_available()
    state, context = fixture(tmp_path, 'row_fixed_max' if learned else 'row_dynamic_max', rotation, 'mps', learned)
    ref = export_model(state.model, context, tmp_path/'native.fp4')
    decoded = load_export(ref)
    final_contexts = evaluation_contexts(state.model, context['evaluation_context'])
    for n,c in final_contexts.items():
        assert torch.equal(decoded.contexts[n].storage_values, c.storage_values.cpu())
    block = context['parity_blocks'][0]
    inputs = torch.tensor([block.inputs]); positions = torch.tensor([block.positions])
    from ralup.models.checks import TOLERANCES
    with torch.no_grad():
        reference_logits = state.model(inputs.to('mps'),positions.to('mps'),contexts=final_contexts).cpu()
        portable_logits = decoded(inputs,positions)
        torch.testing.assert_close(portable_logits, reference_logits,
            atol=TOLERANCES['logits_atol'],rtol=TOLERANCES['logits_rtol'])
    decoded.to('mps')
    assert decoded.metadata['parity']['backend'] == 'mps'
    original = score_blocks(state.model, context['parity_blocks'], evaluation_context=context['evaluation_context'])
    actual = score_blocks(decoded, context['parity_blocks'], evaluation_context=context['evaluation_context'])
    assert original == actual
