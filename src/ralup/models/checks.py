"""Bounded synthetic H04 engineering evidence, never a capacity/quality gate."""
from datetime import datetime, timezone
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import canonical_bytes
from ..records import ArtifactRef, sha256_file
from ..quantization.rotation import hadamard
from .transformer import ByteTransformer, ModelConfig

# Frozen before H04 measurements. Loss/logits/gradient inherit conservative F02
# FP32 complete-forward/backward smoke bounds (observed grad error 3.28e-7).
# Rotation inherits F04 measured context bounds; codes/duplicates inherit zero
# mismatch allowances from F03-F05/F02 respectively. These are fixture bounds,
# not tolerances inferred from the outputs under test or scientific training.
TOLERANCES = {'loss_atol':1e-4, 'loss_rtol':1e-4,
              'logits_atol':1e-4, 'logits_rtol':1e-4,
              'gradient_atol':1e-4, 'gradient_rtol':1e-4,
              'rotation_atol':8e-6, 'rotation_rtol':2e-6,
              'code_mismatches':0, 'duplicate_loss_atol':0.,
              'duplicate_gradient_atol':0., 'duplicate_logits_atol':0.}


def _architecture():
    counts, quantized = [], []
    for shape,total,internal in [((2,128,4),493056,393216),((4,256,4),3347456,3145728),((6,512,8),19281920,18874368)]:
        model = ByteTransformer(ModelConfig(*shape))
        count = sum(p.numel() for p in model.parameters())
        manifest = model.quantized_matrix_manifest()
        names = [f'blocks.{i}.{name}.weight' for i in range(shape[0]) for name in ('q','k','v','attention_output','up','down')]
        assert [item['name'] for item in manifest] == names
        internal_count = sum(dict(model.named_parameters())[name].numel() for name in names)
        assert (count,internal_count) == (total,internal)
        counts.append(count)
        quantized.append(internal_count)
    return {'parameter_counts':counts,'quantized_parameter_counts':quantized,
            'separate_q_k_v':True,'reference_attention':True,'optimized_attention_enabled':False}


def _fixture(shape, device, recipe):
    # Fixture-only CPU creation/loading; no training step copies production W.
    with torch.random.fork_rng():
        torch.manual_seed(104)
        model = ByteTransformer(ModelConfig(*shape)).to(device)
    if recipe is not None:
        model.configure_quantization(recipe)
        model.prepare_quantized_views(0,None)
    inputs = torch.tensor([[256,3,7,11,0],[256,0,0,0,0],[0,0,0,0,0]],device=device)
    positions = torch.arange(5,device=device)[None,:].expand_as(inputs)
    padding = torch.tensor([[False,False,False,False,True],[False,True,True,True,True],[True,True,True,True,True]],device=device)
    selected = ~padding
    logits = model(inputs,positions,padding)
    loss = F.cross_entropy(logits[selected],torch.tensor([3,7,11,13,2],device=device),reduction='sum')
    loss.backward()
    # Detached host comparison is evidence collection only.
    gradients = {name:p.grad.detach().cpu() for name,p in model.named_parameters()}
    codes = {name:c.codes.cpu() for name,c in model.quantization_contexts.items()}
    result = (logits.detach().cpu(),loss.detach().cpu(),gradients,codes)
    assert all(torch.isfinite(value).all() for value in (result[0],result[1],*gradients.values()))
    return result


def _parity(device):
    summary = {'cases':0,'loss_max_error':0.,'logits_max_error':0.,'gradient_max_error':0.,
               'code_mismatches':0,'duplicate_loss_max_error':0.,'duplicate_gradient_max_error':0.,
               'duplicate_logits_max_error':0.}
    cases = [((2,128,4),None),((2,32,4),{'name':'row_dynamic_max'}),
             ((2,32,4),{'name':'row_dynamic_max','rotation':'full_hadamard'}),
             ((2,32,4),{'name':'quest_fp4_row_reference','rotation':'full_hadamard'})]
    rows = []
    for shape,recipe in cases:
        reference = _fixture(shape,'cpu',recipe)
        actual = _fixture(shape,device,recipe)
        duplicate = _fixture(shape,device,recipe)
        logits,loss,gradients,codes = actual
        for kind,left,right in [('logits',logits,reference[0]),('loss',loss,reference[1])]:
            torch.testing.assert_close(left,right,atol=TOLERANCES[f'{kind}_atol'],rtol=TOLERANCES[f'{kind}_rtol'])
            summary[f'{kind}_max_error'] = max(summary[f'{kind}_max_error'],float((left-right).abs().max()))
        for name,gradient in gradients.items():
            torch.testing.assert_close(gradient,reference[2][name],atol=TOLERANCES['gradient_atol'],rtol=TOLERANCES['gradient_rtol'])
            summary['gradient_max_error'] = max(summary['gradient_max_error'],float((gradient-reference[2][name]).abs().max()))
            summary['duplicate_gradient_max_error'] = max(summary['duplicate_gradient_max_error'],float((gradient-duplicate[2][name]).abs().max()))
        summary['duplicate_loss_max_error'] = max(summary['duplicate_loss_max_error'],float((loss-duplicate[1]).abs()))
        summary['duplicate_logits_max_error'] = max(summary['duplicate_logits_max_error'],float((logits-duplicate[0]).abs().max()))
        for name,code in codes.items():
            summary['code_mismatches'] += int((code != reference[3][name]).sum())
            assert torch.equal(code,duplicate[3][name])
        rows.append({'shape':list(shape),'recipe':recipe,'scored_targets':5,'padding_includes_empty_row':True})
        summary['cases'] += 1
    assert summary['code_mismatches'] == TOLERANCES['code_mismatches']
    assert summary['duplicate_loss_max_error'] == TOLERANCES['duplicate_loss_atol']
    assert summary['duplicate_gradient_max_error'] == TOLERANCES['duplicate_gradient_atol']
    assert summary['duplicate_logits_max_error'] == TOLERANCES['duplicate_logits_atol']
    x = torch.arange(32,dtype=torch.float32).reshape(2,16)/31
    rotated = hadamard(x.to(device)).cpu()
    expected = hadamard(x)
    torch.testing.assert_close(rotated,expected,atol=TOLERANCES['rotation_atol'],rtol=TOLERANCES['rotation_rtol'])
    summary['rotation_max_error'] = float((rotated-expected).abs().max())
    summary['fixtures'] = rows
    return summary


def check_model(device: str, output: Path) -> ArtifactRef:
    if device not in ('cpu','mps'):
        raise ValueError('Model checks require cpu or mps')
    source_paths = ['src/ralup/models/transformer.py','src/ralup/models/linear.py','src/ralup/models/checks.py',
                    'src/ralup/quantization/context.py','src/ralup/quantization/rotation.py']
    root = Path(__file__).resolve().parents[3]
    report = {'schema_version':1,'suite':'model','device':device,'dtype':'float32',
              'recorded_utc':datetime.now(timezone.utc).isoformat(),
              'fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','unset'),
              'scope':'H04 bounded synthetic reference model checks; no model quality/capacity claim',
              'scientific_gate':None,'remaining_harness':['H05-H08; complete WP02 pending'],
              'tolerances':dict(TOLERANCES),'blockers':[],
              'source_hashes':{path:sha256_file(root/path) for path in source_paths}}
    if device == 'mps' and not torch.backends.mps.is_available():
        report.update(status='blocked',blockers=['Actual MPS unavailable in this process; sandbox report is not target-device evidence'])
    elif device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        report.update(status='blocked',blockers=['MPS checks require fallback explicitly disabled before torch import'])
    else:
        try:
            report['checks'] = _architecture()
            report['parity'] = _parity(device)
            report['status'] = 'pass'
        except (AssertionError,RuntimeError,ValueError) as error:
            report.update(status='fail',blockers=[f'{type(error).__name__}: {error}'])
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(canonical_bytes(report)+b'\n')
    return ArtifactRef(output,sha256_file(output))
