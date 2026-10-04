"""Actual target-device model loss/gradient/code parity under frozen bounds."""
import os

import pytest
import torch


@pytest.mark.parametrize('device',['cpu','mps'])
def test_reference_model_parity_and_duplicate_variability(device,tmp_path):
    from ralup.models.checks import check_model
    import json
    if device == 'mps':
        if not torch.backends.mps.is_available():
            pytest.skip('Actual MPS unavailable; sandbox skip is not device evidence')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    ref = check_model(device,tmp_path/f'{device}.json')
    report = json.loads(ref.path.read_text())
    assert report['status'] == 'pass', report['blockers']
    assert report['parity']['cases'] == 4
    assert report['parity']['code_mismatches'] == 0
    assert report['parity']['duplicate_loss_max_error'] == 0
    assert report['parity']['duplicate_gradient_max_error'] == 0
