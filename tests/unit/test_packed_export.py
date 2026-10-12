"""Encoding counterexamples: ordinals are not hardware nibbles."""
import struct

import pytest
import torch


def test_every_ordinal_has_the_real_signed_e2m1_nibble():
    from ralup.export import pack_e2m1_codes, unpack_e2m1_codes
    codes = torch.arange(15, dtype=torch.uint8)
    # Low nibble first; positive zero is nibble 0, final unused nibble 0.
    payload = pack_e2m1_codes(codes)
    assert payload == bytes.fromhex('ef cd ab 09 21 43 65 07')
    assert torch.equal(unpack_e2m1_codes(payload, 15), codes)
    assert len(payload) == (len(codes)+1)//2


def test_negative_zero_decodes_positive_and_repacking_canonicalizes():
    from ralup.export import pack_e2m1_codes, unpack_e2m1_codes
    codes = unpack_e2m1_codes(bytes.fromhex('80'), 2)
    assert codes.tolist() == [7, 7]
    assert pack_e2m1_codes(codes) == b'\x00'


@pytest.mark.parametrize('payload,count', [(b'\x10', 1), (b'', 1), (b'\0\0', 1), (b'', -1), (b'', True)])
def test_invalid_count_length_or_odd_padding_rejects(payload, count):
    from ralup.export import unpack_e2m1_codes
    with pytest.raises((TypeError, ValueError)):
        unpack_e2m1_codes(payload, count)


@pytest.mark.parametrize('codes', [torch.tensor([15], dtype=torch.uint8), torch.tensor([7.]), torch.tensor([-1])])
def test_fake_float_or_invalid_ordinal_cannot_be_labeled_packed(codes):
    from ralup.export import pack_e2m1_codes
    with pytest.raises((TypeError, ValueError)):
        pack_e2m1_codes(codes)


def test_empty_codes_are_empty_bytes():
    from ralup.export import pack_e2m1_codes, unpack_e2m1_codes
    assert pack_e2m1_codes(torch.empty(0, dtype=torch.uint8)) == b''
    assert unpack_e2m1_codes(b'', 0).numel() == 0


def test_analytical_core_payload_is_separate_from_container():
    from ralup.export import analytical_payload
    from ralup.models.transformer import ModelConfig
    ledger = analytical_payload(ModelConfig(4, 256, 4), 'row')
    assert ledger['code_bytes'] == 1572864
    assert ledger['auxiliary_bytes'] == 806912
    assert ledger['scale_bytes'] == 36864
    assert ledger['payload_bytes'] == 2416640
    assert ledger['quantized_parameters'] == 3145728
    assert ledger['total_parameters'] == 3347456


def rewrite_metadata(ref, mutate):
    """Damage metadata and rehash envelope so semantic validation is exercised."""
    import hashlib
    import json
    from ralup.config import canonical_bytes
    from ralup.records import ArtifactRef
    raw = ref.path.read_bytes()
    length = struct.unpack('<Q', raw[8:16])[0]
    metadata = json.loads(raw[16:16+length])
    mutate(metadata)
    if metadata['context_sha256'] != '0'*64:
        metadata['context_sha256'] = hashlib.sha256(canonical_bytes({'matrices':metadata['matrices'],
            'evaluation_context':metadata['evaluation_context'],'scale_owner':metadata['scale_owner']})).hexdigest()
    # Keep declared actual bytes correct after metadata mutation.
    for _ in range(8):
        header = canonical_bytes(metadata)
        metadata['actual_container_bytes'] = 16+len(header)+len(raw[16+length:])
    header = canonical_bytes(metadata)
    altered = raw[:8]+struct.pack('<Q', len(header))+header+raw[16+length:]
    ref.path.write_bytes(altered)
    return ArtifactRef(ref.path, hashlib.sha256(altered).hexdigest())
