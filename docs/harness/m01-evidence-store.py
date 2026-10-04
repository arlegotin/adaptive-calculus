"""Task-only exact snapshot evidence archive; no product loader/migration changes.

Large repeated strings, individual safetensors and exact producer bytes are
content-addressed before compression. Every object authenticates stored and raw
identities; duplicate content never implies approximate numerical equivalence.
"""
import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path

from safetensors.torch import load, save
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef, sha256_file
from ralup.state import TrainingSnapshot, _validated, _manifest, _fsync_directory


CHECK_WRITE = None


def asref(ref):return {'path':str(ref.path),'sha256':ref.sha256}


def blob(raw, directory):
    digest=hashlib.sha256(raw).hexdigest();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(digest+'.gz')
    if not path.exists():
        compressed=gzip.compress(raw,compresslevel=6,mtime=0)
        if CHECK_WRITE is not None:CHECK_WRITE(len(compressed))
        staging=directory/('.pending-'+uuid.uuid4().hex)
        with staging.open('xb') as stream:stream.write(compressed);stream.flush();os.fsync(stream.fileno())
        if gzip.decompress(staging.read_bytes()) != raw:raise ValueError('Staged evidence checksum mismatch')
        staging.rename(path);_fsync_directory(directory)
    record={**asref(ArtifactRef(path,sha256_file(path))),'raw_sha256':digest,'raw_bytes':len(raw)}
    assert read_blob(record) == raw
    return record


def read_blob(record):
    path=Path(record['path'])
    if sha256_file(path) != record['sha256']:raise ValueError('Evidence checksum mismatch')
    raw=gzip.decompress(path.read_bytes())
    if len(raw) != record['raw_bytes'] or hashlib.sha256(raw).hexdigest() != record['raw_sha256']:
        raise ValueError('Evidence raw checksum mismatch')
    return raw


def compact(value,directory):
    if isinstance(value,str) and len(value)>4096:
        return {'__m01_large_string__':blob(value.encode(),directory)}
    if isinstance(value,dict):
        if set(value)=={'__m01_large_string__'}:raise ValueError('Reserved evidence marker')
        return {k:compact(v,directory) for k,v in value.items()}
    if isinstance(value,list):return [compact(v,directory) for v in value]
    return value


def inflate(value):
    if isinstance(value,dict):
        if set(value)=={'__m01_large_string__'}:return read_blob(value['__m01_large_string__']).decode()
        return {k:inflate(v) for k,v in value.items()}
    if isinstance(value,list):return [inflate(v) for v in value]
    return value


def archive_snapshot(snapshot,directory):
    _validated(snapshot)
    metadata=canonical_bytes(snapshot.metadata)
    manifest, container=_manifest(snapshot)
    value={'original_manifest_sha256':hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
           'original_tensor_container_sha256':hashlib.sha256(container).hexdigest(),'schema':'m01.complete_snapshot_evidence.v1','metadata':compact(snapshot.metadata,directory),
           'metadata_raw_sha256':hashlib.sha256(metadata).hexdigest(),
           'tensors':{k:blob(save({'value':v}),directory) for k,v in snapshot.tensors.items()},
           'producer_files':{k:blob(v,directory) for k,v in snapshot.producer_files.items()},'milestone':snapshot.milestone}
    descriptor=blob(canonical_bytes(value),directory)
    return ArtifactRef(descriptor['path'],descriptor['sha256'])


def recover_snapshot(ref):
    if sha256_file(ref.path) != ref.sha256:raise ValueError('Evidence descriptor checksum mismatch')
    value=json.loads(gzip.decompress(ref.path.read_bytes()))
    if value['schema'] != 'm01.complete_snapshot_evidence.v1':raise ValueError('Unknown evidence schema')
    metadata=inflate(value['metadata'])
    if hashlib.sha256(canonical_bytes(metadata)).hexdigest() != value['metadata_raw_sha256']:
        raise ValueError('Evidence metadata checksum mismatch')
    snapshot=TrainingSnapshot(metadata,{k:load(read_blob(v))['value'] for k,v in value['tensors'].items()},
                              milestone=value['milestone'],producer_files={k:read_blob(v) for k,v in value['producer_files'].items()})
    _validated(snapshot)
    manifest,container=_manifest(snapshot)
    if hashlib.sha256(container).hexdigest() != value['original_tensor_container_sha256'] or hashlib.sha256(canonical_bytes(manifest)).hexdigest() != value['original_manifest_sha256']:
        raise ValueError('Complete snapshot reconstruction checksum mismatch')
    return snapshot
