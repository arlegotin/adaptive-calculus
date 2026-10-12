"""Authenticate exact archived members and copy essential history; never restore old state."""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path.cwd(); OUT=Path(os.environ['E07_OPERATION'])
PRIOR=Path(os.environ['E07_BASE'])/'preservation-v1'


def sha(raw): return hashlib.sha256(raw).hexdigest()


def ref(p):
    raw=p.read_bytes();return dict(path=str(p),sha256=sha(raw),bytes=len(raw))


def verify(r):
    raw=Path(r['path']).read_bytes()
    assert sha(raw)==r['sha256'],r['path']
    if 'bytes' in r: assert len(raw)==r['bytes'],r['path']
    return raw


def write(name,value):
    (OUT/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


# Locked environment vs lock, separately from editable-project metadata.
import base64
import csv
import importlib.metadata
import re
lock=(ROOT/'requirements.lock').read_text()
expected=dict(re.findall(r'^([\w-]+)==([^ ]+) ',lock,re.M))
installed={d.metadata['Name'].lower().replace('_','-'):d for d in importlib.metadata.distributions()}
packages=[]
for name,version in expected.items():
    dist=installed[name.lower().replace('_','-')];assert dist.version==version
    count=0
    for row in csv.reader(dist.read_text('RECORD').splitlines()):
        path,digest,size=row
        if not digest: continue
        algorithm,want=digest.split('=',1);assert algorithm=='sha256'
        raw=Path(dist.locate_file(path)).read_bytes()
        assert base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()==want
        assert len(raw)==int(size);count+=1
    packages.append(dict(name=name,version=version,authenticated_record_files=count))
install=Path(os.environ['E07_BASE'])/'environment-v1/install-report.json'
for item in json.loads(install.read_bytes())['install']:
    digest=item['download_info']['archive_info']['hashes']['sha256']
    assert digest in lock
import ralup
assert Path(ralup.__file__).resolve().is_relative_to(ROOT/'src')
write('environment-audit.json',dict(python=sys.version,executable=sys.executable,
    packages=packages,lock=ref(ROOT/'requirements.lock'),install_report=ref(install),
    imported_project=str(Path(ralup.__file__).resolve()),
    local_editable_metadata='original ralup0.1.0 not a locked dependency; current source imported explicitly via PYTHONPATH',
    prior_assertion='environment-v1 failure retained; lock install succeeded, broad comparison included unlocked editable metadata'))

# Copy every current scratch file exactly, including raw primary research ledgers.
scratch=ROOT/'.superpowers/sdd/2026-10-03-ralup'; mapping=[]
boundary=datetime.now(timezone.utc).isoformat()
for source in ([] if (PRIOR/'history-mapping.json').exists() else sorted(scratch.rglob('*'))):
    if not source.is_file():continue
    raw=source.read_bytes();dest=OUT/'history'/source.relative_to(scratch)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    assert dest.read_bytes()==raw
    mapping.append(dict(original_path=str(source),sha256=sha(raw),bytes=len(raw),durable_path=str(dest)))
if not mapping:
    prior_mapping=json.loads((PRIOR/'history-mapping.json').read_bytes())
    for item in prior_mapping['files']:
        assert sha(Path(item['durable_path']).read_bytes())==item['sha256']
    mapping=prior_mapping['files']
write('history-mapping.json',dict(schema='ralup.e07_preservation.v1',capture_started_utc=boundary,
    capture_finished_utc=datetime.now(timezone.utc).isoformat(),files=mapping,
    late_root_delta_obligation='Root must capture/hash-link later E07 review/fixes, final whole-branch review/ruling collection and finishing before deleting ONLY this plan scratch. This capture does not authorize cleanup.',
    no_copyright_or_publication_grant=True))

# Preserve already accepted content-addressed M01 archive and verify all thirty
# generations FROM relocated archive objects. Immutable descriptors retain their
# historical path strings; only the read resolver maps them to the copied bytes.
old=ROOT/'artifacts/checks/m01-v1';new=PRIOR/'m01-archive';m01_mapping={}
for p in sorted(old.rglob('*')):
    if not p.is_file():continue
    raw=p.read_bytes();d=new/p.relative_to(old)
    if not d.exists():
        d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(raw)
    assert d.read_bytes()==raw;m01_mapping[str(p)]=ref(d)
spec=importlib.util.spec_from_file_location('m01_archive',ROOT/'docs/harness/m01-evidence-store.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def read_blob(record):
    copied=m01_mapping[record['path']];raw=gzip.decompress(verify(copied))
    assert copied['sha256']==record['sha256']
    assert len(raw)==record['raw_bytes'] and sha(raw)==record['raw_sha256']
    return raw
module.read_blob=read_blob
from ralup.records import ArtifactRef
from ralup.state import _manifest,TrainingSnapshot
from safetensors.torch import load
from ralup.config import canonical_bytes
audit=json.loads((new/'archive-audit-v3.json').read_bytes());proof=[]
for row in audit['snapshots']:
    archived=m01_mapping[row['archive']['path']];assert archived['sha256']==row['archive']['sha256']
    descriptor=json.loads(gzip.decompress(verify(archived)))
    metadata=module.inflate(descriptor['metadata'])
    assert sha(canonical_bytes(metadata))==descriptor['metadata_raw_sha256']
    # Byte reconstruction only: current semantic validation intentionally rejects
    # historical matrix-history owners. Never weaken that validator or restore.
    snap=TrainingSnapshot(metadata,{k:load(read_blob(v))['value'] for k,v in descriptor['tensors'].items()},
        milestone=descriptor['milestone'],producer_files={k:read_blob(v) for k,v in descriptor['producer_files'].items()})
    manifest,container=_manifest(snap)
    assert sha(canonical_bytes(manifest))==row['original']['sha256']
    assert sha(container)==descriptor['original_tensor_container_sha256']
    proof.append(dict(original=row['original'],archive=archived,manifest_sha256=sha(canonical_bytes(manifest)),
        tensor_container_sha256=sha(container),complete_tensor_members=descriptor['tensors'],
        complete_producer_files=descriptor['producer_files']))
assert len(proof)==30
write('m01-extraction-proof.json',dict(mandatory_generations=proof,files=m01_mapping,
    temporary_original_present=Path('/private/tmp/ralup-m01-20261004-v3').exists(),
    absence_attribution=None,observed_utc=datetime.now(timezone.utc).isoformat(),
    scope='All30 mandated original complete manifest/tensor/source/context bytes reconstructed from copied archives; current semantic validator rejected historical MatrixHistory schema in preservation-v1, retained. No model restoration or source-compatibility waiver. Optional non-archived temporary contents unavailable, no assertion of deletion cause.'))

# E06 exceptional dual-root administration: exact typed gzip AND raw identity.
admin=Path('/Volumes/git/legotin/adaptive-calculus/artifacts/ralup-administration/20261012/e06-v1')
start=json.loads((admin/'startup.json').read_bytes());inv=start['inventory']
assert start['schema']=='ralup.e06_compact_administration.v1'
compressed=(admin/inv['path']).read_bytes();raw=gzip.decompress(compressed)
assert sha(compressed)==inv['compressed_sha256'] and len(compressed)==inv['compressed_bytes']
assert sha(raw)==inv['uncompressed_sha256'] and len(raw)==inv['uncompressed_bytes']
assert len(json.loads(raw))==inv['file_count']
verified=[]
def walk(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            verify(value);verified.append(value)
        else:
            for v in value.values():walk(v)
    elif isinstance(value,list):
        for v in value:walk(v)
for name in ('startup.json','finish.json','report-postflight.json'):
    walk(json.loads((admin/name).read_bytes()))
write('e06-dual-root-proof.json',dict(startup=ref(admin/'startup.json'),finish=ref(admin/'finish.json'),
    complete_report=ref(admin/'E06-report-complete.md'),postflight=ref(admin/'report-postflight.json'),
    inventory=inv,verified_refs=verified,preservation='Original-checkout ignored root survives scratch/worktree cleanup; never move/delete it',
    authority='administrative encoding only; original source/runtime/outcomes and reserve failures unchanged'))
print(json.dumps({'history_files':len(mapping),'m01_mandatory_generations':len(proof),'e06_verified_refs':len(verified),'locked_packages':len(packages)}))
