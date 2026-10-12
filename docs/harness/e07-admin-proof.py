"""Exact E06 typed administrative consumer; relative refs resolve at bundle root."""
import gzip
import hashlib
import json
import os
from pathlib import Path

base=Path('/Volumes/git/legotin/adaptive-calculus/artifacts/ralup-administration/20261012/e06-v1')
out=Path(os.environ['E07_OPERATION']);refs=[]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':str(path),'sha256':digest(path.read_bytes()),'bytes':path.stat().st_size}
def walk(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            p=Path(value['path']);p=p if p.is_absolute() else base/p
            raw=p.read_bytes();assert digest(raw)==value['sha256'],str(p)
            if 'bytes' in value:assert len(raw)==value['bytes']
            refs.append({'original':value,'resolved':ref(p)})
        else:
            for v in value.values():walk(v)
    elif isinstance(value,list):
        for v in value:walk(v)
start=json.loads((base/'startup.json').read_bytes());v=start['inventory']
assert start['schema']=='ralup.e06_compact_administration.v1'
compressed=(base/v['path']).read_bytes();raw=gzip.decompress(compressed)
assert len(compressed)==v['compressed_bytes'] and digest(compressed)==v['compressed_sha256']
assert len(raw)==v['uncompressed_bytes'] and digest(raw)==v['uncompressed_sha256']
assert len(json.loads(raw))==v['file_count']
for name in ('startup.json','finish.json','report-postflight.json'):walk(json.loads((base/name).read_bytes()))
report=dict(schema='ralup.e07_e06_authentication.v1',inventory=v,verified_refs=refs,
    objects={n:ref(base/n) for n in ('startup.json','finish.json','E06-report-complete.md','report-postflight.json')},
    preservation='Keep original-checkout ignored root through worktree/scratch cleanup; administrative encoding grants no source/runtime/science waiver.')
(out/'e06-proof.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'refs':len(refs),'raw_inventory_bytes':len(raw),'gzip_bytes':len(compressed)}))
