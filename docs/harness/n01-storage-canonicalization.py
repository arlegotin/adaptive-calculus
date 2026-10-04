"""Root-approved storage-only canonicalization of exact completed N01 copies.

Original compressed identities remain historical. Canonical payload aliases
require independent compressed authentication and complete decompressed hashes.
Never change the primary runner/design or delete checkpoints/failure evidence.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime,timezone

ROOT=Path('artifacts/calibration/n01-v1')
DESIGN=Path('docs/harness/n01-storage-supplement.json')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()


def ref(path):return {'path':str(path),'sha256':digest(path)}


def retained():return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
    assert retained()+len(raw)<=2*2**30
    assert shutil.disk_usage(ROOT).free>=2**30+len(raw)
    path.write_bytes(raw)
    return ref(path)


def authenticate(value):
    path=Path(value['path'])
    assert not path.is_symlink()
    assert digest(path)==value['sha256']
    return path


def payload(value):
    path=authenticate(value)
    checksum=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            checksum.update(block);size+=len(block)
    result={'sha256':checksum.hexdigest(),'bytes':size}
    assert result['sha256']==value['uncompressed_sha256']
    assert size==value['uncompressed_bytes']
    return result


def freeze():
    primary=json.loads(Path('docs/harness/n01-design.json').read_bytes())
    value={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'version':'N01 storage-only canonicalization v1',
           'authority':'root storage ruling following independently authenticated complete step2 U0/U1/R payload equality ccc911bcbfa8778d1b70958904b51fa1e51a187fbe9815eeffb4d5b57d57301d (31172196bytes)',
           'runner':ref(Path(__file__)),'original_design':ref(Path('docs/harness/n01-design.json')),'producer':primary['producer'],
           'numerical_contract':'unchanged72steps/source/recipe/packing/strict resume/bounds; same2GiBcap and1GiBfreefloor; no expanded scope or new outcomes',
           'eligibility':'completed recipe result published and authenticated; ALL-owner/per-step exact PASS with no failures; only U1/R redundant same-step scientifictensorcopy files',
           'required_equivalence':'independently authenticate each original and canonical COMPRESSEDsha, stream DECOMPRESS complete encoded scientificpayload including scalar metadata+all tensor owners, verify recorded completePAYLOADsha+length and equality across originals/U0. Zero observed owner differences alone never sufficient.',
           'retain':'U0 complete raw scientificpayload every step including all pre-forwardU/scales/codes and postcommitW/scales/cachedcodes/context histories; everyarm full reports/manifests/RNG/inputs/rawNLL/cost/source/discrepancies; all4/8checkpointgens; everyfailure/process/futuremismatch witness',
           'predelete_audit':'for everypath publish originalCOMPRESSEDhash/PAYLOADhash, retainedcanonicalref/PAYLOADhash, independently measured equalityproof, complete-summary/authentication/provenance/reconstruction references and DISPOSITION before any deletion',
           'alias_semantics':'resolve complete values through explicit alias index; old compressed identifiers are historical/disposed, never replaced by canonical hashes or false currentArtifactRefs. E07 must disclose canonicalization/discard.',
           'deletion_scope':'ONLY explicit per-recipe U1/R state1..8.safetensors.gz lists after proofs publish; no checkpoints/user/earlier task/failure/other paths',
           'storage_forecast':'approximately720MB initial per-recipe; canonicalization removes~352MB per recipe. Forecast canonical final plus capacity/supplement before releasing next launch. If insufficient HOLD and seek separate prospective ruling, no extra deletion.'}
    print(json.dumps(write(DESIGN,value)))


def canonicalize(recipe):
    start=time.perf_counter();frozen=json.loads(DESIGN.read_bytes())
    assert ref(Path(__file__))==frozen['runner']
    directory=ROOT/recipe;result=json.loads((directory/'result.json').read_bytes())
    assert result['accepted'] and not result['failures']
    for comparison in result['comparisons']:
        for check in comparison['steps'].values():
            assert not check['discrete_mismatches']
            assert all(row['exact'] for row in check['owners'].values())
    reports={}
    for original in result['arms']:
        path=authenticate(original)
        reports[path.parent.name]=json.loads(path.read_bytes())
    assert set(reports)=={'U0','U1','R'}
    listed=[]
    for step in map(str,range(1,9)):
        canonical=reports['U0']['states'][step];canonical_payload=payload(canonical)
        for arm in ('U1','R'):
            original=reports[arm]['states'][step];measured=payload(original)
            path=Path(original['path'])
            assert path.resolve().is_relative_to((directory/arm).resolve())
            assert path.name==f'state-{step}.safetensors.gz'
            assert measured==canonical_payload
            listed.append({'step':int(step),'arm':arm,'original_historical_compressed_ref':{k:original[k] for k in ('path','sha256')},
                           'original_payload':measured,'retained_canonical_ref':{k:canonical[k] for k in ('path','sha256')},
                           'canonical_payload':canonical_payload,'independent_complete_payload_sha_and_length_equal':True,
                           'original_compressed_bytes':path.stat().st_size,
                           'disposition':'N01 operationally completed reconstructible redundant exact scientificcopy; original compressed identity disposed only after this published audit; complete same-step values retained via explicit canonical alias',
                           'reconstruction':'immutable source/parent/seed/input/per-step fullownerhash manifests + same scientific completepayload retainedU0; compressed header identity not claimed reconstructible'})
    audit={'storage_supplement':ref(DESIGN),'completed_exact_result':ref(directory/'result.json'),
           'arm_reports':result['arms'],'original_primary_design':frozen['original_design'],'source':frozen['producer'],
           'explicit_aliases_and_discard':listed,'retained_before_bytes':retained(),'free_before_bytes':shutil.disk_usage(ROOT).free,
           'authentication_decompression_seconds':time.perf_counter()-start,'failure_artifacts_removed':False}
    receipt=write(directory/'canonical-alias-audit.json',audit)
    assert digest(Path(receipt['path']))==receipt['sha256']
    # Reauthenticate compressed files at mutation boundary; no other path is
    # permitted. Publish all immutable proofs before the first unlink.
    for entry in listed:
        authenticate(entry['retained_canonical_ref'])
        authenticate(entry['original_historical_compressed_ref']).unlink()
    done={'audit':receipt,'removed_exact_paths':[x['original_historical_compressed_ref']['path'] for x in listed],
          'removed_bytes':sum(x['original_compressed_bytes'] for x in listed),'retained_after_bytes':retained(),
          'free_after_bytes':shutil.disk_usage(ROOT).free,'project_seconds':time.perf_counter()-start,
          'old_compressed_refs_are_historical_not_current':True,'all_retained_canonical_refs_authenticate':True}
    print(json.dumps(write(directory/'canonicalization-completed.json',done)))


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='canonicalize':canonicalize(sys.argv[2])
