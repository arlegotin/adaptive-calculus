"""Root-authorized NEW typed gzip inventory administration; no numerical imports.

Exact source and original terminal outcomes stay frozen. All bytes are
precomputed/admitted before writes. Allocation bounds are independent of the
logical inventory. This prospective Git-volume location exception owns zero
Data writes, preserves the observed Data shortfall, and retains a1GiB Git floor.
"""
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

GIB = 1024**3
MIB = 1024**2
E07 = 3_072_327_680
TASK_BASE = '7cab914833ac080d3750653f7ac14a249a470997'
ROOT_RULING = '''Ruling: Complete E06 administration in a NEW owned directory using the concrete deterministic gzip(mtime=0) full-inventory sidecar and a small explicitly typed/hash-linked startup envelope. Retain and authenticate the exact uncompressed inventory bytes/hash, compressed bytes/hash, original final-capture startup/finish refs, source archive, runtime/commit/report/finish, and all original outcomes. Precompute exact encoded payloads before writes; include conservative physical allocation and metadata/staging overhead, and admit the proposed complete <=16MiB Data footprint only while fresh raw free space preserves the unchanged64GiB floor plus E07reserve3,072,327,680B. Keep Git's separate bound/floor. No move/delete/overwrite, no retrospective passing-budget claim: report that the original final capture's logical forecast failed to retain its separately planned administration reserve despite preserving the main floor/E07 reserve. Source is frozen; no repeated tests/native/CLI. E07 must explicitly authenticate/decompress this one typed administrative encoding, without relaxing H07/runtime/source/scientific authority. Why: the measured lossless full inventory compresses53,791,175B to1,911,515B, and the measured <=16MiB physical administration fits current29,663,232B pre-E07 headroom while retaining the full evidentiary content. If wrong: filesystem/background allocation could still consume the remaining reserve or downstream readers could reject the new envelope; preserve originals, stop further owned writes at failed boundaries, and leave closure blocked until an authenticated compatible administration path fits. This is an operational evidence-recording change, not a scientific-budget, precision, numerical or eligibility amendment.'''

GIT_VOLUME_RULING = "Ruling: Finish E06's small administrative capture in a NEW uniquely owned directory at /Volumes/git/legotin/adaptive-calculus/artifacts/ralup-administration/20261012/e06-v1 on the Git volume. Root verified that this path is git-ignored; fail if it already exists, retain an ownership marker, and do not touch existing artifacts. Use the concrete complete physical forecast of 6,336,512 bytes with conservative staging/metadata allowances and a fresh Git free-space check; keep the <=16MiB transaction bound and the stronger original 1GiB Git floor. Owned Data-volume writes must be zero: reference the existing immutable Data source archive/inventory evidence, create the new lossless gzip inventory/envelope/helper/report/runtime/commit/finish only on the Git volume, and keep temporary files there. Preserve the actual Data reserve shortfall, both prior HOLDs and unknown background attribution; do not present this as a restored Data reserve or a passing original forecast. Hash-link both storage roots explicitly, authenticate the reused numerical archive and new administrative bundle, and route this exception into E07's discovery, preservation and reproduction. The new directory must survive plan-workspace/worktree cleanup; all existing evidence stays in place. Resume the original E06 worker for this administration and commit only; no numerical source edit or test/native/CLI repeat. Why: a roughly 6.34MB new record fits the measured 6.88GB Git free space while Data no longer has the required reserve, and this reversible local evidence location avoids altering caps or discarding proof. If wrong: physical allocation or downstream path assumptions could break preservation/reproduction; stop new writes on failed boundaries, retain the original proof and incomplete administrative record, and leave closure unresolved rather than deleting files or upgrading scientific claims. This is a prospective administrative location exception, not a scientific or Data-reserve waiver."

def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    raw = path.read_bytes()
    return {'path':str(path),'sha256':digest(raw),'bytes':len(raw)}


def write(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())


def guard(root,store,forecast,logical_bytes):
    if (logical_bytes+forecast > 32*GIB or shutil.disk_usage(root).free-forecast-16*MIB < GIB):
        raise ValueError('Git-volume administration cannot retain its1GiB floor/transaction/store bounds')


def main():
    start = time.perf_counter(); root = Path.cwd()
    store = Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
    original = Path('/Volumes/git/legotin/adaptive-calculus')
    base = original/'artifacts/ralup-administration/20261012/e06-v1'
    if base.exists(): raise ValueError('New immutable administration directory required')
    assert base.resolve().is_relative_to(original.resolve()), 'Destination must resolve within original Git-volume checkout'
    subprocess.run(['git','check-ignore',str(base.relative_to(original))],cwd=original,capture_output=True,check=True)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1', 'Zero owned Data pycache writes required'
    final = store/'e06-20261012'/'final-verification-v1'
    index = json.loads((root/'docs/harness/e06-evidence.json').read_bytes())
    for attempt in index['attempts']:
        for ref in attempt['receipts'].values():
            raw = Path(ref['path']).read_bytes()
            assert len(raw)==ref['bytes'] and digest(raw)==ref['sha256']
    startup = json.loads((final/'startup.json').read_bytes())
    terminal = json.loads((final/'finish.json').read_bytes())
    assert terminal['exit_code']==0 and terminal['child_output_within_allowance']
    for name,item in startup['source'].items():
        if name.startswith(('src/','tests/','configs/')) or name=='requirements.lock':
            assert digest((root/name).read_bytes())==item['sha256'],name
    assert digest((root/'requirements.lock').read_bytes()) == '488142a35a4b1a6abe132f08b5895621ca30bb535b0326e82516446903fcb32b'
    assert subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()==TASK_BASE
    assert subprocess.run(['git','branch','--show-current'],capture_output=True,text=True,check=True).stdout.strip()=='feat/ralup-execution'
    subprocess.run(['git','diff','--cached','--quiet'],check=True)
    subprocess.run(['git','diff','--check'],check=True)
    report = root/'.superpowers/sdd/2026-10-03-ralup/E06-report.md'
    paths = sorted(set(list(root.glob('src/ralup/**/*.py'))+list(root.glob('tests/**/*.py'))+
        list(root.glob('docs/harness/e06*'))+list(root.glob('configs/ablations/*.json'))+
        [root/'requirements.lock',root/'artifacts/environment/environment.json']))
    contents = {str(p.relative_to(root)):p.read_bytes() for p in paths if p.is_file()}
    files = ['src/ralup/export.py','src/ralup/__main__.py','tests/unit/test_packed_export.py',
        'tests/integration/test_export_forward.py','docs/harness/e06-attempt.py','docs/harness/e06-final-check.py',
        'docs/harness/e06-administration.py','docs/harness/e06-contract.md','docs/harness/e06-evidence.json']
    assert sum((root/n).stat().st_size for n in files)+report.stat().st_size <= 16*MIB
    rows = []
    for p in sorted(store.rglob('*')):
        if p.is_file():
            stat = p.stat();rows.append({'path':str(p.relative_to(store)),'bytes':stat.st_size,'allocated_bytes':stat.st_blocks*512})
    inventory = (json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode()
    compressed = gzip.compress(inventory,mtime=0)
    assert gzip.decompress(compressed)==inventory
    runtime = encoded({'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'machine':platform.machine(),
        'torch_distribution_version':importlib.metadata.version('torch'),
        'purpose':'stdlib-only administration; exact native observation remains original final runtime reference'})
    diff = subprocess.run(['git','diff','--binary'],capture_output=True,check=True).stdout
    # Authenticate and hash-link the complete frozen source archive already
    # retained by final capture; only NEW administrative sources need a copy.
    source_archive = {}
    delta_sources = {}
    for name,raw in contents.items():
        prior = final/'producer'/name
        if prior.is_file() and prior.read_bytes()==raw:
            source_archive[name] = reference(prior)
        else:
            delta_sources['producer/'+name] = raw
            source_archive[name] = {'path':str(base/'producer'/name),'sha256':digest(raw),'bytes':len(raw)}
    ownership = encoded({'schema':'ralup.administration_ownership.v1','task':'E06','owner':'original E06 implementer',
        'project_root':str(original),'directory':str(base),'git_volume_only':True,'Data_owned_write_bytes':0,
        'task_base':TASK_BASE,'research_cutoff':'2026-10-03'})
    fixed = {**delta_sources,'ownership.json':ownership,'inventory.json.gz':compressed,
        'runtime-observation.json':runtime,'producer-git-diff.patch':diff,'report-before-commit.md':report.read_bytes()}
    dirs = {str(Path(n).parent) for n in fixed}|{'tmp'}
    for n in tuple(dirs): dirs.update(str(p) for p in Path(n).parents)
    # All staging is in memory. Per-file8KiB overhead and per-directory8KiB,
    # plus2MiB filesystem metadata and2MiB bounded admin/finish writes.
    rounded = sum(((len(raw)+4095)//4096)*4096+8192 for raw in fixed.values())
    allowance = 2*MIB
    physical = rounded+8192*(len(dirs)+3)+2*MIB+allowance  # Three new original-checkout parents.
    envelope = {'schema':'ralup.e06_compact_administration.v1','utc':datetime.now(timezone.utc).isoformat(),
        'local_time':datetime.now().astimezone().isoformat(),'research_cutoff':'2026-10-03','root_ruling':GIT_VOLUME_RULING,'prior_compact_ruling':ROOT_RULING,
        'storage_roots':{'data':{'path':str(store),'identity':reference(store/'store.json'),'owned_new_bytes':0},
            'git':{'path':str(base),'project_root':str(original),'ownership_marker':{'path':'ownership.json','sha256':digest(ownership),'bytes':len(ownership)}}},
        'E07_obligation':'authenticate both roots, typed inventory raw/gzip hashes and complete source references; preserve original-checkout admin through worktree cleanup; no H07/source/scientific authority relaxation',
        'original_final_receipts':{n:reference(final/n) for n in ('startup.json','finish.json','runtime-observation.json','actual-cli-proof.json')},
        'inventory':{'schema':'ralup.complete_store_inventory.v1','encoding':'gzip_json_utf8_mtime0',
            'path':'inventory.json.gz','uncompressed_bytes':len(inventory),'uncompressed_sha256':digest(inventory),
            'compressed_bytes':len(compressed),'compressed_sha256':digest(compressed),'file_count':len(rows)},
        'source':{n:{'bytes':len(raw),'sha256':digest(raw)} for n,raw in contents.items()},
        'source_archive':source_archive,
        'numerical_source_archive':{n:ref for n,ref in source_archive.items()
            if n.startswith(('src/','tests/','configs/')) or n in ('requirements.lock','artifacts/environment/environment.json')},
        'new_administrative_source_archive':{n:ref for n,ref in source_archive.items()
            if str(base/'producer') in ref['path']},
        'fixed_outputs':{n:{'bytes':len(raw),'sha256':digest(raw)} for n,raw in fixed.items()},
        'storage':{'store_logical_bytes_before':sum(r['bytes'] for r in rows),
            'store_allocated_bytes_before':sum(r['allocated_bytes'] for r in rows),
            'raw_data_free_before':shutil.disk_usage(store).free,'git_free_before':shutil.disk_usage(root).free,
            'E07_reserved_bytes':E07,'Data_floor_bytes':64*GIB,'Git_floor_bytes':GIB,
            'small_source_git_bound_bytes':16*MIB,'complete_git_physical_bound_bytes':physical,
            'startup_bytes':0,'fixed_logical_bytes':sum(map(len,fixed.values())),'admin_finish_allowance_bytes':allowance,
            'staging':'in-memory only; any child temporary files restricted to NEW Git-volume tmp; no Data writes/scratch/moves/deletions',
            'Data_owned_write_bytes':0,'actual_Data_reserve_shortfall_before_bytes':max(0,64*GIB+E07-shutil.disk_usage(store).free),'physical_overhead':'rounded4KiB+8KiB/file+8KiB/directory+2MiBmetadata',
            'original_final_admin_reserve_retained':False}}
    for _ in range(16):
        raw_start = encoded(envelope)
        bound = physical+((len(raw_start)+4095)//4096)*4096+8192
        if envelope['storage']['startup_bytes']==len(raw_start) and envelope['storage']['complete_git_physical_bound_bytes']==bound:break
        envelope['storage']['startup_bytes']=len(raw_start);envelope['storage']['complete_git_physical_bound_bytes']=bound
    else:raise ValueError('Compact startup size did not converge')
    forecast = envelope['storage']['complete_git_physical_bound_bytes']
    assert forecast <= 16*MIB
    if sys.argv[1:]==['--forecast-only']:
        print(json.dumps({'storage':envelope['storage'],'inventory':envelope['inventory'],'source_files':len(contents)},sort_keys=True))
        guard(root,store,forecast,envelope['storage']['store_logical_bytes_before'])
        return
    if sys.argv[1:]:raise ValueError('Only --forecast-only is supported')
    guard(root,store,forecast,envelope['storage']['store_logical_bytes_before'])
    base.mkdir(parents=True,exist_ok=False)
    write(base/'ownership.json',fixed['ownership.json'])
    (base/'tmp').mkdir()
    os.environ['TMPDIR']=str(base/'tmp')
    # No execution imports follow: only the exact authorized git administration.
    for name,raw in fixed.items():
        if name != 'ownership.json':write(base/name,raw)
    write(base/'startup.json',raw_start)
    guard(root,store,allowance,sum(r['bytes'] for r in rows)+sum(map(len,fixed.values()))+len(raw_start))
    subprocess.run(['git','add','--',*files],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    result = subprocess.run(['git','commit','-m','feat: add packed model export with decoded-forward verification'],capture_output=True,text=True)
    command = encoded({'command':['git','commit','-m','feat: add packed model export with decoded-forward verification'],
        'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    if len(command)>64*1024:raise ValueError('Commit output exceeds reserved bound; original retained in process memory')
    write(base/'commit-command.json',command)
    if result.returncode:raise SystemExit(result.returncode)
    sha = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    status = subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True).stdout
    assert not status,status
    with report.open('a') as f:
        f.write(f'\nCommit `{sha}` — `feat: add packed model export with decoded-forward verification`. Worktree status `{status!r}`. Coordinator fresh FULL E06 task review pending; root ONE whole-branch review after E07.\n')
        f.write(f'Git-volume compact administration admitted physical bound{forecast}B (<=16MiB), inventory raw{len(inventory)}B / gzip{len(compressed)}B; original allocation/admin-reserve forecast limitation is unchanged. Exact final source/tests/config/lock and all indexed original receipts authenticated before commit.\n')
    write(base/'E06-report.md',report.read_bytes())
    write(base/'commit.json',encoded({'sha':sha,'task_base':TASK_BASE,'status':status,'source_tests_match_final_verification':True,
        'report':reference(base/'E06-report.md'),'scientific_final_model':False,'unknown_enclosing_engineering_cost':None}))
    write(base/'stdout.txt',encoded({'sha':sha,'status':status,'physical_forecast_bytes':forecast}))
    write(base/'stderr.txt',b'')
    objects = [{'path':str(p.relative_to(base)),'bytes':p.stat().st_size,'allocated_bytes':p.stat().st_blocks*512,
        'sha256':digest(p.read_bytes())} for p in sorted(base.rglob('*')) if p.is_file()]
    if sum(r['allocated_bytes'] for r in objects)>forecast:raise ValueError('Actual administration allocation exceeded forecast')
    # Fresh global physical guard includes bounded about-to-write finish.
    current = [p.stat() for p in store.rglob('*') if p.is_file()]
    logical = sum(s.st_size for s in current)
    allocated = sum(s.st_blocks*512 for s in current)
    admin_logical = sum(r['bytes'] for r in objects)
    guard(root,store,512*1024,logical+admin_logical)
    usage = resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    finish = encoded({'schema':'ralup.e06_compact_administration_finish.v1','exit_code':0,'sha':sha,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
        'launcher_user_seconds':usage.ru_utime,'launcher_system_seconds':usage.ru_stime,
        'child_user_seconds':child.ru_utime,'child_system_seconds':child.ru_stime,
        'nested_costs_not_added':True,'unknown_common_engineering_seconds':None,'objects':objects,
        'store_logical_bytes_before_finish':logical,
        'store_allocated_bytes_before_finish':allocated,
        'new_admin_logical_bytes_before_finish':admin_logical,
        'combined_evidence_logical_bytes_before_finish':logical+admin_logical,
        'new_admin_allocated_bytes_before_finish':sum(r['allocated_bytes'] for r in objects),
        'raw_data_free_before_finish':shutil.disk_usage(store).free,'git_free_after':shutil.disk_usage(root).free,
        'forecast_physical_bound_bytes':forecast,'original_final_admin_reserve_retained':False,'location_exception':str(base),'Data_owned_write_bytes':0,
        'actual_Data_reserve_shortfall_before_finish_bytes':max(0,64*GIB+E07-shutil.disk_usage(store).free)})
    if len(finish)>512*1024:raise ValueError('Finish exceeds bounded allowance')
    write(base/'finish.json',finish)
    guard(root,store,0,logical+admin_logical+len(finish))
    final_allocated=sum(p.stat().st_blocks*512 for p in base.rglob('*') if p.is_file())
    if final_allocated>forecast:raise ValueError('Terminal administration physical allocation exceeded forecast')
    print(json.dumps({'sha':sha,'status':status,'path':str(base),'physical_forecast':forecast,
        'actual_admin_allocated_bytes':final_allocated,'startup':reference(base/'startup.json'),
        'finish':reference(base/'finish.json'),'report':reference(base/'E06-report.md'),
        'raw_data_free_after':shutil.disk_usage(store).free,'Data_owned_write_bytes':0,'git_free_after':shutil.disk_usage(root).free}))


if __name__=='__main__':main()
