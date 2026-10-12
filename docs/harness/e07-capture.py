"""E07 bounded, immutable operation receipts; no scientific authorization."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STORE = Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
BASE = STORE / 'e07-20261012'
GIT_COMPLETION = Path('/Volumes/git/legotin/adaptive-calculus/artifacts/ralup-administration/20261012/e07-completion-v1')
GIB = 1024**3
MIB = 1024**2


def encoded(v):
    return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def inventory(root):
    logical = allocated = files = 0
    for directory, _, names in os.walk(root):
        for name in names:
            p = Path(directory)/name
            if p.is_symlink():
                continue
            s = p.stat(); logical += s.st_size; allocated += s.st_blocks*512; files += 1
    return dict(logical_bytes=logical, allocated_bytes=allocated, files=files)


def ref(p):
    raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def allocation_bounds(child_bytes, administration_bytes, child_bound, administration_bound):
    child_ok = child_bytes <= child_bound
    admin_ok = administration_bytes <= administration_bound
    return dict(child_component_bound_pass=child_ok, administration_bound_pass=admin_ok,
        all_declared_allocation_bounds_pass=child_ok and admin_ok)


def route_disposition(resource_bounds, data_floor, zero_write_exception, child_exit):
    admitted = resource_bounds and (data_floor or zero_write_exception)
    return dict(bounds_pass=resource_bounds and data_floor, data_floor_pass=data_floor,
        zero_data_write_route_admitted=bool(zero_write_exception and resource_bounds),
        route_resource_bounds_pass=resource_bounds,
        capture_exit_code=child_exit if admitted else 3)


def child_inventory(root):
    # Capture-owned source/receipts/redirected logs use the separate 16 MiB bound.
    excluded = {'producer', 'startup.json', 'root-location-ruling.txt', 'stdout.txt', 'stderr.txt', 'finish.json'}
    logical = allocated = files = 0
    for entry in root.iterdir():
        if entry.name in excluded or entry.is_symlink(): continue
        value = inventory(entry) if entry.is_dir() else dict(logical_bytes=entry.stat().st_size,
            allocated_bytes=entry.stat().st_blocks*512, files=1)
        logical += value['logical_bytes']; allocated += value['allocated_bytes']; files += value['files']
    return dict(logical_bytes=logical, allocated_bytes=allocated, files=files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--allowance-mib', type=int, required=True)
    parser.add_argument('--reserve-mib', type=int, required=True)
    parser.add_argument('--git-completion', action='store_true', help='requires specific root E07 completion-location authorization')
    parser.add_argument('--zero-data-admin', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    a = parser.parse_args(); command = a.command
    if command[:1] == ['--']: command = command[1:]
    if not command or '/' in a.name: raise ValueError('Explicit command and simple unique name required')
    if a.zero_data_admin and (not a.git_completion or a.name!='final-admin-v1' or command != [str(BASE/'environment-v1/core/bin/python'),'-B','docs/harness/e07-final-admin.py'] or a.allowance_mib!=4 or a.reserve_mib!=1):
        raise ValueError('Zero-Data exception is restricted to exact admitted finite final admin')
    base=GIT_COMPLETION if a.git_completion else BASE
    dest = base/a.name
    if dest.exists(): raise ValueError('Immutable operation directory already exists')
    before = inventory(STORE)
    git_before=inventory(GIT_COMPLETION) if GIT_COMPLETION.exists() else dict(logical_bytes=0,allocated_bytes=0,files=0)
    prior_git=inventory(GIT_COMPLETION.parent/'e06-v1')
    free = shutil.disk_usage(STORE).free
    git_free = shutil.disk_usage(ROOT).free
    # 16 MiB covers capture source archive, receipts, logs and physical metadata;
    # child allowances include all child staging/cache/environment allocations.
    forecast = (a.allowance_mib+16)*MIB
    reserve = a.reserve_mib*MIB
    if max(before['logical_bytes']+git_before['logical_bytes']+prior_git['logical_bytes'], before['allocated_bytes']+git_before['allocated_bytes']+prior_git['allocated_bytes'])+forecast+reserve > 32*GIB:
        raise ValueError('Complete store forecast exceeds 32 GiB')
    if a.git_completion:
        subprocess.run(['git','check-ignore',str(GIT_COMPLETION.relative_to(Path('/Volumes/git/legotin/adaptive-calculus')))],
            cwd='/Volumes/git/legotin/adaptive-calculus',check=True,capture_output=True)
        if (git_before['allocated_bytes']+forecast+reserve>160*MIB or git_free-forecast-reserve-16*MIB< GIB
                or (free<64*GIB and not a.zero_data_admin)):
            raise ValueError('E07 completion aggregate160MiB/Git1GiB bounds cannot admit stage')
        if base.exists() and not (base/'ownership.json').exists():
            raise ValueError('Existing completion root lacks E07 ownership marker')
    elif free-forecast-reserve < 64*GIB or git_free-16*MIB < 512*MIB:
        raise ValueError('Complete physical forecast cannot retain Data/Git floors')
    source = sorted(set(list((ROOT/'src').rglob('*.py')) + list((ROOT/'tests').rglob('*.py')) +
        list((ROOT/'docs/harness').glob('e07*')) + [ROOT/'requirements.lock', ROOT/'pyproject.toml']))
    raws = {str(p.relative_to(ROOT)):p.read_bytes() for p in source if p.is_file()}
    source_alloc = sum(((len(v)+4095)//4096)*4096+8192 for v in raws.values())
    if source_alloc > 8*MIB: raise ValueError('Capture source exceeds allocated source bound')
    start = time.perf_counter(); utc = datetime.now(timezone.utc).isoformat()
    startup = dict(schema='ralup.e07_operation.v1', utc=utc, command=command, cwd=str(ROOT),
        raw_data_free_before=free, git_free_before=git_free, store_before=before,
        completion_root=str(base),git_completion_before=git_before,
        prior_E06_git_evidence=prior_git,
        owned_Data_writes_permitted=not a.git_completion,
        zero_data_admin_exception=a.zero_data_admin, data_floor_pass_before=free>=64*GIB,
        forecast_physical_bytes=forecast, remaining_reserve_bytes=reserve,
        source={k:dict(sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for k,v in raws.items()},
        source_archive='producer', child_allocation_bound_bytes=a.allowance_mib*MIB,
        administration_allocation_bound_bytes=16*MIB, finish_receipt_reserved_bytes=16384,
        allocation_scope='file st_blocks; child excludes producer/startup/ruling/redirected logs/finish; administration includes reserved finish bytes',
        runtime=dict(python=sys.version, executable=sys.executable,
            sw_vers=subprocess.check_output(['sw_vers'], text=True)),
        scope='operation boundary only; enclosing editing/review/project costs unknown',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if a.git_completion and not base.exists():base.mkdir(exist_ok=False)
    dest.mkdir(parents=True)
    if a.git_completion and not (base/'ownership.json').exists():
        (base/'ownership.json').write_bytes(encoded({'task':'E07','owner':'sole E07 implementer',
            'scope':'root-authorized new completion root; earlier Data refs remain immutable',
            'maximum_aggregate_allocated_bytes':160*MIB,'must_survive_worktree_cleanup':True}))
    for k,v in raws.items():
        p=dest/'producer'/k; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(v)
    (dest/'startup.json').write_bytes(encoded(startup))
    if a.git_completion:
        context=(ROOT/'.superpowers/sdd/2026-10-03-ralup/E07-current-context.md').read_text()
        ruling=[line for line in context.splitlines() if line.startswith('Ruling: Complete E07')][-1]
        if a.zero_data_admin:
            ruling += '\n' + [line for line in context.splitlines() if line.startswith("Ruling: Authorize only E07's finite final administrative publication")][-1]
        (dest/'root-location-ruling.txt').write_text(ruling+'\n')
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT/'src'),
        PYTORCH_ENABLE_MPS_FALLBACK='0', E07_OPERATION=str(dest), E07_BASE=str(BASE),
        E07_COMPLETION_ROOT=str(base),
        TMPDIR=str(dest/'tmp'), PIP_NO_CACHE_DIR='1', MPLCONFIGDIR=str(dest/'mplconfig'),
        XDG_CACHE_HOME=str(dest/'cache'),TORCH_HOME=str(dest/'cache/torch'),TORCHINDUCTOR_CACHE_DIR=str(dest/'cache/torchinductor'))
    (dest/'tmp').mkdir()
    with (dest/'stdout.txt').open('xb') as out, (dest/'stderr.txt').open('xb') as err:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    after=inventory(STORE); owned=inventory(dest)
    after_free=shutil.disk_usage(STORE).free
    git_after=inventory(GIT_COMPLETION) if GIT_COMPLETION.exists() else git_before
    bounds=owned['allocated_bytes'] <= forecast and max(after['logical_bytes']+git_after['logical_bytes']+prior_git['logical_bytes'],after['allocated_bytes']+git_after['allocated_bytes']+prior_git['allocated_bytes'])+reserve <32*GIB
    if a.git_completion:
        bounds=bool(bounds and before==after and git_after['allocated_bytes']+reserve<=160*MIB and shutil.disk_usage(ROOT).free-reserve-16*MIB>=GIB)
    else:bounds=bool(bounds and after_free-reserve>=64*GIB)
    data_floor=bool(after_free>=64*GIB if a.git_completion else after_free-reserve>=64*GIB)
    complete_bounds=bounds and data_floor
    child=child_inventory(dest)
    allocation=allocation_bounds(child['allocated_bytes'],owned['allocated_bytes']-child['allocated_bytes']+16384,a.allowance_mib*MIB,16*MIB)
    bounds=bool(bounds and allocation['all_declared_allocation_bounds_pass'])
    disposition=route_disposition(bounds,data_floor,a.zero_data_admin,result.returncode)
    finish=dict(schema='ralup.e07_operation_finish.v3',
        git_floor_pass=shutil.disk_usage(ROOT).free-reserve-16*MIB>=GIB,
        git_aggregate_pass=git_after['allocated_bytes']+reserve<=160*MIB,
        combined_evidence_cap_pass=max(after['logical_bytes']+git_after['logical_bytes']+prior_git['logical_bytes'],after['allocated_bytes']+git_after['allocated_bytes']+prior_git['allocated_bytes'])+reserve<32*GIB, **disposition, complete_stage_bounds_pass=complete_bounds,
        child_owned=child, **allocation, utc=datetime.now(timezone.utc).isoformat(),
        exit_code=result.returncode, startup=ref(dest/'startup.json'),
        stdout=ref(dest/'stdout.txt'), stderr=ref(dest/'stderr.txt'), owned=owned, store_after=after,
        raw_data_free_after=after_free, git_free_after=shutil.disk_usage(ROOT).free,
        git_completion_after=git_after,owned_Data_inventory_unchanged=(before==after) if a.git_completion else None,
        elapsed_seconds=time.perf_counter()-start, child_user_seconds=usage.ru_utime,
        child_system_seconds=usage.ru_stime, common_engineering_seconds=None)
    (dest/'finish.json').write_bytes(encoded(finish))
    print(json.dumps(finish))
    return disposition['capture_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
