"""Immutable 11 October M07 receipts; archive before child imports."""
import hashlib,importlib.metadata,json,os,platform,resource,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
started=time.perf_counter();root=Path.cwd()
store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
assert hashlib.sha256((store/'store.json').read_bytes()).hexdigest()=='16f61f5f5425f0ea95d4416bab16fa47936abaf592844e358013519cb4255af2'
assert hashlib.sha256((root/'requirements.lock').read_bytes()).hexdigest()=='488142a35a4b1a6abe132f08b5895621ca30bb535b0326e82516446903fcb32b'
name,module,*args=sys.argv[1:];base=store/'m07-20261011'/name
forecast=int(os.environ.get("M07_FORECAST_BYTES", str(512*1024**2)));small_forecast=16*1024**2
inventory=[(p,p.stat().st_size) for p in store.rglob('*') if p.is_file()]
git_before=shutil.disk_usage(root).free;data_before=shutil.disk_usage(store).free
assert sum(n for p,n in inventory)+forecast<=32*1024**3 and data_before-forecast>=64*1024**3
assert git_before-small_forecast>=512*1024**2
base.mkdir(parents=True,exist_ok=False)
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
sources={}
paths=sorted(set(list(root.glob('src/ralup/**/*.py'))+list(root.glob('tests/**/*.py'))+list(root.glob('docs/harness/m07*.py'))+list(root.glob('docs/harness/m07*.json'))+list(root.glob('configs/m07*.json'))+[root/'requirements.lock',root/'artifacts/environment/environment.json']))
for path in paths:
    content=path.read_bytes();relative=str(path.relative_to(root));target=base/'producer'/relative
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    sources[relative]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
observed=[]
for command in (['sw_vers'],['uname','-m']):
    result=subprocess.run(command,capture_output=True,text=True)
    observed.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
runtime={'observed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,
    'machine':platform.machine(),'system':platform.system(),'commands':observed,'torch_distribution_version':importlib.metadata.version('torch'),
    'F02_role':'archived historical environment bytes, not a current native admission','scientific_native_admission':'see explicit child purpose and separate current-runtime gate'}
write(base/'runtime-observation.json',runtime)
(base/'tmp').mkdir();args=[a.replace('@ATTEMPT@',str(base)) for a in args]
command=[str(root/'.venv/bin/python')]+([module] if module.endswith('.py') else ['-m',module])+args
env={**os.environ,'TMPDIR':str(base/'tmp'),'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','M07_ATTEMPT_DIR':str(base)}
write(base/'startup.json',{'utc':datetime.now(timezone.utc).isoformat(),'command':command,'fallback':env.get('PYTORCH_ENABLE_MPS_FALLBACK'),
    'source':sources,'runtime_observation':runtime,'storage':{'inventory':[{'path':str(p.relative_to(store)),'bytes':n} for p,n in inventory],
    'allocated_bytes':sum(n for p,n in inventory),'complete_remaining_fix_forecast_bytes':forecast,'forecast_scope':'current M07 attempt archived source/test/helper/lock/F02/native receipts, CPU fixture temps and pointer/commit attempts',
    'git_free_before':git_before,'data_free_before':data_before},'cost_scope':'complete launcher plus child process; enclosing/editing/tool costs unknown'})
code=1
with (base/'stdout.txt').open('xb') as stdout,(base/'stderr.txt').open('xb') as stderr:
    try:code=subprocess.run(command,env=env,stdout=stdout,stderr=stderr).returncode
    except BaseException:
        import traceback
        stderr.write(traceback.format_exc().encode())
    finally:stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
after_git=shutil.disk_usage(root).free;after_data=shutil.disk_usage(store).free
assert after_git>=512*1024**2 and after_data>=64*1024**3
write(base/'finish.json',{'exit_code':code,'elapsed_seconds':time.perf_counter()-started,'launcher_cpu_user_seconds':usage.ru_utime,'launcher_cpu_system_seconds':usage.ru_stime,
    'child_cpu_user_seconds':child.ru_utime,'child_cpu_system_seconds':child.ru_stime,'child_maxrss':child.ru_maxrss,'git_free_after':after_git,'data_free_after':after_data,
    'completed_objects':[{'path':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(base.rglob('*')) if p.is_file()]})
print(json.dumps({'attempt':name,'exit_code':code,'path':str(base)}));raise SystemExit(code)
