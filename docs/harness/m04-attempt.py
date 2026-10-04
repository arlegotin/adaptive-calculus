"""Persist every M04 launch and producing bytes before importing experiment code."""
import contextlib,hashlib,json,os,resource,runpy,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
start=time.perf_counter();root=Path.cwd()
store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
assert hashlib.sha256((store/'store.json').read_bytes()).hexdigest()=='16f61f5f5425f0ea95d4416bab16fa47936abaf592844e358013519cb4255af2'
name,module,*args=sys.argv[1:];base=store/'m04-20261004'/name
inventory=[(p,p.stat().st_size) for p in store.rglob('*') if p.is_file()]
forecast=256*1024**2
assert sum(n for p,n in inventory)+forecast<=32*1024**3
assert shutil.disk_usage(store).free-forecast>=64*1024**3
assert shutil.disk_usage(root).free-16*1024**2>=512*1024**2
base.mkdir(parents=True,exist_ok=False)
def write(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
paths=sorted(set(list(root.glob('src/ralup/**/*.py'))+list(root.glob('tests/**/*.py'))+list(root.glob('docs/harness/m04*.py'))+[root/'requirements.lock',root/'artifacts/environment/environment.json',Path(__file__).resolve()]))
sources={}
for p in paths:
    relative=str(p.relative_to(root));dest=base/'producer'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    content=p.read_bytes();dest.write_bytes(content);sources[relative]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
write(base/'startup.json',{'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'module':module,'arguments':args,'fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),'source_inventory':sources,'storage_inventory':{'files':[{'path':str(p.relative_to(store)),'bytes':n} for p,n in inventory],'allocated_bytes':sum(n for p,n in inventory),'remaining_task_forecast_bytes':forecast,'scope':'all new task launch copies, stdout, checks, snapshots and pytest temporary assets; 256MiB conservative aggregate'},'git_free_before':shutil.disk_usage(root).free,'data_free_before':shutil.disk_usage(store).free,'cost_scope':'this complete helper/module process only; earlier session and project costs unknown'})
# Bulk pytest output and snapshots use this new permanent task attempt directory.
(base/'tmp').mkdir();os.environ['TMPDIR']=str(base/'tmp')
code=0
with (base/'stdout.txt').open('x',buffering=1) as out,contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
    try:
        sys.argv=[module,*args]
        if module.endswith('.py'):runpy.run_path(module,run_name='__main__')
        else:runpy.run_module(module,run_name='__main__',alter_sys=True)
    except SystemExit as e:code=int(e.code) if isinstance(e.code,int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc();code=1
    finally:out.flush();os.fsync(out.fileno())
u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
write(base/'finish.json',{'exit_code':code,'elapsed_seconds':time.perf_counter()-start,'cpu_user_seconds':u.ru_utime,'cpu_system_seconds':u.ru_stime,'child_cpu_user_seconds':c.ru_utime,'child_cpu_system_seconds':c.ru_stime,'maxrss':u.ru_maxrss,'stdout_sha256':hashlib.sha256((base/'stdout.txt').read_bytes()).hexdigest(),'git_free_after':shutil.disk_usage(root).free,'data_free_after':shutil.disk_usage(store).free,'attempt_inventory':[{'path':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(base.rglob('*')) if p.is_file()]})
print(json.dumps({'attempt':name,'exit_code':code,'stdout':str(base/'stdout.txt')}));raise SystemExit(code)
