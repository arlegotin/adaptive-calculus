"""New immutable M06 attempts; archive bytes before child product imports."""
import hashlib,json,os,resource,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
start=time.perf_counter();root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
assert hashlib.sha256((store/'store.json').read_bytes()).hexdigest()=='16f61f5f5425f0ea95d4416bab16fa47936abaf592844e358013519cb4255af2'
name,module,*args=sys.argv[1:];base=store/'m06-20261004'/name
inventory=[(p,p.stat().st_size) for p in store.rglob('*') if p.is_file()]
forecast=128*1024**2;git_before=shutil.disk_usage(root).free;data_before=shutil.disk_usage(store).free
assert sum(n for p,n in inventory)+forecast<=32*1024**3 and data_before-forecast>=64*1024**3
assert git_before-16*1024**2>=512*1024**2
base.mkdir(parents=True,exist_ok=False)
def write(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
paths=sorted(set(list(root.glob('src/ralup/**/*.py'))+list(root.glob('tests/**/*.py'))+list(root.glob('docs/harness/m06*.py'))+[root/'requirements.lock',root/'artifacts/environment/environment.json',Path(__file__).resolve()]))
sources={}
for p in paths:
    relative=str(p.relative_to(root));content=p.read_bytes();dest=base/'producer'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    sources[relative]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
(base/'tmp').mkdir();args=[a.replace('@ATTEMPT@',str(base)) for a in args]
command=[str(root/'.venv/bin/python')]+([module] if module.endswith('.py') else ['-m',module])+args
env={**os.environ,'TMPDIR':str(base/'tmp'),'M06_ATTEMPT_DIR':str(base),'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
write(base/'startup.json',{'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'command':command,'fallback':env.get('PYTORCH_ENABLE_MPS_FALLBACK'),'sources':sources,'storage':{'inventory':[{'path':str(p.relative_to(store)),'bytes':n} for p,n in inventory],'allocated_bytes':sum(n for p,n in inventory),'complete_remaining_task_forecast_bytes':forecast,'forecast_scope':'all M06 attempt source/test/helper/lock/F02 copies, streams, reports, snapshots and temporary assets; conservative 128MiB aggregate','git_free_before':git_before,'data_free_before':data_before},'cost_scope':'complete launcher plus child process; earlier editing/tool/copy and enclosing project costs unknown'})
code=1
with (base/'stdout.txt').open('xb') as out,(base/'stderr.txt').open('xb') as err:
    try:code=subprocess.run(command,stdout=out,stderr=err,env=env).returncode
    except BaseException:
        import traceback
        err.write(traceback.format_exc().encode())
    finally:out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
write(base/'finish.json',{'exit_code':code,'elapsed_seconds':time.perf_counter()-start,'launcher_cpu_user_seconds':u.ru_utime,'launcher_cpu_system_seconds':u.ru_stime,'child_cpu_user_seconds':c.ru_utime,'child_cpu_system_seconds':c.ru_stime,'child_maxrss':c.ru_maxrss,'git_free_after':shutil.disk_usage(root).free,'data_free_after':shutil.disk_usage(store).free,'completed_objects':[{'path':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(base.rglob('*')) if p.is_file()]})
print(json.dumps({'attempt':name,'exit_code':code,'stdout':str(base/'stdout.txt'),'stderr':str(base/'stderr.txt')}));raise SystemExit(code)
