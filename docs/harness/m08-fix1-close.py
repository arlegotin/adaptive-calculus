"""Finite I1 receipt authentication and small authorized local fix commit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path.cwd()
BASE=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m08-20261011')
FIXBASE='ca56636634f01e94d2046f9fe6939179b879c5cf'
INDEX=ROOT/'docs/harness/m08-fix1-evidence.json'


def ref(path):
    path=Path(path)
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


if sys.argv[1]=='index':
    original=json.loads((ROOT/'docs/harness/m08-feature-evidence.json').read_bytes())
    current={name:ref(ROOT/name)['sha256'] for name in original['source_identity']}
    changed=[name for name in current if current[name]!=original['source_identity'][name]]
    assert changed==['src/ralup/state.py']
    attempts=[]
    for name,code in [('fix1-red-common-ema',1),('fix1-green-common-ema',0),('fix1-green-affected-h07-history',0)]:
        path=BASE/name;finish=json.loads((path/'finish.json').read_bytes())
        assert finish['exit_code']==code
        for item in finish['completed_objects']:
            assert ref(path/item['path'])['sha256']==item['sha256']
            assert (path/item['path']).stat().st_size==item['bytes']
        attempts.append({'name':name,'exit_code':code,'elapsed_seconds':finish['elapsed_seconds'],
                         'receipts':{f:ref(path/f) for f in ('startup.json','stdout.txt','stderr.txt','finish.json')}})
    value={'schema':'ralup.m08_fix1_evidence.v1','issue':'I1 required common-gradient EMA ownership',
           'fix_base':FIXBASE,'review':ref(ROOT/'.superpowers/sdd/2026-10-03-ralup/M08-review.md'),
           'prior_index':ref(ROOT/'docs/harness/m08-feature-evidence.json'),'prior_native_primary':original['primary'],
           'source_identity':current,'changed_product_sources':changed,'attempts':attempts,
           'sum_test_attempt_wall_seconds':sum(a['elapsed_seconds'] for a in attempts),
           'scope':'CPU fixture-only common EMA capture/strict restore guard, initialization/raw ownership, and affected H07/history tests; no genuine-data/native/broad-suite execution.',
           'native_applicability':'Historical ProjectMuon measurement remains scoped to original producer. Its collector/momentum/numerical paths are unchanged; I1 only strengthens finite trusted rule/mode ownership validation. No current universal numerical/source claim.',
           'test_results':{'red':'3 failed / 1 passed / 12 deselected (two actual missing guards; existing raw rejection message mismatch)',
                           'green':'4 passed / 12 deselected','affected':'103 passed'},
           'unknown_costs':['editing','tools','enclosing orchestration','total synthetic fixture update/target count'],
           'administrative_attempts':'Separate index/commit receipts; excluded to avoid live/circular hashes.'}
    with INDEX.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({'index':ref(INDEX),'test_wall_seconds':value['sum_test_attempt_wall_seconds']}))
elif sys.argv[1]=='commit':
    start=time.perf_counter();value=json.loads(INDEX.read_bytes())
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==FIXBASE
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='feat/ralup-execution'
    assert all(ref(ROOT/name)['sha256']==digest for name,digest in value['source_identity'].items())
    paths=sorted(p for p in subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','-z']).decode().split('\0') if p)
    assert set(paths)=={'src/ralup/state.py','tests/integration/test_history_lifecycle.py',
                       'docs/harness/m08-fix1-close.py','docs/harness/m08-fix1-evidence.json'}
    size=sum((ROOT/p).stat().st_size for p in paths);before=shutil.disk_usage(ROOT).free
    assert size<=16*1024**2 and before-16*1024**2>=512*1024**2
    commands=[['git','diff','--check'],['git','add','--',*paths],['git','diff','--cached','--check'],
              ['git','commit','-m','fix: require completed common gradient EMA history'],['git','rev-parse','HEAD'],['git','status','--short']]
    results=[]
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True)
        results.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        print(result.stdout,end='');print(result.stderr,end='')
        if result.returncode:break
    after=shutil.disk_usage(ROOT).free;assert after>=512*1024**2
    with (Path(os.environ['M08_ATTEMPT_DIR'])/'commit.json').open('x') as stream:
        json.dump({'commands':results,'selected_paths':paths,'selected_bytes':size,'git_free_before':before,
                   'git_free_after':after,'complete_seconds':time.perf_counter()-start,'index':ref(INDEX)},stream,sort_keys=True,indent=2)
    raise SystemExit(results[-1]['exit_code'])
else:raise ValueError('Finite index or commit operation required')
