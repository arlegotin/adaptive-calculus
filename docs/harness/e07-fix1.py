"""E07 I1/I2 finite report contracts only; no model execution or dependencies."""
import hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path.cwd();OUT=Path(os.environ['E07_OPERATION']);BASE=Path(os.environ['E07_BASE']);GIT=Path(os.environ['E07_COMPLETION_ROOT'])
mode=sys.argv[1]
if mode in ('red','green'):
    subprocess.run([sys.executable,'-B','docs/harness/e07-bound-check.py'],check=True)
    result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(OUT/'fixtures'),'tests/integration/test_report_reproduction.py'])
    raise SystemExit(result.returncode)
if mode=='report':
    def ref(path):
        raw=Path(path).read_bytes();return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    current={str(p.relative_to(ROOT)):ref(p) for p in (ROOT/'src').rglob('*.py')}
    historical=[]
    for operation in (BASE/'final-cpu-v2',GIT/'native-replay-v2'):
        original=json.loads((operation/'startup.json').read_bytes())
        old={k:v for k,v in original['source'].items() if k.startswith('src/')}
        changed=[k for k,v in current.items() if k not in old or old[k]['sha256']!=v['sha256']]
        assert changed==['src/ralup/report.py'],changed
        historical.append({'startup':ref(operation/'startup.json'),'changed_src_paths':changed,
            'same_current_full_source':False,'applicability':'original numerical evidence only; no new model execution or cross-source full-state restoration'})
    audit=OUT/'source-applicability.json'
    audit.write_text(json.dumps({'schema':'ralup.e07_fix1_source_applicability.v1','current_sources':current,
        'historical_comparison':historical,'verification':'focused administrative report-contract tests only; root owns final integration admission'},sort_keys=True,indent=2)+'\n')
    env=dict(os.environ,E07_NATIVE_SUPPLEMENT=str(GIT/'report-v3/native-storage-supplement.json'),E07_REPORT_SOURCE_APPLICABILITY=str(audit))
    subprocess.run([sys.executable,'-B','docs/harness/e07-report-data.py'],env=env,check=True)
    evidence=json.loads((OUT/'evidence.json').read_bytes());registered={(r['path'],r['sha256']) for r in evidence['sources']}
    assert all((r['path'],r['sha256']) in registered for r in evidence['sections']['sources_and_deviations']['primary_ledgers'])
    print('PASS: all three operational primary ledgers registered and authenticated; historical numerical applicability explicit')
elif mode=='plots':
    path=GIT/'fix1-report-v1/report/report.json'
    subprocess.run([str(BASE/'analysis-environment-v1/analysis/bin/python'),'-B','docs/harness/e07-plots.py',str(path),hashlib.sha256(path.read_bytes()).hexdigest()],check=True)
elif mode=='admin':
    subprocess.run([sys.executable,'-B','docs/harness/e07-fix1-admin.py'],check=True)
else:raise ValueError('Mode implementation is installed only before its admitted capture')
