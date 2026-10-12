"""Immutable scoped supplement; never changes original operation receipts."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('e07-capture.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path.cwd();out=Path(os.environ['E07_OPERATION']);native=Path(os.environ['E07_COMPLETION_ROOT'])/'native-replay-v2'
start=json.loads((native/'startup.json').read_bytes());finish=json.loads((native/'finish.json').read_bytes());proof=json.loads((native/'paired-replay.json').read_bytes())
component=sum(m.inventory(native/n)['allocated_bytes'] for n in ('row_fixed_max','row_dynamic_max'))
assert component==36470784 and start['child_allocation_bound_bytes']==33554432
assert finish['exit_code']==0 and finish['bounds_pass'] and all(p['exact_scientific_state'] for p in proof['proof'])
source_checks={}
for operation in (Path(os.environ['E07_BASE'])/'final-cpu-v2',native):
    original=json.loads((operation/'startup.json').read_bytes())
    expected={k:v for k,v in original['source'].items() if k.startswith('src/')}
    current={str(p.relative_to(root)):m.ref(p) for p in (root/'src').rglob('*.py')}
    assert set(current)==set(expected)
    assert all(current[k]['sha256']==v['sha256'] for k,v in expected.items())
    source_checks[str(operation)]={'all_src_python_bytes_identical':True,'files':len(expected),'startup':m.ref(operation/'startup.json')}
context=(root/'.superpowers/sdd/2026-10-03-ralup/E07-current-context.md').read_text()
ruling=[s for s in context.splitlines() if s.startswith("Ruling: Preserve E07 native-replay-v2")][-1]
value={'schema':'ralup.e07_storage_disposition.v1','original_startup':m.ref(native/'startup.json'),
    'original_finish':m.ref(native/'finish.json'),'numerical_proof':m.ref(native/'paired-replay.json'),
    'original_child_exit_code':0,'original_capture_exit_code':0,'numerical_pass':True,
    'complete_stage_bounds_pass':True,'child_component_bound_pass':False,
    'checkpoint_trees_allocated_bytes':component,'child_allocation_bound_bytes':33554432,
    'checkpoint_component_exceeded_by_bytes':component-33554432,
    'complete_child_inventory_now':m.child_inventory(native),
    'complete_stage_allocated_bytes':finish['owned']['allocated_bytes'],
    'complete_stage_admitted_bytes':start['forecast_physical_bytes'],
    'source_applicability':source_checks,'root_ruling_verbatim':ruling,
    'interpretation':'original bounds_pass covered complete stage only; named child bound failed; numerical proof retained; no native repeat or retrospective allowance enlargement'}
path=out/'native-storage-supplement.json';path.write_bytes(m.encoded(value))
env=dict(os.environ,E07_NATIVE_SUPPLEMENT=str(path))
subprocess.run([sys.executable,'-B','docs/harness/e07-report-data.py'],env=env,check=True)
print(json.dumps({'supplement':m.ref(path),'source_applicability':source_checks}))
