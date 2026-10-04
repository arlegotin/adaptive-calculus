"""One source-stable task plus affected-boundary covering, then public gates."""
import json,os,subprocess,sys,time
from pathlib import Path
base=Path(os.environ['M06_ATTEMPT_DIR']);root=Path.cwd()
checks=[('covering',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
    'tests/unit/test_learned_scale.py','tests/unit/test_noise_controls.py',
    'tests/unit/test_method_registry.py','tests/unit/test_transformer.py','tests/unit/test_quantization_context.py',
    'tests/unit/test_checkpoint_integrity.py','tests/integration/test_logical_step.py','tests/integration/test_resume.py',
    'tests/integration/test_stochastic_context.py']),
    ('cpu-cli',[sys.executable,'-m','ralup','verify-baselines','--suite','scale-noise','--device','cpu','--output',str(base/'cpu-results.json')]),
    ('mps-cli',[sys.executable,'-m','ralup','verify-baselines','--suite','scale-noise','--device','mps','--output',str(base/'mps-results.json')])]
rows=[]
for name,command in checks:
    started=time.perf_counter();p=subprocess.run(command,capture_output=True)
    (base/f'{name}.stdout').write_bytes(p.stdout);(base/f'{name}.stderr').write_bytes(p.stderr)
    sys.stdout.buffer.write(p.stdout);sys.stdout.flush();sys.stderr.buffer.write(p.stderr);sys.stderr.flush()
    rows.append({'name':name,'command':command,'exit_code':p.returncode,'complete_child_wall_seconds':time.perf_counter()-started})
    if p.returncode:break
(base/'checks.json').write_text(json.dumps({'checks':rows,'scope':'one scoped source-stable covering; complete native child streams, no universal or scientific gate'},indent=2)+'\n')
raise SystemExit(rows[-1]['exit_code'])
