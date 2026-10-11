"""Preserve diagnostic disposition, pre-existing text logs and unexecuted drafts."""
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
from ralup.policy.checks import _ref,_write

base=Path(os.environ['M11_ATTEMPT_DIR']);store=base.parent
read=lambda p:json.loads(Path(p).read_bytes())
def auth(path):
    value=read(path);finish=read(path.parent/'finish.json')
    entry=next(v for v in finish['completed_objects'] if v['path']==path.name)
    assert _ref(path)['sha256']==entry['sha256'];return value
for name in ('native-final-calibration-v1','interpret-final-calibration-v1','receipt-accounting-v1'):
    auth(store/name/'report.json')
old=read(store/'native-final-calibration-v1'/'startup.json')['source']
changed=[p for p,v in old.items() if p.startswith('src/') and _ref(Path(p))['sha256']!=v['sha256']]
assert changed==['src/ralup/meta/tasks.py'],changed
assert subprocess.check_output(['git','diff','--name-only','5ed817e..16fe874','--','src'],text=True).splitlines()==changed
files=[]
for source in sorted(Path('/private/tmp').glob('m11-*.txt')):
    assert source.stat().st_size<4*1024**2
    target=base/'historical-cpu-stdout'/source.name;target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source,target);files.append({'source_path':str(source),'archived':_ref(target),'scope':'Historical stdout only; original producer/enclosing CPU/tool costs may be unavailable or overlap native work, never inferred zero.'})
drafts=[]
for name in ('m11-learning.py','m11-campaign.py','m11-current-pool.py','m11-selection.py'):
    source=Path('docs/harness')/name;target=base/'unexecuted-drafts'/name;target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source,target);drafts.append({'source_path':str(source),'archived':_ref(target),'scope':'UNEXECUTED/UNFROZEN future draft; not qualified implementation, chosen budget, selected artifact, or evidence of completed learning/selection.'})
source=Path('.superpowers/sdd/2026-10-03-ralup/M11-report.md');target=base/'M11-report.md';shutil.copyfile(source,target)
disposition={'schema':'ralup.m11.disposition.v1','recorded_utc':datetime.now(timezone.utc).isoformat(),'status':'NEEDS_REVISION','implementation_review':'pending full M11 base-to-final-head independent review','base':'c2552c0722110d0c7ed5e5e0e28df11d3d680eda','implementation_commit':'5ed817e8a34640bc0227d9d603947a3ab8933007','logging_fix_commit':'16fe874b3a7e3249ae833fa977a9db974a7191d2','B0':None,'WP04':'inconclusive','confirmation_eligible':False,'selected_controller':None,'final_K':None,'final_m':None,'final_sigma':None,'final_outer_learning_rate':None,'final_iterations':None,'final_refresh_schedule':None,
 'reason':'Both provisional one-step policies rank better atK32/64 but worse atfull496 on both disjoint query panels; residual independent direction sampling variability. Exact brief short-horizon/noisy diagnostic branch. No trained-family defeat, general rejection, failed core transfer, or resource impossibility claim.',
 'evidence':{name:_ref(store/name/'report.json') for name in ('native-final-calibration-v1','interpret-final-calibration-v1','receipt-accounting-v1')},
 'normalizer_data':_ref(store/'native-reference-v1'/'final-reference-normalizer.json'),
 'normalizer_scope':'Original META-only10752-row reference/source authority retained across action-JOURNAL-only fix; no old complete-state import. Future current-source baseline and each family own-policy pools required, generation costs charged.',
 'required_advancement':[{'owner':'E01/reopened WP06','action':'Prospectively justify longer-horizon/objective calibration and resolve evidence-based budgets; restart matched candidate45/LR49/static3 learning with each own-current-policy refresh; complete META-development then full official-development selection and B0/recipe applicability or retraining.'},{'owner':'E01','action':'Robust backbone recipe and independent family backboneLR tuning; exploratory commonMuon.02 is not a competitiveness claim.'},{'owner':'E05','action':'Shared-range learnedLR sensitivity and retuned mechanism/removal contrasts.'},{'owner':'E07','action':'Merge retained source-offset unions and globally deduplicate ancestors/costs; preserve unknown engineering work and all failure scopes.'}],
 'next_action':'Independent full M11 review, then mandatoryM10 beforeE01-E07; no invented final learning budget.',
 'report':_ref(target),'original_report':str(source)}
_write(base/'disposition.json',disposition)
_write(base/'report.json',{'status':'needs_revision_packaged_pending_independent_review','disposition':_ref(base/'disposition.json'),'authoritative_report':_ref(target),'historical_cpu_outputs':files,'unexecuted_drafts':drafts,'source_delta_from_assessment':changed,'source_delta_scope':'Only completed action copying into evidence; no feature/history/normalization/model/quantizer/optimizer numerical change. New complete-source pools not generated because no final campaign accepted or launched.','new_optimizer_updates':0,'new_query_targets':0,'cost_scope':'This receipt measures packaging only; historical stdout is not retroactive complete cost/provenance.'})
print(json.dumps({'report':_ref(base/'report.json'),'disposition':_ref(base/'disposition.json'),'drafts':len(drafts),'stdout_files':len(files)}),flush=True)
