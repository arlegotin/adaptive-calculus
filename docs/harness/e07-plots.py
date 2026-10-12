"""Standalone figures from authenticated report tables in a separate analysis process."""
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path=Path(sys.argv[1]);assert hashlib.sha256(path.read_bytes()).hexdigest()==sys.argv[2]
report=json.loads(path.read_bytes());out=Path(os.environ['E07_OPERATION'])
rows=[r for r in report['tables']['per_seed'] if 'meta_query_nll' in r]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'ralup-e07'})
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for family in sorted(set(r['family'] for r in rows)):
    selected=sorted((r for r in rows if r['family']==family),key=lambda r:r['training_targets'])
    for ax,column,label in zip(axes,['training_targets','observed_rollout_seconds'],['Training targets in rollout','Measured nested rollout seconds']):
        ax.scatter([r[column] for r in selected],[r['meta_query_nll'] for r in selected],label=family,s=25)
        ax.set_xlabel(label);ax.set_ylabel('Held META query NLL (nats / target)');ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
fig.suptitle('Exploratory M3 META cost / quality — shared parent, no final-arm comparison')
fig.text(.02,.01,'Warm ancestry and tuning costs remain separate/unknown; points do not establish deployment or project efficiency.',fontsize=8)
fig.tight_layout(rect=(0,.035,1,.95))
for suffix in ('svg','pdf','png'):fig.savefig(out/f'cost-quality.{suffix}',metadata={'Creator':'RALUP E07'},dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.4));p=report['tables']['precision']
ax.plot([r['n'] for r in p],[r['half_width_in_units_of_paired_sd'] for r in p],marker='o')
ax.set_xscale('log',base=2);ax.set_xlabel('Hypothetical independent paired seeds n')
ax.set_ylabel('97.5% interval half-width / paired SD')
ax.set_title('Analytical precision illustration; empirical pilot unavailable')
ax.grid(alpha=.2)
fig.text(.02,.01,'Student-t formula only. Paired SD, final arms, power and locked n are absent. This curve is not a study design.',fontsize=8)
fig.tight_layout(rect=(0,.055,1,1))
for suffix in ('svg','pdf','png'):fig.savefig(out/f'precision.{suffix}',metadata={'Creator':'RALUP E07'},dpi=160)
plt.close(fig)
(out/'plot-provenance.json').write_text(json.dumps({'report':{'path':str(path),'sha256':sys.argv[2]},
    'matplotlib':matplotlib.__version__,'executable':sys.executable,'raw_rows':len(rows),
    'precision':'analytical standardized formula only; actual pilot unavailable'},indent=2)+'\n')
print(json.dumps({'figures':6,'raw_META_rows':len(rows),'precision_empirical':False}))
