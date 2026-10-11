"""Exploratory paired-article resampling of already recorded development losses."""
import json,math,random,time
from pathlib import Path
from datetime import datetime,timezone
from ralup.records import sha256_file
from ralup.config import canonical_bytes

start=time.perf_counter();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
pairs=[]
for stage,steps in [('query-admission128',(64,128)),('query-extension256',(192,256))]:
    for step in steps:
        paths=[base/stage/f'core-muon-{kind}128-{step}.json' for kind in ('fp32','dynamic')]
        pairs.append((f'matched-Muon-{step}',paths))
for stage in ('query-conversion','query-conversion-expanded'):
    for step in (128,256):
        paths=[base/stage/f'conversion-{step}-{kind}.json' for kind in ('fp32','row_dynamic_max')]
        if stage=='query-conversion' and step==128:paths[0]=base/'query-admission128/core-fp32-warm128-128.json'
        pairs.append((f'{stage}-{step}',paths))
results=[]
for label,paths in pairs:
    scores=[json.loads(p.read_bytes())['score'] for p in paths]
    articles=[s['per_article'] for s in scores];keys=sorted(articles[0]);assert keys==sorted(articles[1])
    counts=[articles[0][k]['target_count'] for k in keys]
    assert counts==[articles[1][k]['target_count'] for k in keys]
    diffs=[articles[1][k]['nll_sum']-articles[0][k]['nll_sum'] for k in keys]
    rng=random.Random(1107);samples=[]
    for _ in range(10000):
        chosen=[rng.randrange(len(keys)) for _ in keys]
        samples.append(math.fsum(diffs[i] for i in chosen)/(math.log(2)*sum(counts[i] for i in chosen)))
    samples.sort()
    results.append({'label':label,'sources':[ref(p) for p in paths],'articles':len(keys),'target_bytes':sum(counts),
        'q_minus_fp_bpb':math.fsum(diffs)/(math.log(2)*sum(counts)),
        'paired_article_resampling_2p5_97p5':[samples[249],samples[9749]],
        'article_byte_range':[min(counts),max(counts)]})
output=Path('docs/harness/m07-uncertainty-v1.json')
payload={'created_utc':datetime.now(timezone.utc).isoformat(),'producer':ref(Path(__file__)),
    'method':'10000 seeded1107 paired article cluster resamples with replacement, weighted pooled NLL/bytes; percentile order statistics249/9749',
    'scope':'exploratory conditional sensitivity on already selected development checkpoints and sampled byte portions; not a confirmatory interval or fresh-training-seed uncertainty',
    'limits':['one training seed707; independent seed variability unidentified','article resampling does not correct adaptive development selection','sampled article portions, not full article census','no convergence/B0/policy-gain inference'],
    'new_model_passes':0,'new_training_steps':0,'results':results,'analysis_seconds':time.perf_counter()-start}
with output.open('xb') as f:f.write(canonical_bytes(payload)+b'\n')
print(json.dumps({'output':ref(output),'results':[{k:v for k,v in r.items() if k!='sources'} for r in results]}))
