"""Two-stage development selection; protected or local-horizon rows fail closed."""
import math
from collections import defaultdict
from ..policy.controllers import ControllerArtifact
from ..config import canonical_bytes


def select_controller(meta_dev: list[dict],official_dev: list[dict]) -> ControllerArtifact:
    if not meta_dev or not official_dev:raise ValueError('Both development selection stages required')
    stages=[]
    for rows,partition in ((meta_dev,'meta-development'),(official_dev,'development')):
        groups=defaultdict(list)
        for row in rows:
            if row.get('partition')!=partition:raise ValueError('Selection partition forbidden')
            required={'partition','controller','seed','terminal_loss','elapsed_seconds','horizon','full_horizon','input_sha256','opportunity_sha256'}
            if set(row)!=required:raise ValueError('Complete selection evidence required')
            if row['horizon']!=row['full_horizon'] or type(row['horizon']) is not int or row['horizon']<=0:raise ValueError('Full continuation selection required')
            if any(not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) or row[k]<0 for k in ('terminal_loss','elapsed_seconds')):raise ValueError('Finite measured selection losses/costs required')
            artifact=row['controller']
            if type(artifact) is not ControllerArtifact:raise ValueError('Authenticated frozen controller artifact required')
            key=canonical_bytes(artifact.to_json())
            groups[key].append(row)
        inventories=[]
        for rows in groups.values():
            inventory={(r['seed'],r['input_sha256'],r['opportunity_sha256'],r['horizon']) for r in rows}
            if len(inventory)!=len(rows):raise ValueError('Duplicate selection seed/opportunity')
            inventories.append(inventory)
        if any(i!=inventories[0] for i in inventories):raise ValueError('Unequal development information/opportunities')
        stages.append(groups)
    # Stage one nominates one artifact per family before official development.
    def rank(rows):return (sum(r['terminal_loss'] for r in rows)/len(rows),sum(r['elapsed_seconds'] for r in rows))
    nominees={}
    for key,rows in stages[0].items():
        family=rows[0]['controller'].family
        if family not in nominees or rank(rows)<rank(stages[0][nominees[family]]):nominees[family]=key
    if set(stages[1])!=set(nominees.values()):raise ValueError('Official stage differs from frozen meta-development nominees')
    winner=min(stages[1],key=lambda k:(rank(stages[1][k]),k))
    return stages[1][winner][0]['controller']
