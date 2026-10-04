"""Fresh trusted interpreter continuation; outer attempt retains startup bytes."""
import hashlib,json,sys
from pathlib import Path
from ralup.records import ArtifactRef
from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.train import train_step
from ralup.data.articles import Article
from ralup.data.blocks import tile_article
from ralup.data.batches import LogicalBatch
path,sha,device=sys.argv[1:]
state=restore_snapshot(load_snapshot(ArtifactRef(path,sha)),device)
raw=b'abcd';batch=LogicalBatch(tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest())),4)
train_step(state,batch)
print(json.dumps({'scientific_hash':exact_cpu_state_hash(state),'step':state.schedule['step']}))
