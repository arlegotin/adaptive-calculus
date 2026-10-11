"""Frozen meta-train reference normalization; file-bound finite provenance."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from torch import Tensor
from ..records import ArtifactRef,sha256_file
from ..config import canonical_bytes


class ProvenanceError(ValueError):pass


def _read(ref):
    if not isinstance(ref,ArtifactRef) or sha256_file(ref.path)!=ref.sha256:raise ProvenanceError('Unauthenticated feature provenance')
    return json.loads(ref.path.read_bytes())


@dataclass(frozen=True)
class FrozenNormalizer:
    mean: Tensor
    std: Tensor
    source_manifest: str

    def __post_init__(self):
        if not isinstance(self.source_manifest,str) or not self.source_manifest:raise ProvenanceError('Normalizer requires frozen provenance')
        for value in (self.mean,self.std):
            if value.shape!=(14,) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):raise ValueError('Normalizer requires fourteen finite scalars')
        if bool((self.std<0).any()):raise ValueError('Normalizer standard deviation cannot be negative')
        object.__setattr__(self,'mean',self.mean.detach().clone())
        object.__setattr__(self,'std',self.std.detach().clone().clamp_min(1e-6))

    @torch.no_grad()
    def normalize(self,values):
        if values.shape!=(14,) or not bool(torch.isfinite(values).all()):raise ValueError('Normalize fourteen finite features')
        standardized=(values-self.mean.to(values.device))/self.std.to(values.device).clamp_min(1e-6)
        return standardized.clamp(-5,5),{'clipped':int((standardized.abs()>5).sum()),'total':14}


def fit_normalizer(trajectories: list[ArtifactRef]) -> FrozenNormalizer:
    if not trajectories:raise ProvenanceError('Meta-train reference trajectories required')
    rows=[];sources=[];seen=set()
    keys={'schema','partition','role','synthetic','source_identity','data_manifest','producer','backbone','parent','continuation_horizon','rows'}
    for ref in trajectories:
        if ref.sha256 in seen:raise ProvenanceError('Duplicate reference trajectory')
        seen.add(ref.sha256)
        d=_read(ref)
        if d.get('partition')!='meta_train':raise ProvenanceError('Normalization only admits meta_train, never target/development/protected data')
        if set(d)!=keys or d['schema']!='ralup.feature_trajectory.v1' or d['role']!='reference' or d['synthetic'] is not False:
            raise ProvenanceError('Genuine reference trajectory provenance required')
        try:
            corpus=_read(ArtifactRef(**d['data_manifest']));producer=_read(ArtifactRef(**d['producer']))
            backbone=_read(ArtifactRef(**d['backbone']));parent=_read(ArtifactRef(**d['parent']))
            if corpus['synthetic'] is not False or 'meta-train' not in corpus['partitions']:raise ProvenanceError('Genuine corpus meta-train partition required')
            if parent['source_partition']!='meta_train' or parent['cursor']['source']['partition']!='meta-train':raise ProvenanceError('Reference parent partition mismatch')
            if parent['cursor']['source']['kind']!='indexed' or parent['cursor']['source']['sha256']!=d['data_manifest']['sha256']:
                raise ProvenanceError('Reference parent/corpus source mismatch')
            if backbone.get('schema')!='ralup.exploratory_backbone.v1' or backbone.get('status')!='EXPLORATORY':raise ProvenanceError('Explicit typed exploratory reference required')
            if backbone['parents']['meta_train']['artifact']!=d['parent']:raise ProvenanceError('Reference parent differs from frozen exploratory protocol')
            if not d['source_identity'] or any(producer['source'][name]['sha256']!=digest for name,digest in d['source_identity'].items()):
                raise ProvenanceError('Trajectory producer source mismatch')
            if type(d['continuation_horizon']) is not int or d['continuation_horizon']<=0:raise ProvenanceError('Finite reference schedule required')
            row_ids=set()
            for row in d['rows']:
                if set(row)!={'step','matrix','values'} or type(row['step']) is not int or not 0<=row['step']<d['continuation_horizon'] or type(row['matrix']) is not str:
                    raise ProvenanceError('Malformed reference feature row')
                if type(row['values']) is not list or len(row['values'])!=14 or any(type(v) not in (float,int) for v in row['values']):raise ProvenanceError('Fourteen plain scalars required')
                identity=(row['step'],row['matrix'])
                if identity in row_ids or abs(row['values'][12]-row['step']/d['continuation_horizon'])>1e-6:raise ProvenanceError('Duplicate row or local-rollout progress')
                row_ids.add(identity)
                rows.append(row['values'])
        except (KeyError,TypeError,ValueError) as error:
            if isinstance(error,ProvenanceError):raise
            raise ProvenanceError('Malformed normalization provenance') from error
        sources.append({'path':str(ref.path.resolve()),'sha256':ref.sha256})
    values=torch.tensor(rows,dtype=torch.float64)
    if values.ndim!=2 or values.shape[0]==0 or not bool(torch.isfinite(values).all()):raise ProvenanceError('Nonempty finite meta-train features required')
    manifest=canonical_bytes({'schema':'ralup.frozen_normalizer_sources.v1','partition':'meta_train','trajectories':sources,'rows':len(rows)}).decode()
    return FrozenNormalizer(values.mean(0).float(),values.std(0,correction=0).float().clamp_min(1e-6),manifest)
