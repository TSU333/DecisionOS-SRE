"""TRAIN-only causal missing-observation views; no new independent cases."""
from pathlib import Path
import math
import numpy as np
from .common import read,file_hash
from .schema import TrainingExample,Metric
from .temporal import temporal_summary
from .dynamics import dynamics_summary

VERSION='causal-missing-v1'


def causal_view(example, frame, onset, kind):
    if example.source_metadata.get('split')!='train':
        raise ValueError('Views are restricted to TRAIN')
    if kind not in ('lag15','gap15_30'):
        raise ValueError('Unknown observation view')
    decision=example.input.decision_time
    if decision!=onset+60:
        raise ValueError('Views require the fixed 60 second decision window')
    frame=frame.loc[(frame.time>=onset-300)&(frame.time<=decision)].copy()
    columns=[m.service+'_'+m.name for m in example.input.evidence.metrics]
    hidden=(frame.time>onset+45) if kind=='lag15' else ((frame.time>=onset+15)&(frame.time<onset+30))
    frame.loc[hidden,columns]=np.nan
    baseline=frame.loc[frame.time<onset];obs=frame.loc[frame.time>=onset]
    ex=example.model_copy(deep=True);metrics=[]
    for old in example.input.evidence.metrics:
        col=old.service+'_'+old.name
        b=baseline[col].replace([np.inf,-np.inf],np.nan).dropna()
        o=obs[col].replace([np.inf,-np.inf],np.nan).dropna()
        bm=float(b.mean()) if len(b) else None;om=float(o.mean()) if len(o) else None
        z=None if bm is None or om is None or len(b)<2 else float(np.clip((om-bm)/max(float(b.std(ddof=0)),abs(bm)*.01,1e-6),-100,100))
        t=temporal_summary(frame,col,onset,decision)
        metrics.append(Metric(service=old.service,name=old.name,unit=old.unit,baseline_mean=bm,observed_mean=om,change_z=z,
            missing_fraction=1-len(o)/max(len(obs),1),baseline_samples=len(b),observed_samples=len(o),
            observed_until=float(obs.loc[o.index,'time'].max()) if len(o) else decision,temporal=t,dynamics=dynamics_summary(frame,col,onset,decision) if old.dynamics is not None else None))
    ex.input.evidence.metrics=metrics
    ex.opaque_incident_id=example.opaque_incident_id+'__'+kind
    ex.parent_incident_id=example.opaque_incident_id
    ex.augmentation_metadata={'version':VERSION,'kind':kind,'parent_run':example.original_run_id,'decision_window_seconds':60,'artificial_missingness':True,'independent_case':False}
    return TrainingExample.model_validate(ex.model_dump())


def load_training_views(config,trainset):
    probability=config.get('training_view_probability',0.)
    if not math.isfinite(probability) or not 0.<=probability<=1.:
        raise ValueError('Invalid training view probability')
    path=config.get('training_views_file')
    if not path:
        if probability:raise ValueError('Training view file is required')
        return [],{'count':0,'version':None}
    if file_hash(path)!=config.get('training_views_sha256'):
        raise ValueError('Training view checksum mismatch')
    parents={e.opaque_incident_id:e for e in trainset}
    views=[TrainingExample.model_validate(e) for e in read(path)]
    ids=set(parents)
    for ex in views:
        parent=parents.get(ex.parent_incident_id)
        if parent is None or parent.source_metadata.get('split')!='train' or ex.source_metadata.get('split')!='train':
            raise ValueError('View parent must belong to TRAIN')
        if ex.original_run_id!=parent.original_run_id or ex.targets!=parent.targets:
            raise ValueError('View cannot alter labels or original run identity')
        if ex.input.candidates!=parent.input.candidates or ex.input.application!=parent.input.application or ex.input.decision_time!=parent.input.decision_time:
            raise ValueError('View cannot alter candidate scope, application or decision time')
        if ex.opaque_incident_id in ids:raise ValueError('Duplicate view identity')
        ids.add(ex.opaque_incident_id)
        if ex.augmentation_metadata.get('version')!=VERSION or ex.augmentation_metadata.get('kind') not in ('lag15','gap15_30'):
            raise ValueError('Unknown view provenance')
    return views,{'count':len(views),'version':VERSION,'path':str(Path(path)),'sha256':file_hash(path),'independent_new_cases':0}


def eligible_parent_indices(templates,n,require_usable):
    indices=[i for i in range(n) if not require_usable or templates[i]['evidence_usable']]
    if not indices:raise ValueError('No eligible TRAIN evidence')
    return indices


def choose_view_indices(parent_indices,by_parent,probability,rng):
    return [rng.choice(by_parent[i]) if by_parent.get(i) and rng.random()<probability else i for i in parent_indices]
