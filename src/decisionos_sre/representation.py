"""Versioned, answer-independent canonical metrics representation.

Aliases/order depend on observable metric signatures, never service names or targets.
Exact signature ties use ID as a deterministic tie-break; no invariance is claimed
for different identities with identical evidence. Numerical branches see ONLY retained
metrics. Vocabulary is recorded in config and derived from TRAIN.
"""
import math
from collections import defaultdict
from .common import canonical

def canonical_encode(serializer,incident):
    from .serializer import Encoded,UnsupportedInput
    enc=lambda x:serializer.tokenizer.encode(x,add_special_tokens=False)
    byservice=defaultdict(list)
    for m in incident.evidence.metrics:byservice[m.service].append(m)
    services=set(byservice)|{c.candidate_id for c in incident.candidates}
    def signature(service):
        ms=byservice[service]
        content=sorted([canonical({k:v for k,v in m.model_dump().items() if k not in ('service','observed_until','temporal')}) for m in ms])
        return (-max([abs(m.change_z or 0) for m in ms]+[0]),canonical(content),service)
    services=sorted(services,key=signature)
    aliases={s:f'C{i}' for i,s in enumerate(services)}
    ids=[serializer.tokenizer.cls_token_id]+enc('application: '+incident.application+'; metrics baseline -300:0 s, observation 0:60 s.\n')
    order=sorted(incident.candidates,key=lambda c:services.index(c.candidate_id))
    spans=[]
    for c in order:
        ids+=enc(' candidate ');start=len(ids);ids+=enc(aliases[c.candidate_id]);spans.append((start,len(ids)))
    if len(ids)+1>serializer.max_length:raise UnsupportedInput('CANDIDATES_EXCEED_TOKEN_BUDGET')
    units=sorted({m.unit for m in incident.evidence.metrics})
    unit_header='\nunits: '+','.join(units)+'; b=baseline, n=observed, z=standardized change, m=missing fraction.\n'
    tokens=enc(unit_header)
    header_fits=len(ids)+len(tokens)+1<=serializer.max_length
    if header_fits:ids+=tokens
    retained=[]
    metrics=sorted(incident.evidence.metrics,key=lambda m:(-abs(m.change_z or 0),services.index(m.service),m.name,canonical({k:v for k,v in m.model_dump().items() if k not in ('service','observed_until','temporal')})))
    fmt=lambda x:'NA' if x is None else f'{x:.3g}'
    for m in metrics:
        if not header_fits: break
        unit='' if len(units)==1 else ' '+m.unit
        line=f'{aliases[m.service]} {m.name}{unit} b{fmt(m.baseline_mean)} n{fmt(m.observed_mean)} z{fmt(m.change_z)} m{m.missing_fraction:.2g}\n'
        tok=enc(line)
        if len(ids)+len(tok)+1<=serializer.max_length:
            ids+=tok;retained.append(m)
    ids.append(serializer.tokenizer.sep_token_id)
    nums=glob=None
    if serializer.numeric_metrics:
        nums,glob=numeric_features(retained,[c.candidate_id for c in order],serializer.numeric_metrics,serializer.numeric_feature_version)
    report={'serializer_version':serializer.version,'tokens':len(ids),'candidate_count':len(order),
            'usable_metrics_retained':sum(m.observed_samples>0 and m.observed_mean is not None for m in retained),
            'metrics_total':len(metrics),'metrics_retained':len(retained),
            'retained_fraction':len(retained)/len(metrics) if metrics else None,
            'retained_metric_keys':[m.service+'/'+m.name for m in retained],'truncated':len(retained)<len(metrics),
            'missing_modalities':[k for k,v in incident.modality_availability.model_dump().items() if not v],
            'token_budgets':{'max':serializer.max_length,'candidate_reservation':True,'logs':0,'traces':0},
            'absolute_timestamps_serialized':False,'service_names_serialized':False,
            'candidate_order':'observable metric signature','numeric_metrics':serializer.numeric_metrics,
            'numeric_evidence_scope':'retained metrics only','descriptions_serialized':False}
    report['numeric_feature_version']=serializer.numeric_feature_version
    report['temporal_metrics_retained']=sum(m.temporal is not None for m in retained)
    report['numeric_evidence_usable']=report['usable_metrics_retained']>0 and (serializer.numeric_feature_version!='temporal-v1' or report['temporal_metrics_retained']>0)
    return Encoded(ids,[c.candidate_id for c in order],spans,report,nums,glob)

def numeric_dimension(config):
    version=config.get('numeric_feature_version','mean-v1')
    if version not in ('mean-v1','temporal-v1'):raise ValueError('unknown numeric feature version')
    return len(config.get('numeric_metrics',[]))*(12 if version=='temporal-v1' else 6) if config.get('numeric_fusion') else 0


def numeric_features(metrics,ids,names,version='mean-v1'):
    # Fixed signed-log transform; no fitted statistics or gold-dependent extraction.
    def squash(v):return 0. if v is None else math.copysign(math.log1p(min(abs(v),1e6)),v)/5
    def values(ms):
        result=[]
        for name in names:
            found=[m for m in ms if m.name==name]
            if not found:result.extend([0.]*6);continue
            # Duplicate name aggregation uses the greatest observable anomaly.
            m=max(found,key=lambda m:(abs(m.change_z or 0),canonical(m.model_dump(exclude={'service','observed_until','temporal'}))))
            rel=None if m.observed_mean is None or m.baseline_mean is None else (m.observed_mean-m.baseline_mean)/max(abs(m.baseline_mean),1e-6)
            result.extend([squash(m.change_z),squash(rel),squash(m.observed_mean),squash(m.baseline_mean),m.missing_fraction,float(m.observed_samples>0 and m.observed_mean is not None)])
        if version=='temporal-v1':
            for name in names:
                found=[m for m in ms if m.name==name]
                m=max(found,key=lambda m:(abs(m.change_z or 0),canonical(m.model_dump(exclude={'service','observed_until','temporal'})))) if found else None
                t=m.temporal if m else None
                result.extend([squash(t.q10_z),squash(t.q90_z),squash(t.std_ratio),squash(t.trend_z),squash(t.late_shift_z),1.] if t else [0.]*6)
        return result
    nums=[values([m for m in metrics if m.service==s]) for s in ids]
    allservices=sorted({m.service for m in metrics})
    allnums=sorted(values([m for m in metrics if m.service==s]) for s in allservices)
    if not allnums:global_values=[0.]*(len(names)*(36 if version=="temporal-v1" else 18))
    else:
        columns=list(zip(*allnums))
        global_values=[x for col in columns for x in (min(col),max(col),sum(col)/len(col))]
    return nums,global_values
