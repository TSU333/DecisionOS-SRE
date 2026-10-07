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
    if serializer.trace_features:services|={t.service for t in incident.evidence.traces or []}
    def signature(service):
        ms=byservice[service]
        content=sorted([canonical({k:v for k,v in m.model_dump().items() if k not in ('service','observed_until','temporal','dynamics')}) for m in ms])
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
    metrics=sorted(incident.evidence.metrics,key=lambda m:(-abs(m.change_z or 0),services.index(m.service),m.name,canonical({k:v for k,v in m.model_dump().items() if k not in ('service','observed_until','temporal','dynamics')})))
    fmt=lambda x:'NA' if x is None else f'{x:.3g}'
    for m in metrics:
        if not header_fits: break
        unit='' if len(units)==1 else ' '+m.unit
        line=f'{aliases[m.service]} {m.name}{unit} b{fmt(m.baseline_mean)} n{fmt(m.observed_mean)} z{fmt(m.change_z)} m{m.missing_fraction:.2g}\n'
        tok=enc(line)
        if len(ids)+len(tok)+1<=serializer.max_length:
            ids+=tok;retained.append(m)
    retained_traces=[]
    if serializer.trace_features:
        from .traces import trace_priority,trace_line
        for t in sorted(incident.evidence.traces or [],key=lambda t:(trace_priority(t),services.index(t.service))):
            token=enc(trace_line(t,aliases[t.service]))
            if len(ids)+len(token)+1<=serializer.max_length:
                ids+=token;retained_traces.append(t)
    ids.append(serializer.tokenizer.sep_token_id)
    nums=glob=None
    if serializer.numeric_metrics:
        nums,glob=numeric_features(retained,[c.candidate_id for c in order],serializer.numeric_metrics,serializer.numeric_feature_version)
    if serializer.trace_features:
        from .traces import trace_features
        tn,tg=trace_features(retained_traces,[c.candidate_id for c in order])
        nums=[a+b for a,b in zip(nums,tn)];glob=glob+tg
    report={'serializer_version' :serializer.version,'tokens':len(ids),'candidate_count':len(order),
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
    report['dynamics_metrics_retained']=sum(m.dynamics is not None for m in retained)
    report['numeric_evidence_usable']=report['usable_metrics_retained']>0 and (serializer.numeric_feature_version=='mean-v1' or report['temporal_metrics_retained']>0) and (serializer.numeric_feature_version!='temporal-dynamics-v1' or report['dynamics_metrics_retained']>0)
    report['trace_summaries_total']=len(incident.evidence.traces or [])
    report['trace_summaries_retained']=len(retained_traces)
    report['trace_services_retained']=[t.service for t in retained_traces]
    report['trace_feature_version']='service-trace-v1' if serializer.trace_features else None
    report['token_budgets']['traces']='remaining budget after metrics; complete summaries only' if serializer.trace_features else 0
    report['modality_evidence_usable']=report['numeric_evidence_usable'] or any(t.all_spans.observed_count>0 for t in retained_traces)
    return Encoded(ids,[c.candidate_id for c in order],spans,report,nums,glob)

def numeric_dimension(config):
    version=config.get('numeric_feature_version','mean-v1')
    if config.get('trace_features') and config.get('trace_feature_version','service-trace-v1')!='service-trace-v1':raise ValueError('unknown trace feature version')
    if version not in ('mean-v1','temporal-v1','temporal-dynamics-v1'):raise ValueError('unknown numeric feature version')
    return (len(config.get('numeric_metrics',[]))*{'mean-v1':6,'temporal-v1':12,'temporal-dynamics-v1':19}[version]+(24 if config.get('trace_features') else 0)) if config.get('numeric_fusion') else 0


def numeric_features(metrics,ids,names,version='mean-v1'):
    # Fixed signed-log transform; no fitted statistics or gold-dependent extraction.
    def squash(v):return 0. if v is None else math.copysign(math.log1p(min(abs(v),1e6)),v)/5
    def values(ms):
        result=[]
        for name in names:
            found=[m for m in ms if m.name==name]
            if not found:result.extend([0.]*6);continue
            # Duplicate name aggregation uses the greatest observable anomaly.
            m=max(found,key=lambda m:(abs(m.change_z or 0),canonical(m.model_dump(exclude={'service','observed_until','temporal','dynamics'}))))
            rel=None if m.observed_mean is None or m.baseline_mean is None else (m.observed_mean-m.baseline_mean)/max(abs(m.baseline_mean),1e-6)
            result.extend([squash(m.change_z),squash(rel),squash(m.observed_mean),squash(m.baseline_mean),m.missing_fraction,float(m.observed_samples>0 and m.observed_mean is not None)])
        if version in ('temporal-v1','temporal-dynamics-v1'):
            for name in names:
                found=[m for m in ms if m.name==name]
                m=max(found,key=lambda m:(abs(m.change_z or 0),canonical(m.model_dump(exclude={'service','observed_until','temporal','dynamics'})))) if found else None
                t=m.temporal if m else None
                result.extend([squash(t.q10_z),squash(t.q90_z),squash(t.std_ratio),squash(t.trend_z),squash(t.late_shift_z),1.] if t else [0.]*6)
        if version=='temporal-dynamics-v1':
            from .dynamics import FIELDS
            for name in names:
                found=[m for m in ms if m.name==name]
                m=max(found,key=lambda m:(abs(m.change_z or 0),canonical(m.model_dump(exclude={'service','observed_until','temporal','dynamics'})))) if found else None
                d=m.dynamics if m else None
                result.extend([getattr(d,k) for k in FIELDS]+[1.] if d else [0.]*7)
        return result
    nums=[values([m for m in metrics if m.service==s]) for s in ids]
    allservices=sorted({m.service for m in metrics})
    allnums=sorted(values([m for m in metrics if m.service==s]) for s in allservices)
    if not allnums:global_values=[0.]*(len(names)*{'mean-v1':18,'temporal-v1':36,'temporal-dynamics-v1':57}[version])
    else:
        columns=list(zip(*allnums))
        global_values=[x for col in columns for x in (min(col),max(col),sum(col)/len(col))]
    return nums,global_values
