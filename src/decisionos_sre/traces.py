"""Fixed Jaeger span reduction; completion cutoff is enforced before aggregation."""
import math,re
import numpy as np
import pandas as pd
from .schema import SpanSummary,ServiceTraceSummary

TRACE_DIM=24

def summarize_traces(frame,onset,decision_time,candidate_ids):
    if decision_time!=onset+60:raise ValueError('service-trace-v1 requires a 60-second observation')
    needed=['serviceName','operationName','startTime','duration','statusCode']
    dedup=frame.drop_duplicates(['traceID','spanID']) if {'traceID','spanID'}<=set(frame.columns) else frame
    df=dedup[needed].copy()
    df=df.dropna(subset=['serviceName','startTime','duration'])
    df=df[df.duration>=0]
    df['start_s']=df.startTime.astype(float)/1e6
    df['end_s']=(df.startTime.astype(float)+df.duration.astype(float))/1e6
    df=df[(df.start_s>=onset-300)&(df.end_s<=decision_time)].copy()
    aliases={'frontendservice':'frontend'} if 'frontend' in candidate_ids else {}
    df['service']=df.serviceName.map(lambda s:aliases.get(s,s))
    def server(row):
        match=re.search(r'(?:^|[./])hipstershop\.([A-Za-z0-9]+Service)/',str(row.operationName))
        return bool(match and aliases.get(match.group(1).lower(),match.group(1).lower())==row.service)
    df['server']=df.apply(server,axis=1) if len(df) else pd.Series(dtype=bool)
    baseline=(df.start_s<onset)&(df.end_s<onset)
    observed=(df.start_s>=onset)&(df.end_s<=decision_time)
    df['baseline']=baseline;df['observed']=observed
    result=[]
    def stats(rows):
        b=rows[rows.baseline];o=rows[rows.observed]
        bk=b.statusCode.dropna();ok=o.statusCode.dropna()
        latency=lambda z,kind:float(z.duration.mean()/1000) if len(z) and kind=='mean' else (float(z.duration.quantile(.95)/1000) if len(z) else None)
        return SpanSummary(baseline_count=len(b),observed_count=len(o),baseline_mean_ms=latency(b,'mean'),observed_mean_ms=latency(o,'mean'),baseline_p95_ms=latency(b,'p95'),observed_p95_ms=latency(o,'p95'),baseline_error_ratio=float((bk!=0).mean()) if len(bk) else None,observed_error_ratio=float((ok!=0).mean()) if len(ok) else None,observed_deadline_ratio=float((ok==4).mean()) if len(ok) else None,observed_unavailable_ratio=float((ok==14).mean()) if len(ok) else None,observed_known_status_fraction=len(ok)/len(o) if len(o) else 0.)
    for service,rows in df.groupby('service',sort=True):
        o=rows[rows.observed]
        result.append(ServiceTraceSummary(service=service,all_spans=stats(rows),server_spans=stats(rows[rows.server]),observed_until=float(o.end_s.max()) if len(o) else float(onset)))
    audit={'raw_spans':len(frame),'duplicate_span_ids_removed':len(frame)-len(dedup),'within_window_completed':len(df),'baseline_spans':int(baseline.sum()),'observed_spans':int(observed.sum()),'max_observed_end':max((r.observed_until for r in result),default=None),'source_services':sorted(frame.serviceName.dropna().unique().tolist()),'aliases':aliases,'unknown_status_not_success':True,'raw_time_unit':'microseconds','output_latency_unit':'milliseconds'}
    return result,audit

def trace_values(trace):
    def squash(v):return 0. if v is None else math.copysign(math.log1p(min(abs(v),1e6)),v)/5
    def relative(a,b):return None if a is None or b is None else (a-b)/max(abs(b),1e-6)
    if trace is None:return [0.]*TRACE_DIM
    vals=[]
    for t in [trace.all_spans,trace.server_spans]:
        vals.extend([squash(relative(t.observed_count/60,t.baseline_count/300)),squash(relative(t.observed_mean_ms,t.baseline_mean_ms)),squash(relative(t.observed_p95_ms,t.baseline_p95_ms)),squash(t.observed_mean_ms),squash(t.observed_p95_ms),t.observed_error_ratio or 0.,t.observed_deadline_ratio or 0.,t.observed_unavailable_ratio or 0.,(t.observed_error_ratio or 0.)-(t.baseline_error_ratio or 0.),t.observed_known_status_fraction,squash(t.observed_count),float(t.observed_count>0)])
    return vals

def trace_features(traces,ids):
    byid={t.service:t for t in traces};nums=[trace_values(byid.get(cid)) for cid in ids]
    allnums=sorted(trace_values(t) for t in traces)
    glob=[v for col in zip(*allnums) for v in (min(col),max(col),sum(col)/len(col))] if allnums else [0.]*(TRACE_DIM*3)
    return nums,glob

def trace_priority(t):
    v=trace_values(t)
    return (-max(abs(x) for x in [v[1],v[2],v[13],v[14]])-10*max(v[5],v[17]),tuple(v))

def trace_line(t,alias):
    # The compact numeric list has a versioned field order documented by trace_values.
    return 'trace '+alias+' service-trace-v1 '+','.join(f'{x:.3g}' for x in trace_values(t))+'\n'
