"""Causal, fixed-window temporal summaries; no fitted or label-derived statistics."""
import numpy as np
from .schema import TemporalSummary


def temporal_summary(frame, column, onset, decision_time):
    if not onset <= decision_time <= onset + 60:
        raise ValueError("temporal-v1 supports observation up to 60 seconds")
    base=frame.loc[(frame.time>=onset-300)&(frame.time<onset),column].replace([np.inf,-np.inf],np.nan).dropna()
    obs=frame.loc[(frame.time>=onset)&(frame.time<=decision_time),['time',column]].replace([np.inf,-np.inf],np.nan).dropna().sort_values('time')
    if len(base)<2 or len(obs)<3 or obs.time.nunique()<2:
        return None
    times=obs.time.to_numpy(dtype=float)-onset
    vals=obs[column].to_numpy(dtype=float)
    early=vals[times<30];late=vals[times>=30]
    if not len(early) or not len(late):return None
    mean=float(base.mean());scale=max(float(base.std(ddof=0)),abs(mean)*.01,1e-6)
    centered=times-times.mean()
    slope=float(np.dot(centered,vals-vals.mean())/np.dot(centered,centered))
    clip=lambda x:float(np.clip(x,-100,100))
    return TemporalSummary(q10_z=clip((np.quantile(vals,.1)-mean)/scale),
        q90_z=clip((np.quantile(vals,.9)-mean)/scale),std_ratio=clip(vals.std(ddof=0)/scale),
        trend_z=clip(slope*60/scale),late_shift_z=clip((late.mean()-early.mean())/scale))
