"""Fixed causal waveform summaries; all scales use only the current baseline."""
import math
import numpy as np
from .schema import DynamicsSummary

VERSION='temporal-dynamics-v1'
FIELDS=('q10_z_log','q90_z_log','roughness_log','max_jump_log','lag1_correlation','excursion_fraction')

def dynamics_summary(frame,column,onset,decision_time):
    if not onset<=decision_time<=onset+60:
        raise ValueError('Dynamics supports observations up to 60 seconds')
    base=frame.loc[(frame.time>=onset-300)&(frame.time<onset),column].replace([np.inf,-np.inf],np.nan).dropna()
    obs=frame.loc[(frame.time>=onset)&(frame.time<=decision_time),['time',column]].replace([np.inf,-np.inf],np.nan).dropna()
    # Duplicate timestamps represent one observation; no interpolation across gaps.
    obs=obs.groupby('time',sort=True)[column].mean()
    if len(base)<2 or len(obs)<6:return None
    t=obs.index.to_numpy(dtype=float)-onset;y=obs.to_numpy(dtype=float)
    if not (t<30).any() or not (t>=30).any():return None
    dt=np.diff(t);adjacent=(dt>0)&(dt<=5)
    if int(adjacent.sum())<3:return None
    mean=float(base.mean());scale=max(float(base.std(ddof=0)),abs(mean)*.01,1e-6)
    if not math.isfinite(mean) or not math.isfinite(scale):return None
    z=(y-mean)/scale;dy=np.abs(np.diff(y)[adjacent])/scale
    a=y[:-1][adjacent];b=y[1:][adjacent];a=a-a.mean();b=b-b.mean()
    denominator=float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
    corr=float(np.clip(np.dot(a,b)/denominator,-1,1)) if denominator>1e-12 else 0.
    signed_log=lambda v:float(np.sign(v)*np.log1p(min(abs(float(v)),1e6))/5.)
    return DynamicsSummary(q10_z_log=signed_log(np.quantile(z,.1)),q90_z_log=signed_log(np.quantile(z,.9)),
        roughness_log=signed_log(np.mean(dy/dt[adjacent])),max_jump_log=signed_log(np.max(dy)),
        lag1_correlation=corr,excursion_fraction=float(np.mean(np.abs(z)>3.)))
