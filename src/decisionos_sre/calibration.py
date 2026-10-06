import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from .common import digest

def probabilities(logits,temperature=1.):
    a=np.asarray(logits,dtype=float)
    if not np.isfinite(temperature) or temperature<=0 or not np.isfinite(a).all():
        raise ValueError("invalid logits/temperature")
    a=a/temperature
    return np.exp(a-logsumexp(a)).tolist()

def fit_temperature(rows,head,min_samples=10):
    usable=[r for r in rows if r[head+"_target"] is not None and r[head+"_target"]>=0]
    if len(usable)<min_samples:
        return {"temperature":1.,"status":"insufficient_samples","n":len(usable)}
    def nll(log_t):
        t=np.exp(log_t)
        return float(np.mean([logsumexp(np.array(r[head+"_logits"])/t)
                   -r[head+"_logits"][r[head+"_target"]]/t for r in usable]))
    opt=minimize_scalar(nll,bounds=(-3,3),method="bounded",options={"xatol":1e-7})
    t=float(np.exp(opt.x))
    return {"temperature":t,"status":"fitted" if opt.success else "fit_failed","n":len(usable),
            "nll_before":nll(0),"nll_after":nll(float(opt.x)),"log_temperature_bounds":[-3,3]}

def fit_calibrator(rows,binding,min_samples=10):
    if any(r["split"]!="calibration" for r in rows):
        raise ValueError("calibration split required")
    result={"binding":binding,"fit_run_ids":sorted({r["run_id"] for r in rows}),
            "root":fit_temperature(rows,"root",min_samples),
            "fault":fit_temperature(rows,"fault",min_samples)}
    result["id"]=digest(result)
    return result

def enrich(rows,calibrator=None):
    for r in rows:
        row=dict(r)
        for head in ["root","fault"]:
            t=calibrator[head]["temperature"] if calibrator and calibrator[head]["status"]=="fitted" else 1.
            row[head+"_raw"]=probabilities(r[head+"_logits"])
            row[head+"_prob"]=probabilities(r[head+"_logits"],t)
            row[head+"_pred"]=int(np.argmax(row[head+"_prob"]))
            target=r[head+"_target"]
            row[head+"_correct"]=None if target is None else row[head+"_pred"]==target
        row["joint_correct"]=None if row["root_correct"] is None or row["fault_correct"] is None else bool(row["root_correct"] and row["fault_correct"])
        row["routing_score"]=min(max(row["root_prob"]),max(row["fault_prob"]))
        yield row
