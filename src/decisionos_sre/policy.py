from .common import digest
import math

def wilson(errors,n,z=1.959963984540054):
    if not n:
        return None
    p=errors/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def select_policy(rows,binding,calibrator,target_risk=.05,min_runs=30):
    if not 0<=target_risk<=1 or min_runs<1:
        raise ValueError("invalid policy constraints")
    if any(r["split"]!="gate_selection" for r in rows):
        raise ValueError("gate_selection split required")
    if len({r["run_id"] for r in rows})!=len(rows):
        raise ValueError("gate requires one observation per independent run group")
    fitted=all(calibrator[h]["status"]=="fitted" for h in ["root","fault"])
    eligible=[r for r in rows if r["joint_correct"] is not None and len(r["candidate_ids"])>1 and r["evidence_usable"]]
    choices=[]
    if fitted:
        for threshold in sorted({r["routing_score"] for r in eligible}):
            accepted=[r for r in eligible if r["routing_score"]>=threshold]
            n=len(accepted)
            errors=sum(not r["joint_correct"] for r in accepted)
            if n>=min_runs and errors/n<=target_risk:
                choices.append({"threshold":threshold,"n":n,"errors":errors,
                                "risk":errors/n,"wilson_95":wilson(errors,n)})
    best=max(choices,key=lambda x:x["n"]) if choices else None
    result={"binding":binding,"calibrator_id":calibrator["id"],"threshold":best["threshold"] if best else None,
        "status":"selected" if best else "no_feasible_threshold","selection":best,
        "target_joint_risk":target_risk,"min_accepted_runs":min_runs,
        "fit_run_ids":sorted({r["run_id"] for r in rows}),
        "search":"all observed routing scores; maximum feasible empirical coverage",
        "guarantee":"none; post-selection interval descriptive only"}
    result["id"]=digest(result)
    return result

def route(root_prob,fault_prob,evidence_usable,calibrator,policy,binding):
    score=min(max(root_prob),max(fault_prob))
    reasons=[]
    if len(root_prob)==1:
        reasons.append("SINGLE_CANDIDATE")
    if not evidence_usable:
        reasons.append("NO_USABLE_EVIDENCE")
    if calibrator is None:
        reasons.append("MISSING_CALIBRATOR")
    elif calibrator["binding"]!=binding:
        reasons.append("CALIBRATOR_VERSION_MISMATCH")
    elif any(calibrator[h]["status"]!="fitted" for h in ["root","fault"]):
        reasons.append("UNCALIBRATED")
    if policy is None:
        reasons.append("MISSING_POLICY")
    elif policy["binding"]!=binding or not calibrator or policy["calibrator_id"]!=calibrator["id"]:
        reasons.append("POLICY_VERSION_MISMATCH")
    elif policy["threshold"] is None:
        reasons.append("NO_FEASIBLE_THRESHOLD")
    elif score<policy["threshold"]:
        reasons.append("BELOW_THRESHOLD")
    return {"destination":"REVIEW" if reasons else "ACCEPT_DIAGNOSIS",
            "routing_score":score,"reason_codes":reasons or ["THRESHOLD_PASSED"]}
