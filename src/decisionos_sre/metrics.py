import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from .common import FAULTS
from .policy import wilson

def probability_metrics(rows,head,key="prob",bins=10):
    usable=[r for r in rows if r[head+"_target"] is not None and r[head+"_target"]>=0]
    if not usable:
        return {"n":0,"nll":None,"ece":None,"brier":None,"reliability":[]}
    nll=[]; brier=[]; conf=[]; correct=[]
    for r in usable:
        p=np.array(r[head+"_"+key]); y=r[head+"_target"]
        nll.append(-np.log(max(p[y],1e-300)))
        onehot=np.zeros(len(p)); onehot[y]=1
        brier.append(float(np.sum((p-onehot)**2)))
        conf.append(float(max(p))); correct.append(float(np.argmax(p)==y))
    records=[]; ece=0.
    for i in range(bins):
        keep=[j for j,c in enumerate(conf) if min(int(c*bins),bins-1)==i]
        acc=float(np.mean([correct[j] for j in keep])) if keep else None
        c=float(np.mean([conf[j] for j in keep])) if keep else None
        if keep:
            ece+=len(keep)/len(usable)*abs(acc-c)
        records.append({"lo":i/bins,"hi":(i+1)/bins,"n":len(keep),"accuracy":acc,"confidence":c})
    return {"n":len(usable),"nll":float(np.mean(nll)),"ece":ece,"brier":float(np.mean(brier)),
            "brier_definition":"mean sum_k (p_k-y_k)^2; not divided by class count",
            "ece_binning":f"{bins} equal width bins; final bin includes 1","reliability":records,
            "condition":"gold covered by candidates" if head=="root" else "known gold fault label"}

def selective(rows):
    eligible=[r for r in rows if r["joint_correct"] is not None]
    accepted=[r for r in eligible if r["routing"]["destination"]=="ACCEPT_DIAGNOSIS"]
    n=len(accepted); errors=sum(not r["joint_correct"] for r in accepted)
    return {"overall_n":len(rows),"joint_gold_n":len(eligible),
            "accepted_total":sum(r["routing"]["destination"]=="ACCEPT_DIAGNOSIS" for r in rows),
            "accepted_joint_gold_n":n,"errors":errors,
            "coverage":sum(r["routing"]["destination"]=="ACCEPT_DIAGNOSIS" for r in rows)/len(rows) if rows else 0.,
            "selective_risk":errors/n if n else None,
            "selective_accuracy":1-errors/n if n else None,
            "risk_wilson_95":wilson(errors,n)}

def rank_metrics(rows):
    valid=[r for r in rows if r["root_target"] is not None]
    ranks=[]
    for r in valid:
        # Stable ID tie breaking, rather than knowledge of the answer.
        order=sorted(range(len(r["root_prob"])),key=lambda j:(-r["root_prob"][j],r["candidate_ids"][j]))
        ranks.append(order.index(r["root_target"])+1 if r["root_target"]>=0 else float("inf"))
    n=len(valid)
    return {"n":n,"acc_at_1":sum(k==1 for k in ranks)/n if n else None,
            "mrr":sum(1/k for k in ranks)/n if n else None,
            "recall_at_3":sum(k<=3 for k in ranks)/n if n else None,
            "candidate_coverage":sum(k!=float("inf") for k in ranks)/n if n else None}

def evaluate_metrics(rows,bins=10):
    root=rank_metrics(rows)
    fault_rows=[r for r in rows if r["fault_target"] is not None and r["fault_target"]>=0]
    truth=[r["fault_target"] for r in fault_rows]; pred=[r["fault_pred"] for r in fault_rows]
    labels=list(range(len(FAULTS)))
    p,rec,f1,support=precision_recall_fscore_support(truth,pred,labels=labels,zero_division=0)
    fault={"n":len(truth),"accuracy":float(np.mean(np.array(truth)==pred)) if truth else None,
       "macro_f1":float(np.mean(f1)) if truth else None,
       "per_class":{name:{"precision":float(p[i]),"recall":float(rec[i]),"f1":float(f1[i]),"support":int(support[i])}
                    for i,name in enumerate(FAULTS)},
       "confusion_matrix":confusion_matrix(truth,pred,labels=labels).tolist()}
    joint=[r for r in rows if r["joint_correct"] is not None]
    curve=[]
    for t in sorted({r["routing_score"] for r in joint},reverse=True):
        accepted=[r for r in joint if r["routing_score"]>=t]
        curve.append({"threshold":t,"coverage":len(accepted)/len(rows),
                      "risk":sum(not r["joint_correct"] for r in accepted)/len(accepted),
                      "n":len(accepted)})
    groups={}
    for field in ["application","candidate_count"]:
        vals={r["application"] if field=="application" else len(r["candidate_ids"]) for r in rows}
        groups[field]={str(v):rank_metrics([r for r in rows if
                 (r["application"] if field=="application" else len(r["candidate_ids"]))==v]) for v in vals}
    return {"runs":len({r["run_id"] for r in rows}),"observations":len(rows),
        "missing_root":sum(r["root_target"] is None for r in rows),
        "candidate_misses":sum(r["root_target"]==-1 for r in rows),
        "missing_fault":sum(r["fault_target"] is None for r in rows),
        "root":root,"fault":fault,"groups":groups,
        "joint_accuracy":sum(r["joint_correct"] for r in joint)/len(joint) if joint else None,
        "joint_n":len(joint),"joint_error_wilson_95":wilson(sum(not r["joint_correct"] for r in joint),len(joint)),
        "probability":{h:{k:probability_metrics(rows,h,k,bins) for k in ["raw","prob"]} for h in ["root","fault"]},
        "selective":selective(rows),"risk_coverage":curve,
        "risk_interval_note":"Descriptive test uncertainty; not a deployment guarantee."}

def plot_metrics(metrics,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,head in zip(axes[:2],["root","fault"]):
        for key,label in [("raw","raw"),("prob","calibrated")]:
            bins=metrics["probability"][head][key]["reliability"]
            usable=[b for b in bins if b["n"]]
            ax.plot([b["confidence"] for b in usable],[b["accuracy"] for b in usable],"o-",label=label)
        ax.plot([0,1],[0,1],"k--",alpha=.4)
        ax.set(title=head+" reliability",xlabel="mean confidence",ylabel="accuracy",xlim=(0,1),ylim=(0,1))
        ax.legend()
    curve=metrics["risk_coverage"]
    axes[2].plot([p["coverage"] for p in curve],[p["risk"] for p in curve],"o-")
    axes[2].set(title="Joint risk / coverage (descriptive)",xlabel="coverage",ylabel="joint error",ylim=(0,1))
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
