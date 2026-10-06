"""Round 3 numerical diagnostic fitted only on TRAIN; no held-out access."""
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from decisionos_sre.common import FAULTS,read,save,file_hash
from decisionos_sre.data import load_split
from decisionos_sre.representation import numeric_features
from decisionos_sre.training import prediction_summary
from numeric_baseline import features

train=load_split("data/round3","train")
dev=load_split("data/round3","model_validation")
names=sorted({m.name for e in train for m in e.input.evidence.metrics})
fault=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.,random_state=42,n_jobs=4)
fault.fit([features(e,names) for e in train],[FAULTS.index(e.targets.fault_type.value) for e in train])
X=[];y=[];weights=[]
for e in train:
    ids=[c.candidate_id for c in e.input.candidates]
    nums,_=numeric_features(e.input.evidence.metrics,ids,names)
    X.extend(nums); y.extend([int(s==e.targets.root_cause.value) for s in ids])
    weights.extend([.5 if s==e.targets.root_cause.value else .5/(len(ids)-1) for s in ids])
root=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.,random_state=42,n_jobs=4)
root.fit(X,y,sample_weight=weights)
rows=[]
for e in dev:
    ids=[c.candidate_id for c in e.input.candidates]
    nums,_=numeric_features(e.input.evidence.metrics,ids,names)
    rp=root.predict_proba(nums)[:,list(root.classes_).index(1)]
    fp=fault.predict_proba([features(e,names)])[0]
    rows.append({"run_id":e.original_run_id,"cohort":e.source_metadata["dataset_suite"],"split":"model_validation","candidate_ids":ids,"root_logits":np.log(np.maximum(rp,1e-12)).tolist(),"fault_logits":np.log(np.maximum(fp,1e-12)).tolist(),"root_target":ids.index(e.targets.root_cause.value),"fault_target":FAULTS.index(e.targets.fault_type.value)})
folder=Path("artifacts/round3/numeric_diagnostic");folder.mkdir(parents=True,exist_ok=True)
with (folder/"models.pkl").open("wb") as f:pickle.dump((root,fault,names),f)
summary=prediction_summary(rows)
save(folder/"validation_logits.json",rows);save(folder/"summary.json",summary)
save(folder/"metadata.json",{"role":"TRAIN-fitted numerical diagnostic only; ineligible for shared-encoder deployment selection","train_run_ids":[e.original_run_id for e in train],"validation_run_ids":[e.original_run_id for e in dev],"features":names,"split_hash":read("data/round3/splits.json")["split_hash"],"model_sha256":file_hash(folder/"models.pkl"),"root_config":root.get_params(),"fault_config":fault.get_params(),"evidence_scope":"all causal summaries; may differ from token-truncated neural input"})
print(summary)
for r in rows:
    if r["cohort"]=="RE2":print(r["run_id"],r["root_target"]==int(np.argmax(r["root_logits"])),FAULTS[r["fault_target"]],FAULTS[int(np.argmax(r["fault_logits"]))])
