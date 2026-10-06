"""TRAIN-fitted numerical control; no test access unless explicitly requested."""
import argparse
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from decisionos_sre.common import FAULTS,read,save,file_hash
from decisionos_sre.data import load_split
from decisionos_sre.training import prediction_summary

def features(ex,names):
    result=[]
    for name in names:
        ms=[m for m in ex.input.evidence.metrics if m.name==name]
        for field in ("change_z","relative_change","observed_mean"):
            values=[]
            for m in ms:
                if field=="relative_change":
                    v=(m.observed_mean-m.baseline_mean)/max(abs(m.baseline_mean),1e-6) if m.observed_mean is not None and m.baseline_mean is not None else None
                else:v=getattr(m,field)
                if v is not None:values.append(np.sign(v)*np.log1p(min(abs(v),1e6)))
            values=sorted(values,reverse=True)
            result.extend((values+[0]*3)[:3]);result.extend((sorted(values)+[0]*3)[:3])
            result.extend([float(np.mean(values)) if values else 0,float(np.std(values)) if values else 0,float(bool(values))])
    return result

def run(split='model_validation'):
    folder=Path('artifacts/retrain/numeric')
    if split=='model_validation':
        train=load_split('data/rcaeval','train')
        names=sorted({m.name for e in train for m in e.input.evidence.metrics})
        X=np.array([features(e,names) for e in train]);y=[FAULTS.index(e.targets.fault_type.value) for e in train]
        model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.0,random_state=42,n_jobs=4)
        model.fit(X,y)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'model.pkl').open('wb') as f:pickle.dump((model,names),f)
        save(folder/'metadata.json',{'train_run_ids':[e.original_run_id for e in train], 'features':names,'config':model.get_params(),'split_hash':read('data/rcaeval/splits.json')['split_hash'],'role':'fault numerical baseline; root remains max absolute z anomaly','model_sha256':file_hash(folder/'model.pkl')})
    else:
        with (folder/'model.pkl').open('rb') as f:model,names=pickle.load(f)
    examples=load_split('data/rcaeval',split)
    probs=model.predict_proba(np.array([features(e,names) for e in examples]))
    rows=[]
    for e,proba in zip(examples,probs):
        ids=[c.candidate_id for c in e.input.candidates]
        scores=[max([abs(m.change_z or 0) for m in e.input.evidence.metrics if m.service==s]+[0]) for s in ids]
        rows.append({'incident_id':e.opaque_incident_id,'run_id':e.original_run_id,'split':split,'root_logits':scores,'fault_logits':np.log(np.maximum(proba,1e-12)).tolist(),'root_target':ids.index(e.targets.root_cause.value),'fault_target':FAULTS.index(e.targets.fault_type.value)})
    save(folder/(split+'_predictions.json'),rows)
    summary=prediction_summary(rows);save(folder/(split+'_summary.json'),summary);print(summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['model_validation','test'],default='model_validation');run(p.parse_args().split)
