"""TRAIN/validation-only lightweight fault classifier diagnostic, no gold roots as input."""
from pathlib import Path
import time,numpy as np,torch
from torch import nn
from sklearn.ensemble import ExtraTreesClassifier
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,FAULTS
from decisionos_sre.data import load_split
from decisionos_sre.serializer import Serializer
from decisionos_sre.training import prediction_summary,selection_key

out=Path('outputs/round7');cfg=read('configs/round5_temporal_seed44.json')
head=nn.Sequential(nn.Linear(120,64),nn.GELU(),nn.Linear(64,1))
state=torch.load('artifacts/round5/temporal_seed44/frozen/best_heads.pt',map_location='cpu',weights_only=True)
head.load_state_dict({k.removeprefix('numeric_root.'):v for k,v in state.items() if k.startswith('numeric_root.')});head.eval()
torch.set_num_threads(2);tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
train=load_split('data/round6','train');val=load_split('data/round6','model_validation');results=[]
for trace in [False,True]:
 ser=Serializer(tok,2048,'metrics-canonical-v2',cfg['numeric_metrics'],'temporal-v1',trace)
 features=[];rows=[];targets=[]
 for ex in train+val:
  enc=ser(ex.input);nums=np.array(enc.candidate_numeric,dtype=np.float32)
  with torch.no_grad():root=head(torch.tensor(nums[:,:120])).squeeze(-1);weights=torch.softmax(root,0).numpy()
  features.append(np.r_[enc.incident_numeric,(nums*weights[:,None]).sum(0),np.sort(weights)[-1],np.sort(weights)[-2]])
  targets.append(FAULTS.index(ex.targets.fault_type.value));rows.append({'run_id':ex.original_run_id,'split':ex.source_metadata['split'],'cohort':ex.source_metadata['dataset_suite'],'candidate_ids':enc.candidate_ids,'root_logits':root.tolist(),'root_target':enc.candidate_ids.index(ex.targets.root_cause.value),'fault_target':targets[-1]})
 x=np.array(features);y=np.array(targets);n=len(train)
 np.savez_compressed(out/('trace_features.npz' if trace else 'metric_features.npz'),x=x,y=y)
 for depth in [8,None]:
  for leaf in [1,2]:
   start=time.perf_counter();model=ExtraTreesClassifier(n_estimators=256,max_depth=depth,min_samples_leaf=leaf,max_features=.5,n_jobs=2,random_state=42)
   model.fit(x[:n],y[:n]);p=model.predict_proba(x[n:]);pred=[]
   for r,probs in zip(rows[n:],p):pred.append({**r,'fault_logits':np.log(np.clip(probs,1e-12,1)).tolist()})
   summary=prediction_summary(pred);record={'trace':trace,'max_depth':depth,'min_samples_leaf':leaf,'validation':summary,'rank':list(selection_key(summary,cfg)),'seconds':time.perf_counter()-start}
   results.append(record);print('DIAGNOSTIC',record,flush=True)
save(out/'diagnostic_results.json',{'diagnostic_only':True,'candidates':results,'best':min(results,key=lambda c:c['rank']),'n_train':165,'n_validation':40,'no_heldout_evaluation':True})
