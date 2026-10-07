"""Derive fixed causal dynamics without modifying labels, partitions or legacy inputs."""
from pathlib import Path
from collections import Counter,defaultdict
import copy,shutil,argparse
import pandas as pd
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash,digest
from decisionos_sre.schema import TrainingExample
from decisionos_sre.dynamics import dynamics_summary,VERSION,FIELDS
from decisionos_sre.serializer import Serializer

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-output',default='data/round12')
parser.add_argument('--audit-output',default='outputs/round12')
parser.add_argument('--parent',default='artifacts/round5/temporal_seed44/frozen')
args=parser.parse_args()
out=Path(args.audit_output);root=Path(args.data_output);oldroot=Path('data/round5')
if (root/'examples.json').exists() or (out/'feature_spec.json').exists():raise FileExistsError('Feature preparation already sealed')
parent=args.parent;meta=read(Path(parent)/'metadata.json');cfg=meta['config']
assert file_hash(Path(parent)/'checkpoint.pt')=='461b93c00f348cc8759060b08f577ea5a7cce25672b0f7de0768d033b16b6423'
original=read(oldroot/'examples.json');examples=copy.deepcopy(original);byid={e['opaque_incident_id']:e for e in examples};manifest=read(oldroot/'manifest.json');split=read(oldroot/'splits.json')
assert len({e['original_run_id'] for e in examples})==400
assert digest({k:v for k,v in manifest.items() if k!='manifest_hash'})==manifest['manifest_hash']
assert digest({k:v for k,v in split.items() if k!='split_hash'})==split['split_hash']
original_files={n:file_hash(oldroot/n) for n in ['manifest.json','splits.json','examples.json']}
assert original_files==read('outputs/round11/data_audit.json')['original_files']
# Seal the formula before reading any new holdout waveform summaries.
save(out/'feature_spec.json',{'version':VERSION,'source_sha256':file_hash('src/decisionos_sre/dynamics.py'),'fields':FIELDS,'baseline':'[onset-300,onset)','observation':'[onset,decision_time], decision_time <= onset+60','scale':'max(baseline population std, abs(baseline mean)*0.01, 1e-6)','q10_q90':'signed log1p of standardized quantiles, saturate at abs 1e6, divide by 5','roughness':'log1p(mean(abs(delta value)/delta seconds)/scale)/5; pairs with 0<dt<=5 only','jump':'log1p(max(abs(delta value))/scale)/5; same adjacent pairs','lag1':'Pearson correlation of valid adjacent observations, constant=0','excursion':'fraction of observed absolute z >3','missing':'None if baseline <2, observations <6, <3 adjacent pairs or missing either 0:30 or 30:60 coverage','duplicates':'mean by timestamp; no interpolation','learned_statistics':False,'labels_used':False,'new_independent_cases':0})
raw=[];coverage=defaultdict(Counter);train_total=Counter();train_valid=Counter();train_clipped=Counter()
for n,entry in enumerate(manifest['entries']):
 path=Path('data/round4')/entry['case']/'metrics.parquet';assert file_hash(path)==entry['sha256']
 frame=pd.read_parquet(path).sort_values('time');frame=frame.loc[(frame.time>=entry['onset']-300)&(frame.time<=entry['decision_time'])]
 ex=byid[entry['opaque_incident_id']];role=split['assignments'][ex['opaque_incident_id']]
 assert ex['source_metadata']['split']==role
 for metric in ex['input']['evidence']['metrics']:
  d=dynamics_summary(frame,metric['service']+'_'+metric['name'],entry['onset'],entry['decision_time']);metric['dynamics']=d.model_dump() if d else None
  coverage[role]['metrics']+=1;coverage[role]['with_dynamics']+=int(d is not None)
  if role=='train':
   name=metric['name'];train_total[name]+=1
   if metric['change_z'] is not None:train_valid[name]+=1;train_clipped[name]+=int(abs(metric['change_z'])>=100)
 TrainingExample.model_validate(ex)
 raw.append({'run_id':ex['original_run_id'],'path':str(path),'sha256':entry['sha256']})
 if (n+1)%50==0:print('dynamics prepared',n+1,flush=True)
tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
legacy=Serializer(tok,2048,cfg['serializer_version'],cfg['numeric_metrics'],'temporal-v1');newser=Serializer(tok,2048,cfg['serializer_version'],cfg['numeric_metrics'],VERSION)
legacy_hashes={};records=[];excluded=[];ids=defaultdict(set)
for old,new in zip(original,examples):
 stripped=copy.deepcopy(new)
 for metric in stripped['input']['evidence']['metrics']:metric.pop('dynamics')
 assert stripped==old
 a=legacy(old['input']);b=legacy(new['input']);c=newser(new['input'])
 info=lambda x:{'ids':x.input_ids,'candidates':x.candidate_ids,'spans':x.spans,'numeric':x.candidate_numeric,'global':x.incident_numeric}
 assert info(a)==info(b)
 assert c.input_ids==a.input_ids and c.candidate_ids==a.candidate_ids and c.spans==a.spans
 assert [row[:120] for row in c.candidate_numeric]==a.candidate_numeric and c.incident_numeric[:360]==a.incident_numeric
 legacy_hashes[new['original_run_id']]=digest(info(a))
 role=new['source_metadata']['split'];ids[role].add(new['original_run_id'])
 records.append({'run_id':new['original_run_id'],'split':role,'cohort':new['source_metadata']['dataset_suite'],'dynamics_retained':c.report['dynamics_metrics_retained'],'usable':c.report['modality_evidence_usable'],'tokens':len(c.input_ids)})
 if role=='train' and not c.report['modality_evidence_usable']:excluded.append({'run_id':new['original_run_id'],'cohort':new['source_metadata']['dataset_suite'],'reason':'no retained usable temporal evidence'})
assert excluded==read('outputs/round11/data_audit.json')['excluded_unusable_train']
assert sum(map(len,ids.values()))==len(set.union(*ids.values()))==400
root.mkdir(parents=True,exist_ok=True)
for name in ['manifest.json','splits.json']:shutil.copyfile(oldroot/name,root/name)
save(root/'examples.json',examples);save(out/'legacy_representation_hashes.json',legacy_hashes)
audit={'version':VERSION,'original_cases':400,'new_independent_cases':0,'allowed_train_cases':165,'eligible_train_cases':164,'training_view_count':0,'excluded_unusable_train':excluded,'original_split_counts':{s:len(i) for s,i in ids.items()},'original_files':original_files,'current_files':{n:file_hash(root/n) for n in original_files},'raw_files':raw,'coverage_by_split':dict(coverage),'records':records,'legacy_inputs_identical_count':400,'legacy_numeric_prefix_identical_count':400,'labels_candidates_and_case_groups_unchanged':True,'future_observations_used':False,'cross_split_overlap':0,'feature_spec_sha256':file_hash(out/'feature_spec.json'),'train_saturation':{n:{'metrics':train_total[n],'nonmissing_z':train_valid[n],'clipped_at_100':train_clipped[n]} for n in train_total},'known_onset':True,'note':'Additional summaries from existing telemetry, not independent new cases. Formula fixed before all-partition preparation. No label or split affects summary values.'}
save(out/'data_audit.json',audit)
print('DATA SEALED','legacy matches',len(legacy_hashes),'eligible',164,'new dims',190,570,flush=True)
