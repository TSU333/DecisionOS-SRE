"""Add causal temporal summaries while preserving all immutable prior partitions."""
from pathlib import Path
from collections import Counter,defaultdict
import pandas as pd
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,digest,file_hash
from decisionos_sre.temporal import temporal_summary
from decisionos_sre.schema import TrainingExample
from decisionos_sre.serializer import Serializer

root=Path('data/round5');out=Path('outputs/round5')
if (root/'examples.json').exists():raise FileExistsError('Prepared data already sealed')
oldroot=Path('data/round4');manifest=read(oldroot/'manifest.json');oldsplit=read(oldroot/'splits.json')
assert digest({k:v for k,v in manifest.items() if k!='manifest_hash'})==manifest['manifest_hash']
assert digest({k:v for k,v in oldsplit.items() if k!='split_hash'})==oldsplit['split_hash']
examples=read(oldroot/'examples.json');byid={e['opaque_incident_id']:e for e in examples}
summary=defaultdict(Counter);raw=[]
for n,entry in enumerate(manifest['entries']):
    path=oldroot/entry['case']/'metrics.parquet'
    assert file_hash(path)==entry['sha256']
    frame=pd.read_parquet(path).sort_values('time')
    ex=byid[entry['opaque_incident_id']];cohort=ex['source_metadata']['dataset_suite']
    for metric in ex['input']['evidence']['metrics']:
        t=temporal_summary(frame,metric['service']+'_'+metric['name'],entry['onset'],entry['decision_time'])
        metric['temporal']=t.model_dump() if t else None
        summary[cohort]['metrics']+=1;summary[cohort]['with_temporal']+=int(t is not None)
    if ex['source_metadata']['split']=='test':ex['source_metadata']['split']='regression_ss'
    TrainingExample.model_validate(ex)
    raw.append({'case':entry['case'],'path':str(path),'sha256':entry['sha256']})
    if (n+1)%50==0:print('temporal prepared',n+1,flush=True)
split={k:v for k,v in oldsplit.items() if k!='split_hash'}
split.update(protocol='No new data; old SS test becomes regression_ss; all other partitions preserved',parent_split_hash=oldsplit['split_hash'])
split['assignments']={oid:('regression_ss' if role=='test' else role) for oid,role in oldsplit['assignments'].items()}
split['counts']=dict(Counter(split['assignments'].values()));split['split_hash']=digest(split)
assert len({e['original_run_id'] for e in examples})==400
assert sorted({m['name'] for e in examples if e['source_metadata']['split']=='train' for m in e['input']['evidence']['metrics']})==read('configs/round4_balanced_seed43.json')['numeric_metrics']
cfg=read('configs/round4_balanced_seed43.json');tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
ser=Serializer(tok,2048,'metrics-canonical-v2',cfg['numeric_metrics'])
golden=read(out/'legacy_representation_hashes.json');checked=0
for ex in examples:
    if ex['original_run_id'] not in golden:continue
    x=ser(TrainingExample.model_validate(ex).input)
    actual=digest({'ids':x.input_ids,'candidates':x.candidate_ids,'spans':x.spans,'numeric':x.candidate_numeric,'global':x.incident_numeric})
    assert actual==golden[ex['original_run_id']];checked+=1
save(root/'manifest.json',manifest);save(root/'splits.json',split);save(root/'examples.json',examples)
save(out/'data_audit.json',{'cases':400,'new_cases':0,'unique_case_groups':len(set(split['groups'].values())),'counts':split['counts'],'temporal_coverage_by_cohort':dict(summary),'legacy_representation_matches':checked,'legacy_hashes':file_hash(out/'legacy_representation_hashes.json'),'source_raw_files':raw,'candidate_coverage':1-manifest['candidate_misses']/400,'parent_integrity':{n:file_hash(oldroot/n) for n in ['manifest.json','splits.json','examples.json']},'current_integrity':{n:file_hash(root/n) for n in ['manifest.json','splits.json','examples.json']},'fresh_test':False,'future_observations_used':False,'temporal_version':'temporal-v1','note':'Same injected cases with known onset; repeated validation and gate reuse. No claim of independent collection lineage.'})
print('DATA SEALED',split['counts'],'legacy matches',checked,dict(summary),flush=True)
