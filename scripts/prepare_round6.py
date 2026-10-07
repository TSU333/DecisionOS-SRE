"""Audit new trace evidence on immutable historical case partitions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import shutil,hashlib
import pandas as pd
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash,digest
from decisionos_sre.data import download,BASE
from decisionos_sre.traces import summarize_traces
from decisionos_sre.schema import TrainingExample
from decisionos_sre.serializer import Serializer

out=Path('outputs/round6');root=Path('data/round6');raw=root/'raw_traces';oldroot=Path('data/round5')
if (root/'examples.json').exists():raise FileExistsError('Data already sealed')
manifest=read(oldroot/'manifest.json');splits=read(oldroot/'splits.json');examples=read(oldroot/'examples.json');byid={e['opaque_incident_id']:e for e in examples}
entries=[e for e in manifest['entries'] if e['suite']=='RE2' and e['application']=='Online Boutique'];assert len(entries)==75
raw.mkdir(parents=True,exist_ok=True)
def fetch(e):
    path=raw/e['case']/'traces.parquet';probe=Path('data/source_audit/trace_probe')/e['case']/'traces.parquet'
    if not path.exists() and probe.exists():path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(probe,path)
    download(BASE+e['case']+'/traces.parquet',path);return e,path
with ThreadPoolExecutor(max_workers=4) as pool:
    for i,f in enumerate(as_completed([pool.submit(fetch,e) for e in entries])):
        f.result()
        if (i+1)%10==0:print('TRACE DOWNLOAD',i+1,'/75',flush=True)
audit=[];seen={}
for i,e in enumerate(entries):
    path=raw/e['case']/'traces.parquet';sha=file_hash(path)
    df=pd.read_parquet(path,columns=['traceID','spanID','serviceName','operationName','startTime','startTimeMillis','duration','statusCode'])
    assert ((df.startTime//1000)==df.startTimeMillis).all(),'inconsistent microsecond timestamps'
    ex=byid[e['opaque_incident_id']];traces,info=summarize_traces(df,e['onset'],e['decision_time'],e['candidate_ids'])
    ex['input']['evidence']['traces']=[t.model_dump() for t in traces] or None
    ex['input']['modality_availability']['traces']=bool(traces)
    TrainingExample.model_validate(ex)
    role=ex['source_metadata']['split']
    if sha in seen and seen[sha]!=role:raise ValueError('Identical trace file crosses partitions')
    seen[sha]=role
    audit.append({'id':e['opaque_incident_id'],'case':e['case'],'split':role,'sha256':sha,'bytes':path.stat().st_size,'summary_count':len(traces),**info})
    if (i+1)%15==0:print('TRACE PREPARED',i+1,'/75',flush=True)
# Legacy input equivalence and trace-budget audit use TRAIN/validation only.
cfg=read('configs/round5_temporal_seed44.json');tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
oldser=Serializer(tok,2048,'metrics-canonical-v2',cfg['numeric_metrics'],'temporal-v1')
newser=Serializer(tok,2048,'metrics-canonical-v2',cfg['numeric_metrics'],'temporal-v1',True)
original={e['opaque_incident_id']:e for e in read(oldroot/'examples.json')};retention=[];checked=0
for rawex in examples:
    if rawex['source_metadata']['split'] not in ['train','model_validation']:continue
    ex=TrainingExample.model_validate(rawex);a=oldser(ex.input);b=oldser(TrainingExample.model_validate(original[ex.opaque_incident_id]).input)
    assert (a.input_ids,a.candidate_ids,a.candidate_numeric,a.incident_numeric)==(b.input_ids,b.candidate_ids,b.candidate_numeric,b.incident_numeric);checked+=1
    new=newser(ex.input)
    retention.append({'id':ex.opaque_incident_id,'split':ex.source_metadata['split'],'cohort':ex.source_metadata['dataset_suite'],'tokens':len(new.input_ids),'trace_total':new.report['trace_summaries_total'],'trace_retained':new.report['trace_summaries_retained'],'metrics_retained_fraction':new.report['retained_fraction']})
manifest['trace_evidence']={'revision':manifest['revision'],'files':audit,'derived_summary_version':'service-trace-v1','source':'Jaeger startTime/duration microseconds; converted latency to ms','no_new_independent_cases':True};manifest.pop('manifest_hash');manifest['manifest_hash']=digest(manifest)
splits.pop('split_hash');splits['parent_split_hash']=read(oldroot/'splits.json')['split_hash'];splits['manifest_hash']=manifest['manifest_hash'];splits['protocol']='All prior partitions unchanged; only eligible completed trace evidence added';splits['split_hash']=digest(splits)
save(root/'manifest.json',manifest);save(root/'splits.json',splits);save(root/'examples.json',examples)
save(out/'data_audit.json',{'cases':400,'new_cases':0,'trace_cases':75,'missing_trace_cases':325,'counts':splits['counts'],'legacy_input_equivalence':checked,'train_validation_retention':retention,'trace_files':audit,'total_trace_bytes':sum(a['bytes'] for a in audit),'candidate_coverage':1.,'split_assignments_unchanged':splits['assignments']==read(oldroot/'splits.json')['assignments'],'integrity':{n:file_hash(root/n) for n in ['manifest.json','splits.json','examples.json']}})
print('SEALED',splits['counts'],'trace bytes',sum(a['bytes'] for a in audit),'legacy input matches',checked,flush=True)
