"""Fetch a pinned public source and assess compatibility without using model predictions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,urllib.request,urllib.parse
import numpy as np
import pandas as pd
from decisionos_sre.common import read,save,file_hash,ONTOLOGY
from decisionos_sre.temporal import temporal_summary
from decisionos_sre.dynamics import dynamics_summary

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--download-dir',default='data/external/opennet_sockshop')
parser.add_argument('--output-dir',default='outputs/round13')
args=parser.parse_args();root=Path(args.download_dir);out=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
revision='1037771bc3c143f95bc615ec8d5b436b33586704'
if not (root/'tree.json').exists():
 url='https://api.github.com/repos/OpenNetAI/Sock-Shop-Dataset/git/trees/'+revision+'?recursive=1'
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DecisionOS-SRE-data-audit'}),timeout=45) as response:save(root/'tree.json',json.load(response))
tree=read(root/'tree.json')
old=read(root/'download_manifest.json') if (root/'download_manifest.json').exists() else {'files':[]}
rows=[r for r in tree['tree'] if r['type']=='blob' and r['path'].startswith('new-version/test/container-chaos/metric-pod-')]
def fetch(r):
 url='https://raw.githubusercontent.com/OpenNetAI/Sock-Shop-Dataset/'+revision+'/'+urllib.parse.quote(r['path'],safe='/')
 dest=root/r['path'].replace(':','_');dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  with urllib.request.urlopen(url,timeout=45) as response:blob=response.read()
  assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==r['sha']
  dest.write_bytes(blob)
 blob=dest.read_bytes();assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==r['sha']
 return {'source_path':r['path'],'local_path':str(dest),'url':url,'git_blob':r['sha'],'sha256':file_hash(dest),'bytes':len(blob)}
metadata_paths=['README.md','LICENSE','new-version/test/container-chaos/container-chaos-info.csv','new-version/test/node-chaos/node-chaos-info.csv']
with ThreadPoolExecutor(max_workers=3) as pool:
 downloaded=list(pool.map(fetch,rows))
 metadata_files=list(pool.map(fetch,[r for r in tree['tree'] if r['path'] in metadata_paths]))
files={r['source_path']:r for r in old['files']}
files.update({r['source_path']:r for r in downloaded+metadata_files});save(root/'download_manifest.json',{'revision':revision,'files':list(files.values())})
frames={};sampling={};sha_existing={r['sha256'] for r in read('data/round12/manifest.json')['entries']}
for r in downloaded:
 path=Path(r['local_path']);f=pd.read_csv(path)
 times=pd.to_datetime(f.pop('timestamp'),format='%d/%m/%Y %H:%M:%S')
 excluded_metadata=[c for c in ['_id'] if c in f]
 f=f.drop(columns=excluded_metadata)
 # Local wall-clock seconds only. No unsupported timezone is assigned.
 parse_failures={}
 for col in f.columns:
  numeric=pd.to_numeric(f[col],errors='coerce');parse_failures[col]=int((f[col].notna()&numeric.isna()).sum());f[col]=numeric
 f.insert(0,'time',times.astype('int64')/1e9);f=f.sort_values('time')
 pod=path.stem.removeprefix('metric-pod-');frames[pod]=f
 dt=np.diff(f.time.to_numpy());sampling[pod]={'excluded_metadata_columns':excluded_metadata,'unparseable_cells':{c:n for c,n in parse_failures.items() if n},'rows':len(f),'median_positive_interval_seconds':float(np.median(dt[dt>0])),'duplicate_timestamps':int(f.time.duplicated().sum()),'nonmonotonic_named_cpu_counter_steps':int((f['container_cpu_usage_seconds_total'].diff()<0).sum()),'raw_columns':list(f.columns[1:])}
 assert r['sha256'] not in sha_existing
labels=pd.read_csv(root/'new-version/test/container-chaos/container-chaos-info.csv');mapping={'cpu-stress':'cpu_stress','memory-stress':'memory_stress','network-delay':'network_delay'}
events=[]
for i,row in labels.iterrows():
 target=str(row['target']).strip();onset=pd.Timestamp(str(row['timestamp']).strip()).value/1e9
 reasons=[];f=frames.get(target)
 if f is None:reasons.append('target_not_in_telemetry')
 observed=baseline=temporal_count=dynamics_count=0
 if f is not None:
  baseline=int(((f.time>=onset-300)&(f.time<onset)).sum());observed=int(((f.time>=onset)&(f.time<=onset+60)).sum())
  for col in f.columns[1:]:
   temporal_count+=temporal_summary(f,col,onset,onset+60) is not None
   dynamics_count+=dynamics_summary(f,col,onset,onset+60) is not None
 if not temporal_count:reasons.append('no_valid_temporal_summary_in_60s')
 reasons.extend(['metric_semantics_not_equated_to_rcaeval','timezone_unspecified'])
 events.append({'event_id':hashlib.sha256((revision+'|'+str(i)+'|'+str(row['timestamp'])+'|'+target).encode()).hexdigest()[:24],'source_row':int(i),'timestamp_local':str(row['timestamp']).strip(),'fault_raw':row['fault-type'],'fault_mapped':mapping.get(row['fault-type']),'target_pod':target,'candidate_present':f is not None,'duration_raw':row['duration'],'collection_group':str(row['timestamp']).strip()[:10],'baseline_rows':baseline,'observed_rows_60s':observed,'temporal_metrics':temporal_count,'dynamics_metrics':dynamics_count,'admitted_to_current_mvp':False,'reasons':reasons})
# Recheck current official RCAEval index without downloading all telemetry.
url='https://huggingface.co/api/datasets/phamquiluan/RCAEval'
with urllib.request.urlopen(url,timeout=45) as response:info=json.load(response)
rev=info['sha'];idxurl='https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/'+rev+'/cases.parquet'
with urllib.request.urlopen(idxurl,timeout=45) as response:blob=response.read()
idxpath=root/'rcaeval_current_cases.parquet';idxpath.write_bytes(blob);idx=pd.read_parquet(idxpath)
selected=idx[idx.system_name.isin(['Online Boutique','Sock Shop'])&idx.fault.isin(ONTOLOGY)]
existing={e['case'] for e in read('data/round12/manifest.json')['entries']}
new_names=sorted(set(selected.case)-existing)
result={'status':'source_audit_completed','retrieved_utc':datetime.now(timezone.utc).isoformat(),'repository':'https://github.com/OpenNetAI/Sock-Shop-Dataset','revision':revision,'license':'MIT','license_sha256':file_hash(root/'LICENSE'),'source_events':len(events),'fault_counts':dict(Counter(e['fault_mapped'] for e in events)),'pod_files':len(downloaded),'download_bytes':sum(r['bytes'] for r in downloaded),'sampling':sampling,'raw_file_hash_overlap_with_rcaeval':0,'events':events,'admitted_new_cases':0,'new_independent_evaluation_cases':0,'existing_split_modified':False,'predictions_executed':False,'model_selection_used':False,'blockers':['Minute-level samples cannot provide >=3 real observations plus early/late coverage within the existing 60-second window.','Public raw metric names/semantics differ; cannot silently reinterpret counters or pod targets as the existing service metrics.','No disk or packet-loss events in this source; absolute timestamp timezone and deeper collection lineage are unspecified.'],'next_data_requirement':'Request or collect raw samples <=5 seconds, documented metric units/rates, pod-to-service metadata, precise injection timing/timezone, clean 300s baseline and independent runs; keep original 60s window.','rcaeval_index_recheck':{'revision':rev,'sha256':file_hash(idxpath),'rows':len(idx),'same_application_five_fault_cases':len(selected),'new_case_names':new_names},'download_manifest_sha256':file_hash(root/'download_manifest.json')}
save(out/'external_source_manifest.json',{'revision':revision,'files':list(files.values())});save(out/'external_data_audit.json',result);save(root/'candidate_events.json',{'status':'quarantined_not_training_data','events':events})
print('EXTERNAL AUDIT',len(events),'events',len(downloaded),'pod files; admitted 0; RCAEval new',len(new_names),flush=True)
