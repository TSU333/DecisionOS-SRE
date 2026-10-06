"""Extend TRAIN data within the five-fault MVP and seal fresh Sock Shop holdouts."""
import hashlib,random,shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from decisionos_sre.common import DATA_REV,ONTOLOGY,read,save,digest,file_hash
from decisionos_sre.data import BASE,download,inspect_frame,prepare

root=Path('data/round4');root.mkdir(parents=True,exist_ok=True)
if (root/'splits.json').exists():raise FileExistsError('Round4 split already sealed')
old=read('data/round3/manifest.json');oldsplit=read('data/round3/splits.json')
idx=pd.read_parquet('data/rcaeval/cases.parquet')
new=idx[idx.dataset.isin(['RE1-SS','RE2-SS']) & idx.fault.isin(ONTOLOGY)].sort_values('case').to_dict('records')
assignments={oid:('regression_re2_ob' if split=='test' else ('regression_re1_ob' if split=='regression' else split)) for oid,split in oldsplit['assignments'].items()}
rng=random.Random(20261008)
# Assign raw cases before fetching/inspecting telemetry. Per service/fault: RE1 three
# training runs, RE2 one training run. Remaining raw runs are stratified by fault/suite.
for suite in ['RE1','RE2']:
    for fault in sorted(ONTOLOGY):
        pool=[]
        services=sorted({r['root_cause_service'] for r in new if r['suite']==suite and r['fault']==fault})
        for service in services:
            group=sorted([r for r in new if r['suite']==suite and r['fault']==fault and r['root_cause_service']==service],key=lambda r:r['case']);rng.shuffle(group)
            count=3 if suite=='RE1' else 1
            for r in group[:count]:assignments[digest({'source':DATA_REV,'case':r['case']})[:24]]='train'
            pool.extend(group[count:])
        rng.shuffle(pool)
        roles=['model_validation']*2+['calibration']*2+['gate_selection']*3+['test']*3
        assert len(pool)==len(roles)==10
        for r,role in zip(pool,roles):assignments[digest({'source':DATA_REV,'case':r['case']})[:24]]=role
save('outputs/round4/pre_download_assignments.json',{'seed':20261008,'assignments':assignments,'new_cases':len(new),'counts':dict(Counter(assignments.values()))})
for e in old['entries']:
    folder=root/e['case'];folder.mkdir(exist_ok=True)
    for name in ['metrics.parquet','inject_time.txt']:shutil.copy2(Path('data/round3')/e['case']/name,folder/name)
def fetch(r):
    for name in ['metrics.parquet','inject_time.txt']:download(BASE+r['case']+'/'+name,root/r['case']/name)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,new))
entries=list(old['entries'])
for r in new:
    case=r['case'];df=pd.read_parquet(root/case/'metrics.parquet').sort_values('time')
    onset=int((root/case/'inject_time.txt').read_text().strip());assert onset==r['inject_time']
    assert df.time.notna().all() and not df.time.duplicated().any()
    cols=sorted(c for c in df.columns if c!='time');b=df[(df.time>=onset-300)&(df.time<onset)][cols];o=df[(df.time>=onset)&(df.time<=onset+60)][cols]
    def sig(frame,rounded=False):
        arr=frame.to_numpy(dtype='float64')
        if rounded:
            counts=np.isfinite(arr).sum(0)
            means=np.divide(np.nansum(np.abs(arr),axis=0),counts,out=np.full(arr.shape[1],np.nan),where=counts>0)
            arr=np.round(arr/np.maximum(means,1e-9),3)
        return hashlib.sha256('|'.join(frame.columns).encode()+arr.tobytes()).hexdigest()
    signs={'telemetry':sig(df[cols]),'baseline':sig(b),'near_window':sig(pd.concat([b.tail(60),o]),True)}
    oid=digest({'source':DATA_REV,'case':case})[:24];services=sorted({c.rsplit('_',1)[0] for c in cols})
    entries.append({'opaque_incident_id':oid,'case':case,'application':r['system_name'],'suite':r['suite'],'onset':onset,'decision_time':onset+60,'oracle_onset':True,'root_raw':r['root_cause_service'],'fault_raw':r['fault'],'repetition':int(r['repetition']),'candidate_ids':services,'candidate_covered':r['root_cause_service'] in services,'metrics':inspect_frame(df),'logs':{'source_present':bool(r['has_logs']),'used':False},'traces':{'source_present':bool(r['has_traces']),'used':False},'sha256':file_hash(root/case/'metrics.parquet'),'onset_sha256':file_hash(root/case/'inject_time.txt'),'signatures':signs,'label_provenance':'official_injection_metadata'})
parent=list(range(len(entries)));seen={}
def find(i):
    while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
    return i
for i,e in enumerate(entries):
    for kind,value in e['signatures'].items():
        if (kind,value) in seen:parent[find(i)]=find(seen[kind,value])
        else:seen[kind,value]=i
clusters={}
for i,e in enumerate(entries):clusters.setdefault(find(i),[]).append(e)
for group in clusters.values():
    if len({assignments[e['opaque_incident_id']] for e in group})>1:raise ValueError('Duplicate telemetry crosses planned partitions; stop before training')
    gid=next((oldsplit['groups'][e['opaque_incident_id']] for e in group if e['opaque_incident_id'] in oldsplit['groups']),min(e['opaque_incident_id'] for e in group))
    for e in group:e['group_id']=gid
assert len(entries)==400 and len(new)==200
manifest={'source':BASE,'revision':DATA_REV,'license':'MIT','index_sha256':file_hash('data/rcaeval/cases.parquet'),'subset':['RE1-OB','RE2-OB five shared faults','RE1-SS','RE2-SS five shared faults'],'collection':'controlled fault injection','runs':len(entries),'unique_telemetry_groups':len(clusters),'derived_windows':len(entries),'augmentations':0,'entries':entries,'mapping':ONTOLOGY,'mapping_version':'rcaeval-re1-v1','new_cases':200,'candidate_misses':sum(not e['candidate_covered'] for e in entries),'modalities_used':['metrics'],'limitations':['within-trained-application evaluation, not unseen-system generalization','old Online Boutique tests are regression only; new Sock Shop test cannot establish fresh Online Boutique target achievement','collection lineage beyond case metadata/fingerprints unproven','oracle injection onset; no detector evaluation','source units unspecified; no conversion']}
manifest['manifest_hash']=digest(manifest);save(root/'manifest.json',manifest)
split={'seed':20261008,'protocol':'preserve old partitions; both previous test cohorts become regression; new raw Sock Shop runs assigned before download','manifest_hash':manifest['manifest_hash'],'assignments':assignments,'groups':{e['opaque_incident_id']:e['group_id'] for e in entries},'counts':dict(Counter(assignments.values())),'parent_split_hash':oldsplit['split_hash']};split['split_hash']=digest(split);save(root/'splits.json',split)
examples=prepare(root);byid={e['opaque_incident_id']:e for e in entries}
for e in examples:
    entry=byid[e['opaque_incident_id']];suffix='OB' if entry['application']=='Online Boutique' else 'SS'
    e['source_metadata']['dataset_suite']=entry['suite']+'-'+suffix
save(root/'examples.json',examples)
save('outputs/round4/data_audit.json',{k:v for k,v in manifest.items() if k!='entries'});save('outputs/round4/splits.json',split)
save('outputs/round4/data_integrity.json',{'manifest_sha256':file_hash(root/'manifest.json'),'splits_sha256':file_hash(root/'splits.json'),'examples_sha256':file_hash(root/'examples.json')})
print('AUDITED',len(entries),'groups',len(clusters),'counts',split['counts'],'candidate misses',manifest['candidate_misses'],flush=True)
print('TRAIN_METRIC_NAMES',sorted({m['name'] for e in examples if e['source_metadata']['split']=='train' for m in e['input']['evidence']['metrics']}),flush=True)
