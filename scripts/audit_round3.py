"""Audit same-application expansion and seal a new RE2 holdout before modeling."""
import shutil,random,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import pandas as pd
import numpy as np
from decisionos_sre.common import DATA_REV,ONTOLOGY,read,save,digest,file_hash
from decisionos_sre.data import BASE,download,inspect_frame,prepare
root=Path('data/round3');root.mkdir(parents=True,exist_ok=True)
if (root/'splits.json').exists():raise FileExistsError('Round3 split already sealed')
idx=pd.read_parquet('data/rcaeval/cases.parquet')
new=idx[idx.dataset.eq('RE2-OB') & idx.fault.isin(ONTOLOGY)].sort_values('case').to_dict('records')
old=read('data/rcaeval/manifest.json');oldsplit=read('data/rcaeval/splits.json')
for e in old['entries']:
    folder=root/e['case'];folder.mkdir(exist_ok=True)
    for fn in ('metrics.parquet','inject_time.txt'):shutil.copy2(Path('data/rcaeval')/e['case']/fn,folder/fn)
def fetch(r):
    for fn in ('metrics.parquet','inject_time.txt'):download(BASE+r['case']+'/'+fn,root/r['case']/fn)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,new))
entries=[]
for r in idx[idx.dataset.isin(['RE1-OB','RE2-OB']) & idx.fault.isin(ONTOLOGY)].sort_values('case').to_dict('records'):
    case=r['case'];df=pd.read_parquet(root/case/'metrics.parquet').sort_values('time')
    onset=int((root/case/'inject_time.txt').read_text().strip());assert onset==r['inject_time']
    assert df.time.notna().all() and not df.time.duplicated().any()
    cols=sorted(c for c in df.columns if c!='time')
    b=df[(df.time>=onset-300)&(df.time<onset)][cols]
    o=df[(df.time>=onset)&(df.time<=onset+60)][cols]
    def sig(frame,rounded=False):
        arr=frame.to_numpy(dtype='float64')
        if rounded:arr=np.round(arr/np.maximum(np.nanmean(np.abs(arr),axis=0),1e-9),3)
        return hashlib.sha256('|'.join(frame.columns).encode()+arr.tobytes()).hexdigest()
    signs={'telemetry':sig(df[cols]),'baseline':sig(b),'near_window':sig(pd.concat([b.tail(60),o]),True)}
    oid=digest({'source':DATA_REV,'case':case})[:24]
    services=sorted({c.rsplit('_',1)[0] for c in cols})
    entries.append({'opaque_incident_id':oid,'case':case,'application':r['system_name'],'suite':r['suite'],
        'onset':onset,'decision_time':onset+60,'oracle_onset':True,'root_raw':r['root_cause_service'],'fault_raw':r['fault'],
        'repetition':int(r['repetition']),'candidate_ids':services,'candidate_covered':r['root_cause_service'] in services,
        'metrics':inspect_frame(df),'logs':{'source_present':bool(r['has_logs']),'used':False},
        'traces':{'source_present':bool(r['has_traces']),'used':False},'sha256':file_hash(root/case/'metrics.parquet'),
        'onset_sha256':file_hash(root/case/'inject_time.txt'),'signatures':signs,'label_provenance':'official_injection_metadata'})
# Conservatively cluster copies across BOTH suites. Abort if they violate old partitions.
parent=list(range(len(entries)));seen={}
def find(i):
    while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
    return i
for i,e in enumerate(entries):
    for kind,sign in e['signatures'].items():
        key=(kind,sign)
        if key in seen:parent[find(i)]=find(seen[key])
        else:seen[key]=i
clusters={}
for i,e in enumerate(entries):clusters.setdefault(find(i),[]).append(e)
assignments={oid:('regression' if s=='test' else s) for oid,s in oldsplit['assignments'].items()}
rng=random.Random(20261007)
# Each new service/fault triplet: one gate, one test, one development run.
new_entries=[e for e in entries if e['suite']=='RE2']
for fault in sorted(ONTOLOGY):
    services=sorted({e['root_raw'] for e in new_entries if e['fault_raw']==fault});rng.shuffle(services)
    # Two schema-inspection pilot cases remain TRAIN, never new test.
    if fault in ('cpu','loss'):
        services.remove('checkoutservice');services.insert(0,'checkoutservice')
    for j,service in enumerate(services):
        group=sorted([e for e in new_entries if e['fault_raw']==fault and e['root_raw']==service],key=lambda e:e['case']);rng.shuffle(group)
        if fault in ('cpu','loss') and service=='checkoutservice':
            pilot=next(e for e in group if e['repetition']==1);group.remove(pilot);group.insert(0,pilot)
        dev='train' if j<3 else ('model_validation' if j==3 else 'calibration')
        for e,split in zip(group,(dev,'gate_selection','test')):assignments[e['opaque_incident_id']]=split
for cluster in clusters.values():
    parts={assignments[e['opaque_incident_id']] for e in cluster}
    if len(parts)>1:raise ValueError('Duplicate/shared telemetry crosses assigned partitions; resolve grouping before training')
    gid=next((oldsplit['groups'][e['opaque_incident_id']] for e in cluster if e['opaque_incident_id'] in oldsplit['groups']),min(e['opaque_incident_id'] for e in cluster))
    for e in cluster:e['group_id']=gid
assert len(entries)==200 and len(new_entries)==75
manifest={'source':BASE,'revision':DATA_REV,'license':'MIT','index_sha256':file_hash('data/rcaeval/cases.parquet'),
 'subset':['RE1-OB','RE2-OB five shared faults'],'collection':'controlled fault injection','runs':len(entries),
 'unique_telemetry_groups':len(clusters),'derived_windows':200,'augmentations':0,'entries':entries,'mapping':ONTOLOGY,
 'mapping_version':'rcaeval-re1-v1','new_cases':75,'excluded_socket':15,'unknown_faults':0,
 'candidate_misses':sum(not e['candidate_covered'] for e in entries),'modalities_used':['metrics'],
 'limitations':['same application, distinct benchmark suite and new telemetry schemas; not a cross-system claim',
                'collection lineage beyond case metadata unavailable; grouping does not prove statistical independence',
                'logs/traces are present for RE2 but intentionally excluded to keep metrics-only MVP',
                'oracle injection onset; no fault detection evaluation','source units unspecified, no conversion']}
manifest['manifest_hash']=digest(manifest);save(root/'manifest.json',manifest)
split={'seed':20261007,'protocol':'preserve old partitions; stratified fresh RE2 gate/test triplets; old test becomes regression',
 'manifest_hash':manifest['manifest_hash'],'assignments':assignments,'groups':{e['opaque_incident_id']:e['group_id'] for e in entries},
 'counts':dict(Counter(assignments.values())),'parent_split_hash':oldsplit['split_hash']}
split['split_hash']=digest(split);save(root/'splits.json',split)
examples=prepare(root)
byid={e['opaque_incident_id']:e for e in entries}
for e in examples:e['source_metadata']['dataset_suite']=byid[e['opaque_incident_id']]['suite']
save(root/'examples.json',examples)
save('outputs/round3/data_audit.json',{k:v for k,v in manifest.items() if k!='entries'})
save('outputs/round3/splits.json',split)
save('outputs/round3/data_integrity.json',{'manifest_sha256':file_hash(root/'manifest.json'),'splits_sha256':file_hash(root/'splits.json'),'examples_sha256':file_hash(root/'examples.json')})
print('AUDITED',len(entries),'groups',len(clusters),'counts',split['counts'],'candidate misses',manifest['candidate_misses'],flush=True)
