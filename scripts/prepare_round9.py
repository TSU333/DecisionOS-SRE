"""Seal TRAIN-only causal views without rewriting any original split or input."""
from pathlib import Path
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash,digest
from decisionos_sre.data import load_split
from decisionos_sre.training_views import causal_view,VERSION
from decisionos_sre.serializer import Serializer

out=Path('outputs/round9');root=Path('data/round9');dest=root/'training_views.json'
if dest.exists():raise FileExistsError('Views already sealed')
parent=read('outputs/latest_model.json')['artifact'];meta=read(Path(parent)/'metadata.json');cfg=meta['config']
assert file_hash(Path(cfg['data_dir'])/'examples.json')==meta['prepared_data_sha256']
manifest=read(Path(cfg['data_dir'])/'manifest.json');split=read(Path(cfg['data_dir'])/'splits.json')
entries={e['opaque_incident_id']:e for e in manifest['entries']};train=load_split(cfg['data_dir'],'train')
tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
ser=Serializer(tok,2048,cfg['serializer_version'],cfg['numeric_metrics'],cfg['numeric_feature_version'])
views=[];records=[];excluded=[];raw=[]
for ex in train:
    entry=entries[ex.opaque_incident_id];path=Path('data/round4')/entry['case']/'metrics.parquet'
    assert file_hash(path)==entry['sha256'];raw.append({'run_id':ex.original_run_id,'path':str(path),'sha256':entry['sha256']})
    assert split['assignments'][ex.opaque_incident_id]=='train'
    enc=ser(ex.input)
    if not enc.report['modality_evidence_usable']:
        excluded.append({'run_id':ex.original_run_id,'cohort':ex.source_metadata['dataset_suite'],'reason':'no retained usable temporal evidence'})
        continue
    frame=pd.read_parquet(path).sort_values('time')
    for kind in ['lag15','gap15_30']:
        view=causal_view(ex,frame,entry['onset'],kind);encoded=ser(view.input)
        usable=encoded.report['modality_evidence_usable']
        records.append({'parent_run':ex.original_run_id,'view_id':view.opaque_incident_id,'kind':kind,'usable':usable,'tokens':encoded.report['tokens'],'metrics_retained':encoded.report['metrics_retained'],'cohort':ex.source_metadata['dataset_suite']})
        if usable:views.append(view.model_dump())
save(dest,views)
audit={'version':VERSION,'original_cases':400,'new_independent_cases':0,'allowed_train_cases':len(train),'eligible_train_cases':len(train)-len(excluded),'excluded_unusable_train':excluded,'training_view_count':len(views),'view_records':records,'raw_train_files':raw,'view_sha256':file_hash(dest),'case_groups_unchanged':True,'original_split_counts':split['counts'],'original_files':{n:file_hash(Path(cfg['data_dir'])/n) for n in ['manifest.json','splits.json','examples.json']},'views_by_cohort':dict(Counter(e['source_metadata']['dataset_suite'] for e in views)),'holdout_views_created':0,'selection_or_calibration_inputs_changed':False,'note':'Missing telemetry views from existing TRAIN runs, not new independent evidence. Labels and parent/run groups preserved. The excluded run remains in the original split and in parent-model history.'}
save(out/'data_audit.json',audit)
print('SEALED',len(views),'views from',audit['eligible_train_cases'],'eligible TRAIN cases; excluded',excluded,flush=True)
