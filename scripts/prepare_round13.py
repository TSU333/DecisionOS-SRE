"""Audit TRAIN and freeze grouped development folds before any training."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import shutil
import numpy as np
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash,digest,MODEL_REV,FAULTS
from decisionos_sre.data import load_split
from decisionos_sre.serializer import Serializer
from decisionos_sre.grouped_validation import make_folds,assert_fresh_config

out=Path('outputs/round13')
if (out/'protocol.json').exists():raise FileExistsError('Round13 protocol already sealed')
meta=read('artifacts/round5/temporal_seed44/frozen/metadata.json');cfg=meta['config'].copy()
for key in ['initialization_artifact','numeric_feature_upgrade']:cfg.pop(key,None)
cfg.update(data_dir='data/round12',head_training_policy='all_heads',head_learning_rate=.0003,learning_rate_frozen=.0003,epochs=160,max_steps=1000,early_stopping_patience=25,batch_size=16,gradient_accumulation=1,loss_weights=[1.,1.],fault_hidden_dropout=.15,fault_label_smoothing=0.,training_view_probability=0.,sampling_strategy='cohort_balanced',trace_features=False)
assert_fresh_config(cfg)
train=load_split(cfg['data_dir'],'train');splits=read('data/round12/splits.json');manifest=read('data/round12/manifest.json');entry={e['opaque_incident_id']:e for e in manifest['entries']}
tok=AutoTokenizer.from_pretrained('artifacts/backbone',local_files_only=True)
ser=Serializer(tok,2048,cfg['serializer_version'],cfg['numeric_metrics'],'temporal-v1')
records=[];excluded=[];issues=[]
for e in train:
 en=ser(e.input);m=entry[e.opaque_incident_id]
 assert e.original_run_id==splits['groups'][e.opaque_incident_id]
 assert e.targets.fault_type.value=={'cpu':'cpu_stress','mem':'memory_stress','disk':'disk_io_stress','delay':'network_delay','loss':'network_packet_loss'}[m['fault_raw']]
 assert e.targets.root_cause.value==m['root_raw']
 assert file_hash(Path('data/round4')/m['case']/'metrics.parquet')==m['sha256']
 present=e.targets.root_cause.value in en.candidate_ids
 root_metrics=[x for x in e.input.evidence.metrics if x.service==e.targets.root_cause.value]
 usable=en.report['numeric_evidence_usable']
 r={'run_id':e.original_run_id,'incident_id':e.opaque_incident_id,'group_id':splits['groups'][e.opaque_incident_id],'split':'train','application':e.input.application,'cohort':e.source_metadata['dataset_suite'],'fault':e.targets.fault_type.value,'root_candidate_present':present,'root_metric_count':len(root_metrics),'root_temporal_count':sum(x.temporal is not None for x in root_metrics),'usable':usable,'tokens':len(en.input_ids)}
 if not usable:excluded.append(r)
 else:records.append(r)
 if not present or not root_metrics or not usable:issues.append(r)
assert len(records)==164 and len(excluded)==1 and not set(r['run_id'] for r in records)&set(r['run_id'] for r in excluded)
folds=make_folds(records)
for f in folds:
 f['counts']={role:dict(Counter(next(r['cohort'] for r in records if r['run_id']==x) for x in f[role])) for role in ['fit','stop','outer']}
save(out/'folds.json',{'records':records,'folds':folds,'split_seed':130,'group_definition':'existing original-run / conservative duplicate group','inner_stopping_partition':True,'global_holdouts_used':False})
audit={'original_cases':400,'original_train_cases':165,'eligible_train_cases':164,'new_independent_cases':0,'allowed_train_run_ids':[r['run_id'] for r in records],'excluded':excluded,'issues':issues,'label_disagreements_with_source':0,'candidate_misses':sum(not r['root_candidate_present'] for r in records),'train_group_overlap_with_other_splits':0,'strata':dict(Counter(r['application']+'|'+r['fault'] for r in records)),'files':{str(p):file_hash(p) for p in [Path('data/round12')/n for n in ['manifest.json','splits.json','examples.json']]},'record_audit':records,'feature_dictionary':'predeclared fixed schema from earlier development, not refitted during CV','label_changes':0}
save(out/'data_audit.json',audit)
backbone=read('artifacts/backbone/manifest.json');assert backbone['revision']==MODEL_REV
for name,sha in backbone['files'].items():assert file_hash(Path('artifacts/backbone')/name)==sha
save('configs/round13_cv.json',cfg)
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'execution_scope':'mvp','config':'configs/round13_cv.json','config_sha256':file_hash('configs/round13_cv.json'),'seeds':[50,51,52],'variants':['temporal','dynamics'],'outer_folds':3,'runs':18,'backbone':backbone,'data_audit_sha256':file_hash(out/'data_audit.json'),'folds_sha256':file_hash(out/'folds.json'),'maximum_updates_per_run':1000,'maximum_total_updates':18000,'inner_selection':'cohort macro joint, joint, sum NLL; inner stop only; patience25','outer_evaluation':'exactly once per trained checkpoint after inner best restore, no outer-based epoch selection','variant_comparison':'mean and sample SD of 3 seed-level OOF metrics, paired folds and consistent failures; no independent significance claim','warm_start':'none; original public pretrained encoder only, fresh heads; zero-pad new dynamics columns for paired initialization','normalization':'fixed per-incident causal summaries; no cross-run fitted preprocessing','promotion':'This round is a TRAIN-only development audit. No production promotion from CV. Existing default, global validation, calibration, gate and historical regressions remain unchanged and unopened for new model evaluation.','source_sampling':'new external source audit and readiness only; no external target-based model tuning','new_independent_cv_cases':0}
save(out/'protocol.json',protocol);shutil.copyfile('outputs/latest_model.json',out/'previous_latest_model.json')
print('SEALED',len(records),'TRAIN','fold sizes',[(len(f['fit']),len(f['stop']),len(f['outer'])) for f in folds],flush=True)
