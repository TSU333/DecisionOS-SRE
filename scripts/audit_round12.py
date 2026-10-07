"""Verify source snapshot, real optimizer draws, migration and exported weights."""
from pathlib import Path
from collections import Counter
import gc,hashlib
import numpy as np
import torch
from dulwich.repo import Repo
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.data import load_split
from decisionos_sre.training import load_checkpoint,predict,initialize_from_artifact
from decisionos_sre.serializer import Serializer
from decisionos_sre.model import build_model
from decisionos_sre.representation import numeric_dimension

out=Path('outputs/round12');sel=read(out/'selection.json');protocol=read(out/'protocol.json');audit=read(out/'data_audit.json');source=read(out/'pretraining_source.json')
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(out/'data_audit.json')==protocol['data_audit_sha256']
assert file_hash(out/'feature_spec.json')==protocol['feature_spec_sha256']
assert read(out/'feature_spec.json')['source_sha256']==file_hash('src/decisionos_sre/dynamics.py')
for name,sha in audit['original_files'].items():assert file_hash(Path('data/round5')/name)==sha
for name,sha in audit['current_files'].items():assert file_hash(Path('data/round12')/name)==sha
for row in audit['raw_files']:assert file_hash(row['path'])==row['sha256']
parent=torch.load(Path(protocol['parent'])/'checkpoint.pt',map_location='cpu',weights_only=True)
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
train=load_split('data/round12','train');validation=load_split('data/round12','model_validation')
byid={e.original_run_id:e.source_metadata['dataset_suite'] for e in train};opaque={e.original_run_id:e.opaque_incident_id for e in train}
excluded={e['run_id'] for e in audit['excluded_unusable_train']};repo=Repo('.');records=[];draw_records=[]
for trial,c in zip(protocol['trials'],sel['candidates']):
 folder=Path(c['artifact']);m=read(folder/'metadata.json');cfg=m['config']
 assert file_hash(trial['config'])==trial['config_sha256'] and read(trial['config'])==cfg
 assert file_hash(folder/'checkpoint.pt')==m['binding']['checkpoint_sha256']
 assert m['prepared_data_sha256']==audit['current_files']['examples.json']
 assert m['code_state']['revision']==source['revision'] and m['code_state']['source_hashes']==source['source_hashes']
 head=repo[m['code_state']['revision'].encode()]
 for p,sha in m['code_state']['source_hashes'].items():
  assert file_hash(p)==sha
  _,oid=repo[head.tree].lookup_path(repo.__getitem__,p.replace(chr(92),'/').encode());assert hashlib.sha256(repo[oid].data).hexdigest()==sha
 assert set(m['train_run_ids'])==set(byid) and set(m['validation_run_ids'])=={e.original_run_id for e in validation}
 assert not set(m['train_run_ids'])&set(m['validation_run_ids'])
 assert len(m['eligible_train_run_ids'])==164 and set(m['excluded_unusable_train_run_ids'])==excluded
 assert m['cache']['precompute_encoder_calls']==m['cache']['expected']==205
 assert m['cache']['validation_full_logit_max_diff']<=1e-5
 if cfg['head_training_policy']=='fault_heads':assert m['root_validation_max_abs_logit_change']==0.
 state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);assert set(state)==set(parent)
 frozen=[n for n in state if n not in m['trainable_parameter_names']];changed=[];padded=[];new_columns=[]
 for n,tensor in state.items():
  ref=parent[n]
  if tensor.shape!=ref.shape:
   assert cfg['numeric_feature_version']=='temporal-dynamics-v1' and n in ['numeric_root.0.weight','numeric_fault.0.weight','numeric_local_fault.0.weight']
   extra=210 if n=='numeric_fault.0.weight' else 70
   assert tensor.shape==(ref.shape[0],ref.shape[1]+extra)
   ref=torch.nn.functional.pad(ref,(0,extra));padded.append(n)
   new_columns.append({'name':n,'nonzero_weights':int(torch.count_nonzero(tensor[:,-extra:]))})
  if n in frozen:assert torch.equal(tensor,ref),n
  elif not torch.equal(tensor,ref):changed.append(n)
 assert changed and not any(n.startswith('backbone.') for n in changed)
 assert set(padded)==set(m['initialization']['zero_padded_temporal_inputs'])
 best=torch.load(folder/'best_heads.pt',map_location='cpu',weights_only=True)
 assert set(best)==set(m['trainable_parameter_names']) and all(torch.equal(state[n],v) for n,v in best.items())
 draws=read(folder/'training_draws.json')['draws'];ids=[i for row in draws for i in row['run_ids']]
 assert set(ids)<=set(m['eligible_train_run_ids']) and not set(ids)&excluded
 assert all(row['view_ids']==[opaque[i] for i in row['run_ids']] for row in draws)
 assert sum((len(row['run_ids'])+15)//16 for row in draws)==c['steps']<=2200
 draw_records.append({'artifact':str(folder),'total_draws':len(ids),'unique_train_cases':len(set(ids)),'outside_train':0,'excluded_unusable_draws':0,'cohort_draws':dict(Counter(byid[i] for i in ids)),'new_independent_cases':0})
 records.append({'artifact':str(folder),'steps':c['steps'],'best_steps':c['best_steps'],'head_training_policy':cfg['head_training_policy'],'numeric_feature_version':cfg['numeric_feature_version'],'trainable_parameters':m['trainable_parameters'],'frozen_tensors_verified':len(frozen),'changed_parameters':changed,'changed_backbone_parameters':[],'new_input_columns':new_columns,'cached_full_delta':m['cache']['validation_full_logit_max_diff'],'root_logits_delta':m['root_validation_max_abs_logit_change'],'source_hashes_and_git_snapshot_verified':True,'checkpoint_and_selected_trainable_weights_verified':True})
 del state,best;gc.collect()
del parent;gc.collect()
# Legacy CPU regression references are used only for loader compatibility, never selection.
model,tok,ser,meta=load_checkpoint(protocol['parent'],'cpu')
rows=predict(model,tok,ser,load_split('data/round5','regression_ss')[:2],'cpu','regression_ss');ref=read(Path(protocol['parent'])/'reload_reference.json')
assert [r['run_id'] for r in rows]==[r['run_id'] for r in ref]
def delta(a,b):return max(float(np.max(np.abs(np.asarray(r[h+'_logits'])-q[h+'_logits']))) for r,q in zip(a,b) for h in ('root','fault'))
legacy_delta=delta(rows,ref);assert legacy_delta<=1e-6
# Check explicit zero padding through real full-model CPU forward, using TRAIN only.
probe=[e for e in train if e.original_run_id not in excluded][:6]
oldrows=predict(model,tok,ser,probe,'cpu','train')
del model,ser;gc.collect()
cfg=read('configs/round12_dynamics_fault.json')
model=build_model(Path(protocol['parent'])/'backbone_config',False,cfg['head_size'],pooling=cfg['pooling'],numeric_dim=numeric_dimension(cfg),root_conditioned_fault=cfg['root_conditioned_fault'],text_logit_weight=cfg['text_logit_weight'])
initialize_from_artifact(model,cfg,train,validation);model.eval()
ser=Serializer(tok,cfg['max_length'],cfg['serializer_version'],cfg['numeric_metrics'],cfg['numeric_feature_version'])
newrows=predict(model,tok,ser,probe,'cpu','train');migration_delta=delta(oldrows,newrows)
assert migration_delta<=1e-5
assert all(np.argmax(a[h+'_logits'])==np.argmax(b[h+'_logits']) for a,b in zip(oldrows,newrows) for h in ('root','fault'))
save(out/'training_integrity.json',{'candidates':records,'original_inputs_unchanged':True,'training_source_revision':source['revision'],'all_training_source_files_match_pretraining_commit':True,'new_independent_cases':0,'new_backbone_sft':False,'legacy_cpu_reload_max_abs_difference':legacy_delta,'migration_full_cpu_max_abs_difference':migration_delta,'migration_probe_train_run_ids':[e.original_run_id for e in probe],'raw_files_reverified':400})
save(out/'training_draw_audit.json',draw_records)
print('INTEGRITY PASS',len(records),'formal updates',sum(r['steps'] for r in records),'legacy delta',legacy_delta,'migration delta',migration_delta,flush=True)
