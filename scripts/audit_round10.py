"""Verify training, immutable inputs, frozen weights, and legacy inference."""
from pathlib import Path
import gc
import numpy as np
import torch
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.data import load_split
from decisionos_sre.training import load_checkpoint,predict

out=Path('outputs/round10');selection=read(out/'selection.json');protocol=read(out/'protocol.json')
assert file_hash(out/'protocol.json')==selection['protocol_sha256']
audit=read(out/'data_audit.json')
for name,sha in audit['original_files'].items():assert file_hash(Path('data/round5')/name)==sha
assert file_hash(out/'data_audit.json')==protocol['data_audit_sha256']
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
parent=torch.load(Path(protocol['parent'])/'checkpoint.pt',map_location='cpu',weights_only=True)
records=[]
for c in selection['candidates']:
 folder=Path(c['artifact']);meta=read(folder/'metadata.json');cfg=meta['config']
 assert meta['code_state']['revision']=='01e9bcd24c6f6b2d71696000f325270d70a1e13b'
 assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
 assert file_hash(folder/'checkpoint.pt')==meta['binding']['checkpoint_sha256']
 assert meta['cache']['precompute_encoder_calls']==meta['cache']['expected']==205
 assert meta['cache']['validation_full_logit_max_diff']<=1e-5
 assert meta['root_validation_max_abs_logit_change']==0.
 assert len(meta['eligible_train_run_ids'])==164 and len(meta['excluded_unusable_train_run_ids'])==1
 assert not set(meta['train_run_ids']) & set(meta['validation_run_ids'])
 assert meta['prepared_data_sha256']==audit['original_files']['examples.json']
 state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
 expert=bool(cfg.get('application_fault_names'));prefixes=('application_fault_heads.',) if expert else ('fault_head.','numeric_fault.','numeric_local_fault.')
 frozen=[n for n in parent if not n.startswith(prefixes)]
 assert all(torch.equal(state[n],parent[n]) for n in frozen)
 if expert:
  assert cfg['application_fault_names']==audit['train_applications']
  assert len(meta['initialization']['copied_application_head_parameters'])==20
  assert len(set(state)-set(parent))==20
 else:assert set(state)==set(parent)
 assert all(n.startswith(prefixes) for n in meta['trainable_parameter_names'])
 changed=[n for n in state if n not in parent or not torch.equal(state[n],parent[n])]
 assert changed and all(n.startswith(prefixes) for n in changed)
 head_state=torch.load(folder/'best_heads.pt',map_location='cpu',weights_only=True)
 assert set(head_state)==set(meta['trainable_parameter_names'])
 assert all(torch.equal(state[n],p) for n,p in head_state.items())
 records.append({'artifact':str(folder),'steps':c['steps'],'trainable_parameters':meta['trainable_parameters'],'parameter_count':meta['parameter_count'],'head_training_policy':cfg['head_training_policy'],'root_logits_delta':meta['root_validation_max_abs_logit_change'],'cached_full_delta':meta['cache']['validation_full_logit_max_diff'],'cached_inputs':205,'eligible_cases':164,'frozen_tensors_verified':len(frozen),'checksum_verified':True,'source_hashes_verified':True,'selected_head_state_verified':True,'application_fault_names':cfg.get('application_fault_names',[])})
 del state,head_state;gc.collect()
del parent;gc.collect()
# Current loader must also reproduce the unmodified incumbent's real CPU outputs.
m,t,s,meta=load_checkpoint(protocol['parent'],'cpu')
rows=predict(m,t,s,load_split(meta['config']['data_dir'],'regression_ss')[:2],'cpu','regression_ss')
ref=read(Path(protocol['parent'])/'reload_reference.json')
assert [r['run_id'] for r in rows]==[r['run_id'] for r in ref]
delta=max(float(np.max(np.abs(np.asarray(r[h+'_logits'])-q[h+'_logits']))) for r,q in zip(rows,ref) for h in ('root','fault'))
assert delta<=1e-6
save(out/'training_integrity.json',{'candidates':records,'original_inputs_unchanged':True,'legacy_cpu_reload_max_abs_difference':delta,'source_revision':'01e9bcd24c6f6b2d71696000f325270d70a1e13b','new_independent_cases':0,'new_backbone_sft':False})
print('INTEGRITY OK',len(records),'legacy reload delta',delta,flush=True)
