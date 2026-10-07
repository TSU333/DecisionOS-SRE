"""Validate all actual optimizer updates, source/data lineage and frozen parameters."""
from pathlib import Path
import gc,hashlib
from collections import Counter
import torch
from dulwich.repo import Repo
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.data import load_split

out=Path('outputs/round11');sel=read(out/'selection.json');protocol=read(out/'protocol.json');audit=read(out/'data_audit.json')
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(out/'data_audit.json')==protocol['data_audit_sha256']
for name,sha in audit['original_files'].items():assert file_hash(Path('data/round5')/name)==sha
parent=torch.load(Path(protocol['parent'])/'checkpoint.pt',map_location='cpu',weights_only=True)
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
train=load_split('data/round5','train');byid={e.original_run_id:e.source_metadata['dataset_suite'] for e in train}
excluded={e['run_id'] for e in audit['excluded_unusable_train']};repo=Repo('.');records=[];draw_records=[]
for trial,c in zip(protocol['trials'],sel['candidates']):
 folder=Path(c['artifact']);m=read(folder/'metadata.json');cfg=m['config']
 assert file_hash(trial['config'])==trial['config_sha256'] and read(trial['config'])==cfg
 assert file_hash(folder/'checkpoint.pt')==m['binding']['checkpoint_sha256']
 assert m['code_state']['revision']=='62db696136a80c4e571cdf3fe2f25eb0e7a1b4c5'
 head=repo[m['code_state']['revision'].encode()]
 for p,sha in m['code_state']['source_hashes'].items():
  assert file_hash(p)==sha
  _,oid=repo[head.tree].lookup_path(repo.__getitem__,p.replace('\\','/').encode());assert hashlib.sha256(repo[oid].data).hexdigest()==sha
 assert set(m['train_run_ids'])==set(byid) and not set(m['train_run_ids'])&set(m['validation_run_ids'])
 assert len(m['eligible_train_run_ids'])==164 and set(m['excluded_unusable_train_run_ids'])==excluded
 assert m['selected_restore_validation_max_abs_difference']<=1e-5
 state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);assert set(state)==set(parent)
 frozen=[n for n in state if n not in m['trainable_parameter_names']]
 assert all(torch.equal(state[n],parent[n]) for n in frozen)
 changed=[n for n in state if not torch.equal(state[n],parent[n])]
 assert changed and set(changed)==set(m['selected_changed_parameters'])<=set(m['trainable_parameter_names'])
 assert bool(m['selected_changed_backbone_parameters'])==bool(cfg['partial_backbone_layers'])
 best=torch.load(folder/'best_trainable.pt',map_location='cpu',weights_only=True)
 assert set(best)==set(m['trainable_parameter_names']) and all(torch.equal(state[n],v) for n,v in best.items())
 draws=read(folder/'training_draws.json')['draws'];ids=[i for row in draws for i in row['run_ids']]
 assert set(ids)<=set(m['eligible_train_run_ids']) and not set(ids)&excluded
 assert sum((len(row['run_ids'])+3)//4 for row in draws)==c['steps']<=246
 draw_records.append({'artifact':str(folder),'total_draws':len(ids),'unique_train_cases':len(set(ids)),'outside_train':0,'excluded_unusable_draws':0,'cohort_draws':dict(Counter(byid[i] for i in ids)),'new_independent_cases':0})
 records.append({'artifact':str(folder),'steps':c['steps'],'best_steps':c['best_steps'],'partial_backbone_layers':cfg['partial_backbone_layers'],'trainable_parameters':m['trainable_parameters'],'frozen_tensors_verified':len(frozen),'changed_parameters':changed,'changed_backbone_parameters':m['selected_changed_backbone_parameters'],'selected_restore_delta':m['selected_restore_validation_max_abs_difference'],'source_hashes_and_git_snapshot_verified':True,'checkpoint_and_selected_trainable_weights_verified':True,'resources':m['resources']})
 del state,best;gc.collect()
save(out/'training_integrity.json',{'candidates':records,'original_inputs_unchanged':True,'training_source_revision':'62db696136a80c4e571cdf3fe2f25eb0e7a1b4c5','all_training_source_files_match_pretraining_commit':True,'probe_micro_steps_separate':3,'new_independent_cases':0})
save(out/'training_draw_audit.json',draw_records)
print('INTEGRITY PASS',len(records),'formal optimizer updates',sum(r['steps'] for r in records),flush=True)
