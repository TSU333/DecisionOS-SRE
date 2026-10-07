"""Reconstruct every CV head, verify lineage and produce TRAIN-only error triage."""
from pathlib import Path
from collections import Counter
import hashlib,math
import numpy as np
import torch
from transformers import AutoTokenizer
from dulwich.repo import Repo
from decisionos_sre.common import read,save,file_hash,seed_all,FAULTS
from decisionos_sre.data import load_split
from decisionos_sre.model import build_model,DecisionModel
from decisionos_sre.serializer import Serializer,collate
from decisionos_sre.grouped_validation import validate_folds

out=Path('outputs/round13');result=read(out/'cv_results.json');protocol=read(out/'protocol.json');source=read(out/'pretraining_source.json');cfg=read(protocol['config']);spec=read(out/'folds.json')
assert len(result['runs'])==protocol['runs']==18
validate_folds(spec['records'],spec['folds'])
split_manifest=read('data/round12/splits.json')
other_groups={g for incident,g in split_manifest['groups'].items() if split_manifest['assignments'][incident]!='train'}
assert not {r['group_id'] for r in spec['records']}&other_groups
repo=Repo('.');head=repo[source['revision'].encode()]
for p,sha in source['source_hashes'].items():
 assert file_hash(p)==sha
 _,oid=repo[head.tree].lookup_path(repo.__getitem__,p.replace(chr(92),'/').encode());assert hashlib.sha256(repo[oid].data).hexdigest()==sha
for p,sha in read(out/'data_audit.json')['files'].items():assert file_hash(p)==sha
assert file_hash('outputs/latest_model.json')==file_hash(out/'previous_latest_model.json')
train={e.original_run_id:e for e in load_split('data/round12','train')};allowed={r['run_id'] for r in spec['records']}
torch.set_num_threads(8);model=build_model(cfg['backbone_dir'],pooling='mean',numeric_dim=120,root_conditioned_fault=True,text_logit_weight=.1).eval();backbone=model.backbone

def tensor_digest(items):
 h=hashlib.sha256()
 for n,p in items:h.update(n.encode());h.update(p.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
assert tensor_digest(backbone.state_dict().items())==result['backbone_tensor_sha256']
tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
ser={v:Serializer(tok,2048,cfg['serializer_version'],cfg['numeric_metrics'],'temporal-v1' if v=='temporal' else 'temporal-dynamics-v1') for v in protocol['variants']}
probes={}
with torch.inference_mode():
 for fold in spec['folds']:
  for rid in fold['outer'][:2]:
   e=train[rid];en=ser['temporal'](e.input);b=collate([en],tok.pad_token_id)
   inc,cand=model.encode_representations(b['input_ids'],b['attention_mask'],b['spans'],b['candidate_mask'])
   for v in protocol['variants']:
    z=ser[v](e.input);n=collate([z],tok.pad_token_id)
    probes[v,rid]={'incident':inc,'candidate':cand,'candidate_mask':n['candidate_mask'],'candidate_numeric':n['candidate_numeric'],'incident_numeric':n['incident_numeric']}
records=[];error_votes={v:{} for v in protocol['variants']};confusions={v:Counter() for v in protocol['variants']}
for run in result['runs']:
 folder=Path(run['artifact']);m=read(folder/'metadata.json');fold=spec['folds'][m['fold']];v=m['variant']
 assert file_hash(folder/'heads.pt')==m['heads_sha256'] and m['protocol_sha256']==file_hash(out/'protocol.json')
 assert m['source_revision']==source['revision']
 for key,role in [('fit_run_ids','fit'),('inner_stop_run_ids','stop'),('outer_run_ids','outer')]:assert m[key]==fold[role]
 draws=read(folder/'training_draws.json');draw_ids=[rid for d in draws for rid in d['run_ids']]
 assert set(draw_ids)<=set(fold['fit']) and not set(draw_ids)&(set(fold['stop'])|set(fold['outer']))
 assert sum(math.ceil(len(d['run_ids'])/16) for d in draws)==m['steps']
 assert m['steps']<=1000 and m['cached_full_logit_max_diff']<=1e-5
 rows=read(folder/'outer_predictions.json');assert {r['run_id'] for r in rows}==set(fold['outer'])
 assert all(r['split']=='train_cv_outer' for r in rows)
 seed_all(m['seed']+100*m['fold'])
 initial=DecisionModel(backbone,head_size=cfg['head_size'],pooling='mean',numeric_dim=120,root_conditioned_fault=True,text_logit_weight=.1)
 initial_state={n:p for n,p in initial.named_parameters() if not n.startswith('backbone.')}
 assert tensor_digest(initial_state.items())==m['paired_initial_head_sha256']
 loaded=DecisionModel(backbone,head_size=cfg['head_size'],pooling='mean',numeric_dim=120 if v=='temporal' else 190,root_conditioned_fault=True,text_logit_weight=.1).eval()
 state=torch.load(folder/'heads.pt',map_location='cpu',weights_only=True)
 assert set(state)==set(m['trainable_parameter_names'])=={n for n,p in loaded.named_parameters() if not n.startswith('backbone.')}
 missing,unexpected=loaded.load_state_dict(state,strict=False);assert not unexpected and all(n.startswith('backbone.') for n in missing)
 changed=[]
 for n,value in state.items():
  ref=initial_state[n]
  if ref.shape!=value.shape:ref=torch.nn.functional.pad(ref,(0,value.shape[1]-ref.shape[1]))
  if not torch.equal(value,ref):changed.append(n)
 assert changed
 differences=[]
 with torch.inference_mode():
  for row in rows[:2]:
   roots,faults=loaded.score_representations(**probes[v,row['run_id']])
   for h,t in [('root',roots),('fault',faults)]:
    values=t[0].tolist();old=row[h+'_logits'];d=float(np.max(np.abs(np.array(values)-old)));differences.append(d)
    assert np.allclose(values,old,atol=1e-4,rtol=1e-5)
    assert int(np.argmax(values))==int(np.argmax(old))
 for row in rows:
  e=train[row['run_id']]
  assert row['candidate_ids'][row['root_target']]==e.targets.root_cause.value and FAULTS[row['fault_target']]==e.targets.fault_type.value
  root=int(np.argmax(row['root_logits']));fault=int(np.argmax(row['fault_logits']))
  item={'seed':m['seed'],'root_correct':root==row['root_target'],'fault_correct':fault==row['fault_target'],'joint_correct':root==row['root_target'] and fault==row['fault_target'],'fault_pred':FAULTS[fault],'root_pred':row['candidate_ids'][root]}
  error_votes[v].setdefault(row['run_id'],[]).append(item)
  if not item['fault_correct']:confusions[v][FAULTS[row['fault_target']]+' -> '+FAULTS[fault]]+=1
 records.append({'variant':v,'seed':m['seed'],'fold':m['fold'],'steps':m['steps'],'best_steps':m['best_steps'],'draw_count':len(draw_ids),'unique_fit_cases_seen':len(set(draw_ids)),'out_of_fit_draws':0,'cpu_vs_gpu_logits_max_abs_difference':max(differences),'cpu_argmax_unchanged':True,'changed_head_tensors':len(changed),'checkpoint_verified':True})
for v,votes in error_votes.items():assert set(votes)==allowed and all(len(items)==3 for items in votes.values())
triage={}
for v,votes in error_votes.items():
 cases=[]
 for rid,items in votes.items():
  e=train[rid];errors=sum(not x['joint_correct'] for x in items)
  if errors:cases.append({'run_id':rid,'application':e.input.application,'cohort':e.source_metadata['dataset_suite'],'fault_gold':e.targets.fault_type.value,'root_gold':e.targets.root_cause.value,'wrong_seeds':errors,'predictions':items,'action':'review source telemetry and independent injection metadata; do not relabel from prediction'})
 triage[v]={'stable_errors_all_three_seeds':sum(c['wrong_seeds']==3 for c in cases),'any_seed_errors':len(cases),'confusions_across_492_repeated_predictions':dict(confusions[v].most_common()),'cases':sorted(cases,key=lambda c:(-c['wrong_seeds'],c['run_id']))}
save(out/'oof_error_audit.json',{'original_train_only':True,'unique_cases':164,'seeds':3,'label_changes':0,'cases_are_not_proven_mislabeled':True,'variants':triage})
save(out/'training_integrity.json',{'status':'passed','runs':records,'total_updates':sum(r['steps'] for r in records),'original_pretrained_backbone_verified':True,'backbone_unchanged':True,'training_source_git_verified':source['revision'],'default_pointer_unchanged':True,'global_holdouts_evaluated':False,'max_cpu_gpu_logit_difference':max(r['cpu_vs_gpu_logits_max_abs_difference'] for r in records),'all_cpu_argmax_unchanged':True})
print('INTEGRITY PASS',len(records),'max CPU/GPU difference',max(r['cpu_vs_gpu_logits_max_abs_difference'] for r in records),'stable errors',{v:x['stable_errors_all_three_seeds'] for v,x in triage.items()},flush=True)
