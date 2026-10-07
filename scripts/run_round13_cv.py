"""Run sealed nested-stopping, grouped TRAIN CV from original pretrained weights."""
from pathlib import Path
from collections import Counter
import copy,gc,hashlib,math,random,time
import numpy as np
import torch
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash,seed_all,FAULTS,environment
from decisionos_sre.data import load_split
from decisionos_sre.serializer import Serializer,collate
from decisionos_sre.model import build_model,DecisionModel,supervised_loss
from decisionos_sre.cached_training import feature_batch
from decisionos_sre.training import prediction_summary,training_order,code_state,predict
from decisionos_sre.fault_regularization import configure_cached_heads
from decisionos_sre.grouped_validation import validate_folds,assert_fresh_config,inner_selection_key

out=Path('outputs/round13');protocol=read(out/'protocol.json');cfg=read(protocol['config']);fold_spec=read(out/'folds.json')
assert_fresh_config(cfg);validate_folds(fold_spec['records'],fold_spec['folds'])
assert file_hash(protocol['config'])==protocol['config_sha256'] and file_hash(out/'folds.json')==protocol['folds_sha256']
assert file_hash(out/'data_audit.json')==protocol['data_audit_sha256']
source=read(out/'pretraining_source.json');assert all(file_hash(p)==v for p,v in source['source_hashes'].items())
assert code_state()['revision']==source['revision']
if (out/'cv_results.json').exists():raise FileExistsError('CV already completed')
for name,sha in protocol['backbone']['files'].items():assert file_hash(Path('artifacts/backbone')/name)==sha
for p,sha in read(out/'data_audit.json')['files'].items():assert file_hash(p)==sha
torch.set_num_threads(8);seed_all(50);device='cuda';start=time.perf_counter()
alltrain=load_split(cfg['data_dir'],'train');byid={e.original_run_id:e for e in alltrain}
examples=[byid[r['run_id']] for r in fold_spec['records']];index={e.original_run_id:i for i,e in enumerate(examples)}
tok=AutoTokenizer.from_pretrained(cfg['backbone_dir'],local_files_only=True)
serializers={v:Serializer(tok,2048,cfg['serializer_version'],cfg['numeric_metrics'],'temporal-v1' if v=='temporal' else 'temporal-dynamics-v1') for v in protocol['variants']}
encoder_model=build_model(cfg['backbone_dir'],pooling=cfg['pooling'],numeric_dim=120,root_conditioned_fault=True,text_logit_weight=.1).to(device).eval()
for p in encoder_model.backbone.parameters():p.requires_grad=False
backbone=encoder_model.backbone

def tensor_digest(items):
 h=hashlib.sha256()
 for n,p in items:
  h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
backbone_digest=tensor_digest(backbone.state_dict().items())
cache={v:[] for v in protocol['variants']};templates={v:[] for v in protocol['variants']}
with torch.no_grad():
 for e in examples:
  enc=serializers['temporal'](e.input);batch=collate([enc],tok.pad_token_id,device)
  incident,candidate=encoder_model.encode_representations(batch['input_ids'],batch['attention_mask'],batch['spans'],batch['candidate_mask'])
  for variant in protocol['variants']:
   en=serializers[variant](e.input)
   assert en.input_ids==enc.input_ids and en.candidate_ids==enc.candidate_ids and en.spans==enc.spans
   b=collate([en],tok.pad_token_id,device)
   cache[variant].append({'incident':incident.detach(),'candidate':candidate.detach(),'candidate_mask':b['candidate_mask'],'candidate_numeric':b['candidate_numeric'],'incident_numeric':b['incident_numeric'],'application_index':None})
   templates[variant].append({'incident_id':e.opaque_incident_id,'run_id':e.original_run_id,'split':'train_cv_outer','cohort':e.source_metadata['dataset_suite'],'application':e.input.application,'candidate_ids':en.candidate_ids,'root_target':en.candidate_ids.index(e.targets.root_cause.value),'fault_target':FAULTS.index(e.targets.fault_type.value)})
assert encoder_model.encoder_calls==164
save(out/'cache_manifest.json',{'original_pretrained_backbone':True,'backbone_tensor_sha256':backbone_digest,'precompute_calls':164,'run_ids':list(index),'variants_share_identical_text':True,'no_cross_case_learned_transform':True,'source_revision':source['revision']})
print('CACHE READY 164 original TRAIN inputs',flush=True)

def fresh_model(variant,seed):
 seed_all(seed)
 base=DecisionModel(backbone,head_size=cfg['head_size'],pooling=cfg['pooling'],numeric_dim=120,root_conditioned_fault=True,text_logit_weight=.1)
 initial={n:p.detach().clone() for n,p in base.named_parameters() if not n.startswith('backbone.')}
 initial_sha=tensor_digest(initial.items())
 if variant=='temporal':model=base
 else:
  model=DecisionModel(backbone,head_size=cfg['head_size'],pooling=cfg['pooling'],numeric_dim=190,root_conditioned_fault=True,text_logit_weight=.1)
  for n,p in model.named_parameters():
   if n.startswith('backbone.'):continue
   old=initial[n]
   if p.shape!=old.shape:old=torch.nn.functional.pad(old,(0,p.shape[1]-old.shape[1]))
   with torch.no_grad():p.copy_(old)
 model.to(device)
 handles=configure_cached_heads(model,cfg)
 # Pair training RNG after construction, independent of expanded layer allocation.
 seed_all(seed+991)
 return model,handles,initial_sha

def cached_rows(model,variant,indices):
 model.eval();rows=[]
 with torch.no_grad():
  for i in indices:
   roots,faults=model.score_representations(**cache[variant][i]);r=copy.deepcopy(templates[variant][i])
   r.update(root_logits=roots[0].cpu().tolist(),fault_logits=faults[0].cpu().tolist());rows.append(r)
 return rows

runs=[];all_oof={v:{str(seed):[] for seed in protocol['seeds']} for v in protocol['variants']}
for variant in protocol['variants']:
 for seed in protocol['seeds']:
  for fold in fold_spec['folds']:
   folder=Path('artifacts/round13')/variant/('seed'+str(seed))/('fold'+str(fold['fold']))
   if folder.exists():raise FileExistsError('Refusing to overwrite '+str(folder))
   folder.mkdir(parents=True)
   fit=[index[i] for i in fold['fit']];stop=[index[i] for i in fold['stop']];outer=[index[i] for i in fold['outer']]
   model,handles,initial_sha=fresh_model(variant,seed+100*fold['fold'])
   trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
   assert all(not n.startswith('backbone.') for n,p in trainable)
   optimizer=torch.optim.AdamW([p for _,p in trainable],lr=cfg['head_learning_rate'],weight_decay=cfg['weight_decay'])
   rng=random.Random(seed+100*fold['fold']);steps=0;history=[];draws=[];best=None;best_state=None;stale=0;trial_start=time.perf_counter()
   torch.cuda.reset_peak_memory_stats()
   print('START',variant,seed,fold['fold'],'fit/stop/outer',len(fit),len(stop),len(outer),flush=True)
   for epoch in range(cfg['epochs']):
    model.train();backbone.eval();order=[fit[i] for i in training_order([examples[j] for j in fit],rng,cfg)];losses=[];used=[]
    for offset in range(0,len(order),cfg['batch_size']):
     ids=order[offset:offset+cfg['batch_size']];used.extend(examples[i].original_run_id for i in ids)
     roots,faults=model.score_representations(**feature_batch([cache[variant][i] for i in ids]))
     rt=torch.tensor([templates[variant][i]['root_target'] for i in ids],device=device);ft=torch.tensor([templates[variant][i]['fault_target'] for i in ids],device=device)
     loss=supervised_loss(roots,faults,rt,ft,cfg['loss_weights'])
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
     optimizer.zero_grad(set_to_none=True);loss.backward()
     norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.,error_if_nonfinite=True)
     optimizer.step();steps+=1;losses.append(float(loss.detach()))
     if steps>=cfg['max_steps']:break
    draws.append({'epoch':epoch+1,'run_ids':used})
    summary=prediction_summary(cached_rows(model,variant,stop));key=inner_selection_key(summary)
    history.append({'epoch':epoch+1,'steps':steps,'fit_loss':float(np.mean(losses)),'inner_stop':summary})
    if best is None or key<best:
     best=key;stale=0;best_epoch=epoch+1;best_steps=steps
     best_state={n:p.detach().cpu().clone() for n,p in trainable}
    else:stale+=1
    if stale>=cfg['early_stopping_patience'] or steps>=cfg['max_steps']:break
   torch.save(best_state,folder/'heads.pt')
   saved=torch.load(folder/'heads.pt',map_location=device,weights_only=True)
   with torch.no_grad():
    for n,p in trainable:p.copy_(saved[n])
   for handle in handles:handle.remove()
   rows=cached_rows(model,variant,outer);summary=prediction_summary(rows)
   # Parity verification does not adapt the model or select another epoch.
   full=predict(model,tok,serializers[variant],[examples[i] for i in outer[:2]],device,'train_cv_outer')
   maxdiff=max(float(np.max(np.abs(np.array(a[h+'_logits'])-b[h+'_logits']))) for a,b in zip(full,rows[:2]) for h in ('root','fault'))
   assert maxdiff<=1e-5
   assert all(torch.equal(saved[n],p) for n,p in trainable)
   meta={'variant':variant,'seed':seed,'fold':fold['fold'],'mode':'development_cv_head_only','artifact':str(folder),'initialization':'original pretrained ModernBERT plus fresh paired heads','paired_initial_head_sha256':initial_sha,'fit_run_ids':fold['fit'],'inner_stop_run_ids':fold['stop'],'outer_run_ids':fold['outer'],'steps':steps,'best_steps':best_steps,'best_epoch':best_epoch,'epochs':len(history),'history':history,'outer_summary':summary,'heads_sha256':file_hash(folder/'heads.pt'),'source_revision':source['revision'],'protocol_sha256':file_hash(out/'protocol.json'),'backbone_tensor_sha256':backbone_digest,'cached_full_logit_max_diff':maxdiff,'trainable_parameters':sum(p.numel() for _,p in trainable),'trainable_parameter_names':[n for n,_ in trainable],'seconds':time.perf_counter()-trial_start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'group_overlap':0,'outer_used_for_stopping':False}
   save(folder/'metadata.json',meta);save(folder/'outer_predictions.json',rows);save(folder/'training_draws.json',draws)
   (out/'trials').mkdir(exist_ok=True);save(out/'trials'/(variant+'_seed'+str(seed)+'_fold'+str(fold['fold'])+'.json'),meta)
   runs.append({k:v for k,v in meta.items() if k!='history'});all_oof[variant][str(seed)].extend(rows)
   print('DONE',variant,seed,fold['fold'],'steps',steps,'best',best_steps,'outer joint',summary['joint_accuracy'],flush=True)
   del model,optimizer,trainable,best_state,saved;gc.collect()
assert tensor_digest(backbone.state_dict().items())==backbone_digest
summaries={}
for variant,seed_rows in all_oof.items():
 values=[]
 for seed,rows in seed_rows.items():
  assert len(rows)==164 and {r['run_id'] for r in rows}==set(index)
  save(out/(variant+'_seed'+seed+'_oof.json'),rows);values.append({'seed':int(seed),'summary':prediction_summary(rows)})
 summaries[variant]={'seeds':values,'joint_mean':float(np.mean([x['summary']['joint_accuracy'] for x in values])),'joint_sample_sd':float(np.std([x['summary']['joint_accuracy'] for x in values],ddof=1)),'root_mean':float(np.mean([x['summary']['root']['accuracy'] for x in values])),'fault_mean':float(np.mean([x['summary']['fault']['accuracy'] for x in values]))}
paired=[]
for a in [r for r in runs if r['variant']=='temporal']:
 b=next(r for r in runs if r['variant']=='dynamics' and r['seed']==a['seed'] and r['fold']==a['fold'])
 assert a['paired_initial_head_sha256']==b['paired_initial_head_sha256']
 paired.append({'seed':a['seed'],'fold':a['fold'],'joint_delta':b['outer_summary']['joint_accuracy']-a['outer_summary']['joint_accuracy']})
result={'status':'completed','execution_scope':'mvp','runs':runs,'summaries':summaries,'paired_comparisons':paired,'total_optimizer_updates':sum(r['steps'] for r in runs),'wall_seconds':time.perf_counter()-start,'new_independent_cases':0,'backbone_unchanged':True,'backbone_tensor_sha256':backbone_digest,'outer_epoch_selection':False,'development_only':True,'default_model_updated':False,'global_holdouts_evaluated':False,'environment':environment(),'limitations':['Fixed architecture and features were designed in earlier rounds using related data. This is development CV, not a fresh independent confirmation.','Seed and fold results reuse cases; do not treat 492 predictions as 492 independent cases.','Runs train on inner-fit subsets. Their scores are not a direct head-to-head comparison with the incumbent trained on the full TRAIN.']}
save(out/'cv_results.json',result)
print('CV COMPLETE',len(runs),'updates',result['total_optimizer_updates'],'summaries',summaries,flush=True)
