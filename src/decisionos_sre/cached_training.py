"""Train heads on exact cached representations from a frozen shared encoder."""
from pathlib import Path
import time,random,copy
import numpy as np
import torch
from transformers import AutoTokenizer
from .common import read,save,file_hash,seed_all,environment,MODEL_NAME,MODEL_REV,FAULTS,SERIALIZER
from .data import load_split
from .representation import numeric_dimension
from .fault_regularization import configure_cached_heads
from .training_views import load_training_views,eligible_parent_indices,choose_view_indices
from .serializer import Serializer,collate
from .model import build_model,labels,supervised_loss
from .training import predict,prediction_summary,selection_key,initialize_from_artifact,config_binding,code_state,training_order

def feature_batch(items):
    max_k=max(x['candidate'].shape[1] for x in items)
    candidates=[];masks=[];numeric=[]
    for x in items:
        k=x['candidate'].shape[1]
        candidates.append(torch.nn.functional.pad(x['candidate'],(0,0,0,max_k-k)))
        masks.append(torch.nn.functional.pad(x['candidate_mask'],(0,max_k-k)))
        if x.get('candidate_numeric') is not None:numeric.append(torch.nn.functional.pad(x['candidate_numeric'],(0,0,0,max_k-k)))
    return {'incident':torch.cat([x['incident'] for x in items]),'candidate':torch.cat(candidates),
            'candidate_mask':torch.cat(masks),'candidate_numeric':torch.cat(numeric) if numeric else None,
            'incident_numeric':torch.cat([x['incident_numeric'] for x in items]) if numeric else None,
            'application_index':torch.cat([x['application_index'] for x in items]) if items[0].get('application_index') is not None else None}

def train_cached(config):
    options=config.get('augmentation',{})
    if options.get('metric_dropout',0) or options.get('evidence_dropout',0) or options.get('candidate_shuffle',False):
        raise ValueError('Cached features require deterministic, unaugmented inputs')
    source=code_state();seed_all(config['seed']);torch.set_num_threads(config['cpu_threads'])
    device=config['train_device'];out=Path(config['artifact_root'])/'frozen'
    if (out/'checkpoint.pt').exists():raise FileExistsError('Refusing to overwrite checkpoint')
    out.mkdir(parents=True,exist_ok=True)
    trainset=load_split(config['data_dir'],'train');valset=load_split(config['data_dir'],'model_validation')
    apps=config.get('application_fault_names',[])
    if apps and apps!=sorted({e.input.application for e in trainset}):
        raise ValueError('Application heads must use the sorted observable TRAIN applications')
    views,view_meta=load_training_views(config,trainset)
    cache_train=trainset+views;val_start=len(cache_train)
    tok=AutoTokenizer.from_pretrained(config['backbone_dir'],local_files_only=True)
    names=config.get('numeric_metrics',[]) if config.get('numeric_fusion') else []
    ser=Serializer(tok,config['max_length'],config.get('serializer_version',SERIALIZER),names,config.get('numeric_feature_version','mean-v1'),config.get('trace_features',False),apps)
    model=build_model(config['backbone_dir'],head_size=config['head_size'],pooling=config.get('pooling','cls'),numeric_dim=numeric_dimension(config),root_conditioned_fault=config.get("root_conditioned_fault",False),text_logit_weight=config.get("text_logit_weight",1.),application_fault_names=apps)
    initialization=initialize_from_artifact(model,config,trainset,valset)
    dropout_handles=configure_cached_heads(model,config)
    model.to(device).eval();start=time.perf_counter()
    cache=[];templates=[];root_targets=[];fault_targets=[]
    with torch.no_grad():
        for ex in cache_train+valset:
            enc=ser(ex.input);batch=collate([enc],tok.pad_token_id,device)
            inc,cand=model.encode_representations(batch['input_ids'],batch['attention_mask'],batch['spans'],batch['candidate_mask'])
            cache.append({'incident':inc.detach(),'candidate':cand.detach(),'candidate_mask':batch['candidate_mask'],
                          'candidate_numeric':batch.get('candidate_numeric'),'incident_numeric':batch.get('incident_numeric'),'application_index':batch.get('application_index')})
            rt,ft=labels([ex],[enc],device);root_targets.append(rt);fault_targets.append(ft)
            root=ex.targets.root_cause.value;fault=ex.targets.fault_type.value
            templates.append({'incident_id':ex.opaque_incident_id,'run_id':ex.original_run_id,'split':ex.source_metadata['split'],
                'cohort':ex.source_metadata.get('dataset_suite','RE1'),'application':ex.input.application,
                'candidate_ids':enc.candidate_ids,'root_target':enc.candidate_ids.index(root) if root in enc.candidate_ids else (-1 if root else None),
                'fault_target':FAULTS.index(fault) if fault in FAULTS else None,'gold':ex.targets.model_dump(),
                'serialization':enc.report,'evidence_usable':enc.report.get('modality_evidence_usable',enc.report.get('numeric_evidence_usable',enc.report['usable_metrics_retained']>0))})
    precompute_calls=model.encoder_calls
    with torch.no_grad():
        root_reference=[model.score_representations(**x)[0].detach().clone() for x in cache[val_start:]]
    parent_index={e.opaque_incident_id:i for i,e in enumerate(trainset)}
    view_indices={i:[] for i in range(len(trainset))}
    for i,ex in enumerate(views,start=len(trainset)):
        if templates[i]['evidence_usable']:view_indices[parent_index[ex.parent_incident_id]].append(i)
    eligible=eligible_parent_indices(templates,len(trainset),config.get('require_train_usable_evidence',False))
    view_rng=random.Random(config['seed']+971)
    heads=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(heads,lr=config['head_learning_rate'],weight_decay=config['weight_decay'])
    rng=random.Random(config['seed']);history=[];best=None;best_heads=None;best_rows=None;stale=0;steps=0
    sampling_log=[]
    for epoch in range(config['epochs']):
        model.train();model.backbone.eval()
        order=[eligible[i] for i in training_order([trainset[i] for i in eligible],rng,config)];losses=[]
        used=[];used_views=[]
        for offset in range(0,len(order),config['batch_size']):
            indices=order[offset:offset+config['batch_size']]
            used.extend(indices)
            indices=choose_view_indices(indices,view_indices,config.get('training_view_probability',0.),view_rng)
            used_views.extend(cache_train[i].opaque_incident_id for i in indices)
            batch=feature_batch([cache[i] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            roots,faults=model.score_representations(**batch)
            loss=supervised_loss(roots,faults,torch.cat([root_targets[i] for i in indices]),torch.cat([fault_targets[i] for i in indices]),config['loss_weights'],fault_label_smoothing=config.get('fault_label_smoothing',0.))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(heads,1.);optimizer.step();steps+=1;losses.append(float(loss.detach()))
            if steps>=config['max_steps']:break
        sampling_log.append({'epoch':epoch+1,'run_ids':[trainset[i].original_run_id for i in used],'view_ids':used_views})
        rows=[];model.eval()
        with torch.no_grad():
            for i in range(val_start,len(cache)):
                roots,faults=model.score_representations(**cache[i]);row=copy.deepcopy(templates[i])
                row.update(root_logits=roots[0].float().cpu().tolist(),fault_logits=faults[0].float().cpu().tolist());rows.append(row)
        summary=prediction_summary(rows);rank=selection_key(summary,config)
        history.append({'epoch':epoch+1,'optimizer_steps':steps,'train_loss':float(np.mean(losses)),
                        'validation_loss':summary['sum_nll'],'validation':summary})
        if best is None or rank<best:
            best=rank;stale=0;best_rows=rows
            best_heads={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            torch.save(best_heads,out/'best_heads.pt')
            save(out/'selected_validation_logits.json',rows)
        else:stale+=1
        if (epoch+1)%5==0:print('cached epoch',epoch+1,'steps',steps,'validation',summary,flush=True)
        if stale>=config['early_stopping_patience'] or steps>=config['max_steps']:break
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in best_heads:p.copy_(best_heads[n].to(device))
    model.eval()
    for handle in dropout_handles:handle.remove()
    with torch.no_grad():
        root_change=max(float((model.score_representations(**x)[0]-old).abs().max()) for x,old in zip(cache[val_start:],root_reference))
    if config.get('head_training_policy') in ('fault_heads','application_fault_heads') and root_change!=0.:
        raise ValueError('Frozen root branch changed during fault-only training')
    # Verify that inference through the full model agrees with the training cache.
    full=predict(model,tok,ser,valset,device,'model_validation')
    maxdiff=max(float(np.max(np.abs(np.array(a[h+'_logits'])-np.array(b[h+'_logits'])))) for a,b in zip(full,best_rows) for h in ('root','fault'))
    if maxdiff>1e-5:raise ValueError('Cached/full logits mismatch '+str(maxdiff))
    torch.save(model.state_dict(),out/'checkpoint.pt');tok.save_pretrained(out/'tokenizer');model.backbone.config.save_pretrained(out/'backbone_config')
    split=read(Path(config['data_dir'])/'splits.json')
    metadata={'mode':'frozen','config':config,'model_name':MODEL_NAME,'model_revision':MODEL_REV,
        'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad],
        'root_validation_max_abs_logit_change':root_change,'training_regularization':{'fault_hidden_dropout':config.get('fault_hidden_dropout',0.),'fault_label_smoothing':config.get('fault_label_smoothing',0.),'hooks_removed_before_export':True},
        'history':history,'elapsed_seconds':time.perf_counter()-start,'environment':environment(),
        'selection':config['selection_metric'],'diagnostic_only':False,'initialization':initialization,
        'binding':config_binding(config,file_hash(out/'checkpoint.pt'),split['split_hash'],file_hash(out/'tokenizer/tokenizer.json')),
        'supported_applications':sorted({e.input.application for e in trainset}),
        'prepared_data_sha256':file_hash(Path(config['data_dir'])/'examples.json'),'code_state':source,
        'training_views':view_meta,'eligible_train_run_ids':[trainset[i].original_run_id for i in eligible],'excluded_unusable_train_run_ids':[e.original_run_id for i,e in enumerate(trainset) if i not in eligible],
        'train_run_ids':[e.original_run_id for e in trainset],'validation_run_ids':[e.original_run_id for e in valset],
        'ontology':FAULTS,'cache':{'precompute_encoder_calls':precompute_calls,'expected':len(cache),'validation_full_logit_max_diff':maxdiff,
        'dtype':'float32','augmentation':'fixed audited TRAIN views' if views else 'none; fixed canonical input','train_and_validation_separate':True}}
    save(out/'metadata.json',metadata);save(out/'resolved_config.json',config);save(out/'split_manifest.json',split)
    save(out/'augmentations.json',[{'view_id':ex.opaque_incident_id,**ex.augmentation_metadata} for ex in views])
    save(out/'training_draws.json',{'strategy':config.get('sampling_strategy','shuffle'),'independent_train_cases':len(eligible),'allowed_train_cases':len(trainset),'training_view_probability':config.get('training_view_probability',0.),'draws':sampling_log,'note':'Repeated TRAIN draws are not independent new cases'})
    return metadata
