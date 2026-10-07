"""Bounded end-to-end tuning of the shared encoder tail, with numeric heads frozen."""
from pathlib import Path
import math,random,time
import numpy as np
import torch
from transformers import AutoTokenizer
from .common import read,save,file_hash,seed_all,environment,MODEL_NAME,MODEL_REV,FAULTS,SERIALIZER
from .data import load_split
from .model import build_model,labels,supervised_loss
from .serializer import Serializer,collate
from .representation import numeric_dimension


def configure_partial_training(model,config):
    count=config['partial_backbone_layers']
    layers=getattr(model.backbone,'layers',None)
    if layers is None or not hasattr(model.backbone,'final_norm'):
        raise ValueError('Partial training requires the pinned ModernBERT layer layout')
    if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=len(layers):
        raise ValueError('Invalid partial backbone layer count')
    if not config.get('initialization_artifact'):
        raise ValueError('Partial training requires an existing parent')
    if config.get('cache_frozen_features') or config.get('application_fault_names'):
        raise ValueError('Partial training cannot use frozen feature caches or application experts')
    if config.get('training_views_file') or config.get('training_view_probability',0.) or any(config.get('augmentation',{}).values()):
        raise ValueError('Partial training requires original unaugmented inputs')
    if config.get('head_training_policy','all_heads')!='all_heads' or config.get('fault_hidden_dropout',0.) or config.get('fault_label_smoothing',0.):
        raise ValueError('Unsupported partial training head controls')
    for parameter in model.parameters():parameter.requires_grad=False
    model.fault_head.requires_grad_(True)
    if count:
        for layer in layers[len(layers)-count:]:layer.requires_grad_(True)
        model.backbone.final_norm.requires_grad_(True)
    return [n for n,p in model.named_parameters() if p.requires_grad]


def accumulation_divisor(index,batch_count,accumulation):
    return min(accumulation,batch_count-(index//accumulation)*accumulation)


def build_partial(config,trainset,valset):
    from .training import initialize_from_artifact
    # Parent supplies every parameter; do not allocate a second pretrained model.
    model=build_model(config['backbone_dir'],pretrained=False,head_size=config['head_size'],pooling=config.get('pooling','cls'),numeric_dim=numeric_dimension(config),root_conditioned_fault=config.get('root_conditioned_fault',False),text_logit_weight=config.get('text_logit_weight',1.))
    initialization=initialize_from_artifact(model,config,trainset,valset)
    configure_partial_training(model,config)
    if config['partial_backbone_layers'] and config.get('gradient_checkpointing'):
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model,initialization


def train_partial(config,mode):
    from .training import predict,prediction_summary,selection_key,code_state,training_order
    if mode!='sft' or config['batch_size']!=1 or config['train_device']!='cuda':
        raise ValueError('Partial trainer requires sft mode, CUDA and batch size 1')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if config['gradient_accumulation']<1 or config['epochs']<1 or config['max_steps']<1:
        raise ValueError('Invalid training budget')
    source=code_state();seed_all(config['seed']);torch.set_num_threads(config['cpu_threads'])
    out=Path(config['artifact_root'])/'sft'
    if (out/'checkpoint.pt').exists():raise FileExistsError('Refusing to overwrite checkpoint')
    out.mkdir(parents=True,exist_ok=True)
    trainset=load_split(config['data_dir'],'train');valset=load_split(config['data_dir'],'model_validation')
    tok=AutoTokenizer.from_pretrained(config['backbone_dir'],local_files_only=True)
    ser=Serializer(tok,config['max_length'],config.get('serializer_version',SERIALIZER),config.get('numeric_metrics',[]),config.get('numeric_feature_version','mean-v1'),config.get('trace_features',False))
    encoded=[ser(e.input) for e in trainset]
    eligible=[i for i,e in enumerate(encoded) if e.report.get('modality_evidence_usable',False)]
    if not eligible:raise ValueError('No usable TRAIN examples')
    model,initialization=build_partial(config,trainset,valset);model.to('cuda')
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    groups=[]
    for name,prefix,lr in [('backbone','backbone.',config['backbone_learning_rate']),('text_fault','fault_head.',config['head_learning_rate'])]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix)]
        if params:groups.append({'params':params,'lr':lr,'name':name})
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'],foreach=False)
    accum=config['gradient_accumulation'];steps=0;history=[];draws=[];best=None;best_state=None;best_rows=None;stale=0
    rng=random.Random(config['seed']);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(config['epochs']):
        model.train()
        if not config['partial_backbone_layers']:model.backbone.eval()
        order=[eligible[i] for i in training_order([trainset[j] for j in eligible],rng,config)]
        losses=[];used=[];optimizer.zero_grad(set_to_none=True)
        for index,i in enumerate(order):
            batch=collate([encoded[i]],tok.pad_token_id,'cuda');targets=labels([trainset[i]],[encoded[i]],'cuda')
            before=model.encoder_calls
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                roots,faults=model(**batch);loss=supervised_loss(roots,faults,*targets,config['loss_weights'])
            if model.encoder_calls-before!=1:raise RuntimeError('Expected one encoder invocation')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite partial training loss')
            (loss/accumulation_divisor(index,len(order),accum)).backward();losses.append(float(loss.detach()));used.append(i)
            if (index+1)%accum==0 or index+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite partial gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
                if steps%10==0:print('partial',config['partial_backbone_layers'],'epoch',epoch+1,'steps',steps,'loss',float(np.mean(losses[-40:])),flush=True)
                if steps>=config['max_steps']:break
        draws.append({'epoch':epoch+1,'run_ids':[trainset[i].original_run_id for i in used]})
        rows=predict(model,tok,ser,valset,'cuda','model_validation');summary=prediction_summary(rows);rank=selection_key(summary,config)
        history.append({'epoch':epoch+1,'optimizer_steps':steps,'train_loss':float(np.mean(losses)),'validation_loss':summary['sum_nll'],'validation':summary})
        print('partial validation',history[-1],flush=True)
        if best is None or rank<best:
            best=rank;stale=0;best_rows=rows
            best_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            torch.save(best_state,out/'best_trainable.pt');save(out/'selected_validation_logits.json',rows)
        else:stale+=1
        if stale>=config['early_stopping_patience'] or steps>=config['max_steps']:break
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in best_state:p.copy_(best_state[n].to('cuda'))
    model.eval()
    full=predict(model,tok,ser,valset,'cuda','model_validation')
    delta=max(float(np.max(np.abs(np.array(a[h+'_logits'])-b[h+'_logits']))) for a,b in zip(full,best_rows) for h in ('root','fault'))
    if delta>1e-5:raise ValueError('Restored selected logits differ')
    changed=[n for n in best_state if not torch.equal(best_state[n],initial[n])]
    if not changed:raise ValueError('Selected training did not update any weights')
    torch.save(model.state_dict(),out/'checkpoint.pt');tok.save_pretrained(out/'tokenizer');model.backbone.config.save_pretrained(out/'backbone_config')
    from .training import config_binding
    split=read(Path(config['data_dir'])/'splits.json')
    metadata={'mode':mode,'training_kind':'partial_backbone_sft' if config['partial_backbone_layers'] else 'frozen_backbone_text_head_control','config':config,'model_name':MODEL_NAME,'model_revision':MODEL_REV,'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_parameter_names':list(initial),'selected_changed_parameters':changed,'selected_changed_backbone_parameters':[n for n in changed if n.startswith('backbone.')],'history':history,'elapsed_seconds':time.perf_counter()-start,'environment':environment(),'selection':config['selection_metric'],'diagnostic_only':False,'initialization':initialization,'binding':config_binding(config,file_hash(out/'checkpoint.pt'),split['split_hash'],file_hash(out/'tokenizer/tokenizer.json')),'supported_applications':sorted({e.input.application for e in trainset}),'prepared_data_sha256':file_hash(Path(config['data_dir'])/'examples.json'),'code_state':source,'eligible_train_run_ids':[trainset[i].original_run_id for i in eligible],'excluded_unusable_train_run_ids':[e.original_run_id for i,e in enumerate(trainset) if i not in eligible],'train_run_ids':[e.original_run_id for e in trainset],'validation_run_ids':[e.original_run_id for e in valset],'ontology':FAULTS,'selected_restore_validation_max_abs_difference':delta,'resources':{'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(),'train_precision':'bf16 autocast; fp32 parameters and validation','gradient_checkpointing':bool(config['partial_backbone_layers'] and config.get('gradient_checkpointing'))}}
    save(out/'metadata.json',metadata);save(out/'resolved_config.json',config);save(out/'split_manifest.json',split);save(out/'augmentations.json',[])
    save(out/'training_draws.json',{'strategy':config.get('sampling_strategy','shuffle'),'independent_train_cases':len(eligible),'allowed_train_cases':len(trainset),'draws':draws,'note':'Only original TRAIN runs; no augmented views'})
    return metadata
