from pathlib import Path
import random
import time
import copy
import math
import numpy as np
import torch
from transformers import AutoTokenizer
from .common import FAULTS, MODEL_REV, MODEL_NAME, SERIALIZER, digest, file_hash, save, read, seed_all, environment
from .data import load_split, augment
from .representation import numeric_dimension
from .serializer import Serializer, collate
from .model import build_model, labels, supervised_loss

def config_binding(config,checkpoint_hash,split_hash,tokenizer_hash):
    pipeline={"serializer":config.get("serializer_version",SERIALIZER),"max_length":config["max_length"],"ontology":FAULTS}
    for key in ("pooling","numeric_fusion","numeric_metrics","root_conditioned_fault","text_logit_weight","numeric_feature_version","trace_features","trace_feature_version"):
        if key in config: pipeline[key]=config[key]
    return {"checkpoint_sha256":checkpoint_hash,"split_hash":split_hash,
            "pipeline_hash":digest(pipeline),
            "tokenizer_sha256":tokenizer_hash,"model_revision":MODEL_REV}

def predict(model,tokenizer,serializer,examples,device,split):
    rows=[]
    model.eval()
    with torch.inference_mode():
        for ex in examples:
            encoded=serializer(ex.input)
            batch=collate([encoded],tokenizer.pad_token_id,device)
            before=model.encoder_calls
            start=time.perf_counter()
            roots,faults=model(**batch)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            assert model.encoder_calls-before==1
            rid=ex.targets.root_cause.value
            fid=ex.targets.fault_type.value
            rows.append({"incident_id":ex.opaque_incident_id,"run_id":ex.original_run_id,"split":split,
              "application":ex.input.application,"cohort":ex.source_metadata.get("dataset_suite","RE1"),"candidate_ids":encoded.candidate_ids,
              "root_logits":roots[0,:len(encoded.candidate_ids)].float().cpu().tolist(),
              "fault_logits":faults[0].float().cpu().tolist(),
              "root_target":encoded.candidate_ids.index(rid) if rid in encoded.candidate_ids else (-1 if rid is not None else None),
              "fault_target":FAULTS.index(fid) if fid in FAULTS else None,
              "gold":ex.targets.model_dump(),"evidence_usable":encoded.report.get("modality_evidence_usable",encoded.report.get("numeric_evidence_usable",encoded.report["usable_metrics_retained"]>0)),
              "serialization":encoded.report,"model_ms":elapsed*1000})
    return rows

def train(config,mode):
    if config.get("cache_frozen_features"):
        if mode!="frozen": raise ValueError("Feature caching requires frozen mode")
        from .cached_training import train_cached
        return train_cached(config)
    if mode not in ["frozen","sft"]:
        raise ValueError("mode must be frozen or sft")
    source_state=code_state()
    seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device=config["train_device"]
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Explicitly choose cpu after reviewing budget.")
    out=Path(config["artifact_root"])/mode
    out.mkdir(parents=True,exist_ok=True)
    if (out/"checkpoint.pt").exists():
        raise FileExistsError("Checkpoint exists. Choose a fresh artifact_root; never silently overwrite a evaluated model.")
    tokenizer=AutoTokenizer.from_pretrained(config["backbone_dir"],local_files_only=True)
    serializer=Serializer(tokenizer,config["max_length"],config.get("serializer_version",SERIALIZER),config.get("numeric_metrics",[]) if config.get("numeric_fusion",False) else [],config.get("numeric_feature_version","mean-v1"),config.get("trace_features",False))
    model=build_model(config["backbone_dir"],head_size=config["head_size"],pooling=config.get("pooling","cls"),numeric_dim=numeric_dimension(config),root_conditioned_fault=config.get("root_conditioned_fault",False),text_logit_weight=config.get("text_logit_weight",1.)).to(device)
    if mode=="frozen":
        for p in model.backbone.parameters():
            p.requires_grad=False
    elif config["gradient_checkpointing"]:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    trainset=load_split(config["data_dir"],"train")
    valset=load_split(config["data_dir"],"model_validation")
    diagnostic=config.get("diagnostic_overfit_per_class",0)
    if diagnostic:
        # Diagnostic only: fit a tiny stratified TRAIN subset, never holdout labels.
        selected=[]
        for fault in FAULTS:
            selected.extend([e for e in trainset if e.targets.fault_type.value==fault][:diagnostic])
        trainset=selected
        valset=selected
    initialization=initialize_from_artifact(model,config,trainset,valset)
    groups=[]
    for name,params,lr in [
        ("backbone",model.backbone.parameters(),config.get("backbone_learning_rate",config["learning_rate_"+mode])),
        ("heads",[p for name,p in model.named_parameters() if not name.startswith("backbone.")],config.get("head_learning_rate",config["learning_rate_"+mode]))]:
        params=[p for p in params if p.requires_grad]
        if params: groups.append({"params":params,"lr":lr,"name":name})
    optimizer=torch.optim.AdamW(groups,weight_decay=config["weight_decay"])
    updates_per_epoch=math.ceil(math.ceil(len(trainset)/config["batch_size"])/config["gradient_accumulation"])
    scheduled_steps=min(config["max_steps"],updates_per_epoch*config["epochs"])
    warmup_steps=int(scheduled_steps*config.get("warmup_fraction",0))
    scheduler=None
    if config.get("schedule")=="linear":
        def scale(step):
            if warmup_steps and step<warmup_steps: return (step+1)/warmup_steps
            return max(0.0,(scheduled_steps-step)/max(1,scheduled_steps-warmup_steps))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
    rng=random.Random(config["seed"])
    history=[]; best=None; stale=0; step=0; total=0; augmentation_log=[]
    start=time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        if mode=="frozen":
            model.backbone.eval()
        order=[trainset[i] for i in training_order(trainset,rng,config)]
        losses=[]
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(order),config["batch_size"]):
            original=order[offset:offset+config["batch_size"]]
            examples=[augment(e,rng,config["seed"],config.get("augmentation")) for e in original]
            augmentation_log.extend({"incident_id":e.opaque_incident_id,"epoch":epoch,**e.augmentation_metadata} for e in examples)
            encoded=[serializer(e.input) for e in examples]
            batch=collate(encoded,tokenizer.pad_token_id,device)
            targets=labels(examples,encoded,device)
            use_amp=device=="cuda"
            with torch.autocast(device_type="cuda" if use_amp else "cpu",dtype=torch.bfloat16,enabled=use_amp):
                outputs=model(**batch)
                loss=supervised_loss(*outputs,*targets,config["loss_weights"])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            total+=1
            batch_index=offset//config["batch_size"]
            batch_count=(len(order)+config["batch_size"]-1)//config["batch_size"]
            accumulation=min(config["gradient_accumulation"],batch_count-(batch_index//config["gradient_accumulation"])*config["gradient_accumulation"])
            (loss/accumulation).backward()
            losses.append(float(loss.detach()))
            last=offset+config["batch_size"]>=len(order)
            if (batch_index+1)%config["gradient_accumulation"]==0 or last:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step+=1
                if scheduler is not None: scheduler.step()
                if step%5==0:
                    print(mode,"epoch",epoch+1,"step",step,"loss",round(float(np.mean(losses[-10:])),4),flush=True)
            if step>=config["max_steps"]:
                break
        val=predict(model,tokenizer,serializer,valset,device,"train_diagnostic" if diagnostic else "model_validation")
        details=prediction_summary(val)
        val_loss=details["sum_nll"]
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,
                        "optimizer_steps":step,"validation":details,
                        "learning_rates":{g["name"]:g["lr"] for g in optimizer.param_groups}})
        print(mode,"validation",history[-1],flush=True)
        rank=selection_key(details,config)
        if best is None or rank<best:
            best=rank; stale=0
            torch.save(model.state_dict(),out/"checkpoint.pt")
            save(out/"selected_validation_logits.json",val)
        else:
            stale+=1
        if stale>=config["early_stopping_patience"] or step>=config["max_steps"]:
            break
    tokenizer.save_pretrained(out/"tokenizer")
    model.backbone.config.save_pretrained(out/"backbone_config")
    split=read(Path(config["data_dir"])/"splits.json")
    tokenizer_hash=file_hash(out/"tokenizer"/"tokenizer.json")
    metadata={"mode":mode,"config":config,"model_name":MODEL_NAME,"model_revision":MODEL_REV,
      "parameter_count":sum(p.numel() for p in model.parameters()),
      "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
      "history":history,"elapsed_seconds":time.perf_counter()-start,"environment":environment(),
      "selection":"diagnostic TRAIN subset fit; not eligible for model selection" if diagnostic else config.get("selection_metric","sum_nll"),
      "initialization":initialization,
      "diagnostic_only":bool(diagnostic),"scheduled_optimizer_steps":scheduled_steps,
      "binding":config_binding(config,file_hash(out/"checkpoint.pt"),split["split_hash"],tokenizer_hash),
      "supported_applications":sorted({e.input.application for e in trainset}),
      "prepared_data_sha256":file_hash(Path(config["data_dir"])/"examples.json"),
      "code_state":source_state,
      "train_run_ids":[e.original_run_id for e in trainset],
      "validation_run_ids":[e.original_run_id for e in valset],"ontology":FAULTS}
    save(out/"metadata.json",metadata)
    save(out/"augmentations.json",augmentation_log)
    save(out/"split_manifest.json",split)
    save(out/"resolved_config.json",config)
    return metadata

def load_checkpoint(artifact_dir,device="cpu"):
    folder=Path(artifact_dir)
    metadata=read(folder/"metadata.json")
    if file_hash(folder/"checkpoint.pt")!=metadata["binding"]["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    if file_hash(folder/"tokenizer"/"tokenizer.json")!=metadata["binding"]["tokenizer_sha256"]:
        raise ValueError("tokenizer checksum mismatch")
    cfg=metadata["config"]
    expected=config_binding(cfg,metadata["binding"]["checkpoint_sha256"],
                            metadata["binding"]["split_hash"],metadata["binding"]["tokenizer_sha256"])
    if expected!=metadata["binding"] or metadata["ontology"]!=FAULTS:
        raise ValueError("pipeline/ontology version mismatch")
    torch.set_num_threads(cfg["cpu_threads"])
    model=build_model(folder/"backbone_config",False,cfg["head_size"],pooling=cfg.get("pooling","cls"),numeric_dim=numeric_dimension(cfg),root_conditioned_fault=cfg.get("root_conditioned_fault",False),text_logit_weight=cfg.get("text_logit_weight",1.))
    model.load_state_dict(torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True),strict=True)
    model.to(device).eval()
    tok=AutoTokenizer.from_pretrained(folder/"tokenizer",local_files_only=True)
    return model,tok,Serializer(tok,cfg["max_length"],cfg.get("serializer_version",SERIALIZER),cfg.get("numeric_metrics",[]) if cfg.get("numeric_fusion",False) else [],cfg.get("numeric_feature_version","mean-v1"),cfg.get("trace_features",False)),metadata

def baseline(data_dir):
    from collections import Counter
    trainset=load_split(data_dir,"train")
    counts=Counter(e.targets.fault_type.value for e in trainset if e.targets.fault_type.value in FAULTS)
    fault_logits=[float(np.log(counts[f]+1)) for f in FAULTS]
    rows=[]
    for e in load_split(data_dir,"test"):
        ids=[c.candidate_id for c in e.input.candidates]
        scores={s:0. for s in ids}
        for m in e.input.evidence.metrics:
            if m.service in scores and m.change_z is not None:
                scores[m.service]=max(scores[m.service],abs(m.change_z))
        root=e.targets.root_cause.value; fault=e.targets.fault_type.value
        rows.append({"incident_id":e.opaque_incident_id,"run_id":e.original_run_id,"split":"test",
          "application":e.input.application,"candidate_ids":ids,
          "root_logits":[scores[s] for s in ids],"fault_logits":fault_logits,
          "root_target":ids.index(root) if root in ids else (-1 if root else None),
          "fault_target":FAULTS.index(fault) if fault in FAULTS else None,"gold":e.targets.model_dump(),
          "evidence_usable":any(m.observed_samples>0 and m.observed_mean is not None for m in e.input.evidence.metrics)})
    return rows


def code_state():
    from dulwich.repo import Repo
    from dulwich import porcelain
    repo=Repo(".")
    try:
        revision=repo.head().decode()
    except KeyError:
        revision=None
    status=porcelain.status(repo)
    return {"revision":revision,"dirty":bool(status.staged.get("add") or status.staged.get("modify") or status.staged.get("delete") or status.unstaged or status.untracked),
            "source_hashes":{str(p):file_hash(p) for p in sorted(Path("src").rglob("*.py"))}}


def prediction_summary(rows):
    """Separate head diagnostics; missing/absent candidate targets are not CE labels."""
    from .calibration import probabilities
    from sklearn.metrics import f1_score
    result={}
    joint=[]
    for head in ("root","fault"):
        valid=[r for r in rows if r[head+"_target"] is not None and r[head+"_target"]>=0]
        y=[r[head+"_target"] for r in valid]
        pred=[int(np.argmax(r[head+"_logits"])) for r in valid]
        nll=[-np.log(max(probabilities(r[head+"_logits"])[r[head+"_target"]],1e-300)) for r in valid]
        result[head]={"n":len(y),"accuracy":float(np.mean(np.array(y)==pred)) if y else None,
                      "nll":float(np.mean(nll)) if y else None}
        if head=="fault":
            result[head]["macro_f1"]=float(f1_score(y,pred,labels=list(range(len(FAULTS))),average="macro",zero_division=0)) if y else None
            result[head]["prediction_counts"]={f:pred.count(i) for i,f in enumerate(FAULTS)}
    for r in rows:
        if r["root_target"] is not None and r["fault_target"] is not None:
            joint.append(int(np.argmax(r["root_logits"]))==r["root_target"] and int(np.argmax(r["fault_logits"]))==r["fault_target"])
    result["joint_accuracy"]=float(np.mean(joint)) if joint else None
    result["sum_nll"]=sum(result[h]["nll"] or 0 for h in ("root","fault"))
    cohorts=sorted({r.get("cohort","RE1") for r in rows})
    per_cohort={}
    for cohort in cohorts:
        subset=[r for r in rows if r.get("cohort","RE1")==cohort and r["root_target"] is not None and r["fault_target"] is not None]
        correct=[int(np.argmax(r["root_logits"]))==r["root_target"] and int(np.argmax(r["fault_logits"]))==r["fault_target"] for r in subset]
        per_cohort[cohort]={"n":len(correct),"joint_accuracy":float(np.mean(correct)) if correct else None}
    result["cohorts"]=per_cohort
    values=[v["joint_accuracy"] for v in per_cohort.values() if v["joint_accuracy"] is not None]
    result["cohort_macro_joint"]=float(np.mean(values)) if values else None
    return result


def selection_key(summary,config):
    criterion=config.get("selection_metric","sum_nll")
    if criterion=="sum_nll":return (summary["sum_nll"],)
    if criterion=="cohort_macro_joint":
        return (-summary["cohort_macro_joint"],-summary["joint_accuracy"],summary["sum_nll"])
    if criterion=="cohort_guarded_retention":
        failures=0
        for app,floor in config['retention_floors'].items():
            suffix='OB' if app=='Online Boutique' else 'SS'
            groups=[v for c,v in summary['cohorts'].items() if c.endswith('-'+suffix)]
            n=sum(g['n'] for g in groups)
            value=sum(g['n']*g['joint_accuracy'] for g in groups)/n if n else 0.
            failures+=int(value+1e-12<floor)
        return (failures,-round(summary['cohort_macro_joint'],12),-round(summary['joint_accuracy'],12),summary['sum_nll'])
    if criterion=="cohort_guarded_joint":
        groups=[summary["cohorts"][c] for c in config["preserve_validation_cohorts"]]
        n=sum(c["n"] for c in groups)
        preserved=sum(c["n"]*c["joint_accuracy"] for c in groups)/n if n else 0.
        return (int(preserved+1e-12<config["preserve_joint_floor"]),-summary["cohort_macro_joint"],-summary["joint_accuracy"],summary["sum_nll"])
    raise ValueError("Unknown selection criterion")


def initialize_from_artifact(model,config,trainset,valset):
    path=config.get("initialization_artifact")
    if not path:return {"kind":"pinned_pretrained_backbone"}
    folder=Path(path);meta=read(folder/"metadata.json")
    if meta.get("diagnostic_only"):raise ValueError("Cannot promote a diagnostic checkpoint")
    if not set(meta["train_run_ids"]).issubset({e.original_run_id for e in trainset}):
        raise ValueError("Parent training examples are not confined to the new train split")
    if not set(meta["validation_run_ids"]).issubset({e.original_run_id for e in valset}):
        raise ValueError("Parent validation examples must remain validation")
    if file_hash(folder/"checkpoint.pt")!=meta["binding"]["checkpoint_sha256"]:raise ValueError("Parent checksum mismatch")
    for field,default in [("pooling","cls"),("serializer_version",SERIALIZER),("head_size",128)]:
        if meta["config"].get(field,default)!=config.get(field,default):raise ValueError("Incompatible parent "+field)
    if meta["config"].get("numeric_fusion") and meta["config"].get("numeric_metrics",[])!=config.get("numeric_metrics",[]):
        raise ValueError("Parent numeric metric vocabulary/order mismatch")
    state=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
    migrated=[]
    old_version=meta['config'].get('numeric_feature_version','mean-v1')
    new_version=config.get('numeric_feature_version','mean-v1')
    if old_version!=new_version:
        if (old_version,new_version)!=('mean-v1','temporal-v1') or config.get('numeric_feature_upgrade')!='zero_pad_temporal_v1':
            raise ValueError('Unsupported numeric feature migration')
        expected=model.state_dict()
        for name in ['numeric_root.0.weight','numeric_fault.0.weight','numeric_local_fault.0.weight']:
            if name not in state:continue
            old=state[name];target=expected[name]
            if target.shape[0]!=old.shape[0] or target.shape[1]!=2*old.shape[1]:raise ValueError('Invalid temporal expansion shape')
            expanded=torch.zeros_like(target);expanded[:,:old.shape[1]]=old
            state[name]=expanded;migrated.append(name)
    old_trace=meta['config'].get('trace_features',False);new_trace=config.get('trace_features',False)
    if old_trace!=new_trace:
        if old_trace or not new_trace or old_version!=new_version or config.get('trace_feature_upgrade')!='zero_pad_trace_v1':
            raise ValueError('Unsupported trace feature migration')
        expected=model.state_dict()
        for name in ['numeric_root.0.weight','numeric_fault.0.weight','numeric_local_fault.0.weight']:
            if name not in state:continue
            old=state[name];target=expected[name];extra=72 if name=='numeric_fault.0.weight' else 24
            if target.shape[0]!=old.shape[0] or target.shape[1]!=old.shape[1]+extra:raise ValueError('Invalid trace expansion shape')
            expanded=torch.zeros_like(target);expanded[:,:old.shape[1]]=old;state[name]=expanded;migrated.append(name)
    missing,unexpected=model.load_state_dict(state,strict=False)
    if unexpected or any(not n.startswith(("numeric_root.","numeric_fault.","numeric_local_fault.")) for n in missing):
        raise ValueError("Unexpected parent architecture mismatch")
    return {"kind":"warm_start_weights_with_new_optimizer","artifact":str(folder),"binding":meta["binding"],"new_random_parameters":missing,"zero_padded_temporal_inputs":migrated}


def training_order(examples,rng,config):
    """Resample TRAIN only; balanced draws never count as new independent cases."""
    from collections import Counter
    strategy=config.get("sampling_strategy","shuffle")
    if strategy=="shuffle":
        order=list(range(len(examples)));rng.shuffle(order);return order
    if strategy=="cohort_balanced":
        cohorts=[e.source_metadata.get("dataset_suite","unknown") for e in examples]
        counts=Counter(cohorts)
        return rng.choices(range(len(examples)),weights=[1/counts[c] for c in cohorts],k=len(examples))
    raise ValueError("Unknown training sampling strategy")
