from pathlib import Path
import random
import time
import copy
import numpy as np
import torch
from transformers import AutoTokenizer
from .common import FAULTS, MODEL_REV, MODEL_NAME, SERIALIZER, digest, file_hash, save, read, seed_all, environment
from .data import load_split, augment
from .serializer import Serializer, collate
from .model import build_model, labels, supervised_loss

def config_binding(config,checkpoint_hash,split_hash,tokenizer_hash):
    return {"checkpoint_sha256":checkpoint_hash,"split_hash":split_hash,
            "pipeline_hash":digest({"serializer":SERIALIZER,"max_length":config["max_length"],"ontology":FAULTS}),
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
              "application":ex.input.application,"candidate_ids":encoded.candidate_ids,
              "root_logits":roots[0,:len(encoded.candidate_ids)].float().cpu().tolist(),
              "fault_logits":faults[0].float().cpu().tolist(),
              "root_target":encoded.candidate_ids.index(rid) if rid in encoded.candidate_ids else (-1 if rid is not None else None),
              "fault_target":FAULTS.index(fid) if fid in FAULTS else None,
              "gold":ex.targets.model_dump(),"evidence_usable":encoded.report["usable_metrics_retained"]>0,
              "serialization":encoded.report,"model_ms":elapsed*1000})
    return rows

def train(config,mode):
    if mode not in ["frozen","sft"]:
        raise ValueError("mode must be frozen or sft")
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
    serializer=Serializer(tokenizer,config["max_length"])
    model=build_model(config["backbone_dir"],head_size=config["head_size"]).to(device)
    if mode=="frozen":
        for p in model.backbone.parameters():
            p.requires_grad=False
    elif config["gradient_checkpointing"]:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    trainset=load_split(config["data_dir"],"train")
    valset=load_split(config["data_dir"],"model_validation")
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
              lr=config["learning_rate_"+mode],weight_decay=config["weight_decay"])
    rng=random.Random(config["seed"])
    history=[]; best=float("inf"); stale=0; step=0; total=0; augmentation_log=[]
    start=time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        if mode=="frozen":
            model.backbone.eval()
        order=list(trainset); rng.shuffle(order)
        losses=[]
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(order),config["batch_size"]):
            original=order[offset:offset+config["batch_size"]]
            examples=[augment(e,rng,config["seed"]) for e in original]
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
                if step%5==0:
                    print(mode,"epoch",epoch+1,"step",step,"loss",round(float(np.mean(losses[-10:])),4),flush=True)
            if step>=config["max_steps"]:
                break
        val=predict(model,tokenizer,serializer,valset,device,"model_validation")
        from .calibration import probabilities
        val_loss=float(np.mean([sum(-np.log(max(probabilities(r[h+"_logits"])[r[h+"_target"]],1e-300))
                   for h in ["root","fault"] if r[h+"_target"] is not None and r[h+"_target"]>=0) for r in val]))
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,"optimizer_steps":step})
        print(mode,"validation",history[-1],flush=True)
        if val_loss<best:
            best=val_loss; stale=0
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
      "selection":"minimum model_validation sum task NLL; no test access",
      "binding":config_binding(config,file_hash(out/"checkpoint.pt"),split["split_hash"],tokenizer_hash),
      "prepared_data_sha256":file_hash(Path(config["data_dir"])/"examples.json"),
      "code_state":code_state(),
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
    model=build_model(folder/"backbone_config",False,cfg["head_size"])
    model.load_state_dict(torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True),strict=True)
    model.to(device).eval()
    tok=AutoTokenizer.from_pretrained(folder/"tokenizer",local_files_only=True)
    return model,tok,Serializer(tok,cfg["max_length"]),metadata

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
