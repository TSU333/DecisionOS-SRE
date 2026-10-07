from pathlib import Path
import copy
import time
import platform
import numpy as np
import torch
import psutil
from .common import read,save,file_hash,digest,environment
from .data import load_split
from .training import load_checkpoint,predict,baseline
from .calibration import enrich,fit_calibrator
from .policy import route,select_policy
from .metrics import evaluate_metrics,plot_metrics

def _load(folder,device="cpu"):
    model,tok,ser,meta=load_checkpoint(folder,device)
    current=read(Path(meta["config"]["data_dir"])/"splits.json")
    if file_hash(Path(meta["config"]["data_dir"])/"examples.json")!=meta["prepared_data_sha256"]:
        raise ValueError("prepared input data changed since training")
    if current["split_hash"]!=meta["binding"]["split_hash"]:
        raise ValueError("current split manifest differs from training checkpoint")
    return model,tok,ser,meta

def calibrate(folder,device="cpu"):
    model,tok,ser,meta=_load(folder,device)
    examples=load_split(meta["config"]["data_dir"],"calibration")
    rows=predict(model,tok,ser,examples,device,"calibration")
    cal=fit_calibrator(rows,meta["binding"],meta["config"]["calibration_min_samples"])
    save(Path(folder)/"calibration_logits.json",rows)
    save(Path(folder)/"calibrator.json",cal)
    return cal

def policy(folder,device="cpu"):
    model,tok,ser,meta=_load(folder,device)
    cal=read(Path(folder)/"calibrator.json")
    assert cal["binding"]==meta["binding"]
    rows=list(enrich(predict(model,tok,ser,load_split(meta["config"]["data_dir"],"gate_selection"),device,"gate_selection"),cal))
    result=select_policy(rows,meta["binding"],cal,meta["config"]["target_joint_risk"],meta["config"]["min_accepted_runs"])
    save(Path(folder)/"gate_predictions.json",rows)
    save(Path(folder)/"policy.json",result)
    return result

def finish_rows(rows,cal,pol,binding):
    out=[]
    for row in enrich(rows,cal):
        row["routing"]=route(row["root_prob"],row["fault_prob"],row["evidence_usable"],cal,pol,binding)
        out.append(row)
    return out

def write_evaluation(rows,destination,bins=10):
    destination=Path(destination)
    save(destination/"predictions.json",rows)
    # Metrics are computed from the persisted round-trip, not a separate code path.
    metrics=evaluate_metrics(read(destination/"predictions.json"),bins)
    save(destination/"metrics.json",metrics)
    plot_metrics(metrics,destination/"diagnostics.png")
    return metrics

def evaluate(folder,device="cpu",ablations=True):
    model,tok,ser,meta=_load(folder,device)
    cal=read(Path(folder)/"calibrator.json")
    pol=read(Path(folder)/"policy.json")
    if cal["binding"]!=meta["binding"] or pol["binding"]!=meta["binding"]:
        raise ValueError("artifact mismatch")
    frozen={"binding":meta["binding"],"calibrator_sha256":file_hash(Path(folder)/"calibrator.json"),
            "policy_sha256":file_hash(Path(folder)/"policy.json")}
    freeze=Path(folder)/"test_freeze.json"
    if freeze.exists() and read(freeze)!=frozen:
        raise ValueError("test previously opened with a different frozen artifact")
    save(freeze,frozen)
    evaluation_split=meta["config"].get("evaluation_split","test")
    examples=load_split(meta["config"]["data_dir"],evaluation_split)
    rows=predict(model,tok,ser,examples,device,evaluation_split)
    metrics=write_evaluation(finish_rows(rows,cal,pol,meta["binding"]),Path(folder)/evaluation_split)
    if ablations:
        for variant in (["mask_all_evidence","reverse_candidates","consistent_service_rename"] if meta["config"].get("trace_features") else ["mask_all_metrics","reverse_candidates","consistent_service_rename"]):
            altered=copy.deepcopy(examples)
            for ex in altered:
                if variant in ("mask_all_metrics","mask_all_evidence"):
                    ex.input.evidence.metrics=[]
                    ex.input.evidence.traces=None
                    ex.input.modality_availability.metrics=False
                    ex.input.modality_availability.traces=False
                elif variant=="reverse_candidates":
                    ex.input.candidates.reverse()
                else:
                    names=sorted({c.candidate_id for c in ex.input.candidates}|{m.service for m in ex.input.evidence.metrics}|{t.service for t in ex.input.evidence.traces or []})
                    mapping={name:f"component_{i:02d}" for i,name in enumerate(names)}
                    for c in ex.input.candidates:
                        c.candidate_id=mapping[c.candidate_id]
                        c.display_name=mapping.get(c.display_name,c.display_name)
                    for m in ex.input.evidence.metrics:
                        m.service=mapping[m.service]
                    for t in ex.input.evidence.traces or []:t.service=mapping[t.service]
                    ex.targets.root_cause.value=mapping.get(ex.targets.root_cause.value,ex.targets.root_cause.value)
            vr=predict(model,tok,ser,altered,device,evaluation_split)
            write_evaluation(finish_rows(vr,cal,pol,meta["binding"]),Path(folder)/variant)
    # Reload reproduction is verified separately without making another model selection.
    save(Path(folder)/"reload_reference.json",rows[:2])
    return metrics

def run_baseline(data_dir,destination):
    rows=finish_rows(baseline(data_dir),None,None,{})
    return write_evaluation(rows,destination)

def benchmark(folder,config):
    from .runtime import Engine
    from .serializer import collate,Serializer
    engine=Engine(folder,"cpu")
    samples=load_split(config["data_dir"],config.get("evaluation_split","test"))
    measurements=[]
    torch.set_num_threads(config["cpu_threads"])
    for ex in samples[:3]:
        for _ in range(config["benchmark_warmup"]):
            engine.decide(ex.input)
        enc=engine.serializer(ex.input)
        batch=collate([enc],engine.tokenizer.pad_token_id,"cpu")
        for i in range(config["benchmark_repeats"]):
            t=time.perf_counter()
            with torch.inference_mode():
                engine.model(**batch)
            core=(time.perf_counter()-t)*1000
            t=time.perf_counter(); result=engine.decide(ex.input); end=(time.perf_counter()-t)*1000
            measurements.append({"incident_id":ex.opaque_incident_id,"iteration":i,
              "tokens":len(enc.input_ids),"candidates":len(enc.candidate_ids),
              "core_ms":core,"end_to_end_ms":end,"rss_bytes":psutil.Process().memory_info().rss})
    # Separate performance probes: synthetic truncations of a real incident, no accuracy claim.
    probes=[]
    ex=samples[0]
    for token_limit,count in [(512,2),(1024,7),(2048,len(ex.input.candidates))]:
        inc=ex.input.model_copy(deep=True); inc.candidates=inc.candidates[:count]
        ser=Serializer(engine.tokenizer,token_limit,engine.serializer.version,engine.serializer.numeric_metrics,engine.serializer.numeric_feature_version,engine.serializer.trace_features); enc=ser(inc)
        batch=collate([enc],engine.tokenizer.pad_token_id)
        with torch.inference_mode():
            engine.model(**batch)
            times=[]
            for _ in range(3):
                t=time.perf_counter(); engine.model(**batch); times.append((time.perf_counter()-t)*1000)
        probes.append({"synthetic_performance_probe":True,"max_tokens":token_limit,"tokens":len(enc.input_ids),
                       "candidate_count":count,"core_ms":times})
    def summary(key):
        vals=[r[key] for r in measurements]
        return {"p50_ms":float(np.percentile(vals,50)),"p95_ms":float(np.percentile(vals,95)),
                "throughput_per_second":1000/float(np.mean(vals))}
    result={"device":"cpu","cpu":platform.processor(),"logical_cpus":psutil.cpu_count(),
       "threads":torch.get_num_threads(),"batch_size":1,"warmup_per_incident":config["benchmark_warmup"],
       "repetitions_per_incident":config["benchmark_repeats"],"cold_model_load_ms":engine.cold_load_ms,
       "core":summary("core_ms"),"end_to_end":summary("end_to_end_ms"),
       "peak_process_ram_bytes":getattr(psutil.Process().memory_info(),"peak_wset",max(r["rss_bytes"] for r in measurements)),
       "peak_ram_scope":"Windows process lifetime peak working set",
       "checkpoint_bytes":(Path(folder)/"checkpoint.pt").stat().st_size,
       "measurements":measurements,"scaling_probes":probes,"environment":environment()}
    save(Path(folder)/"benchmark.json",result)
    return result
