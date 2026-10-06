"""RCAEval RE1-OB adapter. Targets are read ONLY from pinned official index."""
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import time
import random
from collections import Counter
import numpy as np
import pandas as pd
from .common import DATA_REV, ONTOLOGY, digest, file_hash, save, read
from .schema import TrainingExample

BASE = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/" + DATA_REV + "/"

def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(str(path)+".part", "wb") as f:
                while block := r.read(1024*1024):
                    f.write(block)
            Path(str(path)+".part").replace(path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    return path

def inspect_frame(frame):
    numbers = frame.drop(columns="time")
    return {"rows":len(frame), "nonempty_rows":int(numbers.notna().any(axis=1).sum()),
            "columns":list(numbers.columns), "time_start":float(frame.time.min()),
            "time_end":float(frame.time.max()), "time_unit":"unix_seconds",
            "missing_fraction":float(numbers.isna().sum().sum()/numbers.size),
            "duplicate_timestamps":int(frame.time.duplicated().sum())}

def audit(data_dir):
    root = Path(data_dir)
    idx_path = download(BASE+"cases.parquet", root/"cases.parquet")
    card = download(BASE+"README.md", root/"README.source.md")
    idx = pd.read_parquet(idx_path)
    selected = idx[idx.dataset.eq("RE1-OB")].sort_values("case")
    assert len(selected) == 125, "Pinned subset unexpectedly changed"
    rows = selected.to_dict("records")
    def fetch(r):
        for filename in ["metrics.parquet", "inject_time.txt"]:
            download(BASE+r["case"]+"/"+filename, root/r["case"]/filename)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, rows))
    entries = []
    signatures = {}
    for r in rows:
        case = r["case"]
        f = root/case/"metrics.parquet"
        df = pd.read_parquet(f).sort_values("time")
        onset = int((root/case/"inject_time.txt").read_text().strip())
        assert onset == r["inject_time"]
        assert df.time.notna().all() and not df.time.duplicated().any()
        services = sorted({c.rsplit("_",1)[0] for c in df.columns if c != "time"})
        # Signatures exclude absolute timestamp; rounded forms conservatively group near copies.
        before = df[(df.time >= onset-300) & (df.time < onset)].drop(columns="time")
        observed = df[(df.time >= onset) & (df.time <= onset+60)].drop(columns="time")
        body = df.drop(columns="time")
        def signature(x, rounded=False):
            arr = x.to_numpy(dtype="float64")
            if rounded:
                scale = np.nanmean(np.abs(arr), axis=0)
                arr = np.round(arr / np.maximum(scale, 1e-9), 3)
            return hashlib.sha256("|".join(x.columns).encode()+arr.tobytes()).hexdigest()
        sigs = {"telemetry":signature(body), "baseline":signature(before),
                "near_window":signature(pd.concat([before.tail(60),observed]), True)}
        oid = digest({"source":DATA_REV,"case":case})[:24]
        entries.append({"opaque_incident_id":oid, "case":case, "application":r["system_name"],
             "onset":onset, "decision_time":onset+60, "oracle_onset":True,
             "root_raw":r["root_cause_service"], "fault_raw":r["fault"],
             "repetition":int(r["repetition"]), "candidate_ids":services,
             "candidate_covered":r["root_cause_service"] in services,
             "metrics":inspect_frame(df), "logs":{"present":False,"nonempty_rows":0},
             "traces":{"present":False,"nonempty_rows":0},
             "sha256":file_hash(f), "onset_sha256":file_hash(root/case/"inject_time.txt"),
             "signatures":sigs, "label_provenance":"official_injection_metadata"})
    # Union exact export/shared baseline and rounded full observed-window copies.
    parent = list(range(len(entries)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i,e in enumerate(entries):
        for kind,sig in e["signatures"].items():
            key = kind+sig
            if key in signatures:
                parent[find(i)] = find(signatures[key])
            else:
                signatures[key] = i
    for i,e in enumerate(entries):
        e["group_id"] = entries[find(i)]["opaque_incident_id"]
    result = {"source":BASE,"revision":DATA_REV,"license":"MIT; source dataset card",
              "index_sha256":file_hash(idx_path),"card_sha256":file_hash(card),
              "subset":"RE1-OB","collection":"controlled fault injection",
              "runs":len(entries),"unique_telemetry_groups":len(set(e["group_id"] for e in entries)),
              "derived_windows":0,"augmentations":0,"multi_root_excluded":0,
              "unknown_faults":sum(e["fault_raw"] not in ONTOLOGY for e in entries),
              "candidate_misses":sum(not e["candidate_covered"] for e in entries),
              "root_label_coverage":sum(bool(e["root_raw"]) for e in entries)/len(entries),
              "fault_label_coverage":sum(e["fault_raw"] in ONTOLOGY for e in entries)/len(entries),
              "mapping":ONTOLOGY,"mapping_version":"rcaeval-re1-v1","entries":entries,
              "limitations":["Collection lineage beyond published case/repetition is unavailable.",
              "Exact/rounded copy grouping cannot establish independence of all repeated experiments.",
              "Metric units are not specified by the published parquet schema. No unit conversion."]}
    result["manifest_hash"] = digest(result)
    save(root/"manifest.json",result)
    print("audit:",len(entries),"runs;",result["unique_telemetry_groups"],"groups",flush=True)
    return result

def make_splits(data_dir, seed=42):
    manifest = read(Path(data_dir)/"manifest.json")
    entries = manifest["entries"]
    rng = random.Random(seed)
    assignments = {}
    # Stratify by injected fault only. Gold is legal for splitting, never model input.
    groups = {}
    for e in entries:
        groups.setdefault(e["group_id"],[]).append(e)
    for fault in sorted(ONTOLOGY):
        eligible = sorted(g for g,es in groups.items() if es[0]["fault_raw"] == fault)
        rng.shuffle(eligible)
        n = len(eligible)
        # 25 independent runs per fault => 10/3/3/6/3.
        sizes = [int(n*.4),int(n*.12),int(n*.12),int(n*.24)]
        starts = 0
        for split,size in zip(["train","model_validation","calibration","gate_selection","test"],
                              sizes+[n-sum(sizes)]):
            for gid in eligible[starts:starts+size]:
                for e in groups[gid]:
                    assignments[e["opaque_incident_id"]] = split
            starts += size
    result = {"seed":seed,"protocol":"exploratory within-application grouped holdout",
              "manifest_hash":manifest["manifest_hash"],"assignments":assignments,
              "groups":{e["opaque_incident_id"]:e["group_id"] for e in entries},
              "counts":dict(Counter(assignments.values()))}
    assert len(assignments) == len(entries)
    result["split_hash"] = digest(result)
    save(Path(data_dir)/"splits.json",result)
    print("splits:",result["counts"],flush=True)
    return result

def prepare(data_dir):
    root=Path(data_dir)
    manifest=read(root/"manifest.json")
    splits=read(root/"splits.json")
    assert splits["manifest_hash"] == manifest["manifest_hash"]
    examples=[]
    for e in manifest["entries"]:
        df=pd.read_parquet(root/e["case"]/"metrics.parquet").sort_values("time")
        onset=e["onset"]
        baseline=df[(df.time >= onset-300)&(df.time < onset)]
        obs=df[(df.time >= onset)&(df.time <= e["decision_time"])]
        metrics=[]
        for col in df.columns:
            if col=="time":
                continue
            b=baseline[col].replace([np.inf,-np.inf],np.nan).dropna()
            o=obs[col].replace([np.inf,-np.inf],np.nan).dropna()
            bm=float(b.mean()) if len(b) else None
            om=float(o.mean()) if len(o) else None
            z=None
            if bm is not None and om is not None and len(b)>1:
                z=float(np.clip((om-bm)/max(float(b.std(ddof=0)),abs(bm)*.01,1e-6),-100,100))
            service,name=col.rsplit("_",1)
            metrics.append({"service":service,"name":name,"baseline_mean":bm,"observed_mean":om,
                            "change_z":z,"missing_fraction":1-len(o)/max(len(obs),1),
                            "baseline_samples":len(b),"observed_samples":len(o),
                            "observed_until":float(obs.time.max()) if len(obs) else e["decision_time"]})
        def target(value,raw,field):
            return {"value":value,"raw_value":raw,"provenance":{"kind":"gold",
              "source_ref":f"{DATA_REV}/cases.parquet:{e['case']}:{field}",
              "mapping_version":"rcaeval-re1-v1","collection_method":"official_injection_metadata"}}
        ex=TrainingExample.model_validate({"opaque_incident_id":e["opaque_incident_id"],
          "original_run_id":e["group_id"],"parent_incident_id":e["opaque_incident_id"],
          "source_metadata":{"case":e["case"],"revision":DATA_REV,"sha256":e["sha256"],
                             "oracle_onset":True,"synthetic":False,"split":splits["assignments"][e["opaque_incident_id"]]},
          "input":{"application":e["application"],"decision_time":e["decision_time"],
            "candidates":[{"candidate_id":s,"display_name":s} for s in e["candidate_ids"]],
            "evidence":{"metrics":metrics},"modality_availability":{"metrics":bool(metrics)}},
          "targets":{"root_cause":target(e["root_raw"],e["root_raw"],"root_cause_service"),
                     "fault_type":target(ONTOLOGY.get(e["fault_raw"]),e["fault_raw"],"fault")}})
        examples.append(ex.model_dump())
    save(root/"examples.json",examples)
    return examples

def load_split(data_dir, split):
    splits=read(Path(data_dir)/"splits.json")
    return [TrainingExample.model_validate(e) for e in read(Path(data_dir)/"examples.json")
            if splits["assignments"][e["opaque_incident_id"]]==split]

def augment(example, rng, seed, options=None):
    if example.source_metadata.get("split") != "train":
        raise ValueError("Training augmentation restricted to train")
    ex=example.model_copy(deep=True)
    options=options or {}
    shuffle=options.get("candidate_shuffle",True)
    evidence_dropout=options.get("evidence_dropout",.1)
    metric_dropout=options.get("metric_dropout",.1)
    if shuffle: rng.shuffle(ex.input.candidates)
    if rng.random()<evidence_dropout:
        ex.input.evidence.metrics=[]
    else:
        ex.input.evidence.metrics=[m for m in ex.input.evidence.metrics if rng.random()>metric_dropout]
    ex.input.modality_availability.metrics=bool(ex.input.evidence.metrics)
    ex.augmentation_metadata={"seed":seed,"parent_run":example.original_run_id,
                              "candidate_shuffle":shuffle,"metric_dropout":metric_dropout,"evidence_dropout":evidence_dropout}
    return ex
