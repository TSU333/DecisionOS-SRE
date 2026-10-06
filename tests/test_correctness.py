import copy
import json
from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError
from decisionos_sre.schema import IncidentInput, TrainingExample
from decisionos_sre.serializer import Serializer, UnsupportedInput
from decisionos_sre.common import FAULTS, read, digest
from decisionos_sre.calibration import fit_calibrator,enrich
from decisionos_sre.policy import route,select_policy
from decisionos_sre.metrics import probability_metrics,evaluate_metrics
from decisionos_sre.data import augment

class Tokenizer:
    cls_token_id=1
    sep_token_id=2
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):
        return [ord(c)+3 for c in text]

def incident():
    return IncidentInput.model_validate({"application":"app","decision_time":100,
      "candidates":[{"candidate_id":"api","display_name":"api"},{"candidate_id":"db","display_name":"db"}],
      "evidence":{"metrics":[{"service":"db","name":"cpu","baseline_mean":1,"observed_mean":5,
        "change_z":4,"missing_fraction":0,"baseline_samples":50,"observed_samples":10,"observed_until":100}]},
      "modality_availability":{"metrics":True}})

def example():
    target=lambda value: {"value":value,"raw_value":value,"provenance":{"kind":"gold","source_ref":"case_db_cpu_1",
                  "mapping_version":"1","collection_method":"injection"}}
    return TrainingExample.model_validate({"opaque_incident_id":"opaque","original_run_id":"run",
       "parent_incident_id":"opaque","source_metadata":{"case":"case_db_cpu_1","split":"train"},
       "input":incident(),"targets":{"root_cause":target("db"),"fault_type":target(FAULTS[0])}})

def row(i=0,root_target=0):
    return {"incident_id":str(i),"run_id":str(i),"split":"calibration","application":"app",
       "candidate_ids":["api","db"],"root_logits":[2.,0.],"fault_logits":[2.,0.,0.,0.,0.],
       "root_target":root_target,"fault_target":0,"evidence_usable":True}

def test_schema_rejects_labels_and_future_observation():
    x=incident().model_dump(); x["targets"]={"root_cause":"db"}
    with pytest.raises(ValidationError): IncidentInput.model_validate(x)
    x=incident().model_dump(); x["candidates"][1]["candidate_id"]="api"
    with pytest.raises(ValidationError): IncidentInput.model_validate(x)
    x=incident().model_dump(); x["candidates"]=[]
    with pytest.raises(ValidationError): IncidentInput.model_validate(x)
    x=incident().model_dump(); x["evidence"]["metrics"][0]["observed_until"]=101
    with pytest.raises(ValidationError): IncidentInput.model_validate(x)
    x=incident().model_dump(); x["evidence"]["root_cause.txt"]="db"
    with pytest.raises(ValidationError): IncidentInput.model_validate(x)

def test_serializer_whitelist_gold_independence_and_budget():
    ser=Serializer(Tokenizer(),2048)
    ex=example(); first=ser(ex.input)
    ex.targets.root_cause.value="api"; ex.source_metadata["case"]="path_secret"
    ex.original_run_id="injection_answer"; ex.input.decision_time=999999
    assert ser(ex.input).input_ids==first.input_ids
    assert first.candidate_ids==["api","db"]
    for (start,end),cid in zip(first.spans,first.candidate_ids):
        assert cid in "".join(chr(t-3) for t in first.input_ids[start:end])
    # Candidate reservation survives evidence truncation.
    reduced=Serializer(Tokenizer(),170)(incident())
    assert reduced.candidate_ids==first.candidate_ids
    assert reduced.report["truncated"] and reduced.report["metrics_retained"]==0
    with pytest.raises(UnsupportedInput): Serializer(Tokenizer(),20)(incident())

def test_candidate_reorder_alignment_and_augmentation():
    import random
    ex=example(); ser=Serializer(Tokenizer())
    ex.input.candidates.reverse()
    encoded=ser(ex.input)
    assert encoded.candidate_ids.index(ex.targets.root_cause.value)==0
    aug=augment(ex,random.Random(42),42)
    assert aug.original_run_id==ex.original_run_id
    assert aug.parent_incident_id==ex.parent_incident_id
    ex.source_metadata["split"]="test"
    with pytest.raises(ValueError): augment(ex,random.Random(42),42)

def test_shared_forward_masks_and_missing_supervision():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from decisionos_sre.model import DecisionModel,supervised_loss,labels
    from decisionos_sre.serializer import collate
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(hidden_size=8)
            self.emb=nn.Embedding(512,8); self.calls=0
        def forward(self,input_ids,attention_mask):
            self.calls+=1
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    first=incident(); second=incident(); second.candidates=second.candidates[:1]
    tok=Tokenizer(); ser=Serializer(tok)
    enc=[ser(first),ser(second)]
    net=DecisionModel(TinyBackbone())
    r,f=net(**collate(enc,0))
    assert net.backbone.calls==1
    assert r.shape==(2,2) and f.shape==(2,5)
    assert torch.softmax(r,dim=-1)[1,1]==0
    absent=torch.tensor([-100,-100])
    loss=supervised_loss(r,f,absent,absent)
    assert torch.isfinite(loss) and loss==0
    loss.backward()
    r,f=net(**collate(enc,0))
    loss=supervised_loss(r,f,torch.tensor([1,-100]),absent)
    assert torch.isfinite(loss)
    assert float(loss)==pytest.approx(float(torch.nn.functional.cross_entropy(r[:1],torch.tensor([1]))))
    ex=example(); ex.targets.root_cause.value="missing"
    roots,faults=labels([ex],[enc[0]])
    assert roots.item()==-100 and faults.item()==0

def test_temperature_split_guard_and_argmax():
    rows=[row(i) for i in range(12)]
    cal=fit_calibrator(rows,{"model":"a"},10)
    assert cal["root"]["status"]=="fitted" and cal["root"]["temperature"]>0
    enriched=list(enrich(rows,cal))
    assert all(np.argmax(r["root_raw"])==np.argmax(r["root_prob"]) for r in enriched)
    assert cal["root"]["nll_after"]<=cal["root"]["nll_before"]
    rows[0]["split"]="test"
    with pytest.raises(ValueError): fit_calibrator(rows,{},10)
    assert fit_calibrator([row()],{},10)["root"]["status"]=="insufficient_samples"

def test_policy_no_feasible_singleton_and_binding():
    cal=fit_calibrator([row(i) for i in range(12)],{"model":"a"},10)
    rows=list(enrich([row(i) for i in range(30)],cal))
    for r in rows: r["split"]="gate_selection"
    pol=select_policy(rows,{"model":"a"},cal,.05,30)
    assert pol["threshold"] is not None
    assert route([.9,.1],[.9,.025,.025,.025,.025],True,cal,pol,{"model":"b"})["destination"]=="REVIEW"
    assert "SINGLE_CANDIDATE" in route([1.],[1.],True,cal,pol,{"model":"a"})["reason_codes"]
    for r in rows: r["joint_correct"]=False
    pol=select_policy(rows,{"model":"a"},cal,.05,30)
    assert pol["threshold"] is None
    assert route([.999,.001],[.99,.01],True,cal,pol,{"model":"a"})["destination"]=="REVIEW"
    rows[0]["split"]="test"
    with pytest.raises(ValueError): select_policy(rows,{},cal)

def test_hand_computed_probability_metrics_and_missing_candidates():
    rows=list(enrich([row(0),row(1,-1),row(2,None)]))
    for r in rows:
        r["root_prob"]=[.75,.25]; r["root_raw"]=[.75,.25]
        r["routing"]={"destination":"REVIEW"}
    p=probability_metrics(rows,"root")
    assert p["n"]==1
    assert p["nll"]==pytest.approx(-np.log(.75))
    assert p["brier"]==pytest.approx(.125)
    assert p["ece"]==pytest.approx(.25)
    result=evaluate_metrics(rows)
    assert result["root"]["candidate_coverage"]==.5
    assert result["root"]["acc_at_1"]==.5
    assert result["root"]["mrr"]==.5
    assert result["selective"]["coverage"]==0
    assert result["selective"]["selective_risk"] is None
    assert result["selective"]["selective_accuracy"] is None
    rows[1]["routing"]={"destination":"ACCEPT_DIAGNOSIS"}
    assert evaluate_metrics(rows)["selective"]["selective_risk"]==1.
    empty=evaluate_metrics([])
    assert empty["root"]["acc_at_1"] is None

def test_api_unavailable_has_no_random_fallback(tmp_path):
    from fastapi.testclient import TestClient
    from decisionos_sre.api import create_app
    with TestClient(create_app(tmp_path/"missing")) as client:
        assert client.get("/health").status_code==200
        assert client.get("/ready").status_code==503
        assert client.post("/v1/decide",json=incident().model_dump()).status_code==503
        x=incident().model_dump();x["targets"]={"root_cause":"db"}
        assert client.post("/v1/decide",json=x).status_code==422

def test_real_manifest_integration_if_present():
    p=Path("data/rcaeval")
    if not (p/"examples.json").exists():
        pytest.skip("real data integration is opt-in; no downloads")
    from decisionos_sre.data import load_split
    split=read(p/"splits.json"); manifests=read(p/"manifest.json")
    assert split["manifest_hash"]==manifests["manifest_hash"]
    groups={}
    for oid,name in split["assignments"].items():
        group=split["groups"][oid]
        assert groups.setdefault(group,name)==name
    assert set(split["assignments"].values())=={"train","model_validation","calibration","gate_selection","test"}
    for name in set(split["assignments"].values()):
        for ex in load_split(p,name):
            assert max(m.observed_until for m in ex.input.evidence.metrics)<=ex.input.decision_time
            # Only causal telemetry summary is visible to the serializer.
            tokens=Serializer(Tokenizer(),2048)(ex.input).input_ids
            text="".join(chr(t-3) for t in tokens if t>2)
            assert ex.source_metadata["case"] not in text
            assert ex.opaque_incident_id not in text

def test_adapter_future_rows_do_not_affect_summary(tmp_path):
    import pandas as pd
    from decisionos_sre.data import prepare
    from decisionos_sre.common import save
    case="fixture_db_cpu_1"; (tmp_path/case).mkdir()
    frame=pd.DataFrame({"time":[90,99,100,160,161],"db_cpu":[1.,3.,4.,6.,1e12]})
    frame.to_parquet(tmp_path/case/"metrics.parquet")
    entry={"case":case,"opaque_incident_id":"o","group_id":"g","onset":100,"decision_time":160,
           "application":"fixture","candidate_ids":["db"],"root_raw":"db","fault_raw":"cpu","sha256":"fixture"}
    save(tmp_path/"manifest.json",{"manifest_hash":"fixture","entries":[entry]})
    save(tmp_path/"splits.json",{"manifest_hash":"fixture","assignments":{"o":"train"}})
    result=prepare(tmp_path)[0]
    m=result["input"]["evidence"]["metrics"][0]
    assert m["baseline_mean"]==2.
    assert m["observed_mean"]==5.
    frame.loc[4,"db_cpu"]=-1e12
    frame.to_parquet(tmp_path/case/"metrics.parquet")
    assert prepare(tmp_path)[0]["input"]==result["input"]

def test_missing_observations_and_artifact_integrity(tmp_path):
    from decisionos_sre.runtime import Engine
    from decisionos_sre.common import save
    x=incident(); x.evidence.metrics[0].observed_mean=None
    x.evidence.metrics[0].observed_samples=0
    assert Serializer(Tokenizer())(x).report["usable_metrics_retained"]==0
    data={"temperature":1.}; data["id"]=digest(data)
    save(tmp_path/"cal.json",data)
    assert Engine._optional(tmp_path/"cal.json")==data
    data["temperature"]=2.; save(tmp_path/"cal.json",data)
    with pytest.raises(ValueError): Engine._optional(tmp_path/"cal.json")
