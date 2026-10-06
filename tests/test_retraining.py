import random
import copy
import numpy as np
import pytest
from test_correctness import Tokenizer,incident,example
from decisionos_sre.serializer import Serializer,collate
from decisionos_sre.training import config_binding,prediction_summary
from decisionos_sre.common import SERIALIZER,FAULTS,digest
from decisionos_sre.data import augment

def test_legacy_binding_and_new_preprocessing_binding():
    cfg={"max_length":2048}
    old=config_binding(cfg,"weights","split","tokenizer")
    assert old["pipeline_hash"]==digest({"serializer":SERIALIZER,"max_length":2048,"ontology":FAULTS})
    for key,value in [("pooling","mean"),("serializer_version","metrics-canonical-v2"),("numeric_fusion",True),("numeric_metrics",["cpu"])]:
        assert config_binding(dict(cfg,**{key:value}),"weights","split","tokenizer")!=old

def test_canonical_rename_order_gold_invariance_and_ids():
    ser=Serializer(Tokenizer(),2048,"metrics-canonical-v2",["cpu"])
    ex=example();a=ser(ex.input)
    other=ex.model_copy(deep=True)
    mapping={"api":"zebra","db":"apple"}
    for c in other.input.candidates:c.candidate_id=mapping[c.candidate_id];c.display_name=c.candidate_id
    for m in other.input.evidence.metrics:m.service=mapping[m.service]
    other.input.candidates.reverse()
    b=ser(other.input)
    assert a.input_ids==b.input_ids and a.spans==b.spans
    assert b.candidate_ids==[mapping[x] for x in a.candidate_ids]
    assert a.candidate_numeric==b.candidate_numeric and a.incident_numeric==b.incident_numeric
    other.targets.root_cause.value="other";other.source_metadata={"answer":"secret"}
    assert ser(other.input).input_ids==b.input_ids
    assert ex.input.candidates[0].candidate_id=="api"

def test_numeric_only_retained_evidence_and_missing():
    ser=Serializer(Tokenizer(),100,"metrics-canonical-v2",["cpu"])
    # Candidate reservation may need more than 100 char tokens; find first valid budget.
    from decisionos_sre.serializer import UnsupportedInput
    for budget in range(100,500):
        try:e=Serializer(Tokenizer(),budget,"metrics-canonical-v2",["cpu"])(incident());break
        except UnsupportedInput:continue
    assert e.report["metrics_retained"]==0
    assert np.asarray(e.candidate_numeric).sum()==0 and sum(e.incident_numeric)==0
    x=incident();x.evidence.metrics[0].observed_mean=None;x.evidence.metrics[0].observed_samples=0
    e=Serializer(Tokenizer(),2048,"metrics-canonical-v2",["cpu"])(x)
    assert e.report["usable_metrics_retained"]==0
    idx=e.candidate_ids.index("db");assert e.candidate_numeric[idx][-1]==0

def test_numeric_model_single_call_mask_and_gradients():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from decisionos_sre.model import DecisionModel,supervised_loss
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=8);self.emb=nn.Embedding(512,8);self.calls=0
        def forward(self,input_ids,attention_mask):
            self.calls+=1;return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    ser=Serializer(Tokenizer(),2048,"metrics-canonical-v2",["cpu"])
    a=incident();b=incident();b.candidates=b.candidates[:1]
    encoded=[ser(a),ser(b)]
    model=DecisionModel(Backbone(),pooling="mean",numeric_dim=6)
    roots,faults=model(**collate(encoded,0))
    assert model.backbone.calls==1 and torch.softmax(roots,dim=-1)[1,1]==0
    loss=supervised_loss(roots,faults,torch.tensor([encoded[0].candidate_ids.index("db"),-100]),torch.tensor([0,1]))
    loss.backward()
    assert model.numeric_root[0].weight.grad.abs().sum()>0
    assert model.numeric_fault[0].weight.grad.abs().sum()>0
    assert model.backbone.emb.weight.grad.abs().sum()>0
    model.eval()
    one=model(**collate(encoded[:1],0))
    with torch.no_grad(): batch=model(**collate(encoded,0))
    assert torch.allclose(one[0][0],batch[0][0],atol=1e-6)

def test_zero_dropout_and_diagnostic_missing_labels():
    ex=example();a=augment(ex,random.Random(4),4,{"candidate_shuffle":False,"evidence_dropout":0,"metric_dropout":0})
    assert a.input==ex.input and a is not ex
    ex.source_metadata["split"]="calibration"
    with pytest.raises(ValueError):augment(ex,random.Random(4),4,{"metric_dropout":0})
    rows=[{"root_target":-1,"fault_target":0,"root_logits":[1.],"fault_logits":[2.,0,0,0,0]}]
    d=prediction_summary(rows)
    assert d["root"]["n"]==0 and d["joint_accuracy"]==0
