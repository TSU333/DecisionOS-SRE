import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytest
from test_correctness import incident,Tokenizer,example
from decisionos_sre.temporal import temporal_summary
from decisionos_sre.serializer import Serializer
from decisionos_sre.representation import numeric_features
from decisionos_sre.training import initialize_from_artifact,config_binding,selection_key
from decisionos_sre.model import DecisionModel
from decisionos_sre.common import read,save,file_hash


def test_causal_temporal_stats_future_and_old_outside_window_ignored():
    df=pd.DataFrame({'time':[-400,-20,-10,0,15,30,45,60,61], 'x':[999,1,1,1,2,3,4,5,999]})
    result=temporal_summary(df,'x',0,60)
    assert result.q10_z==pytest.approx(40.) and result.q90_z==100.
    assert result.trend_z==100. and result.late_shift_z==100.
    altered=df.copy();altered.loc[[0,8],'x']=-999999
    assert temporal_summary(altered,'x',0,60)==result
    assert temporal_summary(df,'x',0,15) is None
    with pytest.raises(ValueError):temporal_summary(df,'x',0,61)


def test_temporal_missing_and_flat_are_finite():
    df=pd.DataFrame({'time':[-20,-10,0,15,30,45,60],'x':[0]*7})
    result=temporal_summary(df,'x',0,60)
    assert all(x==0 for x in result.model_dump().values())
    df.x=np.nan
    assert temporal_summary(df,'x',0,60) is None


def test_old_text_and_features_unaffected_by_new_temporal_fields():
    inc=incident();ser=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'])
    before=ser(inc)
    from decisionos_sre.schema import TemporalSummary
    inc.evidence.metrics[0].temporal=TemporalSummary(q10_z=1,q90_z=3,std_ratio=2,trend_z=4,late_shift_z=5)
    after=ser(inc)
    assert before.input_ids==after.input_ids and before.candidate_ids==after.candidate_ids
    assert before.candidate_numeric==after.candidate_numeric and before.incident_numeric==after.incident_numeric
    temporal=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],'temporal-v1')(inc)
    assert temporal.input_ids==before.input_ids
    assert temporal.candidate_numeric[0][:6]==before.candidate_numeric[0]
    assert temporal.incident_numeric[:18]==before.incident_numeric
    assert temporal.report['numeric_evidence_usable']
    masked=Serializer(Tokenizer(),170,'metrics-canonical-v2',['cpu'],'temporal-v1')(inc)
    assert not masked.report['numeric_evidence_usable']
    assert not any(masked.incident_numeric)
    assert not Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],'temporal-v1')(incident()).report['numeric_evidence_usable']


def test_temporal_warm_start_preserves_old_logits_before_training(tmp_path):
    backbone=nn.Module();backbone.config=SimpleNamespace(hidden_size=8)
    old=DecisionModel(backbone,numeric_dim=6,root_conditioned_fault=True)
    new=DecisionModel(copy.deepcopy(backbone),numeric_dim=12,root_conditioned_fault=True)
    torch.save(old.state_dict(),tmp_path/'checkpoint.pt')
    config={'numeric_fusion':True,'numeric_metrics':['cpu']}
    save(tmp_path/'metadata.json',{'train_run_ids':['run'],'validation_run_ids':[], 'binding':{'checkpoint_sha256':file_hash(tmp_path/'checkpoint.pt')},'config':config})
    cfg={**config,'initialization_artifact':str(tmp_path),'numeric_feature_version':'temporal-v1','numeric_feature_upgrade':'zero_pad_temporal_v1'}
    migrated=initialize_from_artifact(new,cfg,[example()],[])
    assert len(migrated['zero_padded_temporal_inputs'])==3
    inc=torch.randn(1,8);cand=torch.randn(1,2,8);mask=torch.ones(1,2,dtype=torch.bool);num=torch.randn(1,2,6);glob=torch.randn(1,18)
    a=old.score_representations(inc,cand,mask,num,glob)
    b=new.score_representations(inc,cand,mask,torch.cat([num,torch.randn_like(num)],-1),torch.cat([glob,torch.randn_like(glob)],-1))
    assert all(torch.allclose(x,y,atol=1e-6) for x,y in zip(a,b))
    cfg.pop('numeric_feature_upgrade')
    with pytest.raises(ValueError,match='migration'):initialize_from_artifact(new,cfg,[example()],[])


def test_temporal_version_is_bound_and_unknown_rejected():
    cfg={'max_length':2048,'numeric_fusion':True,'numeric_metrics':['cpu']}
    assert config_binding(cfg,'w','s','t')!=config_binding({**cfg,'numeric_feature_version':'temporal-v1'},'w','s','t')
    with pytest.raises(ValueError):Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],'unknown')


def test_round5_never_promotes_old_test_to_train():
    if not Path('data/round5/splits.json').exists():pytest.skip('local audit not present')
    old=read('data/round4/splits.json');new=read('data/round5/splits.json')
    assert len(new['assignments'])==400 and 'test' not in new['counts']
    for oid,split in old['assignments'].items():assert new['assignments'][oid]==('regression_ss' if split=='test' else split)
