import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from test_correctness import incident,Tokenizer,example
from test_training_views import frame as view_frame
from decisionos_sre.dynamics import dynamics_summary,VERSION
from decisionos_sre.temporal import temporal_summary
from decisionos_sre.serializer import Serializer
from decisionos_sre.representation import numeric_features,numeric_dimension
from decisionos_sre.model import DecisionModel
from decisionos_sre.training import initialize_from_artifact,config_binding
from decisionos_sre.common import save,file_hash
from decisionos_sre.training_views import causal_view


def frame(observed=None):
 t=np.arange(-301,62);y=np.ones(len(t));y[(t>=0)&(t<=60)]=np.arange(61)%2+1 if observed is None else observed
 return pd.DataFrame({'time':t,'x':y})

def test_future_and_prebaseline_data_cannot_change_dynamics():
 f=frame();a=dynamics_summary(f,'x',0,60);assert a is not None
 f.loc[(f.time< -300)|(f.time>60),'x']=1e12
 assert dynamics_summary(f,'x',0,60)==a
 assert dynamics_summary(f.sample(frac=1,random_state=1),'x',0,60)==a
 with pytest.raises(ValueError):dynamics_summary(f,'x',0,61)

def test_log_quantiles_retain_amplitude_lost_at_old_cap():
 a=frame(np.full(61,3.));b=frame(np.full(61,30.))
 assert temporal_summary(a,'x',0,60).q90_z==temporal_summary(b,'x',0,60).q90_z==100.
 assert dynamics_summary(a,'x',0,60).q90_z_log<dynamics_summary(b,'x',0,60).q90_z_log
 assert dynamics_summary(a,'x',0,60).roughness_log==0.

def test_waveform_correlation_and_jumps_distinguish_sequences():
 alternating=dynamics_summary(frame(),'x',0,60)
 smooth=dynamics_summary(frame(np.linspace(1,2,61)),'x',0,60)
 assert alternating.lag1_correlation==pytest.approx(-1.)
 assert smooth.lag1_correlation==pytest.approx(1.)
 assert alternating.max_jump_log>smooth.max_jump_log

def test_missing_flat_duplicate_timestamps_and_gap_pairs():
 f=frame(np.ones(61));a=dynamics_summary(f,'x',0,60)
 assert all(v==0 for v in a.model_dump().values())
 duplicated=pd.concat([f,f.loc[f.time==10]]);assert dynamics_summary(duplicated,'x',0,60)==a
 f.loc[(f.time>=20)&(f.time<=40),'x']=np.nan;f.loc[(f.time>40)&(f.time<=60),'x']=101
 d=dynamics_summary(f,'x',0,60);assert d.max_jump_log==0. and d.roughness_log==0.
 f.loc[f.time>=0,'x']=np.nan;assert dynamics_summary(f,'x',0,60) is None
 assert dynamics_summary(frame(),'x',0,4) is None

def test_legacy_input_and_new_prefix_are_identical_and_missing_zeroes():
 inc=incident();inc.evidence.metrics[0].temporal=temporal_summary(frame(),'x',0,60)
 old=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],'temporal-v1')
 before=old(inc);inc.evidence.metrics[0].dynamics=dynamics_summary(frame(),'x',0,60);after=old(inc)
 assert before.input_ids==after.input_ids and before.candidate_ids==after.candidate_ids and before.spans==after.spans
 assert before.candidate_numeric==after.candidate_numeric and before.incident_numeric==after.incident_numeric
 new=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],VERSION)(inc)
 assert new.input_ids==before.input_ids and len(new.candidate_numeric[0])==19 and len(new.incident_numeric)==57
 assert [row[:12] for row in new.candidate_numeric]==before.candidate_numeric and new.incident_numeric[:36]==before.incident_numeric
 assert new.report['numeric_evidence_usable']
 inc.evidence.metrics[0].dynamics=None;missing=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],VERSION)(inc)
 assert not missing.report['numeric_evidence_usable'] and all(not any(row[12:]) for row in missing.candidate_numeric)
 nums,glob=numeric_features([],['db'],['cpu'],VERSION);assert nums==[[0.]*19] and glob==[0.]*57
 assert numeric_dimension({'numeric_fusion':True,'numeric_metrics':['cpu'],'numeric_feature_version':VERSION})==19

def test_zero_padding_migration_and_version_binding(tmp_path):
 backbone=nn.Module();backbone.config=SimpleNamespace(hidden_size=8)
 old=DecisionModel(backbone,numeric_dim=12,root_conditioned_fault=True);new=DecisionModel(copy.deepcopy(backbone),numeric_dim=19,root_conditioned_fault=True)
 torch.save(old.state_dict(),tmp_path/'checkpoint.pt')
 cfg={'numeric_fusion':True,'numeric_metrics':['cpu'],'numeric_feature_version':'temporal-v1'}
 save(tmp_path/'metadata.json',{'train_run_ids':['run'],'validation_run_ids':[],'binding':{'checkpoint_sha256':file_hash(tmp_path/'checkpoint.pt')},'config':cfg})
 expanded=cfg|{'initialization_artifact':str(tmp_path),'numeric_feature_version':VERSION,'numeric_feature_upgrade':'zero_pad_dynamics_v1'}
 init=initialize_from_artifact(new,expanded,[example()],[]);assert len(init['zero_padded_temporal_inputs'])==3
 inc=torch.randn(2,8);cand=torch.randn(2,3,8);mask=torch.ones(2,3,dtype=torch.bool);numeric=torch.randn(2,3,12);glob=torch.randn(2,36)
 a=old.score_representations(inc,cand,mask,numeric,glob)
 b=new.score_representations(inc,cand,mask,torch.cat([numeric,torch.randn(2,3,7)],-1),torch.cat([glob,torch.randn(2,21)],-1))
 assert all(torch.allclose(x,y,atol=1e-6) for x,y in zip(a,b))
 with pytest.raises(ValueError,match='migration'):initialize_from_artifact(new,expanded|{'numeric_feature_upgrade':None},[example()],[])
 assert config_binding(cfg|{'max_length':2048},'a','b','c')!=config_binding(expanded|{'max_length':2048},'a','b','c')

def test_derived_training_view_never_retains_hidden_dynamics():
 ex=example();f=view_frame();ex.input.evidence.metrics[0].dynamics=dynamics_summary(f,'db_cpu',40,100)
 view=causal_view(ex,f,40,'lag15');assert view.input.evidence.metrics[0].dynamics is not None
 f.loc[f.time>85,'db_cpu']=1e9
 assert causal_view(ex,f,40,'lag15').input==view.input
 assert view.input.evidence.metrics[0].dynamics!=ex.input.evidence.metrics[0].dynamics
