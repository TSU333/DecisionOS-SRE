import pytest
import torch
from torch import nn
from types import SimpleNamespace
from test_correctness import incident,example,Tokenizer
from decisionos_sre.serializer import Serializer,collate
from decisionos_sre.model import DecisionModel,supervised_loss
from decisionos_sre.training import selection_key,initialize_from_artifact
from decisionos_sre.common import save,read

def test_cache_full_logits_and_head_gradients():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=8);self.emb=nn.Embedding(512,8)
        def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    model=DecisionModel(Backbone(),pooling='mean',numeric_dim=6)
    for p in model.backbone.parameters():p.requires_grad=False
    ser=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'])
    batch=collate([ser(incident())],0)
    full=model(**batch)
    with torch.no_grad():i,c=model.encode_representations(batch['input_ids'],batch['attention_mask'],batch['spans'],batch['candidate_mask'])
    before=model.encoder_calls
    cached=model.score_representations(i,c,batch['candidate_mask'],batch['candidate_numeric'],batch['incident_numeric'])
    assert model.encoder_calls==before
    assert all(torch.equal(a,b) for a,b in zip(full,cached))
    supervised_loss(*cached,torch.tensor([0]),torch.tensor([0])).backward()
    assert model.scorer[0].weight.grad.abs().sum()>0
    assert model.backbone.emb.weight.grad is None

def test_cache_rejects_random_augmentation():
    from decisionos_sre.cached_training import train_cached
    with pytest.raises(ValueError,match='deterministic'):
        train_cached({'augmentation':{'metric_dropout':.05}})

def test_parent_test_contamination_rejected(tmp_path):
    save(tmp_path/'metadata.json',{'train_run_ids':['outside_train'],'validation_run_ids':[]})
    with pytest.raises(ValueError,match='train split'):
        initialize_from_artifact(None,{'initialization_artifact':str(tmp_path)},[example()],[])

def test_predeclared_joint_selection_over_nll():
    a={'cohort_macro_joint':.9,'joint_accuracy':.9,'sum_nll':1.2}
    b={'cohort_macro_joint':.8,'joint_accuracy':.95,'sum_nll':.4}
    assert selection_key(a,{'selection_metric':'cohort_macro_joint'})<selection_key(b,{'selection_metric':'cohort_macro_joint'})
    assert selection_key(b,{})<selection_key(a,{})

def test_new_holdout_never_reuses_old_cases():
    from pathlib import Path
    if not Path('data/round3/splits.json').exists():pytest.skip('opt-in real dataset integration')
    old=read('data/rcaeval/splits.json');new=read('data/round3/splits.json')
    assert new['counts']=={'train':65,'model_validation':20,'calibration':20,'gate_selection':55,'regression':15,'test':25}
    for oid,split in old['assignments'].items():assert new['assignments'][oid]==('regression' if split=='test' else split)
    newtest={oid for oid,split in new['assignments'].items() if split=='test'}
    assert not newtest.intersection(old['assignments'])
    groups={}
    for oid,split in new['assignments'].items():assert groups.setdefault(new['groups'][oid],split)==split


def test_root_conditioned_fault_masks_padding_and_permutation():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=8)
    torch.manual_seed(42)
    model=DecisionModel(Backbone(),numeric_dim=6,root_conditioned_fault=True)
    nn.init.normal_(model.numeric_local_fault[-1].weight)
    inc=torch.randn(1,8);cand=torch.randn(1,3,8);nums=torch.randn(1,3,6)
    mask=torch.tensor([[True,True,False]]);glob=torch.randn(1,18)
    r,f=model.score_representations(inc,cand,mask,nums,glob)
    idx=[2,1,0]
    rp,fp=model.score_representations(inc,cand[:,idx],mask[:,idx],nums[:,idx],glob)
    assert torch.isneginf(r[0,2])
    assert torch.allclose(r[:,idx],rp) and torch.allclose(f,fp)
    cand=cand.clone();nums=nums.clone()
    cand[:,2]=1e4;nums[:,2]=1e4
    r2,f2=model.score_representations(inc,cand,mask,nums,glob)
    assert torch.allclose(f,f2)
    f.sum().backward()
    assert model.scorer[0].weight.grad is None
    assert model.numeric_local_fault[-1].weight.grad.abs().sum()>0


def test_root_conditioned_architecture_is_bound():
    from decisionos_sre.training import config_binding
    base={'max_length':2048,'numeric_fusion':True,'numeric_metrics':['cpu']}
    assert config_binding(base,'w','s','t')!=config_binding({**base,'root_conditioned_fault':True},'w','s','t')


def test_text_fusion_weight_bound_and_validated():
    from decisionos_sre.training import config_binding
    base={'max_length':2048}
    assert config_binding(base,'w','s','t')!=config_binding({**base,'text_logit_weight':.1},'w','s','t')
    backbone=nn.Module();backbone.config=SimpleNamespace(hidden_size=8)
    with pytest.raises(ValueError,match='weight'):
        DecisionModel(backbone,text_logit_weight=0)
