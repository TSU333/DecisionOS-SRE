import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from decisionos_sre.model import DecisionModel,supervised_loss
from decisionos_sre.partial_training import configure_partial_training,accumulation_divisor,train_partial

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=8)
        self.emb=nn.Embedding(30,8);self.layers=nn.ModuleList([nn.Linear(8,8) for _ in range(4)]);self.final_norm=nn.LayerNorm(8)
    def forward(self,input_ids,attention_mask):
        h=self.emb(input_ids)
        for layer in self.layers:h=torch.tanh(layer(h))
        return SimpleNamespace(last_hidden_state=self.final_norm(h))

def make():return DecisionModel(TinyBackbone(),numeric_dim=12,root_conditioned_fault=True)
def config(count):return {'partial_backbone_layers':count,'initialization_artifact':'parent'}

@pytest.mark.parametrize('count',[0,2,4])
def test_only_declared_tail_and_fault_text_head_update(count):
    torch.manual_seed(123);m=make();names=configure_partial_training(m,config(count));before=copy.deepcopy(m.state_dict())
    batch=dict(input_ids=torch.tensor([[1,2,3],[4,5,6]]),attention_mask=torch.ones(2,3,dtype=torch.long),spans=torch.tensor([[[0,1],[1,2]]]*2),candidate_mask=torch.ones(2,2,dtype=torch.bool),candidate_numeric=torch.randn(2,2,12),incident_numeric=torch.randn(2,36))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    root,fault=m(**batch);supervised_loss(root,fault,torch.tensor([0,1]),torch.tensor([2,3])).backward()
    assert all(p.grad is None for n,p in m.named_parameters() if n not in names)
    opt.step();changed={n for n,p in m.state_dict().items() if not torch.equal(p,before[n])}
    assert changed and changed<=set(names) and 'fault_head.weight' in changed
    assert m.encoder_calls==1
    assert all(torch.equal(p,before[n]) for n,p in m.state_dict().items() if n.startswith(('numeric_','scorer.','backbone.emb.')))
    actual={int(n.split('.')[2]) for n in changed if n.startswith('backbone.layers.')}
    assert actual==set(range(4-count,4))
    assert ('backbone.final_norm.weight' in changed)==bool(count)

@pytest.mark.parametrize('count',[-1,5,True,1.5])
def test_invalid_layer_selection_rejected(count):
    with pytest.raises(ValueError):configure_partial_training(make(),config(count))

@pytest.mark.parametrize('extra',[{'cache_frozen_features':True},{'application_fault_names':['a']},{'training_view_probability':.5},{'initialization_artifact':None},{'augmentation':{'metric_dropout':.1}},{'fault_label_smoothing':.1}])
def test_incompatible_controls_are_not_silently_ignored(extra):
    with pytest.raises(ValueError):configure_partial_training(make(),config(2)|extra)

def test_partial_training_cannot_run_as_cached_frozen():
    with pytest.raises(ValueError,match='sft mode'):train_partial(config(2),'frozen')

def test_incomplete_accumulation_group_matches_group_mean_gradient():
    accumulated=torch.tensor(1.,requires_grad=True);reference=torch.tensor(1.,requires_grad=True)
    values=[1.,3.,5.];observed=[]
    for i,value in enumerate(values):
        ((accumulated-value)**2/accumulation_divisor(i,len(values),2)).backward()
        if (i+1)%2==0 or i+1==len(values):
            observed.append(accumulated.grad.item());accumulated.grad=None
    expected=[]
    for group in [values[:2],values[2:]]:
        torch.stack([(reference-v)**2 for v in group]).mean().backward();expected.append(reference.grad.item());reference.grad=None
    assert observed==expected
