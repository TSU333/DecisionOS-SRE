import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from decisionos_sre.model import DecisionModel, supervised_loss
from decisionos_sre.application_heads import migrate_application_heads
from decisionos_sre.fault_regularization import configure_cached_heads
from decisionos_sre.serializer import Serializer, collate
from decisionos_sre.cached_training import feature_batch
from decisionos_sre.training import config_binding
from test_correctness import Tokenizer, incident

APPS=["Online Boutique","Sock Shop"]

class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=8);self.embedding=nn.Embedding(30,8)
    def forward(self,input_ids,attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

def make(apps=APPS):
    return DecisionModel(Backbone(),numeric_dim=12,root_conditioned_fault=True,application_fault_names=apps)

def cached(batch_size=3):
    return dict(incident=torch.randn(batch_size,8),candidate=torch.randn(batch_size,2,8),candidate_mask=torch.ones(batch_size,2,dtype=torch.bool),candidate_numeric=torch.randn(batch_size,2,12),incident_numeric=torch.randn(batch_size,36))

def parent_copy(parent):
    model=make();state=copy.deepcopy(parent.state_dict())
    copied=migrate_application_heads(state,model,[],APPS,'copy_parent_fault_v1')
    assert len(copied)==20
    model.load_state_dict(state,strict=True);return model

def test_explicit_migration_preserves_initial_predictions_and_legacy_keys():
    torch.manual_seed(11);old=make([]).eval();new=parent_copy(old).eval()
    assert not any(n.startswith('application_fault_heads.') for n in old.state_dict())
    x=cached(1)
    for index in [-1,0,1]:
        assert all(torch.equal(a,b) for a,b in zip(old.score_representations(**x),new.score_representations(**x,application_index=torch.tensor([index]))))

def test_only_routed_expert_updates_and_root_shared_fallback_are_frozen():
    torch.manual_seed(12);m=make();before=copy.deepcopy(m.state_dict());x=cached(2)
    handles=configure_cached_heads(m,{'head_training_policy':'application_fault_heads','initialization_artifact':'parent'})
    assert not handles
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01,weight_decay=.1)
    roots,faults=m.score_representations(**x,application_index=torch.zeros(2,dtype=torch.long))
    supervised_loss(roots,faults,torch.tensor([0,1]),torch.tensor([2,3]),(0.,1.)).backward();opt.step()
    changed=[n for n,p in m.state_dict().items() if not torch.equal(p,before[n])]
    assert changed and all(n.startswith('application_fault_heads.0.') for n in changed)
    assert torch.equal(roots,m.score_representations(**x,application_index=torch.zeros(2,dtype=torch.long))[0])

def test_mixed_batch_matches_single_routes_and_unknown_uses_shared():
    torch.manual_seed(13);parent=make([]);m=parent_copy(parent);x=cached()
    with torch.no_grad():
        m.application_fault_heads[0]['fault_head'].bias.add_(2.)
        m.application_fault_heads[1]['fault_head'].bias.sub_(3.)
    idx=torch.tensor([0,1,-1]);roots,faults=m.score_representations(**x,application_index=idx)
    for i in range(3):
        one={k:v[i:i+1] for k,v in x.items()}
        actual=m.score_representations(**one,application_index=idx[i:i+1])[1]
        assert torch.allclose(actual,faults[i:i+1],atol=1e-6)
    assert torch.allclose(faults[2],parent.score_representations(**x)[1][2])
    pieces=[{k:v[i:i+1] for k,v in x.items()}|{'application_index':idx[i:i+1]} for i in range(3)]
    combined=feature_batch(pieces)
    assert torch.equal(combined['application_index'],idx)
    assert torch.equal(m.score_representations(**combined)[1],faults)

def test_one_encoder_call_for_mixed_experts():
    m=make();x=cached();x.pop('incident');x.pop('candidate')
    x.update(input_ids=torch.tensor([[1,2,3]]*3),attention_mask=torch.ones(3,3,dtype=torch.long),spans=torch.tensor([[[0,1],[1,2]]]*3),application_index=torch.tensor([0,1,-1]))
    roots,faults=m(**x);assert m.encoder_calls==1 and roots.shape==(3,2) and faults.shape==(3,5)

def test_public_application_routing_preserves_text_and_unknown_index():
    tok=Tokenizer();old=Serializer(tok);ser=Serializer(tok,application_fault_names=APPS)
    xs=[]
    for app,index in [('Online Boutique',0),('Sock Shop',1),('not_seen',-1)]:
        x=incident();x.application=app;enc=ser(x);xs.append(enc)
        assert enc.application_index==index
        assert enc.input_ids==old(x).input_ids and enc.spans==old(x).spans
    assert collate(xs,0)['application_index'].tolist()==[0,1,-1]
    with pytest.raises(ValueError,match='mixed application'):collate([xs[0],old(incident())],0)

def test_mapping_bound_and_reorder_or_implicit_migration_rejected():
    config={'max_length':2048,'application_fault_names':APPS}
    assert config_binding(config,'a','b','c')!=config_binding(config|{'application_fault_names':APPS[::-1]},'a','b','c')
    with pytest.raises(ValueError,match='migration'):migrate_application_heads(make([]).state_dict(),make(),[],APPS,None)
    with pytest.raises(ValueError,match='reordered'):migrate_application_heads(make().state_dict(),make(APPS[::-1]),APPS,APPS[::-1],None)
    with pytest.raises(ValueError,match='unique'):make(['same','same'])

@pytest.mark.parametrize('index',[None,torch.tensor([2]),torch.tensor([-2]),torch.tensor([0.])])
def test_invalid_routing_fails_closed(index):
    with pytest.raises(ValueError,match='[Aa]pplication'):make().score_representations(**cached(1),application_index=index)

def test_expert_dropout_removed_before_exact_reload():
    torch.manual_seed(14);m=make();x=cached();idx=torch.tensor([0,1,-1])
    handles=configure_cached_heads(m,{'head_training_policy':'application_fault_heads','initialization_artifact':'parent','fault_hidden_dropout':.5})
    assert len(handles)==4
    m.train();assert not torch.equal(m.score_representations(**x,application_index=idx)[1],m.score_representations(**x,application_index=idx)[1])
    m.eval();a=m.score_representations(**x,application_index=idx)[1]
    for h in handles:h.remove()
    reload=make().eval();reload.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(a,reload.score_representations(**x,application_index=idx)[1])
