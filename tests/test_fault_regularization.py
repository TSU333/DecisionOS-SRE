import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from decisionos_sre.model import DecisionModel, supervised_loss
from decisionos_sre.fault_regularization import configure_cached_heads
from decisionos_sre.training import train


def tiny():
    backbone=nn.Linear(8,8);backbone.config=SimpleNamespace(hidden_size=8)
    return DecisionModel(backbone,numeric_dim=12,root_conditioned_fault=True)


def inputs():
    return (torch.randn(2,8),torch.randn(2,3,8),torch.tensor([[True,True,True],[True,True,False]]),torch.randn(2,3,12),torch.randn(2,36))


def test_fault_step_keeps_root_weights_and_logits_unchanged():
    torch.manual_seed(12);m=tiny();batch=inputs();m.eval()
    before=copy.deepcopy(m.state_dict());roots_before=m.score_representations(*batch)[0].detach().clone()
    handles=configure_cached_heads(m,{'head_training_policy':'fault_heads','initialization_artifact':'parent','fault_hidden_dropout':.3,'fault_label_smoothing':.05})
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    m.train();root,fault=m.score_representations(*batch)
    loss=supervised_loss(root,fault,torch.tensor([0,1]),torch.tensor([2,3]),(0.,1.),fault_label_smoothing=.05)
    assert torch.isfinite(loss)
    loss.backward();opt.step();m.eval()
    assert torch.equal(roots_before,m.score_representations(*batch)[0])
    assert all(torch.equal(p,before[n]) for n,p in m.state_dict().items() if n.startswith(('backbone.','scorer.','numeric_root.')))
    assert not torch.equal(m.numeric_fault[-1].weight,before['numeric_fault.2.weight'])
    for h in handles:h.remove()


def test_dropout_eval_export_reload_is_exact_and_training_is_stochastic():
    torch.manual_seed(13);m=tiny();batch=inputs();original=copy.deepcopy(m);handles=configure_cached_heads(m,{'fault_hidden_dropout':.5})
    m.train();a=m.score_representations(*batch)[1];b=m.score_representations(*batch)[1];assert not torch.equal(a,b)
    m.eval();original.eval()
    assert all(torch.equal(x,y) for x,y in zip(m.score_representations(*batch),original.score_representations(*batch)))
    for h in handles:h.remove()
    original.load_state_dict(m.state_dict(),strict=True)
    assert all(torch.equal(x,y) for x,y in zip(m.score_representations(*batch),original.score_representations(*batch)))


def test_fault_smoothing_handles_padded_roots_and_missing_fault_labels():
    r=torch.tensor([[1.,2.,float('-inf')]],requires_grad=True);f=torch.randn(1,5,requires_grad=True)
    loss=supervised_loss(r,f,torch.tensor([1]),torch.tensor([0]),fault_label_smoothing=.1)
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(f.grad).all()
    missing=supervised_loss(r,f,torch.tensor([-100]),torch.tensor([-100]),fault_label_smoothing=.1)
    assert missing==0


@pytest.mark.parametrize('bad',[{'head_training_policy':'oops'},{'head_training_policy':'fault_heads'},{'fault_hidden_dropout':1.},{'fault_hidden_dropout':float('nan')},{'fault_label_smoothing':-1.}])
def test_invalid_controls_rejected(bad):
    with pytest.raises(ValueError):configure_cached_heads(tiny(),bad)


def test_uncached_trainer_does_not_silently_ignore_fault_controls():
    with pytest.raises(ValueError,match='cached frozen'):train({'head_training_policy':'fault_heads'},'sft')
