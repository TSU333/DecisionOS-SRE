import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from .common import FAULTS

class DecisionModel(nn.Module):
    def __init__(self, backbone, fault_count=len(FAULTS), head_size=128, pooling="cls", numeric_dim=0):
        super().__init__()
        self.backbone=backbone
        d=backbone.config.hidden_size
        self.scorer=nn.Sequential(nn.Linear(d*3,head_size),nn.GELU(),nn.Linear(head_size,1))
        self.fault_head=nn.Linear(d,fault_count)
        self.encoder_calls=0
        if pooling not in ("cls","mean"): raise ValueError("unknown pooling")
        self.pooling=pooling
        self.numeric_dim=numeric_dim
        if numeric_dim:
            self.numeric_root=nn.Sequential(nn.Linear(numeric_dim,64),nn.GELU(),nn.Linear(64,1))
            self.numeric_fault=nn.Sequential(nn.Linear(numeric_dim*3,128),nn.GELU(),nn.Linear(128,fault_count))

    def forward(self,input_ids,attention_mask,spans,candidate_mask,candidate_numeric=None,incident_numeric=None):
        if not candidate_mask.any(dim=1).all():
            raise ValueError("empty candidates in batch")
        lengths=attention_mask.sum(-1)[:,None]
        invalid=(spans[:,:,0]<0)|(spans[:,:,1]<=spans[:,:,0])|(spans[:,:,1]>lengths)
        if (invalid&candidate_mask).any():
            raise ValueError("span outside valid tokens")
        self.encoder_calls+=1
        h=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        incident=h[:,0,:] if self.pooling=="cls" else (h*attention_mask[:,:,None]).sum(1)/attention_mask.sum(1).clamp_min(1)[:,None]
        positions=torch.arange(h.shape[1],device=h.device)[None,None,:]
        spanmask=(positions>=spans[:,:,0,None])&(positions<spans[:,:,1,None])
        spanmask=spanmask&candidate_mask[:,:,None]
        if ((spanmask.sum(-1)==0)&candidate_mask).any():
            raise ValueError("invalid candidate span")
        candidate=torch.einsum("bkl,bld->bkd",spanmask.to(h.dtype),h)/spanmask.sum(-1).clamp_min(1)[:,:,None]
        shared=incident[:,None,:].expand_as(candidate)
        logits=self.scorer(torch.cat([shared,candidate,shared*candidate],dim=-1)).squeeze(-1)
        fault_logits=self.fault_head(incident)
        if self.numeric_dim:
            if candidate_numeric is None or incident_numeric is None: raise ValueError("numeric features required")
            logits=logits+self.numeric_root(candidate_numeric.to(h.dtype)).squeeze(-1)
            fault_logits=fault_logits+self.numeric_fault(incident_numeric.to(h.dtype))
        logits=logits.masked_fill(~candidate_mask,float("-inf"))
        return logits,fault_logits

def build_model(backbone_path, pretrained=True, head_size=128, pooling="cls", numeric_dim=0):
    conf=AutoConfig.from_pretrained(backbone_path,local_files_only=True)
    conf.reference_compile=False
    conf._attn_implementation="sdpa"
    if pretrained:
        backbone=AutoModel.from_pretrained(backbone_path,config=conf,local_files_only=True)
    else:
        backbone=AutoModel.from_config(conf)
    return DecisionModel(backbone,head_size=head_size,pooling=pooling,numeric_dim=numeric_dim)

def labels(examples, encoded, device="cpu"):
    roots=[]
    faults=[]
    for ex,tok in zip(examples,encoded):
        root=ex.targets.root_cause
        fault=ex.targets.fault_type
        roots.append(tok.candidate_ids.index(root.value)
                     if root.provenance.kind=="gold" and root.value in tok.candidate_ids else -100)
        faults.append(FAULTS.index(fault.value)
                      if fault.provenance.kind=="gold" and fault.value in FAULTS else -100)
    return torch.tensor(roots,device=device),torch.tensor(faults,device=device)

def supervised_loss(root_logits,fault_logits,root_targets,fault_targets,weights=(1.,1.)):
    zero=fault_logits.sum()*0+root_logits[torch.isfinite(root_logits)].sum()*0
    total=zero
    for logits,targets,w in zip((root_logits,fault_logits),(root_targets,fault_targets),weights):
        valid=targets.ne(-100)
        if valid.any():
            total=total+w*F.cross_entropy(logits[valid].float(),targets[valid],reduction="mean")
    return total
