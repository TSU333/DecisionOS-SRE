from dataclasses import dataclass
from .schema import IncidentInput
from .common import SERIALIZER

class UnsupportedInput(ValueError):
    pass

@dataclass
class Encoded:
    input_ids: list[int]
    candidate_ids: list[str]
    spans: list[tuple[int,int]]
    report: dict

class Serializer:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.version=SERIALIZER

    def __call__(self, incident):
        incident=IncidentInput.model_validate(incident)
        tok=self.tokenizer
        enc=lambda text: tok.encode(text,add_special_tokens=False)
        ids=[tok.cls_token_id]+enc("application: "+incident.application+"\n")
        spans=[]
        for c in incident.candidates:
            ids+=enc("\ncandidate: ")
            start=len(ids)
            ids+=enc(c.candidate_id+" "+c.display_name+" "+c.observable_description)
            if len(ids)==start:
                raise ValueError("candidate contains no effective tokens")
            spans.append((start,len(ids)))
        ids+=enc("\nmetrics; baseline previous 300 seconds; observation 60 seconds:\n")
        if len(ids)+1>self.max_length:
            raise UnsupportedInput("CANDIDATES_EXCEED_TOKEN_BUDGET")
        # Answer-independent: observed anomaly magnitude, then service/name tie-break.
        metrics=sorted(incident.evidence.metrics,
                       key=lambda m:(-abs(m.change_z or 0),m.service,m.name))
        kept=[]
        usable=0
        for m in metrics:
            fmt=lambda x: "missing" if x is None else f"{x:.4g}"
            line=f"{m.service} {m.name} unit={m.unit} base={fmt(m.baseline_mean)} now={fmt(m.observed_mean)} z={fmt(m.change_z)} missing={m.missing_fraction:.2g}\n"
            tokens=enc(line)
            if len(ids)+len(tokens)+1<=self.max_length:
                ids+=tokens
                kept.append(m.service+"/"+m.name)
                usable+=int(m.observed_samples>0 and m.observed_mean is not None)
        ids.append(tok.sep_token_id)
        return Encoded(ids,[c.candidate_id for c in incident.candidates],spans,
             {"serializer_version":self.version,"tokens":len(ids),"candidate_count":len(spans),
              "usable_metrics_retained":usable,"metrics_total":len(metrics),"metrics_retained":len(kept),
              "retained_fraction":len(kept)/len(metrics) if metrics else None,
              "retained_metric_keys":kept,"truncated":len(kept)<len(metrics),
              "missing_modalities":[k for k,v in incident.modality_availability.model_dump().items() if not v],
              "token_budgets":{"max":self.max_length,"candidate_reservation":True,"logs":0,"traces":0},
              "absolute_timestamps_serialized":False})

def collate(encoded, pad_token_id, device="cpu"):
    import torch
    n=max(len(x.input_ids) for x in encoded)
    k=max(len(x.spans) for x in encoded)
    ids=torch.full((len(encoded),n),pad_token_id,dtype=torch.long)
    attention=torch.zeros_like(ids)
    spans=torch.zeros((len(encoded),k,2),dtype=torch.long)
    mask=torch.zeros((len(encoded),k),dtype=torch.bool)
    for i,x in enumerate(encoded):
        ids[i,:len(x.input_ids)]=torch.tensor(x.input_ids)
        attention[i,:len(x.input_ids)]=1
        spans[i,:len(x.spans)]=torch.tensor(x.spans)
        mask[i,:len(x.spans)]=True
    return {name:t.to(device) for name,t in
            dict(input_ids=ids,attention_mask=attention,spans=spans,candidate_mask=mask).items()}
