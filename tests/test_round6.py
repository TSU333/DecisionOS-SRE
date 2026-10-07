import copy
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from torch import nn
import pytest
from pydantic import ValidationError
from test_correctness import incident,example,Tokenizer
from decisionos_sre.traces import summarize_traces,trace_features
from decisionos_sre.schema import IncidentInput,SpanSummary,ServiceTraceSummary
from decisionos_sre.serializer import Serializer
from decisionos_sre.model import DecisionModel
from decisionos_sre.training import initialize_from_artifact,config_binding
from decisionos_sre.common import read,save,file_hash


def spans():
    return pd.DataFrame({'serviceName':['emailservice']*6,'operationName':['grpc.hipstershop.EmailService/Send']*6,'startTime':[-20e6,0,10e6,20e6,59e6,61e6],'duration':[1e6,1e6,1e6,1e6,2e6,1e6],'statusCode':[0,0,14,None,4,14]})


def trace(service='db'):
    stats=SpanSummary(baseline_count=20,observed_count=10,baseline_mean_ms=1,observed_mean_ms=8,baseline_p95_ms=2,observed_p95_ms=12,baseline_error_ratio=0,observed_error_ratio=.2,observed_deadline_ratio=.1,observed_unavailable_ratio=.1,observed_known_status_fraction=.8)
    return ServiceTraceSummary(service=service,all_spans=stats,server_spans=stats,observed_until=100)


def test_only_completed_spans_and_known_status_enter_summary():
    df=spans();traces,audit=summarize_traces(df,0,60,['emailservice'])
    t=traces[0]
    assert t.all_spans.observed_count==3 and t.all_spans.baseline_count==1
    assert t.observed_until==21 and t.all_spans.observed_mean_ms==1000
    assert t.all_spans.observed_error_ratio==.5 and t.all_spans.observed_known_status_fraction==pytest.approx(2/3)
    assert t.server_spans==t.all_spans
    altered=df.copy();altered.loc[[4,5],'duration']=999999999;altered.loc[[4,5],'statusCode']=99
    assert summarize_traces(altered,0,60,['emailservice'])[0]==traces
    with pytest.raises(ValueError):summarize_traces(df,0,120,['emailservice'])


def test_schema_rejects_future_trace_wrong_availability_and_duplicate_service():
    x=incident().model_dump();x['evidence']['traces']=[trace().model_dump()]
    with pytest.raises(ValidationError):IncidentInput.model_validate(x)
    x['modality_availability']['traces']=True;IncidentInput.model_validate(x)
    x['evidence']['traces'][0]['observed_until']=101
    with pytest.raises(ValidationError):IncidentInput.model_validate(x)
    x['evidence']['traces']=[trace().model_dump()]*2
    with pytest.raises(ValidationError):IncidentInput.model_validate(x)


def test_trace_features_only_retained_evidence_and_old_prefix_unchanged():
    x=incident();old=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'])(x)
    x.evidence.traces=[trace()];x.modality_availability.traces=True
    ignored=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'])(x)
    assert old.input_ids==ignored.input_ids and old.candidate_numeric==ignored.candidate_numeric
    ser=Serializer(Tokenizer(),2048,'metrics-canonical-v2',['cpu'],'mean-v1',True);full=ser(x)
    assert full.report['trace_summaries_retained']==1
    assert full.candidate_numeric[0][:6]==old.candidate_numeric[0]
    assert full.incident_numeric[:18]==old.incident_numeric and len(full.incident_numeric)==90
    tiny=Serializer(Tokenizer(),len(old.input_ids)+1,'metrics-canonical-v2',['cpu'],'mean-v1',True)(x)
    assert tiny.report['trace_summaries_retained']==0 and not any(tiny.incident_numeric[18:])
    x.candidates.reverse();assert ser(x).input_ids==full.input_ids
    for c in x.candidates:c.candidate_id='renamed_'+c.candidate_id
    for m in x.evidence.metrics:m.service='renamed_'+m.service
    for t in x.evidence.traces:t.service='renamed_'+t.service
    renamed=ser(x);assert renamed.input_ids==full.input_ids and renamed.candidate_numeric==full.candidate_numeric


def test_trace_warm_start_zero_padding_and_binding(tmp_path):
    bb=nn.Module();bb.config=SimpleNamespace(hidden_size=8)
    old=DecisionModel(bb,numeric_dim=6,root_conditioned_fault=True);new=DecisionModel(copy.deepcopy(bb),numeric_dim=30,root_conditioned_fault=True)
    torch.save(old.state_dict(),tmp_path/'checkpoint.pt')
    cfg={'max_length':2048,'numeric_fusion':True,'numeric_metrics':['cpu']}
    save(tmp_path/'metadata.json',{'config':cfg,'train_run_ids':['run'],'validation_run_ids':[],'binding':{'checkpoint_sha256':file_hash(tmp_path/'checkpoint.pt')}})
    upgrade={**cfg,'trace_features':True,'trace_feature_version':'service-trace-v1','trace_feature_upgrade':'zero_pad_trace_v1','initialization_artifact':str(tmp_path)}
    initialize_from_artifact(new,upgrade,[example()],[])
    i=torch.randn(1,8);c=torch.randn(1,2,8);mask=torch.ones(1,2,dtype=torch.bool);n=torch.randn(1,2,6);g=torch.randn(1,18)
    a=old.score_representations(i,c,mask,n,g);b=new.score_representations(i,c,mask,torch.cat([n,torch.randn(1,2,24)],-1),torch.cat([g,torch.randn(1,72)],-1))
    assert all(torch.allclose(x,y,atol=1e-6) for x,y in zip(a,b))
    assert config_binding(cfg,'w','s','t')!=config_binding(upgrade,'w','s','t')
    upgrade.pop('trace_feature_upgrade')
    with pytest.raises(ValueError,match='migration'):initialize_from_artifact(new,upgrade,[example()],[])


def test_empty_trace_input_and_span_id_deduplication():
    df=spans().iloc[:0];result,audit=summarize_traces(df,0,60,[]);assert result==[]
    df=spans();df['traceID']=['trace']*6;df['spanID']=[str(i) for i in range(6)]
    duplicated=pd.concat([df,df.iloc[[1]]],ignore_index=True)
    a=summarize_traces(df,0,60,[])[0];b,audit=summarize_traces(duplicated,0,60,[])
    assert a==b and audit['duplicate_span_ids_removed']==1


def test_round6_preserves_all_groups_and_historical_roles():
    if not Path('data/round6/splits.json').exists():pytest.skip('audit not present yet')
    a=read('data/round5/splits.json');b=read('data/round6/splits.json')
    assert a['assignments']==b['assignments'] and a['groups']==b['groups'] and 'test' not in b['counts']
    rows=read('data/round6/examples.json');with_trace=[r for r in rows if r['input']['modality_availability']['traces']]
    assert len(rows)==400 and len(with_trace)==75 and all(r['source_metadata']['dataset_suite']=='RE2-OB' for r in with_trace)
