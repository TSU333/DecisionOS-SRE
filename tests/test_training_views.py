import copy,random
import numpy as np
import pandas as pd
import pytest
from test_correctness import example
from decisionos_sre.common import save,file_hash
from decisionos_sre.training_views import causal_view,load_training_views,eligible_parent_indices,choose_view_indices


def frame():
    t=np.arange(-301,102)+40
    return pd.DataFrame({'time':t,'db_cpu':np.where(t<40,1.+.01*(t%3),5.+(t-40)/60)})


def test_views_causal_missing_counts_labels_and_identity():
    parent=example();f=frame();lag=causal_view(parent,f,40,'lag15');gap=causal_view(parent,f,40,'gap15_30')
    a=lag.input.evidence.metrics[0];b=gap.input.evidence.metrics[0]
    assert a.observed_samples==46 and b.observed_samples==46
    assert a.missing_fraction==pytest.approx(15/61) and b.missing_fraction==pytest.approx(15/61)
    assert a.observed_until==85 and b.observed_until==100
    assert a.baseline_samples==300 and a.temporal is not None and b.temporal is not None
    assert lag.targets==parent.targets and lag.original_run_id==parent.original_run_id
    assert lag.parent_incident_id==parent.opaque_incident_id and lag.opaque_incident_id!=parent.opaque_incident_id
    assert lag.input.decision_time==parent.input.decision_time
    altered=f.copy();altered.loc[(altered.time>100)|(altered.time< -260),'db_cpu']=99999999
    assert causal_view(parent,altered,40,'lag15').input==lag.input
    assert parent.input.evidence.metrics[0].observed_samples==10


def test_hidden_values_cannot_affect_view_and_no_fabricated_temporal():
    parent=example();f=frame();original=causal_view(parent,f,40,'lag15')
    f.loc[f.time>85,'db_cpu']=-999999
    assert causal_view(parent,f,40,'lag15').input==original.input
    f.loc[f.time>=40,'db_cpu']=np.nan
    empty=causal_view(parent,f,40,'gap15_30').input.evidence.metrics[0]
    assert empty.observed_samples==0 and empty.observed_mean is None and empty.temporal is None and empty.missing_fraction==1


@pytest.mark.parametrize('role',['model_validation','calibration','gate_selection','regression_re2_ob'])
def test_holdout_views_rejected(role):
    ex=example();ex.source_metadata['split']=role
    with pytest.raises(ValueError,match='TRAIN'):causal_view(ex,frame(),40,'lag15')


def test_loader_rejects_tampered_checksum_labels_and_parent(tmp_path):
    parent=example();view=causal_view(parent,frame(),40,'lag15');path=tmp_path/'views.json'
    def cfg():return {'training_view_probability':.5,'training_views_file':str(path),'training_views_sha256':file_hash(path)}
    save(path,[view.model_dump()]);loaded,meta=load_training_views(cfg(),[parent]);assert len(loaded)==1 and meta['independent_new_cases']==0
    wrong=cfg();wrong['training_views_sha256']='bad'
    with pytest.raises(ValueError,match='checksum'):load_training_views(wrong,[parent])
    view.targets.fault_type.value='disk_io_stress';save(path,[view.model_dump()])
    with pytest.raises(ValueError,match='labels'):load_training_views(cfg(),[parent])
    view.targets=parent.targets.model_copy(deep=True);view.parent_incident_id='heldout';save(path,[view.model_dump()])
    with pytest.raises(ValueError,match='TRAIN'):load_training_views(cfg(),[parent])


def test_case_draws_are_not_multiplied_by_views_and_empty_is_excluded():
    eligible=eligible_parent_indices([{'evidence_usable':True},{'evidence_usable':False},{'evidence_usable':True}],3,True)
    assert eligible==[0,2]
    parents=[0,0,2,2];views={0:[3,4],2:[]}
    chosen=choose_view_indices(parents,views,1.,random.Random(46))
    assert len(chosen)==len(parents) and all(x in [3,4] for x in chosen[:2]) and chosen[2:]==[2,2]
    assert choose_view_indices(parents,views,0.,random.Random(46))==parents
    with pytest.raises(ValueError,match='eligible'):eligible_parent_indices([{'evidence_usable':False}],1,True)
    with pytest.raises(ValueError):load_training_views({'training_view_probability':.5},[])
