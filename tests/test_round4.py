import random
from types import SimpleNamespace
from pathlib import Path
import pytest
from decisionos_sre.training import training_order,initialize_from_artifact
from decisionos_sre.common import read,save,file_hash
from test_correctness import example


def test_cohort_sampling_balances_small_train_group_without_new_ids():
    train=[SimpleNamespace(source_metadata={'dataset_suite':'large'}) for _ in range(99)]+[SimpleNamespace(source_metadata={'dataset_suite':'small'})]
    rng=random.Random(42)
    draws=[i for _ in range(100) for i in training_order(train,rng,{'sampling_strategy':'cohort_balanced'})]
    assert set(draws)<=set(range(100))
    assert .45<sum(i==99 for i in draws)/len(draws)<.55
    shuffled=training_order(train,random.Random(42),{})
    assert sorted(shuffled)==list(range(100))


def test_numeric_warm_start_rejects_reordered_metric_semantics(tmp_path):
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'not opened')
    ex=example()
    save(tmp_path/'metadata.json',{'train_run_ids':[ex.original_run_id],'validation_run_ids':[], 'binding':{'checkpoint_sha256':file_hash(checkpoint)},'config':{'numeric_fusion':True,'numeric_metrics':['cpu','mem']}})
    with pytest.raises(ValueError,match='vocabulary'):
        initialize_from_artifact(None,{'initialization_artifact':str(tmp_path),'numeric_fusion':True,'numeric_metrics':['mem','cpu']},[ex],[])


def test_round4_preserves_prior_holdouts_and_seals_new_application_test():
    if not Path('data/round4/splits.json').exists():pytest.skip('real data audit not present')
    old=read('data/round3/splits.json');new=read('data/round4/splits.json')
    assert new['counts']=={'train':165,'model_validation':40,'calibration':40,'gate_selection':85,'regression_re1_ob':15,'regression_re2_ob':25,'test':30}
    for oid,split in old['assignments'].items():
        expected='regression_re2_ob' if split=='test' else ('regression_re1_ob' if split=='regression' else split)
        assert new['assignments'][oid]==expected
    examples=read('data/round4/examples.json')
    test=[e for e in examples if new['assignments'][e['opaque_incident_id']]=='test']
    assert len(test)==30 and all(e['input']['application']=='Sock Shop' for e in test)
    assert not {e['opaque_incident_id'] for e in test}&set(old['assignments'])
    groups={}
    for oid,split in new['assignments'].items():assert groups.setdefault(new['groups'][oid],split)==split


def test_selection_does_not_hide_old_application_regression():
    from decisionos_sre.training import selection_key
    config={'selection_metric':'cohort_guarded_joint','preserve_validation_cohorts':['RE1-OB','RE2-OB'],'preserve_joint_floor':.85}
    degraded={'cohort_macro_joint':.95,'joint_accuracy':.9,'sum_nll':.1,'cohorts':{'RE1-OB':{'n':15,'joint_accuracy':.8},'RE2-OB':{'n':5,'joint_accuracy':.8}}}
    retained={'cohort_macro_joint':.8,'joint_accuracy':.85,'sum_nll':.5,'cohorts':{'RE1-OB':{'n':15,'joint_accuracy':.8666666666666667},'RE2-OB':{'n':5,'joint_accuracy':.8}}}
    assert selection_key(retained,config)<selection_key(degraded,config)
