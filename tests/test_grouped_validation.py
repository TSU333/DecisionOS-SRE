import copy
import pytest
from decisionos_sre.grouped_validation import make_folds,validate_folds,assert_fresh_config,inner_selection_key


def records():
    return [{'run_id':str(i),'group_id':str(i//2),'split':'train','application':'app','fault':str((i//2)%2)} for i in range(96)]


def test_groups_and_inner_stopping_are_disjoint_and_reproducible():
    rows=records();folds=make_folds(rows)
    assert folds==make_folds(rows) and validate_folds(rows,folds)
    assert sum(len(f['outer']) for f in folds)==96
    for f in folds:
        assert not set(f['fit'])&set(f['stop']) and not set(f['fit'])&set(f['outer'])


def test_group_leakage_is_rejected_even_when_run_ids_differ():
    rows=records();folds=make_folds(rows);f=folds[0]
    a=f['fit'][0];b=f['outer'][0]
    f['fit'][0]=b;f['outer'][0]=a
    with pytest.raises(ValueError,match='Group overlap'):validate_folds(rows,folds)


def test_non_train_and_duplicate_inputs_rejected():
    rows=records();rows[0]['split']='model_validation'
    with pytest.raises(ValueError,match='TRAIN'):make_folds(rows)
    rows=records();rows[0]['run_id']=rows[1]['run_id']
    with pytest.raises(ValueError,match='Duplicate'):make_folds(rows)


def test_outer_coverage_and_run_overlap_rejected():
    rows=records();folds=make_folds(rows);folds[1]=copy.deepcopy(folds[0])
    with pytest.raises(ValueError,match='exactly one'):validate_folds(rows,folds)
    folds=make_folds(rows);folds[0]['fit'].append(folds[0]['outer'][0])
    with pytest.raises(ValueError,match='Run overlap'):validate_folds(rows,folds)


def test_fitted_parent_and_views_rejected():
    assert_fresh_config({'initialization_artifact':None})
    for cfg in [{'initialization_artifact':'fifth-round'},{'resume_checkpoint':'x'},{'training_view_probability':.1}]:
        with pytest.raises(ValueError):assert_fresh_config(cfg)


def test_inner_ranking_uses_macro_joint_then_joint_then_nll():
    a={'cohort_macro_joint':.7,'joint_accuracy':.8,'sum_nll':.2}
    assert inner_selection_key(a)<inner_selection_key(a|{'cohort_macro_joint':.6,'sum_nll':.1})
    assert inner_selection_key(a)<inner_selection_key(a|{'sum_nll':.3})
