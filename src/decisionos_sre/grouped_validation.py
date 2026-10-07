"""Development-only grouped CV with a separate inner stopping split."""
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def assert_fresh_config(config):
    if config.get('initialization_artifact') or config.get('resume_checkpoint'):
        raise ValueError('Cross-validation requires fresh heads and the original pretrained backbone')
    if config.get('training_view_probability',0) or config.get('training_views_path'):
        raise ValueError('This protocol uses original TRAIN cases only')


def validate_folds(records, folds):
    ids=[r['run_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate run IDs')
    lookup={r['run_id']:r for r in records};seen=Counter()
    if any(r['split']!='train' for r in records):raise ValueError('CV may use TRAIN only')
    for fold in folds:
        roles=[set(fold[k]) for k in ('fit','stop','outer')]
        if any(not r for r in roles):raise ValueError('Empty CV partition')
        if set.union(*roles)!=set(ids):raise ValueError('CV partition coverage mismatch')
        if any(roles[i]&roles[j] for i in range(3) for j in range(i)):raise ValueError('Run overlap')
        groups=[{lookup[x]['group_id'] for x in role} for role in roles]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Group overlap')
        seen.update(roles[2])
    if set(seen)!=set(ids) or any(n!=1 for n in seen.values()):raise ValueError('Each run must appear in exactly one outer fold')
    return True


def make_folds(records, n_splits=3, split_seed=130, inner_splits=4):
    if len({r['run_id'] for r in records})!=len(records):raise ValueError('Duplicate run IDs')
    if any(r['split']!='train' for r in records):raise ValueError('CV may use TRAIN only')
    y=np.array([r['application']+'|'+r['fault'] for r in records]);groups=np.array([r['group_id'] for r in records])
    indices=np.arange(len(records));folds=[]
    for i,(development,outer) in enumerate(StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=split_seed).split(indices,y,groups)):
        fit,stop=next(StratifiedGroupKFold(n_splits=inner_splits,shuffle=True,random_state=split_seed+100+i).split(development,y[development],groups[development]))
        entry={'fold':i,'fit':[records[j]['run_id'] for j in development[fit]],'stop':[records[j]['run_id'] for j in development[stop]],'outer':[records[j]['run_id'] for j in outer]}
        folds.append(entry)
    validate_folds(records,folds)
    return folds


def inner_selection_key(summary):
    return (-round(summary['cohort_macro_joint'],12),-round(summary['joint_accuracy'],12),summary['sum_nll'])
