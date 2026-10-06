"""Generate the retraining report solely from persisted experimental artifacts."""
from pathlib import Path
import numpy as np
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

selection=read('outputs/retrain/selection.json')
folder=Path(selection['selected']['artifact'])
meta=read(folder/'metadata.json')
assert meta['binding']==selection['selected']['binding']
assert file_hash(folder/'resolved_config.json')==selection['selected']['config_sha256']
assert file_hash(folder/'checkpoint.pt')==meta['binding']['checkpoint_sha256']
cal=read(folder/'calibrator.json');pol=read(folder/'policy.json')
splits=read(Path(meta['config']['data_dir'])/'splits.json')
from decisionos_sre.data import load_split
for artifact,split in [(cal,'calibration'),(pol,'gate_selection')]:
    assert artifact['binding']==meta['binding']
    expected={e.original_run_id for e in load_split(meta['config']['data_dir'],split)}
    assert set(artifact['fit_run_ids'])==expected
    assert not expected.intersection(meta['train_run_ids'])

old=read('artifacts/sft/test/metrics.json');new=read(folder/'test/metrics.json')
oldrows=read('artifacts/sft/test/predictions.json');newrows=read(folder/'test/predictions.json')
oldby={r['run_id']:r for r in oldrows}
assert set(oldby)=={r['run_id'] for r in newrows}
assert len(oldby)==len(newrows)
def task_correct(row,task):
    heads=('root','fault') if task=='joint' else (task,)
    return all(int(np.argmax(row[h+'_logits']))==row[h+'_target'] for h in heads)
paired={}
rng=np.random.default_rng(42)
for task in ('root','fault','joint'):
    pairs=np.array([[task_correct(oldby[r['run_id']],task),task_correct(r,task)] for r in newrows],dtype=int)
    delta=pairs[:,1]-pairs[:,0]
    draws=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1)
    paired[task]={'n_run_groups':len(pairs),'old_correct':int(pairs[:,0].sum()),'new_correct':int(pairs[:,1].sum()),
                  'fixed':int(((pairs[:,0]==0)&(pairs[:,1]==1)).sum()),'regressed':int(((pairs[:,0]==1)&(pairs[:,1]==0)).sum()),
                  'new_accuracy_wilson95':wilson(int(pairs[:,1].sum()),len(pairs)),
                  'paired_accuracy_delta':float(delta.mean()),'paired_bootstrap_delta_95':np.percentile(draws,[2.5,97.5]).tolist(),
                  'interval_scope':'descriptive regression sample only; adaptations were motivated by previously inspected test results'}
ablations={name:read(folder/name/'metrics.json') for name in ('mask_all_metrics','reverse_candidates','consistent_service_rename')}
bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
numeric=read('artifacts/retrain/numeric/test_summary.json')
result={'status':'experiment_completed','execution_scope':'mvp','selection':selection,'paired_comparison':paired,
        'old_metrics':old,'new_metrics':new,'ablations':ablations,'benchmark':bench,
        'integration':integration,'calibrator':cal,'policy':pol,'numeric_control':numeric,
        'representation_audit':read('outputs/retrain/serialization_summary.json'),
        'old_benchmark':read('artifacts/sft/benchmark.json'),
        'limitations':['50 train / 15 validation / 15 calibration / 30 gate / 15 reused regression cases',
                       'controlled injected faults, one application, metrics-only, oracle onset',
                       'test already inspected before this iteration; no claim of new unseen generalization',
                       'no formal future risk guarantee and no remediation'],
        'not_run':['new independent confirmatory holdout','multi-seed stability','cross-system evaluation','KD','LLM cascade','ONNX','INT8']}
inventory=[]
for c in selection['candidates']:
    path=Path(c['artifact'])
    inventory.append({'artifact':str(path),'training':'experiment_completed','validation':'experiment_completed',
        'calibration':'experiment_completed' if (path/'calibrator.json').exists() else 'not_run',
        'gate_selection':'experiment_completed' if (path/'policy.json').exists() else 'not_run',
        'regression':'experiment_completed' if (path/'test/metrics.json').exists() else 'not_run',
        'cpu_benchmark':'experiment_completed' if (path/'benchmark.json').exists() else 'not_run',
        'http_reload':'experiment_completed' if (path/'integration.json').exists() else 'not_run'})
result['experiment_inventory']=inventory
save('outputs/retrain/results.json',result)
save('outputs/retrain/model_artifact.json',{'artifact':str(folder),'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'binding':meta['binding'],'config':meta['config'],'selected_by':'model_validation only'})
print('SELECTED',folder)
print('PAIRED',paired)
print('NEW',{'root':new['root'],'fault':new['fault']['accuracy'],'joint':new['joint_accuracy'],'selective':new['selective']})
print('ABLATIONS',{k:{'root':v['root']['acc_at_1'],'fault':v['fault']['accuracy'],'joint':v['joint_accuracy']} for k,v in ablations.items()})
print('BENCH',bench['end_to_end'])
print('NUMERIC CONTROL',numeric)
