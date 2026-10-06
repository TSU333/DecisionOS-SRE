"""Evaluate frozen fourth-round model on historical regression and fixed controls."""
from pathlib import Path
import gc
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.training import load_checkpoint,predict,prediction_summary
from decisionos_sre.pipeline import finish_rows,write_evaluation,run_baseline
from decisionos_sre.data import load_split

selection=read('outputs/round4/selection.json');folder=Path(selection['selected']['artifact'])
assert (folder/'test_freeze.json').exists()
assert file_hash(folder/'checkpoint.pt')==selection['selected']['binding']['checkpoint_sha256']
model,tok,ser,meta=load_checkpoint(folder,'cpu');cal=read(folder/'calibrator.json');pol=read(folder/'policy.json')
for split in ['regression_re1_ob','regression_re2_ob']:
    rows=predict(model,tok,ser,load_split('data/round4',split),'cpu',split)
    m=write_evaluation(finish_rows(rows,cal,pol,meta['binding']),Path('outputs/round4')/split)
    print(split,m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
del model,tok,ser;gc.collect()
old=Path('artifacts/round3/weighted_0.1/frozen');model,tok,ser,meta=load_checkpoint(old,'cpu')
rows=predict(model,tok,ser,load_split('data/round4','test'),'cpu','test')
m=write_evaluation(finish_rows(rows,read(old/'calibrator.json'),read(old/'policy.json'),meta['binding']),'outputs/round4/previous_model_new_test')
save('outputs/round4/previous_model_new_test/binding.json',{'artifact':str(old),'binding':meta['binding'],'calibrator_sha256':file_hash(old/'calibrator.json'),'policy_sha256':file_hash(old/'policy.json'),'role':'same new Sock Shop heldout cases, previous model not trained on Sock Shop; direct diagnostic predictions, not previous API support'})
print('PREVIOUS_SS',m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
del model,tok,ser;gc.collect()
m=run_baseline('data/round4','outputs/round4/simple_baseline')
print('BASELINE_SS',m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
