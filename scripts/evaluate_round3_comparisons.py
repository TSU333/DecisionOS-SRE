"""Post-freeze comparisons only; never modifies selection or existing weights."""
from pathlib import Path
import gc,pickle
import numpy as np
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.training import load_checkpoint,predict,prediction_summary
from decisionos_sre.pipeline import finish_rows,write_evaluation,run_baseline
from decisionos_sre.data import load_split
from decisionos_sre.representation import numeric_features
from numeric_baseline import features

sel=read('outputs/round3/selection.json');folder=Path(sel['selected']['artifact'])
assert (folder/'test_freeze.json').exists(), 'Run frozen test evaluation first'
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
model,tok,ser,meta=load_checkpoint(folder,'cpu')
rows=predict(model,tok,ser,load_split('data/round3','regression'),'cpu','regression')
reg=write_evaluation(finish_rows(rows,read(folder/'calibrator.json'),read(folder/'policy.json'),meta['binding']),'outputs/round3/regression')
print('REGRESSION',reg['root']['acc_at_1'],reg['fault']['accuracy'],reg['joint_accuracy'],flush=True)
del model,tok,ser;gc.collect()
old=Path('artifacts/retrain/canonical/sft')
model,tok,ser,meta=load_checkpoint(old,'cpu')
rows=predict(model,tok,ser,load_split('data/round3','test'),'cpu','test')
oldmetrics=write_evaluation(finish_rows(rows,read(old/'calibrator.json'),read(old/'policy.json'),meta['binding']),'outputs/round3/previous_model_new_test')
save('outputs/round3/previous_model_new_test/binding.json',{'artifact':str(old),'binding':meta['binding'],'calibrator_sha256':file_hash(old/'calibrator.json'),'policy_sha256':file_hash(old/'policy.json'),'role':'unchanged prior model on identical new held-out examples after current selection freeze'})
print('PREVIOUS_NEW_TEST',oldmetrics['root']['acc_at_1'],oldmetrics['fault']['accuracy'],oldmetrics['joint_accuracy'],flush=True)
del model,tok,ser;gc.collect()
base=run_baseline('data/round3','outputs/round3/simple_baseline')
print('BASELINE',base['root']['acc_at_1'],base['fault']['accuracy'],base['joint_accuracy'],flush=True)
with open('artifacts/round3/numeric_diagnostic/models.pkl','rb') as f:root,fault,names=pickle.load(f)
rows=[]
for e in load_split('data/round3','test'):
    ids=[c.candidate_id for c in e.input.candidates]
    nums,_=numeric_features(e.input.evidence.metrics,ids,names)
    rp=root.predict_proba(nums)[:,list(root.classes_).index(1)]
    fp=fault.predict_proba([features(e,names)])[0]
    rows.append({'incident_id':e.opaque_incident_id,'run_id':e.original_run_id,'split':'test','application':e.input.application,'cohort':e.source_metadata['dataset_suite'],'candidate_ids':ids,'root_logits':np.log(np.maximum(rp,1e-12)).tolist(),'fault_logits':np.log(np.maximum(fp,1e-12)).tolist(),'root_target':ids.index(e.targets.root_cause.value),'fault_target':FAULTS.index(e.targets.fault_type.value),'gold':e.targets.model_dump(),'evidence_usable':True})
numeric=write_evaluation(finish_rows(rows,None,None,{}),'outputs/round3/numeric_diagnostic_new_test')
print('NUMERIC_DIAGNOSTIC_NEW_TEST',numeric['root']['acc_at_1'],numeric['fault']['accuracy'],numeric['joint_accuracy'],flush=True)
