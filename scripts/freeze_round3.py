"""Freeze Round 3 validation selection before any held-out test evaluation."""
from datetime import datetime,timezone
from pathlib import Path
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.training import prediction_summary,selection_key

names=[('cached_fast','frozen'),('cached_slow','frozen'),('finetuned','sft'),('conditioned_fast','frozen'),('conditioned_slow','frozen'),('weighted_0.1','frozen'),('weighted_0.3','frozen')]
choices=[]
expected=None
for name,mode in names:
    folder=Path('artifacts/round3')/name/mode
    meta=read(folder/'metadata.json');rows=read(folder/'selected_validation_logits.json')
    assert not meta.get('diagnostic_only') and len(rows)==20
    assert all(r['split']=='model_validation' for r in rows)
    assert {r['run_id'] for r in rows}==set(meta['validation_run_ids'])
    assert file_hash(folder/'checkpoint.pt')==meta['binding']['checkpoint_sha256']
    assert file_hash(Path(meta['config']['data_dir'])/'examples.json')==meta['prepared_data_sha256']
    if expected is None:expected=set(meta['validation_run_ids'])
    assert set(meta['validation_run_ids'])==expected
    summary=prediction_summary(rows)
    best=min(meta['history'],key=lambda h:selection_key(h['validation'],meta['config']))
    choices.append({'artifact':str(folder),'validation':summary,'binding':meta['binding'],'config_sha256':file_hash(folder/'resolved_config.json'),'steps':meta['history'][-1]['optimizer_steps'],'best_epoch':best['epoch'],'best_steps':best['optimizer_steps'],'seconds':meta['elapsed_seconds']})
selected=min(choices,key=lambda x:selection_key(x['validation'],{'selection_metric':'cohort_macro_joint'}))
out=Path('outputs/round3/selection.json')
if out.exists():raise FileExistsError('Round 3 selection already frozen; never silently reselect')
protocol=read('outputs/round3/protocol.json')
save(out,{'status':'frozen_before_new_test','timestamp_utc':datetime.now(timezone.utc).isoformat(),'criterion':protocol['selection_rule'],'protocol_sha256':file_hash('outputs/round3/protocol.json'),'selected':selected,'candidates':choices,'targets':protocol['targets'],'targets_source':'working assumption; user did not give numeric targets','heldout_n':25,'test_role':'new RE2 controlled-injection cases, not production incidents','no_more_tuning_after_this_file':True,'target_joint_risk':.05,'min_accepted_runs':30})
print('FROZEN',selected)
