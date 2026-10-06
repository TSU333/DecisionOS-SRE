"""Freeze fourth-round choice using guarded model-validation metrics only."""
from pathlib import Path
from datetime import datetime,timezone
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.training import prediction_summary,selection_key
names=['uniform','balanced','balanced_seed43'];choices=[];expected=None
for name in names:
    folder=Path('artifacts/round4')/name/'frozen';meta=read(folder/'metadata.json');rows=read(folder/'selected_validation_logits.json')
    assert not meta.get('diagnostic_only') and len(rows)==40 and all(r['split']=='model_validation' for r in rows)
    assert {r['run_id'] for r in rows}==set(meta['validation_run_ids'])
    assert file_hash(folder/'checkpoint.pt')==meta['binding']['checkpoint_sha256']
    assert file_hash(Path(meta['config']['data_dir'])/'examples.json')==meta['prepared_data_sha256']
    if expected is None:expected=set(meta['validation_run_ids'])
    assert set(meta['validation_run_ids'])==expected
    summary=prediction_summary(rows);rank=selection_key(summary,meta['config'])
    best=min(meta['history'],key=lambda h:selection_key(h['validation'],meta['config']))
    choices.append({'artifact':str(folder),'validation':summary,'rank':list(rank),'eligible_for_promotion':rank[0]==0,'binding':meta['binding'],'config_sha256':file_hash(folder/'resolved_config.json'),'steps':meta['history'][-1]['optimizer_steps'],'best_epoch':best['epoch'],'best_steps':best['optimizer_steps'],'seconds':meta['elapsed_seconds']})
selected=min(choices,key=lambda c:c['rank']);path=Path('outputs/round4/selection.json')
if path.exists():raise FileExistsError('Selection already frozen')
protocol=read('outputs/round4/protocol.json')
save(path,{'status':'frozen_before_new_sock_shop_test','timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash('outputs/round4/protocol.json'),'criterion':protocol['selection'],'selected':selected,'candidates':choices,'fresh_test':protocol['fresh_test'],'old_test_roles':protocol['old_test_roles'],'targets':protocol['targets'],'no_more_tuning_after_freeze':True,'target_joint_risk':.05,'min_accepted_runs':30})
print('FROZEN',selected,flush=True)
