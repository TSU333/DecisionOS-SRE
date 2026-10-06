"""Select by MODEL_VALIDATION only; persist choice before any new regression eval."""
from pathlib import Path
from datetime import datetime,timezone
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.training import prediction_summary

paths=['artifacts/sft','artifacts/retrain/hybrid_frozen/frozen']+['artifacts/retrain/'+name+'/sft' for name in ('budget','canonical','hybrid')]
choices=[]
for path in paths:
    folder=Path(path);metadata=read(folder/'metadata.json')
    assert not metadata.get('diagnostic_only',False)
    rows=read(folder/'selected_validation_logits.json')
    assert all(r['split']=='model_validation' for r in rows)
    assert {r['run_id'] for r in rows}==set(metadata['validation_run_ids'])
    summary=prediction_summary(rows)
    choices.append({'artifact':path,'validation':summary,'binding':metadata['binding'],
                    'config_sha256':file_hash(folder/'resolved_config.json'),
                    'steps':metadata['history'][-1]['optimizer_steps'],
                    'best_epoch':min(metadata['history'],key=lambda h:h['validation_loss'])['epoch']})
selected=min(choices,key=lambda x:x['validation']['sum_nll'])
result={'status':'frozen_before_regression','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'criterion':'minimum model_validation sum of per-head NLL','selected':selected,
        'candidates':choices,'test_role':'previously inspected regression set; exploratory, not new confirmatory evidence',
        'no_more_tuning_after_this_file':True,'target_joint_risk':.05,'min_accepted_runs':30}
path=Path('outputs/retrain/selection.json')
if path.exists():raise FileExistsError('Selection is already frozen')
save(path,result);print(result,flush=True)
