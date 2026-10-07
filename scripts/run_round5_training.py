from pathlib import Path
import os,subprocess,sys,json
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.training import prediction_summary,selection_key
from datetime import datetime,timezone

protocol=read('outputs/round5/protocol.json')
assert Path('data/round5/examples.json').exists()
for trial in protocol['trials']:
    name=trial['name'];config='configs/round5_'+name+'.json';folder=Path('artifacts/round5')/name/'frozen'
    if (folder/'metadata.json').exists():raise FileExistsError('Trial already complete: '+name)
    print('START',name,flush=True)
    with open('outputs/round5/'+name+'.log','wb') as log:
        subprocess.run([sys.executable,'-m','decisionos_sre','--config',config,'train','--mode','frozen'],stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=read(folder/'metadata.json');summary=prediction_summary(read(folder/'selected_validation_logits.json'))
    print('DONE',name,'updates',meta['history'][-1]['optimizer_steps'],'seconds',round(meta['elapsed_seconds'],1),'validation',summary,flush=True)
choices=[]
for trial in protocol['trials']:
    folder=Path('artifacts/round5')/trial['name']/'frozen';meta=read(folder/'metadata.json');rows=read(folder/'selected_validation_logits.json')
    assert {r['run_id'] for r in rows}==set(meta['validation_run_ids']) and len(rows)==40
    assert all(r['split']=='model_validation' for r in rows)
    assert file_hash(folder/'checkpoint.pt')==meta['binding']['checkpoint_sha256']
    assert file_hash(Path(meta['config']['data_dir'])/'examples.json')==meta['prepared_data_sha256']
    best=min(meta['history'],key=lambda h:selection_key(h['validation'],meta['config']))
    s=prediction_summary(rows)
    choices.append({'artifact':str(folder),'validation':s,'rank':list(selection_key(s,protocol['selection_config'])),'binding':meta['binding'],'config_sha256':file_hash(folder/'resolved_config.json'),'steps':meta['history'][-1]['optimizer_steps'],'best_epoch':best['epoch'],'best_steps':best['optimizer_steps'],'seconds':meta['elapsed_seconds']})
inc=Path(protocol['parent']);oldmeta=read(inc/'metadata.json');oldsummary=prediction_summary(read(inc/'selected_validation_logits.json'))
assert set(oldmeta['validation_run_ids'])==set(meta['validation_run_ids'])
oldrank=list(selection_key(oldsummary,protocol['selection_config']));best=min(choices,key=lambda c:c['rank'])
selection={'status':'frozen_before_round5_regression','timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash('outputs/round5/protocol.json'),'selected':best,'candidates':choices,'incumbent':{'artifact':str(inc),'rank':oldrank,'validation':oldsummary,'binding':oldmeta['binding']},'promote_by_validation':best['rank'][0]==0 and best['rank']<oldrank,'no_fresh_test':True,'no_further_tuning':True,'note':'Selected challenger evaluated even if incumbent retained; promotion decided from validation before regression/calibration/gate.'}
path=Path('outputs/round5/selection.json')
if path.exists():raise FileExistsError('Selection already frozen')
save(path,selection)
print('FROZEN',best['artifact'],'PROMOTE',selection['promote_by_validation'],'RANK',best['rank'],'INCUMBENT',oldrank,flush=True)
