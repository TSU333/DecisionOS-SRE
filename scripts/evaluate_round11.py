"""Assess one validation winner; failed validation does not open historical regressions."""
from pathlib import Path
import copy,gc,subprocess,sys
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.pipeline import calibrate,policy,evaluate,benchmark,finish_rows,write_evaluation
from decisionos_sre.training import load_checkpoint,predict
from decisionos_sre.data import load_split

out=Path('outputs/round11');sel=read(out/'selection.json');folder=Path(sel['selected']['artifact'])
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
meta=read(folder/'metadata.json');cfg=meta['config']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
print('CALIBRATION CPU START',flush=True);cal=calibrate(folder,'cpu');print('CALIBRATION DONE',flush=True)
print('GATE CPU START',flush=True);pol=policy(folder,'cpu');print('GATE DONE',pol['status'],pol['selection'],flush=True)
if sel['promote_by_validation']:
 print('SS REGRESSION CPU START',flush=True);m=evaluate(folder,'cpu');print('SS REGRESSION DONE',m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
 model,tok,ser,meta=load_checkpoint(folder,'cpu')
 for split in ['regression_re1_ob','regression_re2_ob']:
  es=load_split(cfg['data_dir'],split);rows=predict(model,tok,ser,es,'cpu',split)
  m=write_evaluation(finish_rows(rows,cal,pol,meta['binding']),out/split)
  print(split,m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
  if split=='regression_re2_ob':
   altered=copy.deepcopy(es)
   for e in altered:
    for metric in e.input.evidence.metrics:metric.temporal=None
   m=write_evaluation(finish_rows(predict(model,tok,ser,altered,'cpu',split),cal,pol,meta['binding']),out/'mask_temporal_ob')
   print('MASK TEMPORAL OB',m['joint_accuracy'],flush=True)
 del model,tok,ser;gc.collect()
 bench_cfg=cfg;integration_args=[];checks={}
 for split in ['regression_re1_ob','regression_re2_ob','regression_ss']:
  new=read((folder if split=='regression_ss' else out)/split/'metrics.json');old=read(Path('outputs/round5')/split/'metrics.json')
  measures=lambda m:{'root':m['root']['acc_at_1'],'fault':m['fault']['accuracy'],'joint':m['joint_accuracy']}
  before,after=measures(old),measures(new)
  checks[split]={h:{'old':before[h],'new':after[h],'non_regressed':after[h]+1e-12>=before[h]} for h in before}
 non_regressed=all(x['non_regressed'] for c in checks.values() for x in c.values())
 guard={'validation_eligible':True,'cohorts':checks,'all_non_regressed':non_regressed,'promote':non_regressed,'regressions_executed':True}
else:
 print('VALIDATION DID NOT BEAT INCUMBENT: regression not opened; engineering checks use validation inputs',flush=True)
 model,tok,ser,meta=load_checkpoint(folder,'cpu');es=load_split(cfg['data_dir'],'model_validation')[:2]
 save(folder/'reload_reference.json',predict(model,tok,ser,es,'cpu','model_validation'))
 del model,tok,ser;gc.collect()
 bench_cfg=cfg|{'evaluation_split':'model_validation'};integration_args=['model_validation']
 guard={'validation_eligible':False,'cohorts':{},'all_non_regressed':None,'promote':False,'regressions_executed':False}
print('BENCHMARK CPU START',flush=True);bench=benchmark(folder,bench_cfg);print('BENCHMARK DONE',bench['end_to_end']['p95_ms'],flush=True)
subprocess.run([sys.executable,'scripts/verify_integration.py',str(folder)]+integration_args,check=True)
guard.update(independent_confirmation=False,rule='Predeclared validation and historical regression guard; no runner-up search')
save(out/'release_guard.json',guard)
print('RELEASE GUARD',guard,'ALL EVALUATION COMPLETED',flush=True)
