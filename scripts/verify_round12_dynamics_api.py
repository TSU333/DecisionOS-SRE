"""Engineering-only real HTTP checks of new dynamics on a validation input."""
from pathlib import Path
import copy,json,socket,subprocess,sys,time,urllib.request,urllib.error
import numpy as np
from decisionos_sre.common import read,save,FAULTS
from decisionos_sre.data import load_split
from decisionos_sre.calibration import probabilities

folder=Path('artifacts/round12/dynamics_fault/frozen');out=Path('outputs/round12')
meta=read(folder/'metadata.json');ex=load_split('data/round12','model_validation')[0]
assert not (folder/'calibrator.json').exists() and not (folder/'policy.json').exists()
row=next(r for r in read(folder/'selected_validation_logits.json') if r['run_id']==ex.original_run_id)
with socket.socket() as sock:
 sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
log=(out/'dynamics-api-server.log').open('w',encoding='utf-8')
proc=subprocess.Popen([sys.executable,'-m','decisionos_sre','serve','--artifact',str(folder),'--port',str(port)],stdout=log,stderr=subprocess.STDOUT)
base='http://127.0.0.1:'+str(port)
def request(body=None,path='/v1/decide'):
 req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=45) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as err:return err.code,json.load(err)
try:
 for _ in range(60):
  if proc.poll() is not None:raise RuntimeError('server exited')
  try:
   if request(path='/ready')[0]==200:break
  except urllib.error.URLError:pass
  time.sleep(1)
 else:raise RuntimeError('server readiness timeout')
 body=ex.input.model_dump();code,reply=request(body)
 assert code==200 and reply['routing']['destination']=='REVIEW' and not reply['execute_remediation']
 assert reply['evidence_status']['dynamics_metrics_retained']>0
 deltas=[]
 for head,field,names in [('root','root_cause',row['candidate_ids']),('fault','fault_type',FAULTS)]:
  delta=float(np.max(np.abs(np.array([reply[field]['probabilities'][n] for n in names])-probabilities(row[head+'_logits']))))
  assert delta<=1e-4;deltas.append(delta)
 checks={'normal':reply['routing']}
 for field,reason in [('dynamics','NO_DYNAMICS_EVIDENCE'),('temporal','NO_TEMPORAL_EVIDENCE')]:
  missing=copy.deepcopy(body)
  for m in missing['evidence']['metrics']:m.pop(field,None)
  code,result=request(missing)
  assert code==200 and result['routing']['destination']=='REVIEW' and reason in result['routing']['reason_codes']
  checks['missing_'+field]=result['routing']
 bad=copy.deepcopy(body);bad['evidence']['metrics'][0]['observed_until']=bad['decision_time']+1
 checks['future_timestamp']=request(bad)[0];assert checks['future_timestamp']==422
 checks['reject_gold']=request(dict(body,targets={'fault_type':'cpu_stress'}))[0];assert checks['reject_gold']==422
 save(out/'dynamics_api.json',{'status':'passed','artifact':str(folder),'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'split':'model_validation','run_id':ex.original_run_id,'used_for_selection':False,'calibration_or_policy_fit':False,'cached_gpu_vs_http_cpu_probability_max_abs_difference':max(deltas),'checks':checks})
 print('DYNAMICS HTTP PASS',list(checks),'probability delta',max(deltas),flush=True)
finally:
 proc.terminate()
 try:proc.wait(timeout=10)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()
