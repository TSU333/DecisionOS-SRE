import json
import subprocess
import sys
from pathlib import Path
import time
import urllib.request
import urllib.error
import numpy as np
from decisionos_sre.common import read,save
from decisionos_sre.data import load_split
from decisionos_sre.training import load_checkpoint,predict
from decisionos_sre.runtime import Engine

folder=Path(sys.argv[1] if len(sys.argv)>1 else "artifacts/sft")
meta=read(folder/"metadata.json")
examples=load_split(meta["config"]["data_dir"],"test")[:2]
model,tok,ser,metadata=load_checkpoint(folder,"cpu")
rows=predict(model,tok,ser,examples,"cpu","test")
reference=read(folder/"reload_reference.json")
deltas=[]
for r,ref in zip(rows,reference):
 assert r["incident_id"]==ref["incident_id"]
 for head in ["root","fault"]:
  delta=float(np.max(np.abs(np.asarray(r[head+"_logits"])-ref[head+"_logits"])))
  assert delta<=1e-6,delta
  deltas.append(delta)
del model,tok,ser
import gc
gc.collect()
# Real HTTP server subprocess; always stop the helper we started.
log=(folder/"api-server.log").open("w",encoding="utf8")
proc=subprocess.Popen([sys.executable,"-m","decisionos_sre","serve","--artifact",str(folder),"--port","8765"],
                       stdout=log,stderr=subprocess.STDOUT)
base="http://127.0.0.1:8765"
def request(path,body=None):
 data=json.dumps(body).encode() if body is not None else None
 req=urllib.request.Request(base+path,data=data,headers={"Content-Type":"application/json"})
 try:
  with urllib.request.urlopen(req,timeout=45) as r: return r.status,json.load(r)
 except urllib.error.HTTPError as e: return e.code,json.load(e)
try:
 for _ in range(60):
  if proc.poll() is not None: raise RuntimeError("server exited")
  try:
   code,health=request("/ready")
   if code==200: break
  except urllib.error.URLError: pass
  time.sleep(1)
 else: raise RuntimeError("server readiness timeout")
 checks={}
 checks["health"]=request("/health")[0]
 checks["ready"]=code
 body=examples[0].input.model_dump()
 code,result=request("/v1/decide",body)
 assert code==200 and result["routing"]["destination"] in {"REVIEW","ACCEPT_DIAGNOSIS"}
 from decisionos_sre.calibration import probabilities
 from decisionos_sre.policy import route
 cal=read(folder/"calibrator.json");pol=read(folder/"policy.json")
 expected=route(probabilities(rows[0]["root_logits"],cal["root"]["temperature"]),
                probabilities(rows[0]["fault_logits"],cal["fault"]["temperature"]),
                rows[0]["evidence_usable"],cal,pol,meta["binding"])
 assert result["routing"]==expected
 assert result["execute_remediation"] is False
 checks["decide"]=code
 checks["decision"]=result
 invalid=dict(body,targets={"root_cause":"secret"})
 checks["reject_gold"]=request("/v1/decide",invalid)[0];assert checks["reject_gold"]==422
 singleton=dict(body,candidates=body["candidates"][:1])
 code,single=request("/v1/decide",singleton)
 assert code==200 and "SINGLE_CANDIDATE" in single["routing"]["reason_codes"]
 checks["single_candidate"]=single["routing"]
 oversized=dict(body,candidates=[{"candidate_id":str(i),"display_name":"long service name "*15} for i in range(1000)])
 code,over=request("/v1/decide",oversized)
 assert code==200 and over["root_cause"] is None and over["routing"]["destination"]=="REVIEW"
 checks["candidate_overflow"]=over["routing"]
 unknown=dict(body,application="unseen_application")
 code,other=request("/v1/decide",unknown)
 assert code==200 and "UNVALIDATED_APPLICATION" in other["routing"]["reason_codes"]
 checks["unvalidated_application"]=other["routing"]
 empty=dict(body,candidates=[])
 checks["empty_candidates"]=request("/v1/decide",empty)[0];assert checks["empty_candidates"]==422
 save(folder/"integration.json",{"status":"experiment_completed","reload_max_abs_logit_diff":max(deltas),"http":checks})
 print("PASS",folder,"reload delta",max(deltas),"HTTP checks",list(checks),flush=True)
finally:
 proc.terminate()
 try:proc.wait(timeout=10)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()
