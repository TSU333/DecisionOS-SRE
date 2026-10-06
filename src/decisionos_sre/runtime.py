import time
import threading
from pathlib import Path
import torch
from .common import FAULTS, SERIALIZER, digest, read
from .schema import IncidentInput, DecisionResponse
from .serializer import collate, UnsupportedInput
from .training import load_checkpoint
from .calibration import probabilities
from .policy import route

class Engine:
    def __init__(self,artifact_dir,device="cpu"):
        self._lock=threading.Lock()
        start=time.perf_counter()
        self.model,self.tokenizer,self.serializer,self.metadata=load_checkpoint(artifact_dir,device)
        self.device=device
        self.binding=self.metadata["binding"]
        self.calibrator=self._optional(Path(artifact_dir)/"calibrator.json")
        self.policy=self._optional(Path(artifact_dir)/"policy.json")
        self.cold_load_ms=(time.perf_counter()-start)*1000

    @staticmethod
    def _optional(path):
        if not path.exists():
            return None
        obj=read(path)
        if digest({k:v for k,v in obj.items() if k!="id"})!=obj["id"]:
            raise ValueError(f"artifact integrity mismatch: {path.name}")
        return obj

    def decide(self,incident):
        with self._lock:
            return self._decide(incident)

    def _decide(self,incident):
        start=time.perf_counter()
        incident=IncidentInput.model_validate(incident)
        versions={"model":self.binding["checkpoint_sha256"],"serializer":self.serializer.version,
                  "ontology":digest(FAULTS),"calibration":self.calibrator["id"] if self.calibrator else None,
                  "policy":self.policy["id"] if self.policy else None}
        try:
            enc=self.serializer(incident)
        except UnsupportedInput as error:
            return DecisionResponse(root_cause=None,fault_type=None,
                routing={"destination":"REVIEW","routing_score":None,"reason_codes":[str(error)]},
                versions=versions,evidence_status={"unsupported":True},
                timings_ms={"end_to_end":(time.perf_counter()-start)*1000}).model_dump()
        batch=collate([enc],self.tokenizer.pad_token_id,self.device)
        serial_ms=(time.perf_counter()-start)*1000
        before=self.model.encoder_calls
        model_start=time.perf_counter()
        with torch.inference_mode():
            r,f=self.model(**batch)
        if self.device=="cuda":
            torch.cuda.synchronize()
        core_ms=(time.perf_counter()-model_start)*1000
        assert self.model.encoder_calls-before==1
        results={}
        probs={}
        for head,logits,names in [("root",r[0],enc.candidate_ids),("fault",f[0],FAULTS)]:
            valid=self.calibrator is not None and self.calibrator["binding"]==self.binding
            status=self.calibrator[head]["status"] if valid else ("missing" if self.calibrator is None else "version_mismatch")
            t=self.calibrator[head]["temperature"] if valid and status=="fitted" else 1.
            p=probabilities(logits.float().cpu().tolist(),t)
            choice=max(range(len(p)),key=p.__getitem__)
            probs[head]=p
            results[head]={"selected":names[choice],"probabilities":dict(zip(names,p)),
                           "confidence":p[choice],"calibration_status":status}
        routing=route(probs["root"],probs["fault"],enc.report["usable_metrics_retained"]>0,
                      self.calibrator,self.policy,self.binding)
        if incident.application not in self.metadata["supported_applications"]:
            routing["destination"]="REVIEW"
            routing["reason_codes"]=[x for x in routing["reason_codes"] if x!="THRESHOLD_PASSED"]+["UNVALIDATED_APPLICATION"]
        return DecisionResponse(root_cause=results["root"],fault_type=results["fault"],routing=routing,
           versions=versions,evidence_status=enc.report,
           timings_ms={"serialization_tokenization":serial_ms,"model":core_ms,
                       "end_to_end":(time.perf_counter()-start)*1000}).model_dump()
