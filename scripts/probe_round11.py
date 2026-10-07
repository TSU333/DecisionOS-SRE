"""Resource probe on the longest retained TRAIN input; no validation or exported model."""
import time,gc
from pathlib import Path
import torch,psutil
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash,seed_all
from decisionos_sre.data import load_split
from decisionos_sre.serializer import Serializer,collate
from decisionos_sre.model import labels,supervised_loss
from decisionos_sre.partial_training import build_partial

seed_all(48);torch.set_num_threads(8)
c=read('configs/round5_temporal_seed44.json');parent=read('outputs/latest_model.json')['artifact']
c.update(initialization_artifact=parent,cache_frozen_features=False,partial_backbone_layers=4)
c.pop('numeric_feature_upgrade',None)
train=load_split(c['data_dir'],'train');validation=load_split(c['data_dir'],'model_validation')
tok=AutoTokenizer.from_pretrained(c['backbone_dir'],local_files_only=True)
ser=Serializer(tok,c['max_length'],c['serializer_version'],c['numeric_metrics'],c['numeric_feature_version'])
encoded=[ser(e.input) for e in train];i=max((i for i,e in enumerate(encoded) if e.report['modality_evidence_usable']),key=lambda i:len(encoded[i].input_ids))
model,init=build_partial(c,train,validation);model.to('cuda').train()
params=[p for p in model.parameters() if p.requires_grad];initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
opt=torch.optim.AdamW(params,lr=1e-5,foreach=False)
batch=collate([encoded[i]],tok.pad_token_id,'cuda');targets=labels([train[i]],[encoded[i]],'cuda')
times=[];losses=[];torch.cuda.reset_peak_memory_stats()
for step in range(3):
 torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
 with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
  loss=supervised_loss(*model(**batch),*targets)
 assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
 opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
changed=[n for n,p in model.named_parameters() if n in initial and not torch.equal(p.detach().cpu(),initial[n])]
assert any(n.startswith('backbone.layers.') for n in changed)
save('outputs/round11/resource_probe.json',{'status':'real_train_only_probe_completed','diagnostic_only':True,'train_run_id':train[i].original_run_id,'tokens':len(encoded[i].input_ids),'partial_backbone_layers':4,'micro_steps':3,'losses':losses,'step_seconds':times,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'trainable_parameters':sum(p.numel() for p in params),'changed_parameters':changed,'ram_available_bytes_after':psutil.virtual_memory().available,'parent_checkpoint_sha256':file_hash(Path(parent)/'checkpoint.pt'),'exported_weights':False,'validation_labels_used':False,'note':'TRAIN-only resource/gradient probe; all updated weights discarded. Validation run IDs are used only for parent lineage checks.'})
print('PROBE PASS','tokens',len(encoded[i].input_ids),'seconds',times,'peak allocated GiB',torch.cuda.max_memory_allocated()/1024**3,'trainable',sum(p.numel() for p in params),flush=True)
