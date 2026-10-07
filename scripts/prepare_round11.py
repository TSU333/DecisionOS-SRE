"""Freeze limited encoder SFT trials using the unchanged original run partitions."""
from pathlib import Path
from datetime import datetime,timezone
import copy,shutil
from collections import Counter
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.data import load_split
from decisionos_sre.serializer import Serializer

out=Path('outputs/round11');out.mkdir(parents=True,exist_ok=True)
if (out/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
parent=read('outputs/latest_model.json')['artifact'];meta=read(Path(parent)/'metadata.json');probe=read(out/'resource_probe.json')
assert file_hash(Path(parent)/'checkpoint.pt')==probe['parent_checkpoint_sha256']=='461b93c00f348cc8759060b08f577ea5a7cce25672b0f7de0768d033b16b6423'
assert probe['status']=='real_train_only_probe_completed' and not probe['exported_weights']
previous=read('outputs/round10/data_audit.json');files={n:file_hash(Path('data/round5')/n) for n in previous['original_files']};assert files==previous['original_files']
sets={s:load_split('data/round5',s) for s in previous['original_split_counts']};ids={s:{e.original_run_id for e in es} for s,es in sets.items()}
assert all(len(ids[s])==len(es) for s,es in sets.items())
assert sum(map(len,ids.values()))==len(set.union(*ids.values()))==400
c=meta['config'];tok=AutoTokenizer.from_pretrained(Path(parent)/'tokenizer',local_files_only=True)
ser=Serializer(tok,c['max_length'],c['serializer_version'],c['numeric_metrics'],c['numeric_feature_version'])
train=sets['train'];records=[]
for e in train:
 enc=ser(e.input);records.append({'run_id':e.original_run_id,'application':e.input.application,'cohort':e.source_metadata['dataset_suite'],'usable':enc.report['modality_evidence_usable'],'tokens':len(enc.input_ids)})
excluded=[{'run_id':r['run_id'],'cohort':r['cohort'],'reason':'no retained usable temporal evidence'} for r in records if not r['usable']]
assert excluded==previous['excluded_unusable_train']
audit={'version':'partial-sft-v1','original_cases':400,'new_independent_cases':0,'allowed_train_cases':165,'eligible_train_cases':164,'excluded_unusable_train':excluded,'training_view_count':0,'original_split_counts':{s:len(es) for s,es in sets.items()},'original_files':files,'train_records':records,'cross_split_run_overlap':0,'known_onset':True,'max_retained_train_tokens':max(r['tokens'] for r in records),'note':'Original split and all source examples preserved; the unusable TRAIN case is excluded only from new gradient draws.'}
save(out/'data_audit.json',audit)
base=copy.deepcopy(c)
for key in ['numeric_feature_upgrade','application_fault_names','application_head_upgrade','head_training_policy','fault_hidden_dropout','fault_label_smoothing','training_view_probability','training_views_file','training_views_sha256']:base.pop(key,None)
base.update(seed=48,initialization_artifact=parent,cache_frozen_features=False,batch_size=1,gradient_accumulation=4,epochs=6,max_steps=246,early_stopping_patience=3,backbone_learning_rate=1e-5,head_learning_rate=3e-4,learning_rate_sft=1e-5,loss_weights=[1.,1.],schedule='constant',warmup_fraction=0.,require_train_usable_evidence=True,trace_features=False)
trials=[]
for name,layers in [('text_control',0),('tail2',2),('tail4',4)]:
 cfg=copy.deepcopy(base);cfg.update(artifact_root='artifacts/round11/'+name,partial_backbone_layers=layers)
 path='configs/round11_'+name+'.json';save(path,cfg);trials.append({'name':name,'config':path,'config_sha256':file_hash(path),'partial_backbone_layers':layers,'mode':'sft'})
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'client_date':'2026-10-08','execution_scope':'mvp','parent':parent,'parent_checkpoint_sha256':meta['binding']['checkpoint_sha256'],'trials':trials,'selection_config':meta['config'],'training_budget_max_updates':738,'resource_probe_micro_steps':3,'data_audit_sha256':file_hash(out/'data_audit.json'),'resource_probe_sha256':file_hash(out/'resource_probe.json'),'targets':{'root_accuracy':.9,'fault_accuracy':.9,'joint_accuracy':.85},'target_origin':'prior working assumption, not explicit user numeric requirements','selection':'Retain OB validation joint >=95%, SS >=90%; then cohort macro joint, overall joint, sum NLL; compare incumbent on the original 40 validation cases.','release_guard':'Only the single frozen validation winner may undergo historical regression. If no candidate beats the incumbent, do not open regression for candidate selection; retain incumbent. Otherwise require each historical cohort root/fault/joint to be non-regressed before promotion. No runner-up or post-regression tuning.','original_cases':400,'new_independent_cases':0,'eligible_train_cases':164,'training_views':0,'known_onset':True,'hypothesis':'The shared encoder has not adapted to the expanded two-application dataset. Train a bounded suffix while fixing the numeric branches. Control changes only text fault head; tail2/tail4 also change shared encoder final norm and the specified last blocks. Root output can change through the shared text representation although root-head weights stay frozen.','regularization':'Frozen numeric branches, small tail learning rate, maximum six epochs, validation early stopping. Original inputs only.','predeclared_trials_complete_before_evaluation':True,'inference_architecture_changed':False,'resource_probe_is_not_formal_model_or_new_data':True}
save(out/'protocol.json',protocol);shutil.copyfile('outputs/latest_model.json',out/'previous_latest_model.json')
Path('docs/round11_protocol.md').write_text("""# 第十一轮：有限共享编码器微调

execution_scope: mvp。保持400个原始run的既定划分、五类故障、temporal-v1输入和原推理架构。

验证侧诊断：第五轮40条验证中的3个错误分别为网络延迟/丢包、磁盘/CPU、磁盘/内存混淆。标签沿用固定公开注入元数据，不因模型错误修改。此前多轮主要更新分类头，编码器仍来自较早的OB数据微调；本轮假设有限末层更新能改善两应用共享文本表示。这个假设允许被实验否定。

TRAIN165条中164条具有有效观测；空证据记录保留原文件和父模型历史，但不再抽入新梯度。验证40、校准40、门控85、历史回归15/25/30不变。本轮不生成视图、不引入独立新样本。cohort仅用于TRAIN均衡抽样，路径、run编号、标签和注入答案不进入输入。

先在最长有效TRAIN输入1659 tokens上做3步临时梯度/资源探测，临时权重全部丢弃。实测4层方案PyTorch峰值分配显存约0.90GiB，稳定单微批次约0.10至0.12秒，无新增依赖。正式固定三组：text_control（只更新3845个文本故障头参数）；tail2（另更新最后2个编码器块与final_norm）；tail4（最后4块与final_norm）。三组均从第五轮重新初始化。

seed48，micro batch1，累积4，最多6个epoch/246个优化器更新，3个epoch无改善早停；backbone lr=1e-5，text fault head lr=3e-4，weight_decay0.01，root/fault loss权重均为1。训练BF16 autocast、FP32参数，验证/CPU推理FP32；末层训练使用非重入梯度检查点。原数值根因、数值全局/局部故障分支及文本根因scorer参数冻结。共享编码器输出变化仍可能改变根因结果，不能声称根因logits固定。一次完整incident推理仍只调用一次编码器。

三组完成后只按原40条验证选一个候选：先要求OB联合至少95%、SS至少90%，再比较四cohort联合均值、总体联合、sum NLL，且须优于现任。若没有候选验证胜出，不为挑选候选打开历史回归。若胜出，只评估这一个封存模型，校准与门控使用原定分区；发布还须三个历史cohort的根因/故障/联合均不下降。失败不改选次优、不依据回归继续调整本轮方案。

验收按原Definition of Done，新增检查真实backbone权重变化、冻结数值分支精确相等、梯度边界、部分累积批次、实际CUDA资源、checkpoint恢复和CPU重载、HTTP及CPU基准。原90%根因/90%故障/85%联合属于先前工程假设，不是原Prompt明确数值。回归与门控均已反复使用，不能作为新独立泛化或自动接受风险证明；门控仍为经验风险≤5%、至少30组接受，无可行阈值则全REVIEW。不执行运维操作。
""",encoding='utf-8')
print('PROTOCOL FROZEN',file_hash(out/'protocol.json'),flush=True)
