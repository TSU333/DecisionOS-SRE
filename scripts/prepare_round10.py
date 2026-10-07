"""Freeze application expert hypotheses and audit existing data before training."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import copy,shutil
from transformers import AutoTokenizer
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.data import load_split
from decisionos_sre.serializer import Serializer

out=Path('outputs/round10');out.mkdir(parents=True,exist_ok=True)
if (out/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
parent=read('outputs/latest_model.json')['artifact'];meta=read(Path(parent)/'metadata.json')
assert file_hash(Path(parent)/'checkpoint.pt')=='461b93c00f348cc8759060b08f577ea5a7cce25672b0f7de0768d033b16b6423'
previous=read('outputs/round9/data_audit.json')
original={n:file_hash(Path('data/round5')/n) for n in previous['original_files']}
assert original==previous['original_files']
sets={s:load_split('data/round5',s) for s in previous['original_split_counts']}
ids={s:{e.original_run_id for e in es} for s,es in sets.items()}
assert all(len(ids[s])==len(es) for s,es in sets.items())
assert sum(map(len,ids.values()))==len(set.union(*ids.values()))==400
train=sets['train'];apps=sorted({e.input.application for e in train})
cfg=meta['config'];tok=AutoTokenizer.from_pretrained(Path(parent)/'tokenizer',local_files_only=True)
ser=Serializer(tok,cfg['max_length'],cfg['serializer_version'],cfg['numeric_metrics'],cfg['numeric_feature_version'],False,apps)
records=[]
for e in train:
 enc=ser(e.input)
 records.append({'run_id':e.original_run_id,'application':e.input.application,'cohort':e.source_metadata['dataset_suite'],'application_index':enc.application_index,'usable':enc.report['numeric_evidence_usable'],'tokens':len(enc.input_ids)})
excluded=[{'run_id':r['run_id'],'cohort':r['cohort'],'reason':'no retained usable temporal evidence'} for r in records if not r['usable']]
assert excluded==previous['excluded_unusable_train']
audit={'version':'application-experts-v1','original_cases':400,'new_independent_cases':0,'allowed_train_cases':165,'eligible_train_cases':164,'excluded_unusable_train':excluded,'training_view_count':0,'original_split_counts':{s:len(es) for s,es in sets.items()},'original_files':original,'train_applications':apps,'train_application_counts':dict(Counter(r['application'] for r in records if r['usable'])),'train_records':records,'cross_split_run_overlap':0,'routing_field':'input.application only; no suite/path/run/target','known_onset':True,'note':'One unusable TRAIN record remains in source and parent history. No fresh data and no augmentation this round.'}
save(out/'data_audit.json',audit)
base=read('configs/round9_views50_fault.json')
for key in ['training_views_file','training_views_sha256']:base.pop(key,None)
base.update(seed=47,training_view_probability=0.,initialization_artifact=parent)
trials=[]
for name,lr,dropout,expert in [('shared_control',1e-4,0.,False),('app_lr100',1e-4,0.,True),('app_lr30',3e-5,0.,True),('app_drop15',1e-4,.15,True)]:
 c=copy.deepcopy(base);c.update(artifact_root='artifacts/round10/'+name,head_learning_rate=lr,learning_rate_frozen=lr,fault_hidden_dropout=dropout,head_training_policy='application_fault_heads' if expert else 'fault_heads')
 if expert:c.update(application_fault_names=apps,application_head_upgrade='copy_parent_fault_v1')
 path='configs/round10_'+name+'.json';save(path,c)
 trials.append({'name':name,'config':path,'config_sha256':file_hash(path),'head_training_policy':c['head_training_policy'],'head_learning_rate':lr,'dropout':dropout})
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'execution_scope':'mvp','parent':parent,'parent_checkpoint_sha256':meta['binding']['checkpoint_sha256'],'trials':trials,'selection_config':meta['config'],'training_budget_max_updates':8800,'data_audit_sha256':file_hash(out/'data_audit.json'),'targets':{'root_accuracy':.9,'fault_accuracy':.9,'joint_accuracy':.85},'target_origin':'prior working assumption, not explicit user numeric requirements','selection':'Retain OB validation joint >=95%, SS >=90%; then cohort joint macro, overall joint, sum NLL; compare incumbent. Only original 40 validation cases.','release_guard':'Evaluate only the frozen validation winner. Promote only if validation improves and each historical cohort root/fault/joint does not regress. Failure keeps incumbent; no runner-up or post-regression tuning.','original_cases':400,'new_independent_cases':0,'eligible_train_cases':164,'training_views':0,'known_onset':True,'no_new_backbone_sft':True,'hypothesis':'Shared fault parameters may interfere between observed applications. Initialize public-application experts as exact parent copies; train only expert parameters. Unknown application uses unchanged shared fallback and REVIEW.','routing_apps_derived_from':'TRAIN input.application','predeclared_trials_complete_before_evaluation':True}
save(out/'protocol.json',protocol);shutil.copyfile('outputs/latest_model.json',out/'previous_latest_model.json')
Path('docs/round10_protocol.md').write_text("""# 第十轮：应用专属故障分类头

execution_scope: mvp。检验不同应用共用故障参数是否存在相互影响，不预设一定改善。

沿用400个案例的既定划分。165条TRAIN中仅164条有有效观测，排除项保留在原始文件及父模型历史。40条验证、40条校准、85条门控和三组历史回归均保持不变；本轮没有新独立样本，也不使用增强视图。

按TRAIN可观测 application 字段建立 Online Boutique / Sock Shop 两组故障头；不输入数据集编号、路径、run_id或标签。每组复制父模型的文本、全局数值、根因条件数值故障头，初始化预测不变。共享编码器、根因头和未知应用回退头冻结。故障头仍使用预测根因权重，正式推理每条事故仅调用一次编码器；未知应用强制 REVIEW。

预先固定四组：shared_control（共享故障头，lr=1e-4）；app_lr100（应用头，lr=1e-4）；app_lr30（应用头，lr=3e-5）；app_drop15（应用头，lr=1e-4，隐藏层dropout=0.15）。全部seed47、batch16、weight_decay0.01、最多2200次更新、80个epoch无改善早停，cohort仅用于TRAIN抽样，loss只训练故障分类。复制两组故障头只增加约13.4万参数；不增加第二个编码器。控制组分离了数据过滤和额外训练的影响，较小学习率与dropout用于限制小样本过拟合。

只按原40条验证选择一次，先要求OB联合准确率至少95%、SS至少90%，再依次比较四cohort联合均值、总联合准确率及sum NLL；纳入第五轮现任比较。选中模型封存后才重做校准、门控和历史回归。仅验证改善且三个历史回归集的根因、故障、联合均不下降时允许替换默认模型。失败不测试次优模型，不依据回归错误继续调参。

反复使用验证和回归集不能证明独立泛化或自动接受安全性。90%根因/90%故障/85%联合是之前工程工作假设，不是原Prompt明确数值。门控仍要求经验联合风险不超过5%、至少30组接受；若无可行阈值则全部REVIEW。系统不执行运维操作。验收包含真实训练、权重与输入审计、缓存/完整模型一致性、CPU重载与实际HTTP、CPU基准和机器可读证据。新分支单测不是正式效果数据。
""",encoding='utf-8')
print('FROZEN',file_hash(out/'protocol.json'),'train applications',apps,'eligible',164,flush=True)
