"""Publish a fully checked result and apply the predeclared release decision."""
from pathlib import Path
import json,math,shutil,xml.etree.ElementTree as ET
import numpy as np
import torch
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

out=Path('outputs/round10');sel=read(out/'selection.json');protocol=read(out/'protocol.json');guard=read(out/'release_guard.json')
folder=Path(sel['selected']['artifact']);meta=read(folder/'metadata.json')
cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
assert meta['cache']['validation_full_logit_max_diff']<=1e-5
from decisionos_sre.data import load_split
for split,filename in [('calibration','calibration_logits.json'),('gate_selection','gate_predictions.json')]:
 rows=read(folder/filename);expected_ids={e.original_run_id for e in load_split(meta['config']['data_dir'],split)}
 assert len(rows)==len(expected_ids) and {r['run_id'] for r in rows}==expected_ids
 assert all(r['split']==split for r in rows)
assert cal['binding']==pol['binding']==meta['binding']
assert all(cal[h]['temperature']>0 for h in ('root','fault'))
assert read(out/'training_integrity.json')['legacy_cpu_reload_max_abs_difference']<=1e-6
dependency_log=(out/'dependency-check.log').read_bytes()
assert 'No broken requirements found.' in dependency_log.decode('utf-16' if dependency_log.startswith(b'\xff\xfe') else 'utf-8-sig')

for name in ['metadata.json','calibrator.json','policy.json','resolved_config.json','benchmark.json','integration.json','test_freeze.json']:
    target=out/'selected_artifact'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,target)
for name in ['regression_ss','reverse_candidates','consistent_service_rename','mask_all_metrics']:
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    for f in ['predictions.json','metrics.json','diagnostics.png']:shutil.copyfile(folder/name/f,dest/f)
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
parent_state=torch.load(Path(protocol['parent'])/'checkpoint.pt',map_location='cpu',weights_only=True)
state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
expert=bool(meta['config'].get('application_fault_names'))
frozen=[n for n in parent_state if expert or n.startswith(('backbone.','scorer.','numeric_root.'))]
assert all(torch.equal(state[n],parent_state[n]) for n in frozen)
changed=[n for n in state if n not in parent_state or not torch.equal(state[n],parent_state[n])]
allowed=('application_fault_heads.',) if expert else ('fault_head.','numeric_fault.','numeric_local_fault.')
assert changed and all(n.startswith(allowed) for n in changed)
weight_audit={'frozen_tensors_verified':len(frozen),'frozen_weights_exactly_equal':True,'changed_head_tensors':changed,'root_branch_frozen':True,'shared_fallback_frozen':expert,'root_validation_max_abs_logit_change':meta['root_validation_max_abs_logit_change'],'application_fault_names':meta['config'].get('application_fault_names',[]),'no_architecture_change':not expert}
del state,parent_state
save(out/'weight_audit.json',weight_audit)
comparisons={};failures={}
for split in ['regression_re1_ob','regression_re2_ob','regression_ss']:
    rows=read(out/split/'predictions.json');oldrows=read(Path('outputs/round5')/split/'predictions.json');byid={r['run_id']:r for r in oldrows}
    assert set(byid)=={r['run_id'] for r in rows}
    current=read(out/split/'metrics.json');old=read(Path('outputs/round5')/split/'metrics.json');paired={}
    for h in ['root','fault','joint']:
        a=np.array([r[h+'_correct'] for r in rows],int);b=np.array([byid[r['run_id']][h+'_correct'] for r in rows],int);d=a-b
        rng=np.random.default_rng(20261007);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
        paired[h]={'n':len(a),'old_correct':int(b.sum()),'new_correct':int(a.sum()),'fixed':int(((b==0)&(a==1)).sum()),'regressed':int(((b==1)&(a==0)).sum()),'difference':float(d.mean()),'paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'new_accuracy_wilson_95':wilson(int(a.sum()),len(a))}
    vals={'root_accuracy':current['root']['acc_at_1'],'fault_accuracy':current['fault']['accuracy'],'joint_accuracy':current['joint_accuracy']}
    targets={k:{'actual':vals[k],'target':v,'met':vals[k]>=v} for k,v in protocol['targets'].items()}
    comparisons[split]={'current':current,'previous':old,'paired':paired,'working_targets':targets,'all_working_targets_met':all(t['met'] for t in targets.values()),'role':'historical regression only; not independent confirmation'}
    failures[split]=[{'run_id':r['run_id'],'cohort':r['cohort'],'root_gold':r['candidate_ids'][r['root_target']],'root_pred':r['candidate_ids'][r['root_pred']],'fault_gold':FAULTS[r['fault_target']],'fault_pred':FAULTS[r['fault_pred']],'routing':r['routing']} for r in rows if not r['joint_correct']]
save(out/'failure_cases.json',failures)
suites=ET.parse(out/'test-results.xml').getroot().findall('testsuite')
passed=sum(int(s.get('tests','0')) for s in suites);failed=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
assert passed>=69 and failed==0
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','tests':{'passed':passed,'failed':failed},'active_model_updated':guard['promote'],'validation_improved':sel['promote_by_validation'],'regression_release_guard':guard['all_non_regressed'],'all_regression_working_targets_met':all(c['all_working_targets_met'] for c in comparisons.values()),'independent_generalization_target_status':'not_reconfirmed_no_new_independent_cases','automatic_acceptance_validated':False,'production_safety_confirmed':False,'gate_empirical_constraint_met':pol['status']=='selected','new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'historical_mvp_checks':read('outputs/round6/status.json')['historical_mvp_checks'],'current_fresh_holdout':'not_run; only historical regression','not_run':['new backbone SFT','fresh independent confirmation','production incidents','KD','external LLM','quantization','ONNX']}
ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename','mask_all_metrics','mask_temporal_ob']}
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'release_guard':guard,'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'benchmark':bench,'integration':integration,'policy':pol,'calibrator':cal,'ablations':ablations,'weight_audit':weight_audit,'training_integrity':read(out/'training_integrity.json'),'source_snapshot':read(out/'source_snapshot.json'),'status_summary':status,'limitation':'Repeated validation and historical regression; no new independent quality or automatic-acceptance safety confirmation.'}
save(out/'status.json',status)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'training_source_hashes_verified':True})
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
assert Path(read(previous)['artifact'])==Path(protocol['parent'])
if guard['promote']:
    save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'predeclared validation retention/ranking followed by historical regression release guard; no runner-up search','report':str(Path('docs/round10_results.md').resolve()),'routing':pol['status'],'quality_target_status':'historical_regression_only_no_fresh_confirmation','test_role':'all historical test sets are regression only','previous_pointer':str(previous)})
else:assert file_hash('outputs/latest_model.json')==file_hash(previous)
active=read('outputs/latest_model.json');result['active_model']=active;save(out/'results.json',result)
pct=lambda v:'null' if v is None else f'{100*v:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
table='\n'.join(f"| {labels[n]} | {c['current']['runs']} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
trials=[]
for c in sel['candidates']:
 cfg=read(Path(c['artifact'])/'resolved_config.json')
 trials.append(f"| {Path(c['artifact']).parent.name} | {cfg['head_learning_rate']} | {cfg['fault_hidden_dropout']} | {c['steps']} | {c['best_epoch']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |")
trialtable='\n'.join(trials)
selective_table='\n'.join(f"| {labels[n]} | {c['current']['selective']['accepted_total']} | {c['current']['selective']['errors']} | {pct(c['current']['selective']['coverage'])} | {pct(c['current']['selective']['selective_risk'])} |" for n,c in comparisons.items())
decision='通过预先固定的替换条件，已更新本地默认模型' if guard['promote'] else '未通过全部替换条件，保留第五轮默认模型'
report=f"""# DecisionOS-SRE 第十轮：应用专属故障分类头

execution_scope: mvp。{decision}。4组真实GPU训练共 **{result['training_updates']} 次参数更新**，训练函数计时合计 {result['summed_training_seconds']:.1f} 秒。没有新增独立数据，也没有新一轮主干SFT。本轮结论仅适用于反复使用的验证和历史回归，不等于达到生产质量。

## 实际效果

以下每格依次为根因准确率 / 故障准确率 / 联合准确率。

| 历史回归集 | 原案例数 | 默认第五轮 | 本轮封存候选 |
|---|---:|---|---|
{table}

选中 `{folder}`，验证联合准确率 {pct(sel['selected']['validation']['joint_accuracy'])}，sum NLL {sel['selected']['validation']['sum_nll']:.6f}；现任验证联合92.5%，sum NLL {sel['incumbent']['validation']['sum_nll']:.6f}。验证晋升条件={sel['promote_by_validation']}，历史回归保护={guard['all_non_regressed']}，最终替换={guard['promote']}。每组根因/故障/联合比较见 release_guard.json；失败后不测试次优模型或追加调参。

| 实验 | 学习率 | dropout | 实际更新 | 最佳epoch | 验证联合 | sum NLL |
|---|---:|---:|---:|---:|---:|---:|
{trialtable}

四组均从第五轮初始化，seed47、batch16、weight_decay0.01、最多2200更新，80个epoch无改善早停；只训练故障头，根因和主干均冻结。shared_control是同预算共享头对照，另外三组为应用专属头。预先按OB验证联合至少95%、SS至少90%，再按四cohort联合均值、总体联合、NLL选一个模型；协议与配置的哈希在训练前封存。验证40例经过多轮使用，选择偏差不能忽略。

## 数据、架构和工程选择

沿用RCAEval固定版本的400例公开故障注入数据。原始run划分为165训练、40验证、40校准、85门控、15 RE1-OB回归、25 RE2-OB回归、30 SS回归。本轮新案例=0，增强视图=0。训练中一条无有效观测记录 `21a11a8fa96a215914feab22` 不参与抽样，但仍保留原始记录及父模型训练历史，因此有效TRAIN为164。manifest、splits、examples哈希均与前轮一致，无跨split原始run。

应用名称只从TRAIN的公开 `input.application` 确定，映射为 Online Boutique / Sock Shop。每个应用单独复制文本、全局数值、根因条件数值故障分类头；新增约13.4万参数。初始参数来自现任模型，根因加权只使用预测根因，不使用gold。独立应用头避免两个应用通过同一故障参数相互影响，但小样本仍可能过拟合；较低学习率和dropout为预先固定的两个约束方案。所有训练只更新所属故障头，未知应用保留旧共享头并强制REVIEW。映射顺序写入模型与校准绑定，不支持静默重排。

API字段不变；编码器仍共享，一次事故只编码一次，候选span/mask、2048 token预算和temporal-v1不变。CPU仅用8线程；本机Ryzen 9 7945HX、16GB RAM、RTX4060 Laptop 8GB，启动前可用RAM约3.6GiB，因此顺序运行、缓存冻结表示，无新增依赖。已知注入起点和60秒观测窗口仍是实验假设，未测试真实事故检测。

## 按原Definition of Done验收

原始MVP工程验收与本轮增量验收分开记录，历史状态保存在 status.json 的 historical_mvp_checks。历史baseline / frozen / SFT及首次holdout证据仍在此前产物，本轮仅重新训练分类头并跑历史回归，不把它们重新称为独立holdout。

| 原始DoD条目 | 本轮核验或历史证据 |
|---|---|
| 公开数据adapter、标签审计、manifest | 已有真实400例；本轮文件哈希、分组与应用输入复核通过 |
| Evidence/gold、路径和注入答案隔离 | schema/serializer测试通过；新路由只取公开application |
| 同run不跨split、增强继承、预处理分离 | 本轮交叉重叠0；无增强；校准与门控仍用指定分区 |
| 动态候选mask/span/ID、无效输入与截断 | 测试及真实HTTP边界验证通过 |
| 每incident一次共享编码 | 模型单测、真实predict和runtime调用断言通过 |
| 缺失标签、有效loss、无NaN | 单测通过；四组实际训练完成 |
| baseline、frozen、SFT路径 | 历史真实实验已完成；本轮不重复SFT |
| 至少一轮真实训练与holdout、注明范围 | 历史首次holdout已完成；本轮实际训练与历史回归完成，未有新独立holdout |
| checkpoint保存重载 | 本轮CPU最大logits差 {integration['reload_max_abs_logit_diff']} |
| 独立校准、argmax保持、版本不匹配处理 | 校准40例；相关单测和绑定检查通过 |
| gate_selection、无可行阈值全REVIEW | 门控85例；策略状态 `{pol['status']}`；零覆盖回退测试通过 |
| 可手算指标、空接受/缺标/候选缺失 | 指标边界测试通过 |
| CPU实测、可运行API、机器结果与复现 | 本轮benchmark.json、integration.json、results.json和下方命令 |
| 区分完成/实现/mock/未运行 | 69项测试含小fixture；正式指标来自真实RCAEval；未运行项如下 |

{passed}项测试通过，失败0；pip check无损坏依赖。测试有一条Starlette依赖弃用提示，不影响结果。本轮候选 {weight_audit['frozen_tensors_verified']} 个冻结张量与父模型完全相等；缓存/完整GPU推理最大logits差 {meta['cache']['validation_full_logit_max_diff']}。真实HTTP验证健康、诊断、拒绝gold、缺时间证据、单/空/超预算候选及未知应用；若选中应用分支模型，另对两个应用分别比对API与离线概率。所有服务验证进程在结束后停止。

CPU batch1、8线程，固定3个SS输入各预热3次、测量12次：P50 {bench['end_to_end']['p50_ms']:.1f}ms，P95 {bench['end_to_end']['p95_ms']:.1f}ms，吞吐 {bench['end_to_end']['throughput_per_second']:.3f}/秒，进程生命周期峰值工作集 {bench['peak_process_ram_bytes']/1024**3:.2f}GiB。不同轮次系统负载不一致，单次计时不能证明性能提高。

## 门控和局限

候选门控结果 `{pol['selection']}`；预设经验联合风险不超过5%、至少接受30组。以下是候选在历史回归上的实际选择性结果，不是安全承诺，也不能自动外推门控集风险。

| 历史回归集 | 接受数 | 接受后错误数 | 覆盖率 | 接受后错误率 |
|---|---:|---:|---:|---:|
{selective_table}

三个历史回归集是否全部达到工作假设90%根因/90%故障/85%联合：{status['all_regression_working_targets_met']}。这些数值是之前的工程假设，不是原Prompt明确要求。工程验收通过也不等于独立泛化、自动接受风险或生产安全已确认。固定来源中与当前两个应用、五类故障兼容的400例已用完；独立确认仍需要新的、故障定义一致的运行案例。

本轮未运行：新主干SFT、KD、外部LLM、跨新应用测试、生产事故实验、ONNX、INT8量化。未执行任何运维操作。保留全部未晋升候选和失败证据，默认指针按预先固定的保护规则处理。

## 产物与复现

默认模型：`{active['artifact']}`。
本轮候选 checkpoint SHA256：`{meta['binding']['checkpoint_sha256']}`。
模型本地权重与数据由Git忽略，源码、配置、协议、逐案例指标和报告归档。训练时Git工作区标记为dirty：训练前提交只包含新文件及协议，漏暂存的已修改文件在归档时补齐。22个实际训练源文件均由训练时SHA256封存，事后验证与实现快照 `{read(out/'source_snapshot.json')['implementation_snapshot_commit']}` 完全一致；未改写训练元数据或权重。仅检出训练元数据中的旧Git HEAD不足以复现，应使用该实现快照或当前完整仓库；详见 source_snapshot.json。验收证据包括 protocol.json、selection.json、training_integrity.json、training_draw_audit.json、weight_audit.json、artifact_manifest.json、release_guard.json、status.json、results.json。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round10_{folder.parent.name}.json -Mode frozen
```

复现依赖本地 data/round5、pinned backbone及第五轮父模型，通用脚本使用新输出目录。四组完整过程由 prepare_round10.py、run_round10_training.py、evaluate_round10.py、report_round10.py保存；原封存输出不覆盖。
"""
Path('docs/round10_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT',str(Path('docs/round10_results.md').resolve()),'PROMOTED',guard['promote'],'UPDATES',result['training_updates'],flush=True)
for n,c in comparisons.items():print(n,triple(c['current']),c['current']['selective'],flush=True)
