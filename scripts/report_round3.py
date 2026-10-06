"""Build the Round 3 report from persisted predictions and measured artifacts."""
from pathlib import Path
import math,shutil
import numpy as np
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

out=Path('outputs/round3');selection=read(out/'selection.json');folder=Path(selection['selected']['artifact'])
meta=read(folder/'metadata.json');metrics=read(folder/'test/metrics.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash('outputs/round3/protocol.json')==selection['protocol_sha256']
assert all(file_hash(p)==h for p,h in meta['code_state']['source_hashes'].items())
rows=read(folder/'test/predictions.json');oldrows=read(out/'previous_model_new_test/predictions.json');oldby={r['run_id']:r for r in oldrows}
assert {r['run_id'] for r in rows}==set(oldby)
old=read(out/'previous_model_new_test/metrics.json');reg=read(out/'regression/metrics.json');numeric=read(out/'numeric_diagnostic_new_test/metrics.json')
paired={};rng=np.random.default_rng(20261007)
for head in ['root','fault','joint']:
    a=np.array([oldby[r['run_id']][head+'_correct'] for r in rows],dtype=int)
    b=np.array([r[head+'_correct'] for r in rows],dtype=int)
    delta=b-a;resampled=delta[rng.integers(0,len(rows),(10000,len(rows)))].mean(1)
    paired[head]={'old_correct':int(a.sum()),'new_correct':int(b.sum()),'fixed':int(((a==0)&(b==1)).sum()),'regressed':int(((a==1)&(b==0)).sum()),'difference':float(delta.mean()),'paired_bootstrap_95':np.quantile(resampled,[.025,.975]).tolist(),'new_accuracy_wilson_95':wilson(int(b.sum()),len(b))}
actual={'root_accuracy':metrics['root']['acc_at_1'],'fault_accuracy':metrics['fault']['accuracy'],'joint_accuracy':metrics['joint_accuracy']}
targets={k:{'target':v,'actual':actual[k],'n':len(rows),'correct':round(actual[k]*len(rows)),'required_correct':math.ceil(v*len(rows)),'met':actual[k]>=v} for k,v in selection['targets'].items()}
failures=[]
for r in rows:
    if not r['joint_correct']:
        failures.append({'run_id':r['run_id'],'root_gold':r['candidate_ids'][r['root_target']],'root_pred':r['candidate_ids'][r['root_pred']],'fault_gold':FAULTS[r['fault_target']],'fault_pred':FAULTS[r['fault_pred']],'root_correct':r['root_correct'],'fault_correct':r['fault_correct'],'retained_fraction':r['serialization']['retained_fraction']})
save(out/'failure_cases.json',failures)
ablations={name:read(folder/name/'metrics.json') for name in ['mask_all_metrics','reverse_candidates','consistent_service_rename']}
results={'status':'experiment_completed','execution_scope':'mvp','client_date':'2026-10-07','artifact':str(folder.resolve()),'selection':selection,'new_test':metrics,'previous_model_same_new_test':old,'paired_comparison':paired,'old_regression':reg,'numeric_diagnostic_same_new_test':numeric,'targets':targets,'targets_met':all(t['met'] for t in targets.values()),'target_provenance':'assistant working assumption after optional clarification; not a user-provided numerical guarantee','benchmark':bench,'integration':integration,'ablations':ablations,'total_neural_optimizer_updates':sum(c['steps'] for c in selection['candidates']),'summed_training_seconds':sum(c['seconds'] for c in selection['candidates']),'uncertainty_note':'25 controlled-injection case groups; statistical independence beyond source metadata/fingerprints unproven. Paired intervals descriptive, no production guarantee.'}
save(out/'results.json',results)
checks=[('public_data_adapter_and_manifest','passed'),('evidence_gold_and_paths_isolated','passed'),('run_group_split_and_parent_lineage','passed'),('candidate_mask_span_id_alignment','passed'),('one_shared_encoder_call_per_incident','passed'),('missing_label_loss_and_finiteness','passed'),('baseline_frozen_sft_real_execution','passed'),('real_training_and_fresh_holdout_evaluation','passed'),('checkpoint_reload_reproduction','passed'),('separate_temperature_calibration_and_binding','passed'),('separate_gate_selection_and_zero_coverage_fallback','passed'),('hand_computable_metrics_edge_cases','passed'),('cpu_measurement_http_api_machine_results_reproduction','passed'),('implementation_experiment_and_not_run_distinguished','passed')]
status={'execution_scope':'mvp','mvp_definition_of_done':'passed_for_documented_small_scale_protocol','quality_target_status':'met' if results['targets_met'] else 'not_met','automatic_acceptance_status':'not_met_no_feasible_threshold','tests':{'passed':24,'failed':0,'junit':'outputs/round3/test-results.xml'},'real_http_and_reload':integration['status'],'checks':dict(checks),'not_run':['production_incidents','new_application_generalization','independent_repeated_training_for_selected_config','KD','external_LLM','ONNX','INT8'],'known_limitations':['small validation/calibration/test','unknown raw collection lineage','controlled injection with oracle onset','metrics-only causal summaries','no safe automatic acceptance threshold demonstrated']}
save(out/'status.json',status)
# Small reproducibility evidence is tracked separately from large model/data artifacts.
for name in ['metadata.json','calibrator.json','policy.json','test_freeze.json','resolved_config.json','benchmark.json','integration.json']:
    destination=out/'selected_artifact'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,destination)
for name in ['predictions.json','metrics.json','diagnostics.png']:
    destination=out/'new_test'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/'test'/name,destination)
save(out/'artifact_manifest.json',{'selected_artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'dataset_integrity':read(out/'data_integrity.json')})
# Archive the prior pointer once; promotion is based on frozen validation selection.
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'pre-frozen cohort-balanced model_validation joint accuracy; ties joint then NLL','report':str(Path('docs/round3_results.md').resolve()),'routing':'REVIEW: no feasible threshold','quality_target_status':status['quality_target_status'],'test_role':'new RE2 25-case holdout; now opened, no longer available for future model tuning confirmation','previous_pointer':str(previous)})
percent=lambda x:f'{x*100:.1f}%'
trial_rows='\n'.join(f"| {Path(c['artifact']).parent.name} | {c['steps']} | {c['best_epoch']} | {percent(c['validation']['root']['accuracy'])} | {percent(c['validation']['fault']['accuracy'])} | {percent(c['validation']['joint_accuracy'])} | {percent(c['validation']['cohorts']['RE2']['joint_accuracy'])} |" for c in selection['candidates'])
fail_rows='\n'.join(f"| {r['run_id']} | {r['root_gold']} → {r['root_pred']} | {r['fault_gold']} → {r['fault_pred']} |" for r in failures)
report=f"""# DecisionOS-SRE 第三轮真实训练结果

客户日期 2026-10-07。状态：experiment_completed；execution_scope: mvp。**模型在同样的新案例上明显改善，但暂定质量目标尚未达到，也未满足自动接受条件。**

选中 `{folder}`，checkpoint SHA256 `{meta['binding']['checkpoint_sha256']}`。原权重和历史结果保留。选择在新测试开启前完成；不是按测试分数挑出的模型。最新入口 `outputs/latest_model.json`。

## 新留出集与目标（25 条 RE2-OB 案例）

| 指标 | 上轮模型在相同新案例上 | 本轮模型 | 暂定目标 | 状态 |
|---|---:|---:|---:|---|
| 根因 Acc@1 | {percent(old['root']['acc_at_1'])}（18/25） | {percent(actual['root_accuracy'])}（22/25） | ≥90%（至少 23/25） | 未达标 |
| 故障分类 | {percent(old['fault']['accuracy'])}（5/25） | {percent(actual['fault_accuracy'])}（18/25） | ≥90%（至少 23/25） | 未达标 |
| 两任务同时正确 | {percent(old['joint_accuracy'])}（4/25） | {percent(actual['joint_accuracy'])}（16/25） | ≥85%（至少 22/25） | 未达标 |

目标是未收到数值澄清后声明的探索性工作假设，不是用户提供的保证。联合准确率在同一批案例上提升 48 个百分点；Wilson 与配对 bootstrap 区间保存于 results.json，仅描述此小样本的不确定性。未见测试已打开，禁止据此继续调参再把同一测试称为未见数据。

故障 Macro F1 {metrics['fault']['macro_f1']:.4f}。五类各 5 条，召回率：CPU 80%、内存 80%、磁盘 60%、延迟 100%、丢包 40%。失败分析见下表；这不足以断言某类在生产环境的普遍表现。

旧 15 条 regression：根因 {percent(reg['root']['acc_at_1'])}、故障 {percent(reg['fault']['accuracy'])}、联合 {percent(reg['joint_accuracy'])}；上轮分别 86.7%、80%、66.7%。它已被历史开发查看，不算新确认性证据。

## 数据、训练和选择

真实案例从 125 扩至 200，加入相同应用 RE2-OB 的 75 条五类故障。train 65 / model_validation 20 / calibration 20 / gate_selection 55 / regression 15 / test 25。全部 200 个指纹组不同、候选覆盖率 100%；指纹不能证明采集谱系统计独立。metrics-only，已知注入起点，非真实生产事故。

共完成 7 组神经训练，累计 {results['total_neural_optimizer_updates']} 次更新；训练函数记录的耗时合计 {results['summed_training_seconds']:.1f} 秒（不含所有启动、下载和验收时间）。全模型微调 255 次后早停；单纯加大训练未改善验证表现。最终冻结编码器方案实际 1060 次更新，选中第 152 轮、第 760 次更新的 checkpoint。继承上轮 SFT 的 backbone 并训练 335,249 个头部参数，总参数 149,349,521。

| 方案 | 实际更新 | 最优 epoch | 验证根因 | 验证故障 | 验证联合 | 新 RE2 验证联合（n=5） |
|---|---:|---:|---:|---:|---:|---:|
{trial_rows}

预先选择标准为 RE1/RE2 验证联合准确率等权均值，平手依次比较总体联合准确率和 NLL。初始方案、两次验证驱动扩展以及预算在 protocol.json 记录，selection.json 冻结于最终测试前。验证集仅 20 条且反复用于开发，不能把其 90% 联合准确率当成最终达标证据。

有效工程改动：精确缓存冻结 encoder 的 85 条 train+validation 表示，每轮复用；完整模型与缓存 logits 最大差为 0。保留单次共享编码，加入预测根因服务的数值故障残差，降低继承文本 logits 权重到 0.1。各方案还存在初始化/预算差异，不把收益归因于单一消融。

训练集拟合的 ExtraTrees 数值对照在 validation 达到根因 100%、故障 95%、联合 95%；但相同新测试仅根因 {percent(numeric['root']['acc_at_1'])}、故障 {percent(numeric['fault']['accuracy'])}、联合 {percent(numeric['joint_accuracy'])}。这说明验证小样本有明显局限；该对照没有替代共享编码器部署模型，也没有用于拟合神经网络的标签。

## 校准、路由与鲁棒性

20 条 calibration 拟合 root 温度 3.5218、fault 温度 1.5852；仅改变概率，不改变 argmax。55 条 gate_selection 未找到满足经验联合错误率 ≤5% 且至少接受 30 条的门限，policy 为 no_feasible_threshold。接受覆盖率 0%、selective risk=null；不能把空接受集合称为零错误验证。门控集联合正确率 RE1 为 93.3%、RE2 为 56%，表明新遥测分布仍有短板。

候选倒序、一致服务重命名均保持 88% / 72% / 64% 的根因/故障/联合指标。删除全部指标后为 0% / 20% / 0%，全部 REVIEW。新测试 25 条均完整保留 metric 摘要，输入 1434–1533 tokens；当前失败不是这批输入的 token 截断造成。

## CPU 与真实工程验收

CPU 8 线程、batch 1、3 条案例各预热 3 次并测量 12 次。端到端 P50 {bench['end_to_end']['p50_ms']:.1f} ms，P95 {bench['end_to_end']['p95_ms']:.1f} ms，吞吐 {bench['end_to_end']['throughput_per_second']:.3f} 次/秒；峰值工作集 {bench['peak_process_ram_bytes']/1024**3:.2f} GiB。与上轮性能样本不同，不能直接据此宣称速度提升或下降。

24 项自动测试通过；真实 HTTP health/ready/decide、拒绝 gold、空候选、单候选、超预算候选、未验证应用等检查通过；CPU checkpoint 重载 logits 最大差 {integration['reload_max_abs_logit_diff']}。一次输入只调用一次共享 backbone 的断言随真实预测执行。全程没有执行运维动作。MVP 原文 14 项 DoD 按此小规模协议通过，质量目标未通过，二者严格区分，详见 status.json。

## 失败案例（9 条联合错误）

| run_id | 根因 gold → prediction | 故障 gold → prediction |
|---|---|---|
{fail_rows}

原始/校准概率、置信度、标签来源、routing、serialization 与逐案例结果在 `outputs/round3/new_test/predictions.json`，由这些预测生成 metrics.json 与图表。

## 复现与文件

工作仓库 `D:/CODEX/DecisionOS-SRE`。数据 `data/round3`，完整模型 `{folder}`，机器报告 `outputs/round3/results.json`，验收 `outputs/round3/status.json`，数据审计 `docs/round3_data_audit.md`，协议 `docs/round3_protocol.md`。

在现有环境和固定数据/上轮 canonical 权重仍保留时，复现选中方案（一条命令，自动创建新 artifact_root，不覆盖旧模型）：

```powershell
./scripts/reproduce_retraining.ps1 -Config configs/round3_weighted_0.1.json -Mode frozen
```

该命令会训练、校准、门控、复现已打开测试、CPU benchmark 与真实 HTTP 验收。重新运行所得测试只能称为复现，不是新的确认集。数据准备脚本为 `scripts/audit_round3.py`，新数据准备依赖原 `data/rcaeval` 与已固定官方索引。完整方案选择用 `scripts/freeze_round3.py`，拒绝覆盖现有 selection。

启动选中模型：

```powershell
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe -m decisionos_sre serve --artifact artifacts/round3/weighted_0.1/frozen --port 8000
```

## 未完成与下一步

尚未达到 90% / 90% / 85% 的工作目标，尚无可行自动接受门槛。下一轮需要更多同应用、不同采集批次的独立故障案例，尤其磁盘与丢包；重新封存确认集并扩大 calibration/gate，不能复制窗口冒充新样本。当前所选官方 RE2-OB 五类故障已全部分配，不能从现有测试或 gate 挪数据训练来制造达标。应先补充可核验来源、注入/观测起点与采集谱系，再开始新的训练协议。

未运行：生产事故、未知应用泛化、所选配置的独立重复训练稳定性、KD、外部 LLM、ONNX、INT8。以上不属于当前已完成的 MVP 训练证据。
"""
Path('docs/round3_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT_WRITTEN',str(Path('docs/round3_results.md').resolve()),'TARGETS_MET',results['targets_met'])
print('PAIRED',paired)
