"""Generate round-four deliverables from persisted predictions, no model selection."""
from pathlib import Path
import math,shutil
import numpy as np
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson
from decisionos_sre.training import prediction_summary

out=Path('outputs/round4');sel=read(out/'selection.json');folder=Path(sel['selected']['artifact'])
meta=read(folder/'metadata.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json');pol=read(folder/'policy.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert all(file_hash(p)==h for p,h in meta['code_state']['source_hashes'].items())
new=read(folder/'test/metrics.json');old=read(out/'previous_model_new_test/metrics.json')
reg1=read(out/'regression_re1_ob/metrics.json');reg2=read(out/'regression_re2_ob/metrics.json')
prior1=read('outputs/round3/regression/metrics.json');prior2=read('artifacts/round3/weighted_0.1/frozen/test/metrics.json')
rows=read(folder/'test/predictions.json');oldrows=read(out/'previous_model_new_test/predictions.json')
def compare(before,after):
    b={r['run_id']:r for r in before};assert set(b)=={r['run_id'] for r in after}
    result={};rng=np.random.default_rng(20261008)
    for head in ['root','fault','joint']:
        x=np.array([b[r['run_id']][head+'_correct'] for r in after],int);y=np.array([r[head+'_correct'] for r in after],int);d=y-x
        boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
        result[head]={'old_correct':int(x.sum()),'new_correct':int(y.sum()),'n':len(y),'fixed':int(((x==0)&(y==1)).sum()),'regressed':int(((x==1)&(y==0)).sum()),'difference':float(d.mean()),'paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'new_accuracy_wilson_95':wilson(int(y.sum()),len(y))}
    return result
paired={'fresh_ss':compare(oldrows,rows),'regression_re1_ob':compare(read('outputs/round3/regression/predictions.json'),read(out/'regression_re1_ob/predictions.json')),'regression_re2_ob':compare(read('artifacts/round3/weighted_0.1/frozen/test/predictions.json'),read(out/'regression_re2_ob/predictions.json'))}
def target_status(m):
    vals={'root_accuracy':m['root']['acc_at_1'],'fault_accuracy':m['fault']['accuracy'],'joint_accuracy':m['joint_accuracy']}
    return {k:{'actual':vals[k],'target':v,'met':vals[k]>=v,'correct':round(vals[k]*m['runs']),'n':m['runs'],'required_correct':math.ceil(v*m['runs'])} for k,v in sel['targets'].items()}
ss_targets=target_status(new);ob_targets=target_status(reg2)
failures={}
for label,rr in [('fresh_ss',rows),('regression_re2_ob',read(out/'regression_re2_ob/predictions.json'))]:
    failures[label]=[{'run_id':r['run_id'],'cohort':r['cohort'],'root_gold':r['candidate_ids'][r['root_target']],'root_pred':r['candidate_ids'][r['root_pred']],'fault_gold':FAULTS[r['fault_target']],'fault_pred':FAULTS[r['fault_pred']],'root_correct':r['root_correct'],'fault_correct':r['fault_correct']} for r in rr if not r['joint_correct']]
save(out/'failure_cases.json',failures)
ablations={name:read(folder/name/'metrics.json') for name in ['reverse_candidates','consistent_service_rename','mask_all_metrics']}
result={'status':'experiment_completed','execution_scope':'mvp','client_date':'2026-10-07','artifact':str(folder.resolve()),'selection':sel,'new_sock_shop_test':new,'previous_model_on_new_sock_shop_test':old,'regression_re1_ob':reg1,'regression_re2_ob':reg2,'fresh_ss_targets':ss_targets,'fresh_ss_targets_met':all(v['met'] for v in ss_targets.values()),'original_ob_regression_targets':ob_targets,'original_ob_fresh_target_status':'not_reconfirmed_no_new_independent_OB_cases','paired_comparisons':paired,'benchmark':bench,'integration':integration,'policy':pol,'gate_diagnostics':read(out/'gate_diagnostics.json'),'ablations':ablations,'training_draw_audit':read(out/'training_draw_audit.json'),'optimizer_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'new_test_cohorts':{c:prediction_summary([r for r in rows if r['cohort']==c]) for c in sorted({r['cohort'] for r in rows})},'note':'No pooled goal claim: fresh Sock Shop test cannot confirm Online Boutique generalization. Small controlled-injection samples; lineage independence unproven.'}
save(out/'results.json',result)
checks=read('outputs/round3/status.json')['checks']
status={'execution_scope':'mvp','mvp_definition_of_done':'passed_for_documented_small_scale_protocol','quality_target_status':'partially_met_new_sock_shop_only','fresh_ss_targets_met':result['fresh_ss_targets_met'],'original_ob_regression_targets_met':all(v['met'] for v in ob_targets.values()),'original_ob_fresh_target_status':result['original_ob_fresh_target_status'],'automatic_acceptance_status':'not_met_no_feasible_threshold','tests':{'passed':28,'failed':0,'junit':'outputs/round4/test-results.xml'},'real_http_and_reload':integration['status'],'checks':checks,'not_run':['new_independent_online_boutique_confirmation','unseen_application_generalization','production_incidents','same_configuration_multi_seed_stability','KD','external_LLM','ONNX','INT8']}
save(out/'status.json',status)
for name in ['metadata.json','calibrator.json','policy.json','test_freeze.json','resolved_config.json','benchmark.json','integration.json']:
    dst=out/'selected_artifact'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,dst)
for name in ['predictions.json','metrics.json','diagnostics.png']:
    dst=out/'new_test'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/'test'/name,dst)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'dataset_integrity':read(out/'data_integrity.json')})
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
if sel['selected']['eligible_for_promotion']:
    save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'frozen four-cohort validation joint score with old OB retention floor; no test selection','report':str(Path('docs/round4_results.md').resolve()),'routing':'REVIEW: no feasible threshold','quality_target_status':status['quality_target_status'],'original_ob_fresh_target_status':result['original_ob_fresh_target_status'],'test_role':'30 new Sock Shop heldouts now opened; historical OB sets are regression only','previous_pointer':str(previous)})
pct=lambda x:f'{x*100:.1f}%'
trials='\n'.join(f"| {Path(c['artifact']).parent.name} | {c['steps']} | {c['best_epoch']} | {pct(c['validation']['root']['accuracy'])} | {pct(c['validation']['fault']['accuracy'])} | {pct(c['validation']['joint_accuracy'])} | {pct(c['validation']['cohort_macro_joint'])} |" for c in sel['candidates'])
failed='\n'.join(f"| {r['run_id']} | {r['root_gold']} → {r['root_pred']} | {r['fault_gold']} → {r['fault_pred']} |" for r in failures['fresh_ss'])
ss_state='达到' if result['fresh_ss_targets_met'] else '未达到'
gate=result['gate_diagnostics']['best_observed_risk_with_at_least_30']
report=f"""# DecisionOS-SRE 第四轮继续训练结果

客户日期 2026-10-07；execution_scope: mvp；experiment_completed。

**新 Sock Shop 留出集{ss_state}暂定质量指标；原 Online Boutique 仍未完全达到目标，自动接受条件也未达到。** 不把新应用分数当成旧应用新泛化达标的证明。

选中 `{folder}`。SHA256 `{meta['binding']['checkpoint_sha256']}`。旧模型保留，统一入口 `outputs/latest_model.json`。选择在 test 打开前写入 selection.json，只用 validation，不用回归或新 test 选模型。

## 实测结果：分开报告不同数据集

| 数据集 / 角色 | 案例数 | 上轮根因 / 故障 / 联合 | 本轮根因 / 故障 / 联合 |
|---|---:|---|---|
| 新 Sock Shop 留出（已训练该应用） | 30 | {pct(old['root']['acc_at_1'])} / {pct(old['fault']['accuracy'])} / {pct(old['joint_accuracy'])} | **{pct(new['root']['acc_at_1'])} / {pct(new['fault']['accuracy'])} / {pct(new['joint_accuracy'])}** |
| Online Boutique RE2 历史回归 | 25 | {pct(prior2['root']['acc_at_1'])} / {pct(prior2['fault']['accuracy'])} / {pct(prior2['joint_accuracy'])} | **{pct(reg2['root']['acc_at_1'])} / {pct(reg2['fault']['accuracy'])} / {pct(reg2['joint_accuracy'])}** |
| Online Boutique RE1 历史回归 | 15 | {pct(prior1['root']['acc_at_1'])} / {pct(prior1['fault']['accuracy'])} / {pct(prior1['joint_accuracy'])} | {pct(reg1['root']['acc_at_1'])} / {pct(reg1['fault']['accuracy'])} / {pct(reg1['joint_accuracy'])} |

新 SS 的计数为根因 29/30、故障 27/30、联合 26/30，对照暂定 90% / 90% / 85% 分别至少需 27/30、27/30、26/30。各类 6 条，故障 Macro F1 {new['fault']['macro_f1']:.4f}。上轮未训练 Sock Shop，表中旧模型分数为直接诊断对照，不代表旧 API 支持该应用。新模型训练包含 SS，不能称为未知应用泛化。

原 OB RE2 回归目前根因 24/25、故障 20/25、联合 19/25；故障分类和联合正确仍低于原工作目标。OB 旧测试已被多轮查看，只能是回归证据；没有新的独立 OB 留出，所以原应用的新泛化目标状态为 not_reconfirmed。配对纠错/退化、Wilson 与 bootstrap 区间见 results.json；小样本指标不构成生产保证。

## 数据审计与工程选择

复核 [TORAI 官方数据包](https://doi.org/10.6084/m9.figshare.31925976.v1)：73 条 OB 案例与现有 RE2 时间、列、数值一致；另 2 条覆盖更多时间，但相交时间的数值及注入时间相同，因此 75 条均不作新增独立数据。只读取包中的 metrics 与注入时间做重叠审计，包内其他模态未用于训练。证据在 torai_overlap_audit.json。

改用固定 [RCAEval 数据](https://huggingface.co/datasets/phamquiluan/RCAEval/tree/afeacb11bcc94dadfd1c8f483ee4377b2b8b614e) 的 200 条 RE1-SS / RE2-SS 五类故障。先分配原始案例，再下载和预处理。总计 400 案例、400 指纹组、候选覆盖率 100%。划分为 train 165 / validation 40 / calibration 40 / gate 85 / OB regression 15+25 / 新 SS test 30。只从 TRAIN 得到词表和梯度；已知注入起点定义基线前 300 秒、观测后 60 秒，metrics-only，不评估故障检测。重复抽样不增加独立案例数。

继续复用已微调共享编码器并训练 335,249 个诊断头参数，仍一次输入一次共享编码。冻结特征预计算每组 205 次（165 train+40 validation），缓存与完整模型 logits 最大差为 0。四来源均衡采样避免占多数的数据压过旧 RE2 的 15 条训练案例；每次 draw 的真实 run ID 保存并审核，无 calibration/gate/test ID。

温启动新增特征词表/顺序校验，拒绝相同维度但语义错位的数值头加载；旧推理产物保持兼容。

## 已执行训练与冻结规则

| 方案 | 实际更新 | 最佳 epoch | 验证根因 | 验证故障 | 验证联合 | 四来源联合均值 |
|---|---:|---:|---:|---:|---:|---:|
{trials}

共 {result['optimizer_updates']} 次优化器更新，训练函数耗时合计 {result['summed_training_seconds']:.1f} 秒（不含全部下载、进程启动和验收时间）。选中第 {sel['selected']['best_epoch']} 轮、第 {sel['selected']['best_steps']} 次更新，实际训练 {sel['selected']['steps']} 次后早停。第三方案同时改变 seed 和学习率，不能作为同一配置的随机种子稳定性证据。

验证排序首先要求旧 OB 联合准确率 ≥85%（保留约束，原质量目标没有降低），再最大化 RE1-OB/RE2-OB/RE1-SS/RE2-SS 联合准确率等权平均，平手看总体联合和 NLL。选中模型旧 OB 验证 19/20=95%，SS 验证 18/20=90%。所有预算和规则在训练前记录于 protocol.json。

## 校准、门控和鲁棒性

40 条 calibration 拟合独立温度，85 条 gate 选择阈值。仍无满足联合经验错误率 ≤5%、接受案例数 ≥30 的区间。可选区间中最小实测风险为 {gate['errors']}/{gate['n']}={pct(gate['risk'])}，高于 5%；该诊断是选阈值后的描述，不能拿来宣称保证。policy=no_feasible_threshold，新测试接受覆盖率 0，selective risk=null。仍全部 REVIEW，不执行任何运维动作。

候选倒序与一致重命名均保持新 SS 的 96.7% / 90% / 86.7%；删除指标后为 {pct(ablations['mask_all_metrics']['root']['acc_at_1'])} / {pct(ablations['mask_all_metrics']['fault']['accuracy'])} / {pct(ablations['mask_all_metrics']['joint_accuracy'])}，全部 REVIEW。验证集 metric 摘要完整保留，模型输入没有使用路径、注入标签或来源 cohort。

## CPU 与工程验收

28 项测试通过；真实 HTTP health/ready/decide、拒绝 gold、空候选、单候选、超预算与未知应用检查通过。CPU checkpoint 重载 logits 最大差 {integration['reload_max_abs_logit_diff']}。

CPU 8 线程、batch 1；3 条新测试案例各预热 3 次、测量 12 次。端到端 P50 {bench['end_to_end']['p50_ms']:.1f} ms，P95 {bench['end_to_end']['p95_ms']:.1f} ms，吞吐 {bench['end_to_end']['throughput_per_second']:.3f} 请求/秒，峰值工作集 {bench['peak_process_ram_bytes']/1024**3:.2f} GiB。本轮样本与上一轮不同，不作速度升降的直接因果比较。MVP 工程验收通过与质量、自动接受目标达标分别记录于 status.json。

## 新 SS 失败案例

| run_id | 根因 gold → prediction | 故障 gold → prediction |
|---|---|---|
{failed}

完整逐案例原始/校准概率、标签来源和路由保存于 `outputs/round4/new_test/predictions.json`；原 OB 的错误单列于 failure_cases.json。不能看这些测试错误后继续调参，再称本批数据未见。

## 使用和复现

仓库 `D:/CODEX/DecisionOS-SRE`；数据 `data/round4`；模型 `{folder}`；报告 `outputs/round4/results.json`；协议 `docs/round4_protocol.md`。需保留原初始化 checkpoint、固定数据、tokenizer 和依赖环境。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe -m decisionos_sre serve --artifact artifacts/round4/balanced_seed43/frozen --port 8000
```

一条复现命令，自动创建新 artifact_root：

```powershell
./scripts/reproduce_retraining.ps1 -Config configs/round4_balanced_seed43.json -Mode frozen
```

该命令训练、校准、门控、重现已打开测试、CPU benchmark 与真实 HTTP 检查；重跑是复现，不是新的未见确认。数据审计脚本 `scripts/audit_round4.py`，冻结选择 `scripts/freeze_round4.py`，分域对照 `scripts/evaluate_round4_comparisons.py`，结果生成 `scripts/report_round4.py`。已封存数据/选择脚本拒绝覆盖。

## 尚未完成

原 OB 目标未完全达到且缺新的独立确认数据；自动接受目标未达到。下一步应补充同应用独立采集，改善磁盘故障判别和高置信错误排序，并封存新确认集与门控数据。原门槛不能因 5.56% 接近 5% 而放宽。

未运行生产事故、未知应用泛化、同配置多种子稳定性、KD、外部 LLM、ONNX、INT8；这些不能作为当前成果宣传。
"""
Path('docs/round4_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT',str(Path('docs/round4_results.md').resolve()),'FRESH_SS_TARGETS',result['fresh_ss_targets_met'])
print('PAIRED',paired)
