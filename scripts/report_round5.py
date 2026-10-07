"""Assemble actual fifth-round outcomes after all checks, without further tuning."""
from pathlib import Path
import math,shutil,json
import numpy as np
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

out=Path('outputs/round5');sel=read(out/'selection.json');protocol=read(out/'protocol.json');folder=Path(sel['selected']['artifact'])
meta=read(folder/'metadata.json');cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash(out/'protocol.json')==sel['protocol_sha256'] and file_hash(out/'selection_precision_fix.json')==sel['precision_fix_sha256']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
for name in ['metadata.json','calibrator.json','policy.json','resolved_config.json','benchmark.json','integration.json','test_freeze.json']:
    target=out/'selected_artifact'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,target)
for name in ['regression_ss','reverse_candidates','consistent_service_rename','mask_all_metrics']:
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    for file in ['predictions.json','metrics.json','diagnostics.png']:shutil.copyfile(folder/name/file,dest/file)
sets={'regression_re1_ob':'outputs/round4/regression_re1_ob','regression_re2_ob':'outputs/round4/regression_re2_ob','regression_ss':'outputs/round4/new_test'}
comparisons={};failures={}
for name,oldpath in sets.items():
    rows=read(out/name/'predictions.json');oldrows=read(Path(oldpath)/'predictions.json');oldbyid={r['run_id']:r for r in oldrows}
    assert set(oldbyid)=={r['run_id'] for r in rows}
    metrics=read(out/name/'metrics.json');oldmetrics=read(Path(oldpath)/'metrics.json');paired={}
    for head in ['root','fault','joint']:
        b=np.array([oldbyid[r['run_id']][head+'_correct'] for r in rows],int);a=np.array([r[head+'_correct'] for r in rows],int);d=a-b;rng=np.random.default_rng(20261009)
        boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
        paired[head]={'old_correct':int(b.sum()),'new_correct':int(a.sum()),'n':len(a),'fixed':int(((b==0)&(a==1)).sum()),'regressed':int(((b==1)&(a==0)).sum()),'difference':float(d.mean()),'paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'new_accuracy_wilson_95':wilson(int(a.sum()),len(a))}
    values={'root_accuracy':metrics['root']['acc_at_1'],'fault_accuracy':metrics['fault']['accuracy'],'joint_accuracy':metrics['joint_accuracy']}
    targets={k:{'actual':values[k],'target':v,'met':values[k]>=v,'required_correct':math.ceil(v*len(rows))} for k,v in protocol['targets'].items()}
    comparisons[name]={'current':metrics,'previous':oldmetrics,'paired':paired,'working_targets':targets,'all_working_targets_met':all(t['met'] for t in targets.values()),'role':'previously opened regression; no independent confirmation'}
    failures[name]=[{'run_id':r['run_id'],'cohort':r['cohort'],'root_gold':r['candidate_ids'][r['root_target']],'root_pred':r['candidate_ids'][r['root_pred']],'fault_gold':FAULTS[r['fault_target']],'fault_pred':FAULTS[r['fault_pred']],'routing':r['routing']} for r in rows if not r['joint_correct']]
save(out/'failure_cases.json',failures)
import xml.etree.ElementTree as ET
suites=ET.parse(out/'test-results.xml').getroot();passed=sum(int(s.get('tests','0'))-int(s.get('failures','0'))-int(s.get('errors','0'))-int(s.get('skipped','0')) for s in suites.findall('testsuite'))
assert passed==35
allmet=all(c['all_working_targets_met'] for c in comparisons.values())
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','tests':{'passed':passed,'failed':0},'original_ob_regression_targets_met':comparisons['regression_re2_ob']['all_working_targets_met'],'all_regression_targets_met':allmet,'independent_generalization_target_status':'not_reconfirmed_no_new_independent_cases','gate_empirical_constraint_met':pol['status']=='selected','production_safety_confirmed':False,'automatic_acceptance_validated':False,'original_ob_regression_selective_risk':comparisons['regression_re2_ob']['current']['selective']['selective_risk'],'new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'mvp_definition_of_done':'engineering regression passed; original heldout evidence remains documented in earlier rounds; no fresh holdout in round5','checks':read('outputs/round4/status.json')['checks'],'not_run':['fresh independent Online Boutique confirmation','fresh independent gate safety evaluation','production incidents','new application generalization','external LLM','KD','quantization','ONNX']}
save(out/'status.json',status)
ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename','mask_all_metrics']}
if (out/'mask_temporal_ob/metrics.json').exists():ablations['mask_temporal_ob']=read(out/'mask_temporal_ob/metrics.json')
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'benchmark':bench,'integration':integration,'calibrator':cal,'policy':pol,'gate_diagnostics':read(out/'gate_diagnostics.json'),'ablations':ablations,'status_summary':status,'limitation':'Every evaluation set is historical regression. Repeated validation selection and gate reuse; no independent quality or production safety confirmation.'}
save(out/'results.json',result)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'source_precision_fix':read(out/'selection_precision_fix.json')})
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
if sel['promote_by_validation']:
    save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'frozen validation: retain OB>=95%, SS>=90%, equal cohort/overall accuracy then lower NLL; no regression selection','report':str(Path('docs/round5_results.md').resolve()),'routing':pol['status'],'quality_target_status':'regression_only_no_fresh_confirmation','original_ob_fresh_target_status':status['independent_generalization_target_status'],'test_role':'all historical test sets are regression only','previous_pointer':str(previous)})
pct=lambda x:'null' if x is None else f'{x*100:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
table='\n'.join(f"| {labels[n]} | {c['current']['runs']} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
trials='\n'.join(f"| {Path(c['artifact']).parent.name} | {c['steps']} | {c['best_epoch']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |" for c in sel['candidates'])
selective='\n'.join(f"| {labels[n]} | {pct(c['current']['selective']['coverage'])} | {pct(c['current']['selective']['selective_risk'])} |" for n,c in comparisons.items())
bestgate=result['gate_diagnostics']['best_observed_risk_with_at_least_30']
newrisk='没有至少 30 个有效案例的候选阈值' if bestgate is None else f"候选阈值中最小观测错误率 {bestgate['errors']}/{bestgate['n']}={pct(bestgate['risk'])}"
gate_state='存在满足门槛的经验阈值' if pol['status']=='selected' else '没有满足门槛的阈值，全部 REVIEW'
ob=comparisons['regression_re2_ob'];obselective=ob['current']['selective'];obstate='达到暂定数值目标' if ob['all_working_targets_met'] else '仍未达到全部暂定数值目标'
report=f"""# DecisionOS-SRE 第五轮真实执行结果

execution_scope: mvp；experiment_completed。共完成 4 组训练、{result['training_updates']} 次更新；训练函数计时合计 {result['summed_training_seconds']:.1f} 秒，不含数据审计、启动和完整 CPU 验收。

**原 Online Boutique RE2 历史回归{obstate}。本轮没有新增独立案例，不能宣称新的泛化目标已被确认。** 工作质量目标沿用 90% 根因 / 90% 故障 / 85% 联合准确率，该数值来自前轮工作假设，并非用户提供的精确指标。

## 实际分数

| 历史回归集 | 案例数 | 上轮根因 / 故障 / 联合 | 本轮根因 / 故障 / 联合 |
|---|---:|---|---|
{table}

所有表中数据都已在此前轮次打开。本轮冻结选择后才重新评估，不使用回归结果再次训练或选择。逐案例预测、修复与退化计数、Wilson 区间和配对 bootstrap 区间保存于 results.json；小样本、重复查看以及案例采集独立性未证实均限制结论。不能把提高历史分数等同于独立泛化。

## 本轮改变与选择理由

原表示仅给出均值变化，新数值表示 temporal-v1 补充 p10、p90、波动、趋势、前后半段差异和存在标记。只用保留指标在 [-300,0) 基线及 [0,60] 秒观测内的值，裁剪并固定变换，无标签、路径或 cohort 输入。仍依赖公开注入起点，未评估真实起点发现。候选预留和一次共享 ModernBERT 编码保持不变，新增列使用零权重初始化，原数字列与文本保持原样。

原有 400 个案例组全部保留：train165 / validation40 / calibration40 / gate85 / OB历史回归15+25 / SS历史回归30。源文件哈希复核通过，训练及验证 205 条旧表示指纹一致。三个时间模型使用同配置的不同 seed，均获得 92.5% 验证联合准确率；这是有限种子下的验证稳定性观察，没有进行独立多种子测试确认。

| 试验 | 实际更新 | 最佳 epoch | 验证联合 | 验证 sum NLL |
|---|---:|---:|---:|---:|
{trials}

选择规则训练前封存：保留 OB 验证联合≥95%、SS≥90%，再按四 cohort 联合均值、总体联合、sum NLL 排序，包含上轮现任对照。选中 `{folder}`，最佳 epoch {sel['selected']['best_epoch']} / 更新 {sel['selected']['best_steps']}；实际更新 {sel['selected']['steps']}。上一轮 sum NLL 为 {sel['incumbent']['validation']['sum_nll']:.6f}，本轮 {sel['selected']['validation']['sum_nll']:.6f}。验证准确率持平时，概率损失改善构成选中依据。

修复了离散准确率在 0.9 与 0.8999999999999999 之间的浮点平局问题；对准确率排序取 12 位精度，再比较 NLL。修复发生在校准、门控和回归前；四组完整训练历史复核表明最优 epoch 与已保存权重均不变，无须重训。原选择记录、代码前后哈希与核查记录保存在 selection_before_precision_fix.json 和 selection_precision_fix.json。

缓存只包含训练和验证输入；所有抽样 ID 均属于 TRAIN，重复抽样不是独立新案例。参数量 {meta['parameter_count']:,}，训练参数 {meta['trainable_parameters']:,}，缓存与完整模型的验证 logits 最大差 {meta['cache']['validation_full_logit_max_diff']}。共享主干本轮冻结；是诊断头续训，不是重新训练整个基础模型。

## 校准与门控

40 个 calibration 案例单独拟合温度：根因 {cal['root']['temperature']:.4f}，故障 {cal['fault']['temperature']:.4f}。85 个 gate 案例选择阈值：{gate_state}。{newrisk}。原要求仍为经验联合错误率≤5%、至少接受30个案例组。

**原 OB RE2 历史回归中，接受 {obselective['accepted_total']} 条、错误 {obselective['errors']} 条，错误率 {pct(obselective['selective_risk'])}，高于 5%。因此门控集经验达标并不等于跨案例自动接受风险达标；不能宣称自动接受已获验证。**

policy 状态 `{pol['status']}`，选择详情 `{pol['selection']}`。已有 calibration 和 gate 被复用；选出的经验风险及区间是描述性结果，没有生产安全保证。只接受诊断，不执行任何运维动作。缺少时间证据的旧格式输入会 REVIEW。

| 历史回归集 | 接受覆盖率 | 已接受案例联合错误率 |
|---|---:|---:|
{selective}

## 验收与性能

{passed} 项测试通过；真实 HTTP health/ready/decide、拒绝标签字段、空候选、单候选、候选超预算、未知应用及缺失时间证据检查通过。CPU checkpoint 重载最大 logits 差 {integration['reload_max_abs_logit_diff']}。时间截断、常数/缺失指标、旧序列兼容、数值维度迁移、标签隔离、候选顺序与候选重命名均有检查。

CPU 8 线程、batch1：3个SS历史案例各预热3次、测量12次；端到端 P50 {bench['end_to_end']['p50_ms']:.1f} ms / P95 {bench['end_to_end']['p95_ms']:.1f} ms，吞吐 {bench['end_to_end']['throughput_per_second']:.3f} 请求/秒，进程峰值工作集 {bench['peak_process_ram_bytes']/1024**3:.2f} GiB。测量受机器负载影响，不是严格配对的硬件速度实验。

候选倒序、统一重命名、删除所有指标以及删除 OB 时间特征的结果见 results.json 的 ablations；不据此再调参。原 MVP 工程 DoD 的数据审计、共享推理、损失/缺失标签、checkpoint重载、校准和门控绑定、CPU/API及机器结果要求已回归验收；本轮没有新的未打开 holdout，独立质量确认保持未完成。

## 产物与复现

仓库 D:/CODEX/DecisionOS-SRE。选中 checkpoint SHA256 `{meta['binding']['checkpoint_sha256']}`，split hash `{meta['binding']['split_hash']}`，pipeline hash `{meta['binding']['pipeline_hash']}`。旧模型保留，outputs/latest_model.json 按冻结的验证选择更新。模型、数据与训练日志均留在本机；大型文件不进入 Git。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe -m decisionos_sre serve --artifact artifacts/round5/temporal_seed44/frozen --port 8000
./scripts/reproduce_retraining.ps1 -Config configs/round5_temporal_seed44.json -Mode frozen
```

复现命令建立新的 artifact_root，依赖保留的 data/round5、固定主干及 round4 父 checkpoint；再次评估仍是历史回归。数据准备脚本 scripts/prepare_round5.py、训练 scripts/run_round5_training.py、评测 scripts/evaluate_round5.py 和报告 scripts/report_round5.py 保存于仓库，原封存结果不覆盖。

详细数据审计、选择协议、逐案例概率和错误、训练抽样审计、门控、机器状态、模型清单位于 outputs/round5。协议见 docs/round5_protocol.md。尚未运行新独立同应用确认、新独立 gate 安全确认、生产事故、KD、外部 LLM、量化及 ONNX；不将其写成成果。
"""
Path('docs/round5_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT',str(Path('docs/round5_results.md').resolve()),flush=True)
for name,c in comparisons.items():print(name,triple(c['previous']),'->',triple(c['current']),'selective',c['current']['selective'],flush=True)
print('POLICY',pol['status'],pol['selection'],'UPDATES',result['training_updates'],'BENCH P95',bench['end_to_end']['p95_ms'],flush=True)
