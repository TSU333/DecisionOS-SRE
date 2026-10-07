"""Assemble actual sixth-round outcomes after all checks, without further tuning."""
from pathlib import Path
import math,shutil,json
import numpy as np
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

out=Path('outputs/round6');sel=read(out/'selection.json');protocol=read(out/'protocol.json');evaluation=read(out/'evaluation_selection.json');folder=Path(evaluation['artifact'])
meta=read(folder/'metadata.json');cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(folder/'checkpoint.pt')==evaluation['candidate']['binding']['checkpoint_sha256']
for name in ['metadata.json','calibrator.json','policy.json','resolved_config.json','benchmark.json','integration.json','test_freeze.json']:
    target=out/'selected_artifact'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,target)
mask_name='mask_all_evidence' if meta['config'].get('trace_features') else 'mask_all_metrics'
for name in ['regression_ss','reverse_candidates','consistent_service_rename',mask_name]:
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    for file in ['predictions.json','metrics.json','diagnostics.png']:shutil.copyfile(folder/name/file,dest/file)
sets={'regression_re1_ob':'outputs/round5/regression_re1_ob','regression_re2_ob':'outputs/round5/regression_re2_ob','regression_ss':'outputs/round5/regression_ss'}
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
    targets={k:{'actual':values[k],'target':v,'met':values[k]>=v,'required_correct':math.ceil(v*len(rows))} for k,v in {'root_accuracy':.9,'fault_accuracy':.9,'joint_accuracy':.85}.items()}
    comparisons[name]={'current':metrics,'previous':oldmetrics,'paired':paired,'working_targets':targets,'all_working_targets_met':all(t['met'] for t in targets.values()),'role':'previously opened regression; no independent confirmation'}
    failures[name]=[{'run_id':r['run_id'],'cohort':r['cohort'],'root_gold':r['candidate_ids'][r['root_target']],'root_pred':r['candidate_ids'][r['root_pred']],'fault_gold':FAULTS[r['fault_target']],'fault_pred':FAULTS[r['fault_pred']],'routing':r['routing']} for r in rows if not r['joint_correct']]
save(out/'failure_cases.json',failures)
import xml.etree.ElementTree as ET
suites=ET.parse(out/'test-results.xml').getroot();passed=sum(int(s.get('tests','0'))-int(s.get('failures','0'))-int(s.get('errors','0'))-int(s.get('skipped','0')) for s in suites.findall('testsuite'))
assert passed==41
allmet=all(c['all_working_targets_met'] for c in comparisons.values())
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','tests':{'passed':passed,'failed':0},'original_ob_regression_targets_met':comparisons['regression_re2_ob']['all_working_targets_met'],'all_regression_targets_met':allmet,'independent_generalization_target_status':'not_reconfirmed_no_new_independent_cases','gate_empirical_constraint_met':pol['status']=='selected','production_safety_confirmed':False,'automatic_acceptance_validated':False,'original_ob_regression_selective_risk':comparisons['regression_re2_ob']['current']['selective']['selective_risk'],'new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'mvp_definition_of_done':'engineering regression passed; original heldout evidence remains documented in earlier rounds; no fresh holdout in round6','historical_mvp_checks':read('outputs/round5/status.json')['checks'],'current_fresh_holdout':'not_run; only historical regression','not_run':['fresh independent Online Boutique confirmation','fresh independent gate safety evaluation','production incidents','new application generalization','external LLM','KD','quantization','ONNX']}
status['active_model_updated']=False
status['experimental_candidate_only']=True
save(out/'status.json',status)
ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename',mask_name]}
for n in ['mask_traces_ob','reverse_candidates_ob','consistent_service_rename_ob']:
    if (out/n/'metrics.json').exists():ablations[n]=read(out/n/'metrics.json')
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'evaluated_candidate':evaluation,'active_model_retained':read('outputs/latest_model.json'),'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'benchmark':bench,'integration':integration,'calibrator':cal,'policy':pol,'gate_diagnostics':read(out/'gate_diagnostics.json'),'ablations':ablations,'status_summary':status,'limitation':'Every evaluation set is historical regression. Repeated validation selection and gate reuse; no independent quality or production safety confirmation.'}
save(out/'results.json',result)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'source_hashes_verified':all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())})
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
assert not sel['promote_by_validation'], 'This report describes rejected experimental candidates only'

pct=lambda x:'null' if x is None else f'{100*x:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
table='\n'.join(f"| {labels[n]} | {c['current']['runs']} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
trials='\n'.join(f"| {Path(c['artifact']).parent.name} | {c['steps']} | {c['best_epoch']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |" for c in sel['candidates'])
diag=read('outputs/round7/diagnostic_results.json');active=read('outputs/latest_model.json')
assert not sel['promote_by_validation'] and active['artifact']==sel['incumbent']['artifact']
result['lightweight_diagnostic']=diag;result['source_investigation']=read(out/'source_investigation.json');save(out/'results.json',result)
ob=comparisons['regression_re2_ob']['current'];risk=ob['selective']
trace_audit=read(out/'data_audit.json');kept=[r for r in trace_audit['train_validation_retention'] if r['trace_total']]
retention=sum(r['trace_retained'] for r in kept)/sum(r['trace_total'] for r in kept)
report=f"""# DecisionOS-SRE 继续执行与成功判定

execution_scope: mvp。原 Prompt 的工程流程已通过历史验收；诊断质量与自动接受风险尚不能宣布全部成功。本轮完成新的真实实验，但没有选出优于第五轮的模型，默认入口仍为 `{active['artifact']}`，旧权重不覆盖。

## 本次实际执行

核查了代码、依赖和资源；新增审计固定 RCAEval revision 的75个RE2-OB调用链文件，共 {trace_audit['total_trace_bytes']:,} 字节。原400个案例、所有split及group均保留，新增证据不算新增独立案例。其余325例缺少traces，继续仅使用指标。

实现 completed-span 服务摘要与24个数值通道，零填充迁移原数值头，单例仍只调用一次共享ModernBERT。训练前封存预算和选择规则。四组续训共 **{result['training_updates']} 次更新**、训练函数计时 {result['summed_training_seconds']:.1f} 秒；另完成8组ExtraTrees轻量基线拟合，属于只使用训练/验证数据的诊断，不是已部署模型。

| 方案 | 实际更新 | 最佳 epoch | 验证联合准确率 | 验证 sum NLL |
|---|---:|---:|---:|---:|
{trials}

第五轮现任验证联合准确率92.5%，sum NLL={sel['incumbent']['validation']['sum_nll']:.6f}。本轮所有神经候选准确率相同，NLL均更差；8个轻量基线均为87.5%。根据预先规定的验证规则拒绝替换。不能因训练次数增加、加入新模态或某个回归分数变化就称为模型增强成功。

## 冻结后的实验候选验收

为验证新调用链实现，额外验收验证集最好的调用链候选 `{folder}`。它是实验产物，未晋升；不是用回归结果重新选模型。下面比较的是第五轮默认模型与该实验候选。

| 历史回归集 | n | 第五轮根因 / 故障 / 联合 | 本轮调用链候选根因 / 故障 / 联合 |
|---|---:|---|---|
{table}

全部都是此前已打开的案例。本轮没有新独立测试，重复验证选择也可能过拟合。逐案例概率、修复/退化计数和描述性区间在 results.json；failure_cases.json 保存错误，不能用它们继续拟合再称为未见测试。

候选门控状态 `{pol['status']}`，只在85个gate案例上按经验错误率≤5%、至少30例选择，详情 `{pol['selection']}`。OB RE2历史回归接受 {risk['accepted_total']} 例、错误 {risk['errors']} 例，错误率 {pct(risk['selective_risk'])}。门控、校准均被复用，经验结果不构成独立自动接受安全验证。系统不执行任何运维动作。默认第五轮策略及其已记录的OB接受风险1/7=14.3%保持原记录，不拿实验候选结果替换它。

## 数据与工程正确性

- 使用span结束时间过滤：baseline开始≥onset−300且结束<onset；观测开始≥onset且结束≤onset+60。持续到决策之后的span不参与统计。
- Jaeger时间/持续时长以微秒读取，延迟转成ms；重复traceID/spanID去重；缺失状态码不当成成功。实际入库延迟未知，完成时间仅近似可见性。
- 全span与operation目标服务匹配自身的RPC子集分别汇总；后者是明确的名称匹配规则，源数据没有提供统一span-kind保证。
- 别名、排序和预算不依赖gold；traceID、spanID、operationName、标签及路径不进入模型文本。只用完整保留的调用链摘要作为数值输入。
- 205条旧训练/验证指标表示完全一致。含trace的训练/验证20例共保留 {sum(r['trace_retained'] for r in kept)}/{sum(r['trace_total'] for r in kept)} 个服务摘要（{pct(retention)}），输入长度 {min(r['tokens'] for r in kept)}–{max(r['tokens'] for r in kept)}，不超过2048。
- {passed}项测试通过；缓存与完整推理最大logits差 {meta['cache']['validation_full_logit_max_diff']}；CPU重载差 {integration['reload_max_abs_logit_diff']}。
- 真实HTTP检查包含指标输入、调用链输入、拒绝未来trace、标签泄漏、空/单一/超预算候选、缺少时间证据与未知应用。候选倒序、统一改名和去掉traces等实际消融见 results.json。

CPU benchmark本次改为3个含调用链的OB回归输入，各预热3次/测量12次：P50 {bench['end_to_end']['p50_ms']:.1f} ms、P95 {bench['end_to_end']['p95_ms']:.1f} ms、吞吐 {bench['end_to_end']['throughput_per_second']:.3f}/秒，峰值进程工作集 {bench['peak_process_ram_bytes']/1024**3:.2f} GiB。样本与第五轮SS性能测试不同，不作直接速度因果比较。输入选择记录于 outputs/round6/benchmark_config.json。

## 为什么还不算完整成功

当前默认模型的OB RE2根因96%、故障80%、联合76%，未达到此前工作假设90%/90%/85%；该假设并非用户原Prompt中的明确数值。默认模型的自动接受错误率在该历史子集中为14.3%，不能宣称≤5%得到独立验证。

本轮4组续训和8组轻量基线均未提高验证表现。现有RE2-OB训练集每类只有3个案例，限制结论；不能保证继续加轮次会获得收益。公开AIOps2025的400个标签已核查，只有42个能严格映射CPU/内存两类；网络链路与磁盘填满/读写错误不能冒充当前单服务网络故障与磁盘压力。因此本轮未将它们混入或作为完整五分类确认。

下一步需要新增同应用、五类定义一致、按原始运行分组的真实故障数据；提前封存独立确认集和gate确认集。缺少这些证据时，可以说MVP工程流程完成，不能说诊断质量和自动接受已经全面达标。原MVP历史验收与本轮没有fresh holdout的状态在 status.json 分开记录。

## 产物与复现

默认模型：`{active['artifact']}`。实验调用链模型：`{folder}`；checkpoint SHA256 `{meta['binding']['checkpoint_sha256']}`。实验代码、配置、逐案例输出与测试记录已保存，权重和原始数据留在本机并忽略Git。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round6_trace_seed44.json -Mode frozen
./work/.venv/Scripts/python.exe scripts/diagnose_round7.py
```

复现建立新artifact_root并依赖保留的data/round6与第五轮父模型；已打开案例只能作为回归复现。通用复现脚本默认测SS性能；复现本轮含trace的CPU测量，另传 `--config outputs/round6/benchmark_config.json` 执行benchmark。没有运行新的SFT主干微调、KD、外部LLM、量化或ONNX；本轮只训练诊断头，旧SFT实验仍保留。

来源与单位核验：[固定RCAEval数据](https://huggingface.co/datasets/phamquiluan/RCAEval/tree/afeacb11bcc94dadfd1c8f483ee4377b2b8b614e)、[Jaeger字段定义](https://github.com/jaegertracing/jaeger/blob/v1.54.0/model/json/model.go)、[gRPC状态码](https://grpc.io/docs/guides/status-codes/)、[AIOps2025官方说明](https://www.aiops.cn/gitlab/aiops-live-benchmark/agenticopseval/-/blob/main/AIOps2025/README.md)。
"""
Path('docs/round6_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
save('outputs/round7/status.json',{'diagnostic_only':True,'experiments':8,'validation_joint_accuracy':.875,'promoted':False,'reason':'Below incumbent validation; no calibration, gate or regression evaluation','report':'docs/round6_results.md'})
print('REPORT',str(Path('docs/round6_results.md').resolve()),'ACTIVE RETAINED',active['artifact'],flush=True)
for n,c in comparisons.items():print(n,triple(c['current']),c['current']['selective'],flush=True)
print('UPDATES',result['training_updates'],'TRAIN SECONDS',result['summed_training_seconds'],'CPU P95',bench['end_to_end']['p95_ms'],flush=True)
