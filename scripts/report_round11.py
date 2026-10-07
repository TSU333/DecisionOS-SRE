"""Publish actual partial-SFT results and apply the frozen release decision."""
from pathlib import Path
import shutil,xml.etree.ElementTree as ET
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.data import load_split

out=Path('outputs/round11');sel=read(out/'selection.json');protocol=read(out/'protocol.json');guard=read(out/'release_guard.json')
folder=Path(sel['selected']['artifact']);meta=read(folder/'metadata.json');cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json');integrity=read(out/'training_integrity.json')
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
assert integrity['all_training_source_files_match_pretraining_commit']
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert meta['selected_restore_validation_max_abs_difference']<=1e-5
assert cal['binding']==pol['binding']==meta['binding']
for split,name in [('calibration','calibration_logits.json'),('gate_selection','gate_predictions.json')]:
 rows=read(folder/name);ids={e.original_run_id for e in load_split(meta['config']['data_dir'],split)}
 assert len(rows)==len(ids) and {r['run_id'] for r in rows}==ids and all(r['split']==split for r in rows)
for name in ['metadata.json','calibrator.json','policy.json','resolved_config.json','benchmark.json','integration.json','test_freeze.json']:
 if (folder/name).exists():
  dest=out/'selected_artifact'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,dest)
comparisons={};failures={};ablations={}
if guard['regressions_executed']:
 for name in ['regression_ss','reverse_candidates','consistent_service_rename','mask_all_metrics']:
  dest=out/name;dest.mkdir(parents=True,exist_ok=True)
  for f in ['predictions.json','metrics.json','diagnostics.png']:shutil.copyfile(folder/name/f,dest/f)
 for split in ['regression_re1_ob','regression_re2_ob','regression_ss']:
  old=read(Path('outputs/round5')/split/'metrics.json');new=read(out/split/'metrics.json');rows=read(out/split/'predictions.json');oldrows={r['run_id']:r for r in read(Path('outputs/round5')/split/'predictions.json')}
  assert set(oldrows)=={r['run_id'] for r in rows}
  paired={h:{'fixed':sum(not oldrows[r['run_id']][h+'_correct'] and r[h+'_correct'] for r in rows),'regressed':sum(oldrows[r['run_id']][h+'_correct'] and not r[h+'_correct'] for r in rows)} for h in ['root','fault','joint']}
  targets={'root':new['root']['acc_at_1']>=.9,'fault':new['fault']['accuracy']>=.9,'joint':new['joint_accuracy']>=.85}
  comparisons[split]={'previous':old,'current':new,'paired':paired,'working_targets_met':targets,'role':'historical regression; not independent confirmation'}
  failures[split]=[{'run_id':r['run_id'],'cohort':r['cohort'],'root_gold':r['candidate_ids'][r['root_target']],'root_pred':r['candidate_ids'][r['root_pred']],'fault_gold':FAULTS[r['fault_target']],'fault_pred':FAULTS[r['fault_pred']],'routing':r['routing']} for r in rows if not r['joint_correct']]
 ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename','mask_all_metrics','mask_temporal_ob']}
save(out/'failure_cases.json',{'regressions_executed':guard['regressions_executed'],'cases':failures})
suites=ET.parse(out/'test-results.xml').getroot().findall('testsuite');passed=sum(int(s.get('tests','0')) for s in suites);failed=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
assert passed>=84 and failed==0
raw=(out/'dependency-check.log').read_bytes();assert 'No broken requirements found.' in raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8-sig')
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','tests':{'passed':passed,'failed':failed},'active_model_updated':guard['promote'],'validation_improved':sel['promote_by_validation'],'regressions_executed':guard['regressions_executed'],'regression_release_guard':guard['all_non_regressed'],'candidate_all_regression_working_targets_met':all(all(v['working_targets_met'].values()) for v in comparisons.values()) if comparisons else None,'independent_generalization_target_status':'not_confirmed_no_new_independent_cases','automatic_acceptance_validated':False,'production_safety_confirmed':False,'gate_empirical_constraint_met':pol['status']=='selected','new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'historical_mvp_checks':read('outputs/round6/status.json')['historical_mvp_checks'],'current_fresh_holdout':'not_run; no new independent cases','new_partial_backbone_sft_executed':True,'not_run':['full-backbone SFT','fresh independent confirmation','production incidents','KD','external LLM','quantization','ONNX']}
assert Path(read(out/'previous_latest_model.json')['artifact'])==Path(protocol['parent'])
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
if guard['promote']:
 save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'predeclared validation ranking and all-cohort historical regression guard','report':str(Path('docs/round11_results.md').resolve()),'routing':pol['status'],'quality_target_status':'historical_regression_only_no_fresh_confirmation','test_role':'all historical sets remain regression only','previous_pointer':str(out/'previous_latest_model.json')})
else:assert file_hash('outputs/latest_model.json')==file_hash(out/'previous_latest_model.json')
active=read('outputs/latest_model.json')
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'release_guard':guard,'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'resource_probe_micro_steps':3,'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'training_integrity':integrity,'benchmark':bench,'integration':integration,'policy':pol,'calibrator':cal,'ablations':ablations,'status_summary':status,'active_model':active,'limitation':'Repeated validation and historical data; no new independent quality or automatic-acceptance safety confirmation.'}
save(out/'status.json',status);save(out/'results.json',result)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'training_source_hashes_verified':True,'pretraining_commit':meta['code_state']['revision']})
pct=lambda v:'null' if v is None else f'{100*v:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
rows=[]
for c,a in zip(sel['candidates'],integrity['candidates']):
 rows.append(f"| {Path(c['artifact']).parent.name} | {a['partial_backbone_layers']} | {a['trainable_parameters']:,} | {c['steps']} | {c['best_steps']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |")
trial_table='\n'.join(rows)
if comparisons:
 regression_text='每格依次为根因 / 故障 / 联合准确率。\n\n| 历史回归集 | 第五轮现任 | 本轮候选 |\n|---|---|---|\n'+'\n'.join(f"| {labels[n]} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
 selective_text='| 历史回归集 | 接受数 | 错误数 | 接受后错误率 |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {labels[n]} | {c['current']['selective']['accepted_total']} | {c['current']['selective']['errors']} | {pct(c['current']['selective']['selective_risk'])} |" for n,c in comparisons.items())
else:
 regression_text='没有候选在验证集上胜过现任，按训练前协议跳过本轮历史回归；不通过反复查看回归选择模型。现任已知历史联合准确率为RE1-OB 86.7%、RE2-OB 76.0%、Sock Shop 93.3%，这些是第五轮旧结果，本轮没有重新测得新的回归成绩。'
 selective_text='本轮未打开历史回归，因此不报告候选在回归集上的接受后风险。'
decision='通过固定替换条件，已更新默认模型' if guard['promote'] else '未通过全部替换条件，保留第五轮默认模型'
selected_audit=next(a for a in integrity['candidates'] if Path(a['artifact'])==folder)
bench_split='历史Sock Shop回归' if comparisons else 'model_validation'
report=f"""# DecisionOS-SRE 第十一轮：有限共享编码器微调

客户日期：2026-10-08（Australia/Sydney）。execution_scope: mvp。**{decision}。** 完成3组真实GPU训练，共 **{result['training_updates']} 次优化器更新**，记录的训练阶段耗时合计 {result['summed_training_seconds']:.1f} 秒（不含进程启动和模型加载）；另有3个仅用TRAIN的资源探测微步，其权重已丢弃。

## 验证与实际效果

选中 `{folder}`。现任验证联合92.5%、sum NLL {sel['incumbent']['validation']['sum_nll']:.6f}；候选验证联合 {pct(sel['selected']['validation']['joint_accuracy'])}、sum NLL {sel['selected']['validation']['sum_nll']:.6f}。验证晋升条件={sel['promote_by_validation']}，是否执行历史回归={guard['regressions_executed']}，回归保护={guard['all_non_regressed']}，最终替换={guard['promote']}。

| 实验 | 可训练编码器末层数 | 可训练参数 | 实际更新 | 选中checkpoint更新步 | 验证联合 | sum NLL |
|---|---:|---:|---:|---:|---:|---:|
{trial_table}

{regression_text}

即使验证或历史回归有改善，也不等于新独立泛化或生产质量已确认。候选三个历史回归集是否均达到先前工作假设90%根因/90%故障/85%联合：{status['candidate_all_regression_working_targets_met']}；None表示未重测。这些阈值不是原Prompt明确的用户数值。

## 为什么这样训练

第五轮原40条验证的3个错误为网络延迟/丢包、磁盘/CPU、磁盘/内存混淆。此前多轮主要改变分类头，而共享编码器仍来自较早的OB数据微调。本轮检查有限主干更新能否改善两应用表示，控制组只更新文本故障头，另外两组更新最后2或4个编码器块及final_norm。没有改标签、故障定义、输入表示或推理架构。

所有方案从第五轮重新初始化；原数值根因、全局/局部数值故障头及文本根因scorer参数冻结。共享编码器输出改变仍会影响根因logits和预测根因条件权重，因此不声称根因结果固定。每次正式推理仍只编码一次。微调代码真实更新了相应backbone权重，逐张量变化和冻结核对见 training_integrity.json；text_control没有主干更新，不称为主干SFT。

seed48，micro batch1、累积4，最多6轮/246更新，3轮无改善早停；编码器lr=1e-5、文本故障头lr=3e-4、weight_decay0.01、root/fault损失权重均为1。训练BF16 autocast和FP32参数，验证与CPU推理FP32，使用非重入梯度检查点。训练样本有限，选择小学习率和有限层数控制参数变化；不把占满显存作为效果目标。

本机Ryzen 9 7945HX、16GB RAM、RTX4060 Laptop 8GB，启动前可用RAM约5.0GiB。最长有效TRAIN输入1659 tokens的3步资源测试通过，4层方案PyTorch峰值分配约0.90GiB，热身后单微批次约0.10至0.12秒。正式各组峰值显存和实际时间由metadata记录；这些数值不包含桌面程序或驱动的全部显存。

## 数据与固定验收规则

RCAEval固定版本400个原始run，划分仍为165 TRAIN、40验证、40校准、85门控、15 RE1-OB回归、25 RE2-OB回归、30 SS回归。无新增独立样本，无增强视图。无有效观测的TRAIN记录 `21a11a8fa96a215914feab22` 保留在源数据及父模型历史，但不参与新梯度抽样；有效TRAIN164条。每次抽样核查无跨split与被排除记录，manifest/splits/examples哈希未变。

三组完整训练后，只用原40条验证选一个候选：OB联合至少95%、SS至少90%，再比较四cohort联合均值、总体联合、sum NLL，且必须优于现任。验证未胜出则跳过回归；胜出后仅一个封存候选进入校准、门控及历史回归，并要求每组根因/故障/联合均不退步才替换。失败不改选次优，不依据回归错误继续调整本轮方案。

协议和全部实际训练源码在训练前显式提交，Git `{meta['code_state']['revision']}`。全部源码SHA256与该提交及当前文件一致；训练日志等工作区文件不会被误当作新的训练源码。训练完整保存最佳可训练参数并还原后再导出全量checkpoint，没有用最终epoch冒充最佳epoch。

## 按原Definition of Done验收

| 条目 | 本轮结果 / 历史依据 |
|---|---|
| 真实公开adapter、标签审计、manifest | 原400例固定数据；本轮哈希和分组审计通过 |
| Evidence/gold、路径与答案隔离 | 原schema/serializer及本轮测试通过；模型输入未扩展 |
| 同run不跨split、增强继承、预处理分离 | 跨split重叠0；本轮无增强；校准40与门控85单独使用 |
| 动态候选mask/span/ID、无效和截断行为 | 单测及真实HTTP边界通过 |
| 每incident一次共享编码 | 完整训练forward、predict及runtime调用断言通过 |
| 缺标监督、有效loss、无NaN | 单测及三组实际训练通过；梯度有限性检查通过 |
| baseline、frozen、SFT真实执行路径 | baseline和完整SFT历史已运行；本轮文本冻结对照及末层SFT实际运行 |
| 真实训练与holdout评测、范围声明 | 历史首次holdout已完成；本轮训练与验证完成，历史回归执行={guard['regressions_executed']}，无新独立holdout |
| checkpoint保存重载 | 最佳权重还原后GPU差 {meta['selected_restore_validation_max_abs_difference']}；CPU重载差 {integration['reload_max_abs_logit_diff']} |
| 校准数据、argmax保持、绑定与异常 | 原标量温度和版本测试通过；候选校准仅40例 |
| gate_selection及无可行阈值全REVIEW | 原边界测试通过；候选门控状态 `{pol['status']}` |
| 手算指标、空接受/缺标/遗漏候选 | 既有指标测试通过 |
| CPU实测、API、机器结果、复现 | 本轮benchmark、integration、results及下方命令 |
| 区分实现、真实实验、fixture和未运行 | {passed}项测试含小fixture；正式训练及指标使用真实数据 |

{passed}项测试通过，失败0，pip check通过。只有一条Starlette依赖弃用提示。选中模型 {selected_audit['frozen_tensors_verified']} 个冻结张量与父模型完全相等。真实HTTP检查健康、诊断、拒绝标签、缺时间证据、单/空/超预算候选和未知应用；验证服务结束后停止。未知应用仍强制REVIEW，不执行运维动作。

CPU batch1、8线程，使用{bench_split}的固定3个输入各预热3次、测量12次：端到端P50 {bench['end_to_end']['p50_ms']:.1f}ms、P95 {bench['end_to_end']['p95_ms']:.1f}ms、吞吐 {bench['end_to_end']['throughput_per_second']:.3f}/秒；进程生命周期峰值工作集 {bench['peak_process_ram_bytes']/1024**3:.2f}GiB。系统负载与输入可能不同，不据单轮计时宣布速度提升。

## 门控与限制

候选门控在85条既有门控数据上选择，结果 `{pol['selection']}`；目标仍为经验联合风险≤5%、至少接受30个run，找不到可行阈值则全REVIEW。

{selective_text}

门控/验证/历史回归均被多轮复用，不能视为新独立安全证据。当前五类故障及两应用兼容的400例已用完，需要新的运行案例来确认泛化。已知注入起点、60秒观测、受控故障标签仍是适用范围；不是生产事故检测实验。本轮未运行全主干微调、KD、外部LLM、ONNX或量化。

## 产物与复现

默认模型 `{active['artifact']}`。
本轮候选SHA256 `{meta['binding']['checkpoint_sha256']}`。
协议、选择、训练抽样、逐参数审计、发布保护、CPU/API结果、逐案例预测、机器状态、完整产物校验和均保存在 outputs/round11。模型权重、原始数据由Git忽略并保留本机。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round11_{folder.parent.name}.json -Mode sft
```

依赖本地data/round5、固定主干及第五轮父模型，通用脚本使用新输出目录。该单配置复现脚本默认会执行历史回归；完整本轮的验证门槛和跳过规则由 run_round11_training.py / evaluate_round11.py / report_round11.py保存。复现旧案例不增加独立证据，不能借复现改选候选。
"""
Path('docs/round11_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT COMPLETE','updates',result['training_updates'],'PROMOTED',guard['promote'],'REGRESSION',guard['regressions_executed'],flush=True)
