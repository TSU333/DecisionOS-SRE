"""Publish actual dynamics-feature results and apply the frozen release decision."""
from pathlib import Path
import shutil,xml.etree.ElementTree as ET
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.data import load_split

out=Path('outputs/round12');sel=read(out/'selection.json');protocol=read(out/'protocol.json');guard=read(out/'release_guard.json')
folder=Path(sel['selected']['artifact']);meta=read(folder/'metadata.json');cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json');integrity=read(out/'training_integrity.json')
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
assert integrity['all_training_source_files_match_pretraining_commit']
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert meta['cache']['validation_full_logit_max_diff']<=1e-5
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
 ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename','mask_all_metrics','mask_temporal_ob','mask_dynamics_ob']}
save(out/'failure_cases.json',{'regressions_executed':guard['regressions_executed'],'cases':failures})
suites=ET.parse(out/'test-results.xml').getroot().findall('testsuite');passed=sum(int(s.get('tests','0')) for s in suites);failed=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
assert passed>=91 and failed==0
raw=(out/'dependency-check.log').read_bytes();assert 'No broken requirements found.' in raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8-sig')
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','dynamics_api_checks':'passed','tests':{'passed':passed,'failed':failed},'active_model_updated':guard['promote'],'validation_improved':sel['promote_by_validation'],'regressions_executed':guard['regressions_executed'],'regression_release_guard':guard['all_non_regressed'],'candidate_all_regression_working_targets_met':all(all(v['working_targets_met'].values()) for v in comparisons.values()) if comparisons else None,'independent_generalization_target_status':'not_confirmed_no_new_independent_cases','automatic_acceptance_validated':False,'production_safety_confirmed':False,'gate_empirical_constraint_met':pol['status']=='selected','new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'mvp_definition_of_done':'engineering passed; original heldout evidence documented in earlier rounds; no fresh holdout in round12','historical_mvp_checks':read('outputs/round6/status.json')['historical_mvp_checks'],'current_fresh_holdout':'not_run; no new independent cases','new_partial_backbone_sft_executed':False,'new_frozen_head_training_executed':True,'not_run':['new backbone SFT','fresh independent confirmation','production incidents','KD','external LLM','quantization','ONNX']}
assert Path(read(out/'previous_latest_model.json')['artifact'])==Path(protocol['parent'])
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
if guard['promote']:
 save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'predeclared validation ranking and all-cohort historical regression guard','report':str(Path('docs/round12_results.md').resolve()),'routing':pol['status'],'quality_target_status':'historical_regression_only_no_fresh_confirmation','test_role':'all historical sets remain regression only','previous_pointer':str(out/'previous_latest_model.json')})
else:assert file_hash('outputs/latest_model.json')==file_hash(out/'previous_latest_model.json')
active=read('outputs/latest_model.json')
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'release_guard':guard,'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'training_integrity':integrity,'dynamics_api':read(out/'dynamics_api.json'),'benchmark':bench,'integration':integration,'policy':pol,'calibrator':cal,'ablations':ablations,'status_summary':status,'active_model':active,'limitation':'Repeated validation and historical data; no new independent quality or automatic-acceptance safety confirmation.'}
save(out/'status.json',status);save(out/'results.json',result)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'training_source_hashes_verified':True,'pretraining_commit':meta['code_state']['revision']})

pct=lambda v:'null' if v is None else f'{100*v:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
trial_table='\n'.join(f"| {Path(c['artifact']).parent.name} | {a['trainable_parameters']:,} | {c['steps']} | {c['best_steps']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |" for c,a in zip(sel['candidates'],integrity['candidates']))
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
regression_text='本轮验证未胜出，按协议未打开候选历史回归。'
selective_text='未执行本轮候选历史回归，不报告相应接受风险。'
if comparisons:
 regression_text='每格为根因 / 故障 / 联合准确率。\n\n| 历史回归集 | 第五轮现任 | 本轮候选 |\n|---|---|---|\n'+'\n'.join(f"| {labels[n]} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
 selective_text='| 历史回归集 | 接受数 | 错误数 | 接受后错误率 |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {labels[n]} | {c['current']['selective']['accepted_total']} | {c['current']['selective']['errors']} | {pct(c['current']['selective']['selective_risk'])} |" for n,c in comparisons.items())
decision='通过固定替换条件，已更新默认模型' if guard['promote'] else '未通过全部替换条件，保留第五轮默认模型'
selected_audit=next(a for a in integrity['candidates'] if Path(a['artifact'])==folder)
legacy=next(c for c in sel['candidates'] if Path(c['artifact']).parent.name=='legacy_control')
dynamic_best=min((c for c in sel['candidates'] if Path(c['artifact']).parent.name!='legacy_control'),key=lambda c:c['rank'])
feature_better=dynamic_best['rank']<legacy['rank']
api_probe=read(out/'dynamics_api.json')
assert api_probe['status']=='passed'
report=f"""# DecisionOS-SRE 第十二轮：因果动态特征对照

客户日期：2026-10-08（Australia/Sydney）。execution_scope: mvp。**{decision}。** 本轮完成4组真实GPU冻结编码器分类头训练，共 **{result['training_updates']} 次优化器更新**，训练函数计时合计 {result['summed_training_seconds']:.1f} 秒（包含缓存、优化、验证与导出，不含进程启动与最初模型加载）。全部候选保留，未进行新的主干SFT。

## 实际效果

仅按原40例验证选中 `{folder}`。验证联合准确率由现任 {pct(sel['incumbent']['validation']['joint_accuracy'])} 到候选 {pct(sel['selected']['validation']['joint_accuracy'])}，sum NLL {sel['incumbent']['validation']['sum_nll']:.6f} → {sel['selected']['validation']['sum_nll']:.6f}。验证胜出={sel['promote_by_validation']}，历史回归执行={guard['regressions_executed']}，所有回归项不退步={guard['all_non_regressed']}，默认模型替换={guard['promote']}。

| 实验 | 可训练参数 | 实际更新 | 最佳checkpoint更新步 | 验证联合 | sum NLL |
|---|---:|---:|---:|---:|---:|
{trial_table}

新增特征最佳方案按预定验证排名是否优于匹配旧特征对照：**{feature_better}**。不能把旧特征对照也出现的改善归因于新特征。仅一个seed、40例反复使用的验证集，不构成显著性或泛化证明。

{regression_text}

候选三个历史回归集是否均达到既有90%根因/90%故障/85%联合工作假设：{status['candidate_all_regression_working_targets_met']}（None为未评测）。这些数值不是用户原Prompt明确要求。历史数据多轮重复使用，本轮无新独立质量确认。

预定消融仅对选中的旧特征对照实施：Sock Shop候选反序和服务一致重命名均保持联合90.0%；屏蔽全部指标降至3.3%。RE2-OB移除旧temporal后联合76.0%，移除dynamics仍为72.0%；该候选本来不读取dynamics，所以后者只是兼容性对照，不能评判新增特征贡献。消融不用于重新选择模型。

## 这轮做了什么及原因

先复查现有仓库、固定依赖和资源。沿用Python3.13、PyTorch2.7.1/cu128、transformers4.51.3；Ryzen9 7945HX、16GB RAM、RTX4060 Laptop 8GB。未安装新包。冻结主干允许每组205次预计算（165原TRAIN+40验证），随后只优化小型分类头；每incident完整推理仍只共享编码一次。训练中一次nvidia-smi采样为GPU利用率94%、显存1769/8188MiB，属于瞬时整机采样，非峰值或整轮利用率。

TRAIN审计发现部分旧摘要触及±100截断，例如latency-90的变化z有150/1215项达到截断界限。提出保留幅度和局部波动形态的可证伪假设：新增q10/q90的signed-log幅度、平均相邻变化率、最大跳变、相邻相关和|z|>3比例，另加存在标识。原基线尺度为max(std,abs(mean)*.01,1e-6)，signed-log为sign(v)*log1p(min(abs(v),1e6))/5。

每项都来自本案例相对onset的[-300,0)基线、[0,decision_time-onset]观测，decision不超过onset+60。重复时间取均值；只用间隔≤5秒相邻对，不跨长缺测间隔。至少2个基线点、6个观测时间点、早晚窗口都有观测且至少3对有效相邻采样。缺少动态摘要时使用存在标识，并由新版本运行时强制REVIEW。公式在处理holdout波形前封存，无跨案例拟合统计和gold输入。

新候选数值维度120→190、全局360→570。新增列初始化为0，原文本、候选、span、排序和原数值前缀逐条保持一致。TRAIN6个真实输入的扩维初始完整CPU logits最大差 {integrity['migration_full_cpu_max_abs_difference']}，argmax一致；旧默认checkpoint的CPU重载差 {integrity['legacy_cpu_reload_max_abs_difference']}。

四组共同seed49、batch16、lr1e-4、weight_decay.01、cohort均衡抽样、最多2200更新、80轮无改善早停。从第五轮重新初始化。前三组仅故障头，第四组根因/故障头同时训练；drop15组仅训练时隐藏层dropout=.15。主干张量全部冻结并核查相等，故障头组根因logits变化严格为0。实际最佳权重经还原导出，保存步数见表；不能把执行了更多更新解释为选中模型更强。

## 数据、选择和发布约束

400个固定原run，165 TRAIN、40验证、40校准、85门控、15 RE1-OB回归、25 RE2-OB回归、30 SS回归。原空观测TRAIN `21a11a8fa96a215914feab22` 保留记录，但从新梯度抽样中排除，有效164。400个原始Parquet哈希重验；manifest/splits逐字节保持，examples只增加动态摘要，删除新字段后与原字典相同，400例旧表示及数值前缀一致。无新增独立案例、无增强视图、无跨split抽样。

先完成全部训练，再只用40验证选一个候选：OB联合≥95%、SS≥90%，之后cohort宏观联合、整体联合、sum NLL，且必须胜过现任。验证失败跳过回归；验证胜出仅封存候选进入历史回归，各组根因/故障/联合必须不退步，失败不改选次优、不按本轮回归调参。校准仅用40，门控仅用85；95%验证不代表95%未知案例准确率。

训练前源码提交 `{meta['code_state']['revision']}`，24个源文件哈希与提交及实际文件一致。新特征/训练配置/协议均提前封存；报告与审计脚本在训练期间补齐，不改变训练源码。源码、配置、数据、checkpoint、校准器及policy版本绑定验证通过。

## 原Definition of Done验收

| 条目 | 本轮结果与证据范围 |
|---|---|
| 真实公开adapter、标签审计、manifest | 原RCAEval400例固定版本；新摘要和原文件哈希核查通过 |
| Evidence/gold和路径/答案隔离 | 输入扩展只含遥测数值；原标签和分组不变；序列化隔离测试通过 |
| 同run不跨split、增强继承和预处理隔离 | 分组重叠0；无增强；无跨案例拟合预处理；校准/门控指定分区 |
| 动态候选mask/span/ID/target及截断 | 单测、400例兼容比较、真实HTTP边界通过 |
| 每incident一次共享编码 | 缓存205次与完整forward调用断言、runtime断言通过 |
| 缺标监督与有限loss | 既有边界单测及四组真实训练有限loss通过 |
| baseline、frozen、SFT真实路径 | baseline/完整SFT见历史验收；本轮实际训练冻结头，不冒充主干SFT |
| 真实训练及holdout评测 | 历史首次holdout已完成；本轮真实训练/验证/历史回归执行={guard['regressions_executed']}；无新holdout |
| checkpoint保存和重载 | 选中完整forward/缓存差 {meta['cache']['validation_full_logit_max_diff']}；CPU重载差 {integration['reload_max_abs_logit_diff']} |
| 校准分区、argmax和版本异常 | 标量温度/绑定测试通过；真实拟合40条指定校准例 |
| gate_selection与无阈值零接受 | 85条指定门控例；状态 `{pol['status']}`；无可行阈值回退测试通过 |
| 可手算指标、缺标和候选遗漏 | 既有指标边界测试通过 |
| CPU实测、API、机器结果、复现 | benchmark/integration/results和下方命令齐全 |
| 实现、正式实验、fixture和未运行分开 | {passed}项单测包括fixture；正式训练/评测用真实模型与数据 |

**{passed}项测试通过，失败0，pip check通过。** 唯一提示为既有Starlette弃用警告。候选实际HTTP涵盖正常诊断、拒绝gold、缺时间证据、单/空/超预算候选、未知应用。另对固定dynamics_fault工件使用验证输入运行新特征真实HTTP工程检查：原输入、缺dynamics、缺temporal、未来时间戳、拒绝gold均通过；该检查不拟合门控、不查看历史回归、不选模型，缺校准/策略时强制REVIEW。所有测试服务完成后停止。

CPU batch1、8线程、3条固定输入各预热3次并计时12次：端到端P50 {bench['end_to_end']['p50_ms']:.1f}ms、P95 {bench['end_to_end']['p95_ms']:.1f}ms、吞吐 {bench['end_to_end']['throughput_per_second']:.3f}/秒；进程生命周期峰值工作集 {bench['peak_process_ram_bytes']/1024**3:.2f}GiB。不同系统负载的单次基准不作为速度提升证明。

## 门控与适用范围

候选门控选择结果 `{pol['selection']}`，约束为经验联合风险≤5%、至少30个接受run；不可行时全REVIEW。

{selective_text}

以上是重复使用的历史样本风险，不是自动接受的独立安全保证。本机这两应用/五故障的400例均已纳入既定分区，本轮没有取得新独立run；新摘要没有增加独立信息来源。原工程MVP流程已经完成；未知案例质量、生产安全和自动接受风险仍未确认。需要新增独立运行案例，保持已知注入起点/60秒观测与受控故障范围声明。不执行运维动作。

本轮未执行新主干SFT、KD、外部LLM、ONNX、量化或生产事故实验。当前没有已授权MVP工程步骤因缺算力而未完成；缺的是独立质量证据。

## 产物与复现

默认工件 `{active['artifact']}`。本轮候选 `{folder}`，checkpoint SHA256 `{meta['binding']['checkpoint_sha256']}`。完整结果、训练抽样、冻结参数、发布保护、逐案例预测、CPU/API结果、校验清单在 outputs/round12；新输入在 data/round12，原始数据在 data/round4，权重与数据由Git忽略并留存本机。

实际执行 prepare_round12_data.py → run_round12_training.py → evaluate_round12.py → audit_round12.py → verify_round12_dynamics_api.py → report_round12.py；全部通过后提交最终报告。训练前执行pytest，另执行pip check。执行日志保存在outputs/round12。

单配置复现使用新输出目录：

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round12_{folder.parent.name}.json -Mode frozen
```

依赖本地data/round12、固定主干及第五轮父模型。该通用单配置脚本会执行历史回归；完整本轮四组封存和门槛逻辑保存在run_round12_training.py/evaluate_round12.py。重复训练旧案例只验证可复现性，不构成新的未见数据或重新选择候选的理由。prepare_round12_data.py拒绝覆盖本轮已经封存的数据和特征公式。若本机新摘要缺失，可从原数据按封存公式重建到新目录，逐项核对outputs/round12/data_audit.json中的SHA256，再将复现配置data_dir指向新目录：

```powershell
./work/.venv/Scripts/python.exe scripts/prepare_round12_data.py --data-output data/reproductions/round12 --audit-output outputs/reproductions/round12
```

数据准备脚本在训练完成后补充了可选输出路径和固定父工件参数，特征计算与模型源码未变；训练使用的版本仍由上述训练前Git提交保存。
"""
Path('docs/round12_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT COMPLETE','updates',result['training_updates'],'PROMOTED',guard['promote'],'REGRESSION',guard['regressions_executed'],flush=True)
