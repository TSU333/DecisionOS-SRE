"""Publish a fully checked result and apply the predeclared release decision."""
from pathlib import Path
import json,math,shutil,xml.etree.ElementTree as ET
import numpy as np
import torch
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

out=Path('outputs/round9');sel=read(out/'selection.json');protocol=read(out/'protocol.json');guard=read(out/'release_guard.json')
folder=Path(sel['selected']['artifact']);meta=read(folder/'metadata.json')
cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
assert meta['cache']['validation_full_logit_max_diff']<=1e-5
for name in ['metadata.json','calibrator.json','policy.json','resolved_config.json','benchmark.json','integration.json','test_freeze.json']:
    target=out/'selected_artifact'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,target)
for name in ['regression_ss','reverse_candidates','consistent_service_rename','mask_all_metrics']:
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    for f in ['predictions.json','metrics.json','diagnostics.png']:shutil.copyfile(folder/name/f,dest/f)
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
parent_state=torch.load(Path(protocol['parent'])/'checkpoint.pt',map_location='cpu',weights_only=True)
state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
fault_only=meta['config']['head_training_policy']=='fault_heads'
frozen=[n for n in state if n.startswith(('backbone.','scorer.','numeric_root.') if fault_only else ('backbone.',))]
assert set(state)==set(parent_state) and all(torch.equal(state[n],parent_state[n]) for n in frozen)
changed=[n for n in state if not torch.equal(state[n],parent_state[n])]
assert changed and all(n.startswith(('fault_head.','numeric_fault.','numeric_local_fault.') if fault_only else ('scorer.','numeric_root.','fault_head.','numeric_fault.','numeric_local_fault.')) for n in changed)
weight_audit={'frozen_tensors_verified':len(frozen),'frozen_weights_exactly_equal':True,'changed_head_tensors':changed,'root_branch_frozen':fault_only,'root_validation_max_abs_logit_change':meta['root_validation_max_abs_logit_change'],'no_architecture_change':True}
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
assert passed==58 and failed==0
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','tests':{'passed':passed,'failed':failed},'active_model_updated':guard['promote'],'validation_improved':sel['promote_by_validation'],'regression_release_guard':guard['all_non_regressed'],'all_regression_working_targets_met':all(c['all_working_targets_met'] for c in comparisons.values()),'independent_generalization_target_status':'not_reconfirmed_no_new_independent_cases','automatic_acceptance_validated':False,'production_safety_confirmed':False,'gate_empirical_constraint_met':pol['status']=='selected','new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'historical_mvp_checks':read('outputs/round6/status.json')['historical_mvp_checks'],'current_fresh_holdout':'not_run; only historical regression','not_run':['new backbone SFT','fresh independent confirmation','production incidents','KD','external LLM','quantization','ONNX']}
ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename','mask_all_metrics','mask_temporal_ob']}
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'release_guard':guard,'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'benchmark':bench,'integration':integration,'policy':pol,'calibrator':cal,'ablations':ablations,'weight_audit':weight_audit,'training_integrity':read(out/'training_integrity.json'),'source_index_reaudit':read(out/'source_index_reaudit.json'),'status_summary':status,'limitation':'Repeated validation and historical regression; no new independent quality or automatic-acceptance safety confirmation.'}
save(out/'status.json',status)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'training_source_hashes_verified':True})
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
assert Path(read(previous)['artifact'])==Path(protocol['parent'])
if guard['promote']:
    save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'predeclared validation retention/ranking followed by historical regression release guard; no runner-up search','report':str(Path('docs/round9_results.md').resolve()),'routing':pol['status'],'quality_target_status':'historical_regression_only_no_fresh_confirmation','test_role':'all historical test sets are regression only','previous_pointer':str(previous)})
else:assert file_hash('outputs/latest_model.json')==file_hash(previous)
active=read('outputs/latest_model.json');result['active_model']=active;save(out/'results.json',result)
pct=lambda v:'null' if v is None else f'{100*v:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
table='\n'.join(f"| {labels[n]} | {c['current']['runs']} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
trials=[]
for c in sel['candidates']:
    cfg=read(Path(c['artifact'])/'resolved_config.json')
    trials.append(f"| {Path(c['artifact']).parent.name} | {cfg['training_view_probability']} | {cfg['head_training_policy']} | {c['steps']} | {c['best_epoch']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |")
trialtable='\n'.join(trials)
selective_table='\n'.join(f"| {labels[n]} | {c['current']['selective']['accepted_total']} | {c['current']['selective']['errors']} | {pct(c['current']['selective']['coverage'])} | {pct(c['current']['selective']['selective_risk'])} |" for n,c in comparisons.items())
decision='通过预设验证选择与历史回归保护，已更新本地默认模型' if guard['promote'] else '未通过全部晋升条件，保留第五轮默认模型'
report=f"""# DecisionOS-SRE 第九轮训练数据增强结果

execution_scope: mvp。{decision}。4组真实GPU训练共 **{result['training_updates']} 次更新**，训练函数计时合计 {result['summed_training_seconds']:.1f} 秒。原数据仍为400个案例，没有新增独立测试。验证NLL改善若未通过历史回归保护，不计作模型增强成功。

## 实际效果

| 历史回归集 | 案例数 | 第五轮根因 / 故障 / 联合 | 本轮候选根因 / 故障 / 联合 |
|---|---:|---|---|
{table}

本轮回归重新打开前已冻结模型选择。发布保护要求每个cohort的三个准确率均不下降；失败则保留现任，不试次优模型、不再调参。逐案例概率、配对修复/退化计数及描述性区间见 results.json 与各回归目录。全部是此前已打开的历史案例，不能当作未见泛化确认。

## 数据审计与增强

165个原TRAIN案例中，`21a11a8fa96a215914feab22`（RE1-OB）没有故障发生后的有效观测。本轮保留其原始记录、标签、分组和划分，只不再抽入训练。父模型历史仍包含该例，不声称消除了过去学习影响。

其余164个有效案例生成328个训练视图；仍只有164个原案例组，不是新增328个独立案例。lag15遮蔽(45,60]秒，模拟末尾观测延迟；gap15_30遮蔽[15,30)秒，模拟中间缺测。决策时刻为onset+60，baseline为[-300,0)，不读未来数据。缺失比例以原60秒窗口完整采样数为分母，均值、z值、观测数、截止时间和temporal-v1均由剩余观测重算。标签、候选、应用和原run/group不变，视图元数据不进入模型输入。

先按cohort平衡抽取原案例，再选择其原输入或视图；一次抽样只提供一个表示，不因视图数量改变案例权重。验证、校准、门控和历史回归不生成视图，所有原文件哈希不变。仍使用2048 token预算，一次正式推理只调用一次共享ModernBERT。

## 四组固定实验

| 方案 | 视图概率 | 训练头范围 | 更新数 | 最佳epoch | 验证联合 | sum NLL |
|---|---:|---|---:|---:|---:|---:|
{trialtable}

全部从第五轮初始化，seed46、lr0.0001、batch16、weight_decay0.01，最多2200更新、80个无改进epoch早停，没有Dropout或标签平滑。四组均过滤无观测案例，包含不做增强的对照，不能将多个改动混为单一因素。共享主干冻结；all_heads更新根因与故障头，fault_heads只更新故障头。

现任验证联合92.5%、sum NLL={sel['incumbent']['validation']['sum_nll']:.6f}。选中 `{folder}`，验证联合 {pct(sel['selected']['validation']['joint_accuracy'])}、sum NLL={sel['selected']['validation']['sum_nll']:.6f}。优先保留OB验证联合≥95%、SS≥90%，再比较四cohort联合均值、总体联合、NLL。验证晋升条件={sel['promote_by_validation']}，历史回归保护={guard['all_non_regressed']}，最终晋升={guard['promote']}。重复使用40条验证集可能过拟合。

## 工程验收与性能

{passed}项测试通过，覆盖时间边界、缺失比例、TRAIN限定、标签/分组不变及空证据过滤。实际draws逐条核对run_id与view_id；排除案例抽样为0，详情见 training_draw_audit.json。全部候选的真实更新、权重与源码哈希见 training_integrity.json。

选中模型 {weight_audit['frozen_tensors_verified']} 个冻结张量与父模型精确相同；根因分支是否冻结={weight_audit['root_branch_frozen']}。缓存/完整推理最大logits差 {meta['cache']['validation_full_logit_max_diff']}，CPU重载差 {integration['reload_max_abs_logit_diff']}。真实HTTP健康、诊断、拒绝标签、缺少时间证据、单/空/超预算候选及未知应用检查通过。导出架构与API不变，旧checkpoint兼容。

CPU batch1、8线程，3个与第五轮相同的SS输入各预热3次/测量12次：P50 {bench['end_to_end']['p50_ms']:.1f}ms、P95 {bench['end_to_end']['p95_ms']:.1f}ms，吞吐 {bench['end_to_end']['throughput_per_second']:.3f}/秒，峰值进程工作集 {bench['peak_process_ram_bytes']/1024**3:.2f}GiB。不据单次计时宣称速度提升。

## 门控与限制

候选门控状态 `{pol['status']}`；85条门控集经验结果 `{pol['selection']}`，条件仍为联合错误率≤5%、至少30条。门控集被反复使用，独立自动接受风险未确认。以下为本轮候选结果；未晋升时默认模型策略保持原样。

| 历史回归集 | 接受数 | 接受后错误数 | 覆盖率 | 接受后错误率 |
|---|---:|---:|---:|---:|
{selective_table}

系统不执行运维动作。90%根因/90%故障/85%联合是此前工作假设，非用户原Prompt明确数值；三组历史回归是否全部达到：{status['all_regression_working_targets_met']}。原工程MVP的历史DoD与本轮验收均已记录，未运行新的主干SFT、KD、外部LLM、量化或ONNX。

固定源索引735条记录已复审：当前应用和五类故障兼容的400例均已使用；剩余同应用标签为socket exhaustion或代码级F1–F5，不能强行映射。Train Ticket属于其他应用，本轮未使用，也不算新的同应用确认。详见source_index_reaudit.json。

增强视图不能替代新独立事故。仍需新采集、故障定义一致的运行案例，并提前封存质量与门控确认集。

## 产物与复现

默认模型：`{active['artifact']}`。本轮候选checkpoint SHA256：`{meta['binding']['checkpoint_sha256']}`。原始数据、视图和权重保留本机并由Git忽略；源码、配置与逐案例证据已保存。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round9_{folder.parent.name}.json -Mode frozen
```

复现依赖data/round5、data/round9/training_views.json与第五轮父模型，通用脚本创建新目录。prepare_round9.py保留视图准备过程，已封存数据不覆盖；完整四组比较与发布保护见run_round9_training.py、evaluate_round9.py、protocol.json。
"""
Path('docs/round9_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT',str(Path('docs/round9_results.md').resolve()),'PROMOTED',guard['promote'],'UPDATES',result['training_updates'],flush=True)
for n,c in comparisons.items():print(n,triple(c['current']),c['current']['selective'],flush=True)

