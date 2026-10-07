"""Publish a fully checked result and apply the predeclared release decision."""
from pathlib import Path
import json,math,shutil,xml.etree.ElementTree as ET
import numpy as np
import torch
from decisionos_sre.common import read,save,file_hash,FAULTS
from decisionos_sre.policy import wilson

out=Path('outputs/round8');sel=read(out/'selection.json');protocol=read(out/'protocol.json');guard=read(out/'release_guard.json')
folder=Path(sel['selected']['artifact']);meta=read(folder/'metadata.json')
cal=read(folder/'calibrator.json');pol=read(folder/'policy.json');bench=read(folder/'benchmark.json');integration=read(folder/'integration.json')
assert integration['status']=='experiment_completed' and integration['reload_max_abs_logit_diff']<=1e-6
assert file_hash(out/'protocol.json')==sel['protocol_sha256']
assert file_hash(folder/'checkpoint.pt')==sel['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
assert meta['root_validation_max_abs_logit_change']==0 and meta['cache']['validation_full_logit_max_diff']<=1e-5
for name in ['metadata.json','calibrator.json','policy.json','resolved_config.json','benchmark.json','integration.json','test_freeze.json']:
    target=out/'selected_artifact'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,target)
for name in ['regression_ss','reverse_candidates','consistent_service_rename','mask_all_metrics']:
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    for f in ['predictions.json','metrics.json','diagnostics.png']:shutil.copyfile(folder/name/f,dest/f)
assert file_hash(Path(protocol['parent'])/'checkpoint.pt')==protocol['parent_checkpoint_sha256']
parent_state=torch.load(Path(protocol['parent'])/'checkpoint.pt',map_location='cpu',weights_only=True)
state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
frozen=[n for n in state if n.startswith(('backbone.','scorer.','numeric_root.'))]
assert set(state)==set(parent_state) and all(torch.equal(state[n],parent_state[n]) for n in frozen)
changed=[n for n in state if not torch.equal(state[n],parent_state[n])]
assert changed and all(n.startswith(('fault_head.','numeric_fault.','numeric_local_fault.')) for n in changed)
weight_audit={'frozen_tensors_verified':len(frozen),'frozen_weights_exactly_equal':True,'changed_fault_tensors':changed,'root_validation_max_abs_logit_change':meta['root_validation_max_abs_logit_change'],'no_architecture_change':True}
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
assert passed==50 and failed==0
status={'execution_scope':'mvp','experiment':'completed','engineering_checks':'passed','tests':{'passed':passed,'failed':failed},'active_model_updated':guard['promote'],'validation_improved':sel['promote_by_validation'],'regression_release_guard':guard['all_non_regressed'],'all_regression_working_targets_met':all(c['all_working_targets_met'] for c in comparisons.values()),'independent_generalization_target_status':'not_reconfirmed_no_new_independent_cases','automatic_acceptance_validated':False,'production_safety_confirmed':False,'gate_empirical_constraint_met':pol['status']=='selected','new_cases':0,'shared_backbone_calls_per_incident':1,'execute_remediation':False,'historical_mvp_checks':read('outputs/round6/status.json')['historical_mvp_checks'],'current_fresh_holdout':'not_run; only historical regression','not_run':['new backbone SFT','fresh independent confirmation','production incidents','KD','external LLM','quantization','ONNX']}
ablations={n:read(out/n/'metrics.json') for n in ['reverse_candidates','consistent_service_rename','mask_all_metrics','mask_temporal_ob']}
result={'status':'experiment_completed','execution_scope':'mvp','artifact':str(folder.resolve()),'selection':sel,'release_guard':guard,'regression_comparisons':comparisons,'training_updates':sum(c['steps'] for c in sel['candidates']),'summed_training_seconds':sum(c['seconds'] for c in sel['candidates']),'data_audit':read(out/'data_audit.json'),'training_draw_audit':read(out/'training_draw_audit.json'),'benchmark':bench,'integration':integration,'policy':pol,'calibrator':cal,'ablations':ablations,'weight_audit':weight_audit,'training_integrity':read(out/'training_integrity.json'),'status_summary':status,'limitation':'Repeated validation and historical regression; no new independent quality or automatic-acceptance safety confirmation.'}
save(out/'status.json',status)
save(out/'artifact_manifest.json',{'artifact':str(folder.resolve()),'binding':meta['binding'],'files':{str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()},'training_source_hashes_verified':True})
previous=out/'previous_latest_model.json'
if not previous.exists():shutil.copyfile('outputs/latest_model.json',previous)
assert Path(read(previous)['artifact'])==Path(protocol['parent'])
if guard['promote']:
    save('outputs/latest_model.json',{'artifact':str(folder),'absolute_artifact':str(folder.resolve()),'binding':meta['binding'],'checkpoint_sha256':meta['binding']['checkpoint_sha256'],'config':meta['config'],'selected_by':'predeclared validation retention/ranking followed by historical regression release guard; no runner-up search','report':str(Path('docs/round8_results.md').resolve()),'routing':pol['status'],'quality_target_status':'historical_regression_only_no_fresh_confirmation','test_role':'all historical test sets are regression only','previous_pointer':str(previous)})
else:assert file_hash('outputs/latest_model.json')==file_hash(previous)
active=read('outputs/latest_model.json');result['active_model']=active;save(out/'results.json',result)
pct=lambda v:'null' if v is None else f'{100*v:.1f}%'
triple=lambda m:' / '.join([pct(m['root']['acc_at_1']),pct(m['fault']['accuracy']),pct(m['joint_accuracy'])])
labels={'regression_re1_ob':'Online Boutique RE1','regression_re2_ob':'Online Boutique RE2','regression_ss':'Sock Shop'}
table='\n'.join(f"| {labels[n]} | {c['current']['runs']} | {triple(c['previous'])} | {triple(c['current'])} |" for n,c in comparisons.items())
trials=[]
for c in sel['candidates']:
    cfg=read(Path(c['artifact'])/'resolved_config.json')
    trials.append(f"| {Path(c['artifact']).parent.name} | {cfg['fault_hidden_dropout']} | {cfg['fault_label_smoothing']} | {c['steps']} | {c['best_epoch']} | {pct(c['validation']['joint_accuracy'])} | {c['validation']['sum_nll']:.6f} |")
trialtable='\n'.join(trials)
selective_table='\n'.join(f"| {labels[n]} | {c['current']['selective']['accepted_total']} | {c['current']['selective']['errors']} | {pct(c['current']['selective']['coverage'])} | {pct(c['current']['selective']['selective_risk'])} |" for n,c in comparisons.items())
decision='已通过预先规定的验证选择与历史回归保护，更新本地默认模型' if guard['promote'] else '未通过全部晋升条件，保留第五轮默认模型'
quality_interpretation='验证和旧案例回归均通过本轮发布要求，但仍缺新独立确认。' if guard['promote'] else ('验证集提高没有转化为稳定的旧案例表现，触发回归保护；这是未成功晋升的实验，不能称为模型已经增强。' if sel['promote_by_validation'] else '验证表现未超过现任，不能称为模型已经增强。')
report=f"""# DecisionOS-SRE 第八轮专项强化结果

execution_scope: mvp。{decision}。本轮完成六组真实GPU训练，共 **{result['training_updates']} 次更新**，训练函数计时合计 {result['summed_training_seconds']:.1f} 秒。{quality_interpretation} 模型质量仍缺新的独立确认，不能宣称生产自动接受安全。

## 实际效果

| 历史回归集 | 案例数 | 第五轮根因 / 故障 / 联合 | 本轮候选根因 / 故障 / 联合 |
|---|---:|---|---|
{table}

这些都是此前已打开的案例。选择在本次回归打开前冻结；发布保护要求每个cohort三个准确率均不下降，未通过则不尝试次优模型。配对修复/退化计数、描述性区间及全部概率在 outputs/round8/results.json 和各回归目录；不能把这些旧案例分数当成未见泛化。

## 为什么这样强化

父模型验证集根因40/40正确，故障分类37/40；其训练损失低于验证损失。固定共享主干和根因分支，只训练故障文本头、全局数值故障头和按预测根因加权的局部故障头，尝试减轻小样本过拟合。仍不使用正确根因作为输入。每次推理一次共享编码，temporal-v1指标与2048 token预算不变，本轮未使用traces。

Dropout只在数值故障隐藏层训练时开启，推理关闭并移除训练hook；标签平滑只作用五类故障，不平滑带padding的根因候选。所有方案从第五轮同一权重初始化，seed45，lr0.0001，batch16，weight_decay0.01，最多2200更新、60个无改进epoch早停；配置/预算在训练前封存。

| 方案 | Dropout | 标签平滑 | 更新数 | 最佳epoch | 验证联合 | sum NLL |
|---|---:|---:|---:|---:|---:|---:|
{trialtable}

现任第五轮验证联合92.5%、sum NLL={sel['incumbent']['validation']['sum_nll']:.6f}；本轮选中 `{folder}`，验证联合 {pct(sel['selected']['validation']['joint_accuracy'])}，sum NLL={sel['selected']['validation']['sum_nll']:.6f}。优先保留OB≥95%、SS≥90%，再比较四cohort联合均值、总体联合、NLL。验证晋升条件={sel['promote_by_validation']}，历史回归保护={guard['all_non_regressed']}，最终晋升={guard['promote']}。重复使用40条验证集仍可能过拟合，训练次数不等同于独立证据。全部候选的训练完整性检查在 training_integrity.json，曲线在 outputs/round8/training_curves.png。

## 数据、验收与性能

仍是原400个案例：165训练、40验证、40校准、85门控及15+25+30历史回归。案例数、分组和prepared inputs哈希不变；本轮没有新增独立数据。实际抽样ID均属于TRAIN，根因分支本轮没有更新。

{passed}项测试通过。选中模型 {weight_audit['frozen_tensors_verified']} 个冻结状态张量与父模型逐一精确相同；验证根因logits最大变化 {meta['root_validation_max_abs_logit_change']}，缓存/完整推理最大差 {meta['cache']['validation_full_logit_max_diff']}，CPU重载最大差 {integration['reload_max_abs_logit_diff']}。真实HTTP检查覆盖health/ready/decide、拒绝标签、缺少时间证据、单/空/超预算候选及未知应用。权重、tokenizer、split、配置、校准和策略绑定保留；旧checkpoint兼容。

CPU batch1、8线程，3个与第五轮相同的SS输入各预热3次、测量12次：端到端P50 {bench['end_to_end']['p50_ms']:.1f}ms、P95 {bench['end_to_end']['p95_ms']:.1f}ms，吞吐 {bench['end_to_end']['throughput_per_second']:.3f}/秒，峰值进程工作集 {bench['peak_process_ram_bytes']/1024**3:.2f}GiB。不据一次计时宣称速度提升。

## 校准、门控与限制

候选门控状态 `{pol['status']}`；85条门控集经验结果 `{pol['selection']}`。条件仍为经验联合错误率≤5%、至少30条。这些门控案例被多次复用，独立自动接受风险未验证；每个历史cohort的区间保留在机器报告中。系统不执行任何运维动作。

| 历史回归集 | 接受数 | 接受后错误数 | 覆盖率 | 接受后错误率 |
|---|---:|---:|---:|---:|
{selective_table}

90%根因/90%故障/85%联合仍是此前工作假设，并非原Prompt的明确数值。三组历史回归是否全部达到该假设：{status['all_regression_working_targets_met']}。原工程MVP历史DoD与本轮新增工程检查均有记录，未运行新的主干SFT、KD、外部LLM、量化或ONNX。

后续需要同应用、故障定义一致且按原始运行分组的新案例，提前封存独立质量与门控确认集；本次没有用旧回归标签继续拟合。

## 产物与复现

当前默认：`{active['artifact']}`。本轮候选checkpoint SHA256：`{meta['binding']['checkpoint_sha256']}`。权重与原始数据保留本机、不进入Git；报告、配置和逐案例结果进入本地仓库。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round8_{folder.parent.name}.json -Mode frozen
```

复现脚本新建输出目录，依赖已保留的data/round5与第五轮父模型；重复已打开案例只是回归复现。完整六组选择和发布保护见 scripts/run_round8_training.py、scripts/evaluate_round8.py 与冻结protocol。
"""
Path('docs/round8_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT',str(Path('docs/round8_results.md').resolve()),'PROMOTED',guard['promote'],'UPDATES',result['training_updates'],flush=True)
for n,c in comparisons.items():print(n,triple(c['current']),c['current']['selective'],flush=True)