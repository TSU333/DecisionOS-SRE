"""Publish actual grouped-CV results, source readiness and remaining evidence gaps."""
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
from decisionos_sre.common import read,save,file_hash

out=Path('outputs/round13');r=read(out/'cv_results.json');p=read(out/'protocol.json');audit=read(out/'training_integrity.json');data=read(out/'data_audit.json');external=read(out/'external_data_audit.json');errors=read(out/'oof_error_audit.json');source=read(out/'pretraining_source.json')
assert audit['status']=='passed' and len(r['runs'])==18 and r['total_optimizer_updates']==audit['total_updates']
assert all(file_hash(path)==sha for path,sha in source['source_hashes'].items())
assert file_hash('outputs/latest_model.json')==file_hash(out/'previous_latest_model.json')
assert file_hash(out/'data_audit.json')==p['data_audit_sha256'] and file_hash(out/'folds.json')==p['folds_sha256']
suites=ET.parse(out/'test-results.xml').getroot().findall('testsuite');tests=sum(int(s.get('tests','0')) for s in suites);failed=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites);assert tests==97 and failed==0
raw=(out/'dependency-check.log').read_bytes();assert 'No broken requirements found.' in raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8-sig')
manifest={str(path):file_hash(path) for path in Path('artifacts/round13').rglob('*') if path.is_file()}
save(out/'artifact_manifest.json',{'files':manifest,'base_model_manifest_sha256':file_hash('artifacts/backbone/manifest.json'),'training_source_revision':source['revision'],'head_only_artifacts_require_original_pretrained_encoder':True})
status={'execution_scope':'mvp','grouped_cv':'completed','train_error_audit':'completed','external_source_preparation':'completed_with_no_admissible_cases','new_usable_independent_data':'not_obtained','formal_training_runs':18,'formal_optimizer_updates':r['total_optimizer_updates'],'tests_passed':tests,'engineering_checks':'passed','global_holdouts_evaluated':False,'default_model_updated':False,'new_model_strength_confirmed':False,'fresh_independent_evaluation_completed':False,'new_independent_cases':0,'source_events_audited':94,'source_events_admitted':0,'mvp_definition_of_done':'Prior engineering MVP retained; this round verifies grouped CV and data readiness. Prior CPU/API benchmarks are not rerun or relabeled as new measurements.','not_run':['new full-TRAIN deployment candidate','new global-validation or historical-regression model evaluation','new calibration or gate selection','new API latency benchmark','new independent quality confirmation','production fault injection','KD','external LLM','ONNX','quantization']}
save(out/'status.json',status)
result={'status':status,'cv':r,'training_integrity':audit,'data_audit':data,'error_audit':errors,'external_data_audit':external,'active_model':read('outputs/latest_model.json'),'previous_runtime_checks':'outputs/round12/status.json; no production inference code changed this round'}
save(out/'results.json',result)
pct=lambda v:f'{100*v:.1f}%'
rows='\n'.join(f"| {v} | {pct(s['root_mean'])} | {pct(s['fault_mean'])} | {pct(s['joint_mean'])} | {s['joint_sample_sd']*100:.2f}个百分点 |" for v,s in r['summaries'].items())
seed_rows='\n'.join(f"| {seed} | {pct(next(x['summary']['joint_accuracy'] for x in r['summaries']['temporal']['seeds'] if x['seed']==seed))} | {pct(next(x['summary']['joint_accuracy'] for x in r['summaries']['dynamics']['seeds'] if x['seed']==seed))} |" for seed in p['seeds'])
cohort_rows=[]
for cohort in ['RE1-OB','RE2-OB','RE1-SS','RE2-SS']:
 values={v:np.mean([x['summary']['cohorts'][cohort]['joint_accuracy'] for x in s['seeds']]) for v,s in r['summaries'].items()}
 cohort_rows.append(f"| {cohort} | {pct(values['temporal'])} | {pct(values['dynamics'])} |")
cohort_table='\n'.join(cohort_rows)
best_steps=[x['best_steps'] for x in r['runs']];positive=sum(x['joint_delta']>1e-12 for x in r['paired_comparisons']);negative=sum(x['joint_delta']< -1e-12 for x in r['paired_comparisons'])
case_rows='\n'.join(f"| {v} | {x['stable_errors_all_three_seeds']} | {x['any_seed_errors']} |" for v,x in errors['variants'].items())
confusion_rows='\n'.join(f"- {v}: "+'; '.join(f'{k}: {n}' for k,n in list(x['confusions_across_492_repeated_predictions'].items())[:4]) for v,x in errors['variants'].items())
report=f"""# DecisionOS-SRE 第十三轮：分组交叉验证、训练数据审计与新案例准备

execution_scope: mvp。**完成18次真实GPU训练和数据审计；继续保留第五轮默认模型。** 没有因开发CV的均值改善替换默认模型，没有新增可用独立案例。

## 实际执行与结论

固定两种特征 × 三折 × 三个随机种子，实际 **{r['total_optimizer_updates']} 次优化器更新**，训练程序计时 {r['wall_seconds']:.1f}秒（含预计算、训练、验证与导出，不含进程启动）。主干为原始ModernBERT预训练权重、全程冻结；根因和故障分类头每折重新初始化。没有使用已经见过整个TRAIN的第五轮或后续权重作为CV起点。

| 方案 | 根因平均准确率 | 故障平均准确率 | 联合平均准确率 | 联合准确率的seed间样本标准差 |
|---|---:|---:|---:|---:|
{rows}

| seed | 旧时间摘要联合 | 新动态摘要联合 |
|---|---:|---:|
{seed_rows}

新动态特征平均联合提高 {(r['summaries']['dynamics']['joint_mean']-r['summaries']['temporal']['joint_mean'])*100:.2f}个百分点，9个配对折中{positive}个提高、{negative}个下降、其余持平；seed50下下降，seed51/52提高。它有继续研究的信号，但随机种子波动较大，不能宣布稳定收益或生产改进。

**这些数字不是第五轮模型的准确率重测，也不能与此前40例验证的95%直接比较。** 每折实际拟合只用81–82个案例，初始化和评估集合也不同。每个seed有164个折外预测，3个seed重复使用同一164个run，不能记作492个独立案例。标准差描述3个seed的波动，不是95%置信区间。既有特征和架构曾基于这些相关数据开发，所以本轮仍是开发评估，不是新独立确认。

## 分区与训练隔离

原400例和七路分区不变。TRAIN165条中原空观测记录仍保留、从新梯度抽样中排除，使用164条。外层按原run/保守重复组做3折，按application×fault分层；每折另设inner-stop选epoch，outer在最佳inner权重恢复后才评测。各折fit/stop/outer为82/28/54、82/27/55、81/28/55，三个随机种子使用同一封存折划分。

只用inner-fit进行cohort均衡抽样。lr3e-4、batch16、weight_decay.01、故障隐藏层dropout.15、根因与故障loss均1，最多160轮/1000更新，inner-stop连续25轮无改善早停。最佳步数范围{min(best_steps)}–{max(best_steps)}、中位数{float(np.median(best_steps)):.0f}；说明上一轮从已训练权重出发的22步不能直接套用于从头训练。

两方案对应seed/fold的旧分类头初值完全一致，动态输入新增列以0初始化。只对原始公开预训练编码器作一次固定缓存提取，164次共享编码；无跨案例拟合归一化。外层输入被无监督编码不构成梯度或早停使用，所有标签仅在对应合法分区消费。27个实际训练相关文件（25个src与2个脚本）匹配训练前Git `{source['revision']}`；所有原数据与默认指针字节不变。

## 训练数据与反复错误审计

原TRAIN标签与官方注入元数据一致，没有发现转录不一致，候选遗漏为{data['candidate_misses']}。唯一已知无观测案例仍被排除。这里验证的是标签来源一致性，不能证明每次注入都成功或标签语义绝无问题；未根据模型输出修改标签。

| 方案 | 三个seed都联合出错的run | 至少一个seed出错的run |
|---|---:|---:|
{case_rows}

主要故障混淆计数（每方案492个重复预测，非独立案例数）：

{confusion_rows}

| TRAIN来源组 | 旧时间摘要联合均值 | 新动态摘要联合均值 |
|---|---:|---:|
{cohort_table}

逐run、逐seed的根因/故障预测在oof_error_audit.json；反复错误是待复核线索，不是错标结论。下一步可用独立注入记录核查故障是否实际作用于目标、目标是否有足够观测及指标语义是否一致。本轮不根据这些新诊断再调参或重选结果。

## 新案例准备：已下载，但尚未通过准入

重新核查官方RCAEval当前索引：版本仍为 `{external['rcaeval_index_recheck']['revision']}`，735条中同应用/同五故障兼容400条均已使用，新增兼容case为0。索引SHA256与旧版本一致。[官方来源](https://huggingface.co/datasets/phamquiluan/RCAEval)

另固定OpenNetAI Sock-Shop-Dataset提交 `1037771bc3c143f95bc615ec8d5b436b33586704`，核对MIT许可、Git blob和SHA256，下载19个pod遥测CSV，合计{external['download_bytes']/1024**2:.2f}MiB。公开标注有94个容器故障事件：CPU29、内存31、网络延迟34，另有30条节点故障标注未纳入当前服务级任务。[原始来源](https://github.com/OpenNetAI/Sock-Shop-Dataset/tree/1037771bc3c143f95bc615ec8d5b436b33586704)

94个目标都能在遥测文件中找到，但60秒窗口内90例只有1行、2例有2行、2例无观测；原temporal-v1至少需要3个有效点及早晚覆盖，因此没有可用时间摘要，新动态摘要也不满足条件。`_id`列是导出记录元数据，已明确排除，不能把它当作指标或用于模型输入。原始字段保留未改。

还存在原指标名/单位/速率语义未与RCAEval等同、pod与service层级不同、绝对时区未说明、缺少磁盘和丢包类别等问题。故94条只是已审计的外部事件，**准入训练0条、独立评估0条**。不插值伪造高频数据，不把观察窗口改为5分钟来凑数，不在该源上选模型或拟合门控。

全部下载与逐事件原因在data/external/opennet_sockshop，审计与来源清单在outputs/round13。后续需提供或采集≤5秒间隔原始遥测、明确单位/速率和pod到service映射、注入时刻及时区、300秒无故障基线与60秒观测。现有已汇总CSV无法可靠恢复缺失的细粒度观测。准备规范见docs/round13_new_data_requirements.md。

## 验收与适用范围

- {tests}项测试通过（新增6项分组/泄漏/初始化防护），失败0，pip check通过。既有Starlette弃用提示1条。
- 18个模型的原始预训练主干哈希匹配且未更新，分类头确实改变；文件SHA、配对初值、抽样次数和最佳步数逐项核查。
- 梯度只来自各折fit；stop与outer没有进入抽样。每个有效run在每seed恰好有一次outer预测，覆盖完整。
- 每个最佳分类头都已保存、重新加载，并由CPU原始主干重建推理；两条固定outer输入的CPU/GPU最大logit差 {audit['max_cpu_gpu_logit_difference']}，argmax全部一致。训练内完整forward与缓存差均≤1e-5。
- 默认第五轮工件与指针保持，原40验证、40校准、85门控和历史回归未做新模型评测。原MVP的API、校准、门控和性能检查见第十二轮证据；本轮未重测CPU P95，不把旧数值标成新测量。
- 本轮产物是开发CV分类头，不是可直接交给生产API的完整发布模型。未进行新全TRAIN部署候选训练、主干SFT、生产注入、KD、外部LLM、ONNX或量化。

原工程MVP状态保持。此次完成了分组CV、训练错误审计和外部资料准备；“取得可训练的新独立样本”和“证明模型稳定变强”尚未完成，具体原因如上。

## 复现与产物

工件在artifacts/round13（18份heads.pt与抽样/元数据）；OOF预测、分组、协议、校验、错误列表、来源审计和机器结果在outputs/round13。源数据及原始预训练模型保留本机，权重和原数据不入Git。

以下审计命令可以直接重跑，不改变模型：

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe scripts/audit_round13_cv.py
./work/.venv/Scripts/python.exe scripts/audit_round13_external.py
./work/.venv/Scripts/python.exe scripts/report_round13.py
```

完整训练复现入口如下，预检已通过；本轮没有额外重复整套18次训练。脚本在work/cv_reproductions的新子目录还原训练前Git快照，复制约630MB必需数据与原始主干，保持固定分组/配置/哈希，原产物不覆盖：

```powershell
./scripts/reproduce_round13.ps1 -CheckOnly
./scripts/reproduce_round13.ps1 -OutputDir D:/CODEX/DecisionOS-SRE/work/cv_reproductions/recheck-01
```

每次重跑都是复现旧数据，不增加独立案例。
"""
Path('docs/round13_results.md').write_text(report,encoding='utf-8');(out/'execution_report.md').write_text(report,encoding='utf-8')
print('REPORT COMPLETE 18 CV runs; default unchanged; external admitted 0',flush=True)
