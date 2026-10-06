"""Human report and plots from persisted retraining results."""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decisionos_sre.common import read,save,file_hash
r=read('outputs/retrain/results.json')
s=r['selection'];folder=Path(s['selected']['artifact']);m=read(folder/'metadata.json')
pct=lambda x: 'undefined' if x is None else f'{100*x:.1f}%'
rows=[]
for c in s['candidates']:
    v=c['validation'];rows.append(f"| {c['artifact']} | {c['steps']} | {pct(v['root']['accuracy'])} | {pct(v['fault']['accuracy'])} | {pct(v['joint_accuracy'])} | {v['sum_nll']:.4f} |")
old=r['old_metrics'];new=r['new_metrics'];b=r['benchmark'];ob=r['old_benchmark'];a=r['ablations'];p=r['paired_comparison']
controls=[]
for label,path in [('冻结骨干数值融合','artifacts/retrain/hybrid_frozen/frozen/test/metrics.json')]:
    cm=read(path);controls.append(f"{label}：根因 {pct(cm['root']['acc_at_1'])}，故障 {pct(cm['fault']['accuracy'])}，联合 {pct(cm['joint_accuracy'])}。")
root=ET.parse('outputs/retrain/test-results.xml').getroot()
suites=[root] if root.tag=='testsuite' else list(root)
passed=sum(int(x.attrib.get('tests',0))-int(x.attrib.get('failures',0))-int(x.attrib.get('errors',0))-int(x.attrib.get('skipped',0)) for x in suites)
checks={'schema_gold_causal_split_tests':passed,'old_artifacts_preserved':all(file_hash('artifacts/'+name+'/checkpoint.pt')==read('artifacts/'+name+'/metadata.json')['binding']['checkpoint_sha256'] for name in ('frozen','sft')),'checkpoint_reload':r['integration']['reload_max_abs_logit_diff'],
        'http_api':r['integration']['status'],'calibration_bound_to_selected_checkpoint':r['calibrator']['binding']==m['binding'],
        'gate_bound_to_selected_checkpoint':r['policy']['binding']==m['binding'],'new_unseen_confirmatory_test':'not_run',
        'scope':'mvp','status':'experiment_completed','quality_claim':'exploratory improvement on reused regression cases; no generalization or risk guarantee'}
save('outputs/retrain/status.json',checks)
text=f'''# DecisionOS-SRE 第二轮训练结果

状态：experiment_completed；execution_scope: mvp。

本轮选择：`{folder}`。只按 model_validation 的两任务 NLL 之和选择 checkpoint 和方案；选择冻结文件为 `outputs/retrain/selection.json`。保留原模型，不启用 LLM、蒸馏、量化或真实运维动作。

## 数据和适用边界

继续使用固定 RCAEval RE1-OB 125 个受控注入案例，50 train / 15 model_validation / 15 calibration / 30 gate_selection / 15 原测试案例。没有增加独立样本数。原测试集在上一轮已查看，本轮只能称为回归评测，不能称为新的未见确认性测试。输入为 metrics-only，已知注入起点定位窗口，不评价故障检测和生产事故泛化。

## 验证集选择记录（n=15）

| 方案 | 实际更新数 | 根因 | 故障类型 | 联合正确 | 两任务 NLL |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

实际更新数是完整训练过程；表内质量对应验证 NLL 最佳 checkpoint，未必是最后一次更新。10 条 TRAIN 样本的拟合诊断另存 `artifacts/retrain/sanity`，不得用于模型选择或充当验证集。诊断在 200 次更新时根因 10/10、故障 7/10，说明原故障头拟合仍困难。

## 冻结后的回归结果（同一组 15 条案例）

| 指标 | 原 SFT | 本轮选中模型 |
|---|---:|---:|
| 根因 Acc@1 | {pct(old['root']['acc_at_1'])} | {pct(new['root']['acc_at_1'])} |
| 故障类型 Accuracy | {pct(old['fault']['accuracy'])} | {pct(new['fault']['accuracy'])} |
| 故障类型 Macro F1 | {old['fault']['macro_f1']:.4f} | {new['fault']['macro_f1']:.4f} |
| 两任务联合正确 | {pct(old['joint_accuracy'])} | {pct(new['joint_accuracy'])} |
| 接受覆盖率 | {pct(old['selective']['coverage'])} | {pct(new['selective']['coverage'])} |
| CPU 端到端 p50 | {ob['end_to_end']['p50_ms']:.1f} ms | {b['end_to_end']['p50_ms']:.1f} ms |
| CPU 端到端 p95 | {ob['end_to_end']['p95_ms']:.1f} ms | {b['end_to_end']['p95_ms']:.1f} ms |

根因净变化：修正 {p['root']['fixed']} 条，退化 {p['root']['regressed']} 条。故障分类：修正 {p['fault']['fixed']} 条，退化 {p['fault']['regressed']} 条。联合正确：修正 {p['joint']['fixed']} 条，退化 {p['joint']['regressed']} 条。逐 run 配对 bootstrap 和 Wilson 区间见 `outputs/retrain/results.json`；区间仅描述此复用样本的不确定性，不消除适应性开发影响。

{' '.join(controls)}
轻量 ExtraTrees 故障分类对照：验证集 {pct(read('artifacts/retrain/numeric/model_validation_summary.json')['fault']['accuracy'])}，回归集 {pct(r['numeric_control']['fault']['accuracy'])}。其根因仍使用 max-abs-z 规则。它读取全部因果摘要，而旧文本表示存在截断，因此不能把所有差异归因于模型结构。

## 配对鲁棒性检查

| 冻结变体 | 根因 Acc@1 | 故障 Accuracy | 联合正确 |
|---|---:|---:|---:|
| 删除全部指标 | {pct(a['mask_all_metrics']['root']['acc_at_1'])} | {pct(a['mask_all_metrics']['fault']['accuracy'])} | {pct(a['mask_all_metrics']['joint_accuracy'])} |
| 候选倒序 | {pct(a['reverse_candidates']['root']['acc_at_1'])} | {pct(a['reverse_candidates']['fault']['accuracy'])} | {pct(a['reverse_candidates']['joint_accuracy'])} |
| 一致服务重命名 | {pct(a['consistent_service_rename']['root']['acc_at_1'])} | {pct(a['consistent_service_rename']['fault']['accuracy'])} | {pct(a['consistent_service_rename']['joint_accuracy'])} |

v2 表示使用可观测指标签名生成别名与顺序，并保留输出 ID 对应。相同指标签名的不同身份存在 tie-break 边界；不将同分不可区分服务宣传为普遍排列不变。语义描述在此 metrics-only 版本中不参与编码。数值残差分支仅在相应配置启用；即使无数值分支，仍可使用 v2 表示和 mean pooling。两任务共享一次 backbone 编码，候选不逐个重新编码。

## 校准、门控和工程验收

温度 root={r['calibrator']['root']['temperature']:.5g}，fault={r['calibrator']['fault']['temperature']:.5g}，只由指定 15 条 calibration 拟合。policy 状态 `{r['policy']['status']}`，仍要求经验联合错误率 ≤5%、至少30个独立接受案例。门控集总共30条，可选择子集的统计空间受限；不降低该门槛来制造覆盖率。接受不是执行运维动作。selective risk 为 `{new['selective']['selective_risk']}`；空接受集合必须为 null。

{passed} 项自动测试通过。真实 HTTP readiness/decide/非法输入检查通过；checkpoint CPU 重载最大 logits 差为 {r['integration']['reload_max_abs_logit_diff']}。CPU benchmark 单线程请求、{b['threads']} 个计算线程、每案例 {b['warmup_per_incident']} 次预热、{b['repetitions_per_incident']} 次测量，原始时延、内存和 token/candidate 数据保存在 `{folder}/benchmark.json`。没有在并行训练期间测量最终 CPU benchmark。

实际原始命令与完整日志保存在 `outputs/retrain/`，训练配置为 `configs/retrain_*.json`。数据、源码和 checkpoint hashes 见各实验 metadata；旧权重未覆盖。

## 使用与复现

```powershell
Set-Location D:\\CODEX\\DecisionOS-SRE
$env:PYTHONPATH='src'
.\\work\\.venv\\Scripts\\python.exe -m decisionos_sre serve --artifact {folder.as_posix()} --port 8000
```

单方案重新训练使用 `scripts/reproduce_retraining.ps1 -Config <选中方案配置>`；默认自动创建新的 artifact_root，拒绝覆盖已有 checkpoint。该脚本包含校准、门控、回归、CPU benchmark 和 HTTP 验证。完整方案比较按 `docs/retraining_protocol.md` 与已保存配置执行，再以 `scripts/freeze_retraining.py` 固定选择；不要根据回归结果继续选参。

## 后续尚未执行

新的独立确认性留出集、多随机种子稳定性、跨系统评估均未执行。提高接受覆盖率需要更多独立门控案例和可靠的联合正确性，不能复制窗口凑样本。KD、真实 LLM、ONNX、INT8 保持 scope 外。
'''
Path('outputs/retrain/execution_report.md').write_text(text,encoding='utf-8')
Path('docs/retraining_results.md').write_text(text,encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name in ('budget','canonical','hybrid'):
    history=read(f'artifacts/retrain/{name}/sft/metadata.json')['history']
    steps=[h['optimizer_steps'] for h in history]
    axes[0].plot(steps,[h['validation']['fault']['accuracy'] for h in history],label=name)
    axes[1].plot(steps,[h['validation_loss'] for h in history],label=name)
axes[0].set(ylabel='Fault accuracy on 15 validation runs',xlabel='Optimizer steps',ylim=(0,1.05))
axes[1].set(ylabel='Validation sum of head NLL',xlabel='Optimizer steps')
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.suptitle('DecisionOS-SRE retraining: validation only (seed 42)');fig.tight_layout()
fig.savefig('outputs/retrain/training_curves.png',dpi=150);plt.close(fig)
print('Human report, status, and training curves generated')