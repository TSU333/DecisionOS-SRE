# DecisionOS-SRE 第二轮训练结果

状态：experiment_completed；execution_scope: mvp。

本轮选择：`artifacts\retrain\canonical\sft`。只按 model_validation 的两任务 NLL 之和选择 checkpoint 和方案；选择冻结文件为 `outputs/retrain/selection.json`。保留原模型，不启用 LLM、蒸馏、量化或真实运维动作。

## 数据和适用边界

继续使用固定 RCAEval RE1-OB 125 个受控注入案例，50 train / 15 model_validation / 15 calibration / 30 gate_selection / 15 原测试案例。没有增加独立样本数。原测试集在上一轮已查看，本轮只能称为回归评测，不能称为新的未见确认性测试。输入为 metrics-only，已知注入起点定位窗口，不评价故障检测和生产事故泛化。

## 验证集选择记录（n=15）

| 方案 | 实际更新数 | 根因 | 故障类型 | 联合正确 | 两任务 NLL |
|---|---:|---:|---:|---:|---:|
| artifacts/sft | 26 | 66.7% | 20.0% | 20.0% | 2.9760 |
| artifacts/retrain/hybrid_frozen/frozen | 312 | 93.3% | 93.3% | 93.3% | 0.6978 |
| artifacts/retrain/budget/sft | 247 | 100.0% | 40.0% | 40.0% | 0.9537 |
| artifacts/retrain/canonical/sft | 312 | 93.3% | 80.0% | 80.0% | 0.6224 |
| artifacts/retrain/hybrid/sft | 312 | 86.7% | 86.7% | 80.0% | 0.7249 |

实际更新数是完整训练过程；表内质量对应验证 NLL 最佳 checkpoint，未必是最后一次更新。10 条 TRAIN 样本的拟合诊断另存 `artifacts/retrain/sanity`，不得用于模型选择或充当验证集。诊断在 200 次更新时根因 10/10、故障 7/10，说明原故障头拟合仍困难。

## 冻结后的回归结果（同一组 15 条案例）

| 指标 | 原 SFT | 本轮选中模型 |
|---|---:|---:|
| 根因 Acc@1 | 66.7% | 86.7% |
| 故障类型 Accuracy | 20.0% | 80.0% |
| 故障类型 Macro F1 | 0.1000 | 0.7943 |
| 两任务联合正确 | 20.0% | 66.7% |
| 接受覆盖率 | 0.0% | 0.0% |
| CPU 端到端 p50 | 1194.8 ms | 683.5 ms |
| CPU 端到端 p95 | 1389.1 ms | 825.9 ms |

根因净变化：修正 4 条，退化 1 条。故障分类：修正 9 条，退化 0 条。联合正确：修正 7 条，退化 0 条。逐 run 配对 bootstrap 和 Wilson 区间见 `outputs/retrain/results.json`；区间仅描述此复用样本的不确定性，不消除适应性开发影响。

冻结骨干数值融合：根因 86.7%，故障 80.0%，联合 80.0%。
轻量 ExtraTrees 故障分类对照：验证集 100.0%，回归集 100.0%。其根因仍使用 max-abs-z 规则。它读取全部因果摘要，而旧文本表示存在截断，因此不能把所有差异归因于模型结构。

## 配对鲁棒性检查

| 冻结变体 | 根因 Acc@1 | 故障 Accuracy | 联合正确 |
|---|---:|---:|---:|
| 删除全部指标 | 6.7% | 20.0% | 0.0% |
| 候选倒序 | 86.7% | 80.0% | 66.7% |
| 一致服务重命名 | 86.7% | 80.0% | 66.7% |

v2 表示使用可观测指标签名生成别名与顺序，并保留输出 ID 对应。相同指标签名的不同身份存在 tie-break 边界；不将同分不可区分服务宣传为普遍排列不变。语义描述在此 metrics-only 版本中不参与编码。数值残差分支仅在相应配置启用；即使无数值分支，仍可使用 v2 表示和 mean pooling。两任务共享一次 backbone 编码，候选不逐个重新编码。

## 校准、门控和工程验收

温度 root=1.5633，fault=1.5141，只由指定 15 条 calibration 拟合。policy 状态 `no_feasible_threshold`，仍要求经验联合错误率 ≤5%、至少30个独立接受案例。门控集总共30条，可选择子集的统计空间受限；不降低该门槛来制造覆盖率。接受不是执行运维动作。selective risk 为 `None`；空接受集合必须为 null。

16 项自动测试通过。真实 HTTP readiness/decide/非法输入检查通过；checkpoint CPU 重载最大 logits 差为 0.0。CPU benchmark 单线程请求、8 个计算线程、每案例 3 次预热、12 次测量，原始时延、内存和 token/candidate 数据保存在 `artifacts\retrain\canonical\sft/benchmark.json`。没有在并行训练期间测量最终 CPU benchmark。

实际原始命令与完整日志保存在 `outputs/retrain/`，训练配置为 `configs/retrain_*.json`。数据、源码和 checkpoint hashes 见各实验 metadata；旧权重未覆盖。

## 使用与复现

```powershell
Set-Location D:\CODEX\DecisionOS-SRE
$env:PYTHONPATH='src'
.\work\.venv\Scripts\python.exe -m decisionos_sre serve --artifact artifacts/retrain/canonical/sft --port 8000
```

单方案重新训练使用 `scripts/reproduce_retraining.ps1 -Config <选中方案配置>`；默认自动创建新的 artifact_root，拒绝覆盖已有 checkpoint。该脚本包含校准、门控、回归、CPU benchmark 和 HTTP 验证。完整方案比较按 `docs/retraining_protocol.md` 与已保存配置执行，再以 `scripts/freeze_retraining.py` 固定选择；不要根据回归结果继续选参。

## 后续尚未执行

新的独立确认性留出集、多随机种子稳定性、跨系统评估均未执行。提高接受覆盖率需要更多独立门控案例和可靠的联合正确性，不能复制窗口凑样本。KD、真实 LLM、ONNX、INT8 保持 scope 外。
