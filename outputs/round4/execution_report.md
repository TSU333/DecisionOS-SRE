# DecisionOS-SRE 第四轮继续训练结果

客户日期 2026-10-07；execution_scope: mvp；experiment_completed。

**新 Sock Shop 留出集达到暂定质量指标；原 Online Boutique 仍未完全达到目标，自动接受条件也未达到。** 不把新应用分数当成旧应用新泛化达标的证明。

选中 `artifacts\round4\balanced_seed43\frozen`。SHA256 `0b49ae1d44f7c5b5e9181fb51dadb396774dfeaa350dcfe272d14ed171fca7fa`。旧模型保留，统一入口 `outputs/latest_model.json`。选择在 test 打开前写入 selection.json，只用 validation，不用回归或新 test 选模型。

## 实测结果：分开报告不同数据集

| 数据集 / 角色 | 案例数 | 上轮根因 / 故障 / 联合 | 本轮根因 / 故障 / 联合 |
|---|---:|---|---|
| 新 Sock Shop 留出（已训练该应用） | 30 | 86.7% / 50.0% / 50.0% | **96.7% / 90.0% / 86.7%** |
| Online Boutique RE2 历史回归 | 25 | 88.0% / 72.0% / 64.0% | **96.0% / 80.0% / 76.0%** |
| Online Boutique RE1 历史回归 | 15 | 86.7% / 100.0% / 86.7% | 86.7% / 100.0% / 86.7% |

新 SS 的计数为根因 29/30、故障 27/30、联合 26/30，对照暂定 90% / 90% / 85% 分别至少需 27/30、27/30、26/30。各类 6 条，故障 Macro F1 0.8967。上轮未训练 Sock Shop，表中旧模型分数为直接诊断对照，不代表旧 API 支持该应用。新模型训练包含 SS，不能称为未知应用泛化。

原 OB RE2 回归目前根因 24/25、故障 20/25、联合 19/25；故障分类和联合正确仍低于原工作目标。OB 旧测试已被多轮查看，只能是回归证据；没有新的独立 OB 留出，所以原应用的新泛化目标状态为 not_reconfirmed。配对纠错/退化、Wilson 与 bootstrap 区间见 results.json；小样本指标不构成生产保证。

## 数据审计与工程选择

复核 [TORAI 官方数据包](https://doi.org/10.6084/m9.figshare.31925976.v1)：73 条 OB 案例与现有 RE2 时间、列、数值一致；另 2 条覆盖更多时间，但相交时间的数值及注入时间相同，因此 75 条均不作新增独立数据。只读取包中的 metrics 与注入时间做重叠审计，包内其他模态未用于训练。证据在 torai_overlap_audit.json。

改用固定 [RCAEval 数据](https://huggingface.co/datasets/phamquiluan/RCAEval/tree/afeacb11bcc94dadfd1c8f483ee4377b2b8b614e) 的 200 条 RE1-SS / RE2-SS 五类故障。先分配原始案例，再下载和预处理。总计 400 案例、400 指纹组、候选覆盖率 100%。划分为 train 165 / validation 40 / calibration 40 / gate 85 / OB regression 15+25 / 新 SS test 30。只从 TRAIN 得到词表和梯度；已知注入起点定义基线前 300 秒、观测后 60 秒，metrics-only，不评估故障检测。重复抽样不增加独立案例数。

继续复用已微调共享编码器并训练 335,249 个诊断头参数，仍一次输入一次共享编码。冻结特征预计算每组 205 次（165 train+40 validation），缓存与完整模型 logits 最大差为 0。四来源均衡采样避免占多数的数据压过旧 RE2 的 15 条训练案例；每次 draw 的真实 run ID 保存并审核，无 calibration/gate/test ID。

温启动新增特征词表/顺序校验，拒绝相同维度但语义错位的数值头加载；旧推理产物保持兼容。

## 已执行训练与冻结规则

| 方案 | 实际更新 | 最佳 epoch | 验证根因 | 验证故障 | 验证联合 | 四来源联合均值 |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 2000 | 143 | 100.0% | 87.5% | 87.5% | 85.0% |
| balanced | 1452 | 72 | 100.0% | 92.5% | 92.5% | 90.0% |
| balanced_seed43 | 1364 | 64 | 100.0% | 92.5% | 92.5% | 90.0% |

共 4816 次优化器更新，训练函数耗时合计 163.0 秒（不含全部下载、进程启动和验收时间）。选中第 64 轮、第 704 次更新，实际训练 1364 次后早停。第三方案同时改变 seed 和学习率，不能作为同一配置的随机种子稳定性证据。

验证排序首先要求旧 OB 联合准确率 ≥85%（保留约束，原质量目标没有降低），再最大化 RE1-OB/RE2-OB/RE1-SS/RE2-SS 联合准确率等权平均，平手看总体联合和 NLL。选中模型旧 OB 验证 19/20=95%，SS 验证 18/20=90%。所有预算和规则在训练前记录于 protocol.json。

## 校准、门控和鲁棒性

40 条 calibration 拟合独立温度，85 条 gate 选择阈值。仍无满足联合经验错误率 ≤5%、接受案例数 ≥30 的区间。可选区间中最小实测风险为 2/36=5.6%，高于 5%；该诊断是选阈值后的描述，不能拿来宣称保证。policy=no_feasible_threshold，新测试接受覆盖率 0，selective risk=null。仍全部 REVIEW，不执行任何运维动作。

候选倒序与一致重命名均保持新 SS 的 96.7% / 90% / 86.7%；删除指标后为 23.3% / 20.0% / 3.3%，全部 REVIEW。验证集 metric 摘要完整保留，模型输入没有使用路径、注入标签或来源 cohort。

## CPU 与工程验收

28 项测试通过；真实 HTTP health/ready/decide、拒绝 gold、空候选、单候选、超预算与未知应用检查通过。CPU checkpoint 重载 logits 最大差 0.0。

CPU 8 线程、batch 1；3 条新测试案例各预热 3 次、测量 12 次。端到端 P50 814.9 ms，P95 858.8 ms，吞吐 1.226 请求/秒，峰值工作集 1.83 GiB。本轮样本与上一轮不同，不作速度升降的直接因果比较。MVP 工程验收通过与质量、自动接受目标达标分别记录于 status.json。

## 新 SS 失败案例

| run_id | 根因 gold → prediction | 故障 gold → prediction |
|---|---|---|
| 8903099dad6881f322724c14 | catalogue → catalogue | cpu_stress → disk_io_stress |
| 23f5b4dedda1211f36fac0a3 | catalogue → catalogue | disk_io_stress → cpu_stress |
| 5a229245ed1166f68a3a16b6 | carts → carts | disk_io_stress → memory_stress |
| 509b6b3697a2bfa9aefa62f1 | orders → payment | network_delay → network_delay |

完整逐案例原始/校准概率、标签来源和路由保存于 `outputs/round4/new_test/predictions.json`；原 OB 的错误单列于 failure_cases.json。不能看这些测试错误后继续调参，再称本批数据未见。

## 使用和复现

仓库 `D:/CODEX/DecisionOS-SRE`；数据 `data/round4`；模型 `artifacts\round4\balanced_seed43\frozen`；报告 `outputs/round4/results.json`；协议 `docs/round4_protocol.md`。需保留原初始化 checkpoint、固定数据、tokenizer 和依赖环境。

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
