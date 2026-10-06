# DecisionOS-SRE 第三轮真实训练结果

客户日期 2026-10-07。状态：experiment_completed；execution_scope: mvp。**模型在同样的新案例上明显改善，但暂定质量目标尚未达到，也未满足自动接受条件。**

选中 `artifacts\round3\weighted_0.1\frozen`，checkpoint SHA256 `c1588a4dcb764fd3db3504b53c654f5ff515e498a4290dd52d19cc92f1159e4e`。原权重和历史结果保留。选择在新测试开启前完成；不是按测试分数挑出的模型。最新入口 `outputs/latest_model.json`。

## 新留出集与目标（25 条 RE2-OB 案例）

| 指标 | 上轮模型在相同新案例上 | 本轮模型 | 暂定目标 | 状态 |
|---|---:|---:|---:|---|
| 根因 Acc@1 | 72.0%（18/25） | 88.0%（22/25） | ≥90%（至少 23/25） | 未达标 |
| 故障分类 | 20.0%（5/25） | 72.0%（18/25） | ≥90%（至少 23/25） | 未达标 |
| 两任务同时正确 | 16.0%（4/25） | 64.0%（16/25） | ≥85%（至少 22/25） | 未达标 |

目标是未收到数值澄清后声明的探索性工作假设，不是用户提供的保证。联合准确率在同一批案例上提升 48 个百分点；Wilson 与配对 bootstrap 区间保存于 results.json，仅描述此小样本的不确定性。未见测试已打开，禁止据此继续调参再把同一测试称为未见数据。

故障 Macro F1 0.7090。五类各 5 条，召回率：CPU 80%、内存 80%、磁盘 60%、延迟 100%、丢包 40%。失败分析见下表；这不足以断言某类在生产环境的普遍表现。

旧 15 条 regression：根因 86.7%、故障 100.0%、联合 86.7%；上轮分别 86.7%、80%、66.7%。它已被历史开发查看，不算新确认性证据。

## 数据、训练和选择

真实案例从 125 扩至 200，加入相同应用 RE2-OB 的 75 条五类故障。train 65 / model_validation 20 / calibration 20 / gate_selection 55 / regression 15 / test 25。全部 200 个指纹组不同、候选覆盖率 100%；指纹不能证明采集谱系统计独立。metrics-only，已知注入起点，非真实生产事故。

共完成 7 组神经训练，累计 3580 次更新；训练函数记录的耗时合计 385.4 秒（不含所有启动、下载和验收时间）。全模型微调 255 次后早停；单纯加大训练未改善验证表现。最终冻结编码器方案实际 1060 次更新，选中第 152 轮、第 760 次更新的 checkpoint。继承上轮 SFT 的 backbone 并训练 335,249 个头部参数，总参数 149,349,521。

| 方案 | 实际更新 | 最优 epoch | 验证根因 | 验证故障 | 验证联合 | 新 RE2 验证联合（n=5） |
|---|---:|---:|---:|---:|---:|---:|
| cached_fast | 310 | 32 | 85.0% | 80.0% | 75.0% | 40.0% |
| cached_slow | 460 | 62 | 90.0% | 75.0% | 75.0% | 40.0% |
| finetuned | 255 | 5 | 95.0% | 70.0% | 70.0% | 40.0% |
| conditioned_fast | 235 | 17 | 85.0% | 80.0% | 75.0% | 40.0% |
| conditioned_slow | 250 | 20 | 90.0% | 70.0% | 70.0% | 40.0% |
| weighted_0.1 | 1060 | 152 | 100.0% | 90.0% | 90.0% | 60.0% |
| weighted_0.3 | 1010 | 142 | 95.0% | 90.0% | 90.0% | 60.0% |

预先选择标准为 RE1/RE2 验证联合准确率等权均值，平手依次比较总体联合准确率和 NLL。初始方案、两次验证驱动扩展以及预算在 protocol.json 记录，selection.json 冻结于最终测试前。验证集仅 20 条且反复用于开发，不能把其 90% 联合准确率当成最终达标证据。

有效工程改动：精确缓存冻结 encoder 的 85 条 train+validation 表示，每轮复用；完整模型与缓存 logits 最大差为 0。保留单次共享编码，加入预测根因服务的数值故障残差，降低继承文本 logits 权重到 0.1。各方案还存在初始化/预算差异，不把收益归因于单一消融。

训练集拟合的 ExtraTrees 数值对照在 validation 达到根因 100%、故障 95%、联合 95%；但相同新测试仅根因 88.0%、故障 72.0%、联合 64.0%。这说明验证小样本有明显局限；该对照没有替代共享编码器部署模型，也没有用于拟合神经网络的标签。

## 校准、路由与鲁棒性

20 条 calibration 拟合 root 温度 3.5218、fault 温度 1.5852；仅改变概率，不改变 argmax。55 条 gate_selection 未找到满足经验联合错误率 ≤5% 且至少接受 30 条的门限，policy 为 no_feasible_threshold。接受覆盖率 0%、selective risk=null；不能把空接受集合称为零错误验证。门控集联合正确率 RE1 为 93.3%、RE2 为 56%，表明新遥测分布仍有短板。

候选倒序、一致服务重命名均保持 88% / 72% / 64% 的根因/故障/联合指标。删除全部指标后为 0% / 20% / 0%，全部 REVIEW。新测试 25 条均完整保留 metric 摘要，输入 1434–1533 tokens；当前失败不是这批输入的 token 截断造成。

## CPU 与真实工程验收

CPU 8 线程、batch 1、3 条案例各预热 3 次并测量 12 次。端到端 P50 891.0 ms，P95 945.0 ms，吞吐 1.117 次/秒；峰值工作集 1.82 GiB。与上轮性能样本不同，不能直接据此宣称速度提升或下降。

24 项自动测试通过；真实 HTTP health/ready/decide、拒绝 gold、空候选、单候选、超预算候选、未验证应用等检查通过；CPU checkpoint 重载 logits 最大差 0.0。一次输入只调用一次共享 backbone 的断言随真实预测执行。全程没有执行运维动作。MVP 原文 14 项 DoD 按此小规模协议通过，质量目标未通过，二者严格区分，详见 status.json。

## 失败案例（9 条联合错误）

| run_id | 根因 gold → prediction | 故障 gold → prediction |
|---|---|---|
| 5edcafb57f99d79d2ee7d28d | checkoutservice → paymentservice | network_delay → network_delay |
| 5e32212427a2c11858169a1d | checkoutservice → checkoutservice | network_packet_loss → network_delay |
| 36c5f09960143aaf4d2ea4d7 | currencyservice → currencyservice | disk_io_stress → memory_stress |
| 9d82c8b8d583c89e35470808 | currencyservice → recommendationservice | network_packet_loss → network_packet_loss |
| 4a02e503ec61ac851da05f55 | currencyservice → currencyservice | memory_stress → cpu_stress |
| c89e1cd0cb846af202640a40 | emailservice → emailservice | network_packet_loss → memory_stress |
| 119ae31af074a86d82098bed | productcatalogservice → recommendationservice | disk_io_stress → cpu_stress |
| 600fe120a642ee3205886d12 | recommendationservice → recommendationservice | cpu_stress → network_delay |
| a83caa602745ace294bda03d | recommendationservice → recommendationservice | network_packet_loss → network_delay |

原始/校准概率、置信度、标签来源、routing、serialization 与逐案例结果在 `outputs/round3/new_test/predictions.json`，由这些预测生成 metrics.json 与图表。

## 复现与文件

工作仓库 `D:/CODEX/DecisionOS-SRE`。数据 `data/round3`，完整模型 `artifacts\round3\weighted_0.1\frozen`，机器报告 `outputs/round3/results.json`，验收 `outputs/round3/status.json`，数据审计 `docs/round3_data_audit.md`，协议 `docs/round3_protocol.md`。

在现有环境和固定数据/上轮 canonical 权重仍保留时，复现选中方案（一条命令，自动创建新 artifact_root，不覆盖旧模型）：

```powershell
./scripts/reproduce_retraining.ps1 -Config configs/round3_weighted_0.1.json -Mode frozen
```

该命令会训练、校准、门控、复现已打开测试、CPU benchmark 与真实 HTTP 验收。重新运行所得测试只能称为复现，不是新的确认集。数据准备脚本为 `scripts/audit_round3.py`，新数据准备依赖原 `data/rcaeval` 与已固定官方索引。完整方案选择用 `scripts/freeze_round3.py`，拒绝覆盖现有 selection。

启动选中模型：

```powershell
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe -m decisionos_sre serve --artifact artifacts/round3/weighted_0.1/frozen --port 8000
```

## 未完成与下一步

尚未达到 90% / 90% / 85% 的工作目标，尚无可行自动接受门槛。下一轮需要更多同应用、不同采集批次的独立故障案例，尤其磁盘与丢包；重新封存确认集并扩大 calibration/gate，不能复制窗口冒充新样本。当前所选官方 RE2-OB 五类故障已全部分配，不能从现有测试或 gate 挪数据训练来制造达标。应先补充可核验来源、注入/观测起点与采集谱系，再开始新的训练协议。

未运行：生产事故、未知应用泛化、所选配置的独立重复训练稳定性、KD、外部 LLM、ONNX、INT8。以上不属于当前已完成的 MVP 训练证据。
