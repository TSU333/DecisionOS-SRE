> 第三轮最新记录见 [round3_results.md](round3_results.md)；以下保留初始阶段历史记录。

> 本文件保留第一轮历史结果。第二轮真实重训与回归对比见 `retraining_results.md`。

# DecisionOS-SRE MVP 实际执行报告

已完成文档中的 MVP 工程验收和一轮小规模真实实验。研究结论为负：当前训练模型未超过简单根因异常排序基线，且未找到符合预设风险条件的自动接受阈值。所有诊断转 REVIEW。

本地新仓库：D:\CODEX\DecisionOS-SRE。原 E 盘写入失败后，经目录级授权迁移。没有远程发布、外部付费调用或运维变更。

真实数据：RCAEval RE1-OB，125 个故障注入案例，按现有去重规则 125 组；metrics-only、oracle_onset=true。split={'train': 50, 'model_validation': 15, 'calibration': 15, 'gate_selection': 30, 'test': 15}。

| 方法 | 根因 Acc@1 | MRR | 故障 Accuracy | Macro F1 | 联合正确率 | 接受覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 66.7% | 0.8133 | 20.0% | 0.0667 | 20.0% | 0.0% |
| frozen | 20.0% | 0.4367 | 20.0% | 0.0667 | 0.0% | 0.0% |
| sft | 66.7% | 0.7778 | 20.0% | 0.1000 | 20.0% | 0.0% |

测试集只有 15 例，每例约占 6.7 个百分点；不能据此声称广泛泛化、模型优越性或生产风险保证。空接受集合的 selective accuracy/risk 均为 null。

## 训练、校准和实测

两个模型均为 149,313,286 参数，固定 seed=42、两轮、50 个训练案例、26 个 optimizer steps；frozen 训练 299,014 个 head 参数。checkpoint 只按 15 例 model_validation NLL 选择，绝不按 test 选择。每个训练过程记录 100 次 train presentation 的增强元数据，这些不是新的独立事故。

- frozen：训练 11.55 秒；root/fault 温度={'root': 0.6666494353966735, 'fault': 2.166254900256277}；CPU 端到端 p50/p95=1249.12/1412.32 ms；吞吐=0.781 请求/秒；峰值进程工作集=1.82 GiB；checkpoint=569.64 MiB。
- sft：训练 27.43 秒；root/fault 温度={'root': 0.3566517028197195, 'fault': 20.08553528937195}；CPU 端到端 p50/p95=1194.75/1389.14 ms；吞吐=0.808 请求/秒；峰值进程工作集=1.82 GiB；checkpoint=569.64 MiB。

CPU：Ryzen 9 7945HX，8 线程、batch=1；3 个实际输入各 3 次 warmup、12 次计时。core 与端到端为分别计时的调用，抖动可能导致端到端中位数略小于 core；不能将两者相减推算固定开销。原始计时和独立输入缩放探针保存于 artifact/benchmark.json。

校准各使用 15 例，只优化 calibration NLL。SFT fault 温度接近 logT 上界，说明本轮故障分类概率缺乏有用区分力；不称作可靠校准。gate_selection 为 30 例，最低接受 30 例且经验 joint error<=5%；两模型均无可行阈值，保留原门槛。

## 校准前后测试概率质量

| 模型/任务 | raw NLL | cal NLL | raw ECE | cal ECE | raw Brier | cal Brier |
|---|---:|---:|---:|---:|---:|---:|
| frozen/root | 1.7225 | 1.8436 | 0.1949 | 0.3479 | 0.8234 | 0.8921 |
| frozen/fault | 1.6257 | 1.5880 | 0.1481 | 0.0658 | 0.8141 | 0.7932 |
| sft/root | 1.4068 | 1.1124 | 0.3728 | 0.2332 | 0.6756 | 0.5357 |
| sft/fault | 1.6058 | 1.6085 | 0.0510 | 0.0024 | 0.8001 | 0.7996 |

## 冻结配对消融

| 模型 / 变体 | 根因 Acc@1 | 联合正确率 |
|---|---:|---:|
| frozen/mask_all_metrics | 26.7% | 0.0% |
| frozen/reverse_candidates | 13.3% | 0.0% |
| frozen/consistent_service_rename | 0.0% | 0.0% |
| sft/mask_all_metrics | 26.7% | 13.3% |
| sft/reverse_candidates | 46.7% | 13.3% |
| sft/consistent_service_rename | 0.0% | 0.0% |

删除全部证据、倒序候选和一致服务匿名化均未用于再训练或阈值修改。候选倒序有差异是普通 Transformer 的顺序敏感性；不宣称架构排列不变。原始/校准概率和 routing score 可从逐例文件比较。

## 失败案例（固定测试集前 3 个联合错误案例）

- frozen/21fe50b25236688ea0b6594e：root 预测 currencyservice，gold adservice；fault 预测索引 2，gold cpu_stress；['NO_FEASIBLE_THRESHOLD']。
- frozen/4a40a5b9c9edacbf102d3de3：root 预测 productcatalogservice，gold adservice；fault 预测索引 2，gold disk_io_stress；['NO_FEASIBLE_THRESHOLD']。
- frozen/c4dfa4744b8839cd3ce67a49：root 预测 productcatalogservice，gold adservice；fault 预测索引 2，gold memory_stress；['NO_FEASIBLE_THRESHOLD']。
- sft/21fe50b25236688ea0b6594e：root 预测 cartservice，gold adservice；fault 预测索引 4，gold cpu_stress；['NO_FEASIBLE_THRESHOLD']。
- sft/c4dfa4744b8839cd3ce67a49：root 预测 currencyservice，gold adservice；fault 预测索引 4，gold memory_stress；['NO_FEASIBLE_THRESHOLD']。
- sft/4b6cd49950d26020a44cbfba：root 预测 cartservice，gold cartservice；fault 预测索引 4，gold cpu_stress；['NO_FEASIBLE_THRESHOLD']。

## 验证：11 项测试全部通过

真实 HTTP 验证包括 /health、/ready、正常诊断、拒绝 gold 字段和空候选、单一候选 REVIEW、候选超预算 REVIEW、未知 application REVIEW。缺 checkpoint 返回 503 的分支由本地测试覆盖。checkpoint 重载 CPU logits 的最大差异见 integration.json，均不超过 1e-6。

完整 Definition of Done：
- [x] 真实公开子集 adapter / 标签审计 / 可复现 manifest
- [x] Schema 和 serializer 隔离 gold；泄漏检查
- [x] 同 run 不跨 split；训练增强继承；拟合 holdout 隔离
- [x] 动态候选 mask / span / ID / target 对齐和截断
- [x] 一次共享 encoder forward 调用检查
- [x] 缺失标签 loss mask 有效、无 NaN
- [x] 简单 baseline / frozen heads / SFT 真实数据路径
- [x] 真实训练和独立 holdout 评测
- [x] Checkpoint 重载复现预测
- [x] 独立温度校准、argmax 与失配处理
- [x] 独立阈值选择；不可行时零接受
- [x] 可手算指标；空接受、缺标签和候选遗漏
- [x] CPU 实测、本地 HTTP API、机器可读结果、复现命令
- [x] 区分实现、fixture/smoke、真实实验与未运行项

## 复现与产物

运行 scripts/setup.ps1 后运行 scripts/reproduce.ps1。已有 checkpoint 时 train 会明确拒绝覆盖；新实验复制 configs/mvp.json、设置新的 artifact_root，再用 scripts/reproduce.ps1 -Config 新配置路径。脚本依次下载/审计、划分、准备、测试、baseline、frozen/SFT、校准、gate、评测、CPU benchmark 和 HTTP 验收。

data/rcaeval 包含原始文件、哈希、split 与样本；artifacts/backbone 固定原始模型；artifacts/frozen 与 artifacts/sft 含完整 checkpoint、tokenizer、config、metadata、calibrator、policy、预测、指标、图表和 timing。requirements-lock.txt 保存实际依赖，训练 metadata 保存代码 revision、dirty 状态与源码 SHA-256。

KD、跨系统、AnoMod、LLM cascade、ONNX/INT8、large 和展示均为 not_run，属于后续范围。当前实现能证明完整实验链路可运行；本轮结果不能支持自动接受诊断的部署结论。

## Paired routing-score changes (from saved predictions)

- frozen/mask_all_metrics: {'mean_routing_score_delta': -0.015681354967869574, 'confidence_increased_n': 0, 'n': 15}
- frozen/reverse_candidates: {'mean_routing_score_delta': 0.0012055332109794333, 'confidence_increased_n': 8, 'n': 15}
- frozen/consistent_service_rename: {'mean_routing_score_delta': 0.0002290615838163775, 'confidence_increased_n': 10, 'n': 15}
- sft/mask_all_metrics: {'mean_routing_score_delta': -0.0013521579084811886, 'confidence_increased_n': 0, 'n': 15}
- sft/reverse_candidates: {'mean_routing_score_delta': -0.0003922427946406348, 'confidence_increased_n': 9, 'n': 15}
- sft/consistent_service_rename: {'mean_routing_score_delta': 0.0007838843349061819, 'confidence_increased_n': 14, 'n': 15}

SFT timing note: the initial measurement could overlap the start of checkpoint reload verification. It is retained as benchmark-initial-possible-overlap.json. benchmark.json and this report use the subsequent isolated rerun.
