> 第三轮最新记录见 [round3_protocol.md](round3_protocol.md)；以下保留初始阶段历史记录。

# 冻结评测协议 v1

固定 seed=42。RE1-OB 的 run 在确定性摘要和训练增强之前划分。按 fault 分层，对 exact telemetry、共享基线和近似窗口指纹相同者 union 分组。各组只属于一个 split。只有 train 可以做候选打乱、10% 指标 masking 和 10% 全 evidence dropout。增强元数据记录 parent run 与 seed。

数据为单根因官方注入标签。未知 fault 保留 null；多根因不能取第一个，当前所选官方子集均为单一标量。候选由遥测列构造；root 不在候选中时监督 loss、根因校准与 NLL/ECE/Brier 被 mask，端到端排名与 joint 按失败计入。missing gold 从对应指标分母剔除且记录数量。

每个案例已知注入时间 t0；正常基线 [t0-300,t0)，故障观测 [t0,t0+60]，decision_time=t0+60。不做未来填充；忽略 NaN/Inf 后计算均值和基线总体标准差。z=(observed_mean-baseline_mean)/max(baseline_std,abs(baseline_mean)*0.01,1e-6)，clip 到 [-100,100]。缺失基线或不足两个基线点时 z=null。没有使用整个数据集拟合的标准化参数。

模型在每条 incident 一次共享 backbone forward 中编码全部候选和观测；candidate span 均值与 CLS representation 拼接、逐元素乘积送入共享 scorer。两个任务 CE 分别按有效标签归一化。batch_size、累积、学习率、训练步数、停止条件都在 resolved config。

以 model_validation 的两任务 NLL 之和选 checkpoint。calibration 只拟合该 checkpoint 的两个正标量温度。gate_selection 搜索所有出现的 routing score 阈值，取满足经验 joint error <=5%、至少 30 独立接受组的最大覆盖阈值；默认 bundle=根因与故障都正确。无可行阈值=全部 REVIEW。单一候选、无可用 evidence、校准或 policy 缺失/失配也 REVIEW。阈值不能读取 gold 判断候选是否完备。

测试前保存 test_freeze.json；之后产物变化会拒绝继续把同一 test 当未见测试集。评测主结果和 frozen 配对的全 metrics masking、候选倒序、一致服务匿名化都从保存的逐例预测生成，禁止根据这些结果反向修改模型。消融不要求单例置信度单调。

指标：
- root Acc@1 / MRR / Recall@3 的分母包含已知根因被遗漏的案例；另报候选覆盖。
- fault macro F1 为固定五类平均，逐类 precision/recall/support 与混淆矩阵并列。
- 每 head NLL、10 个等宽 bin ECE、未除类数的 multiclass Brier。root 条件于 gold 在候选中。
- joint 只在两个任务有 gold 时计分，同时保留整体分母。
- 空接受集合 selective risk/accuracy 为 null，不填 0/100%。
- 当前每独立组一例，Wilson 95% 描述区间；若新增多窗口必须先按组聚合或 cluster bootstrap，不能直接使用该独立样本公式。阈值选择集区间为描述性，未校正选择过程；测试区间亦不是部署保证。
- Risk–coverage 曲线按观测 score 的完整并列组计算，不拆分 ties。

CPU benchmark：batch=1、8 线程、3 warmup、每个真实输入 12 次；分别记录预分词 model core 和完整 serializer+tokenization+model+后处理。另有明确标注的 token/candidate 数变化性能探针，不用于质量指标。冷启动加载单列，进程 RAM 是 Windows 进程峰值工作集，模型磁盘体积是完整 backbone+heads checkpoint。没有远程 LLM 对比。
