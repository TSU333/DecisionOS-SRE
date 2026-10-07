# DecisionOS-SRE 第六轮协议

execution_scope: mvp。原工程 DoD 已完成，本轮继续改善诊断质量，不进入 KD、外部 LLM 或部署优化。

沿用 400 个案例及原始分组，补充其中 75 个 RE2-OB 案例原本已有但尚未使用的 traces。案例数不增加，不把新增模态当成新独立样本；全部历史测试仍是回归。数据固定 RCAEval revision afeacb11bcc94dadfd1c8f483ee4377b2b8b614e，MIT。

新增 service-trace-v1：分别汇总每个服务的全部 span，以及 RPC operation 的目标服务等于该 span 服务的服务端 span。基线要求 start≥onset−300 且 end<onset，观测要求 start≥onset 且 end≤onset+60。startTime、duration 为 Jaeger 微秒，延迟转换成 ms。去除重复 traceID/spanID；状态缺失不算成功。实际采集入库延迟未知，以 span 完成时间近似可见时间，保留这项限制。

每个 scope 12 个通道：速率相对变化、均值延迟相对变化、p95相对变化、观测均值ms、观测p95ms、非零状态比例、deadline比例、unavailable比例、错误比例变化、已知状态比例、观测数量、存在标记。前五项和数量使用固定 signed-log/5，幅度上限1e6，比例保持原值。共24通道，拼接在原120个指标通道之后；全局同样保持原360通道前缀后追加72通道。只输入预算内保留的完整摘要，不输入 traceID/spanID/operationName/路径/标签。

保留全部候选及原指标优先预算，调用链摘要按可观察延迟变化和错误率排序，占用剩余2048 token预算。没有 gold 驱动的保留规则。仅在现有候选包含 frontend 时，把可观察 trace 名 frontendservice 对齐 frontend；其余服务名不推断根因。原指标输入忽略 traces 时，需与第五轮完全一致。

四组实验提前封存：纯指标续训 seed44，增加 traces 的 seed42/43/44；相同学习率0.0003，batch16，最多2500次更新，70个无改进epoch早停。温启动第五轮权重，新增输入列零填充；共享主干冻结，每例一次编码，仅训练诊断头。代码、权重、配置、数据及实际抽样ID均记录。

选择包含第五轮现任模型：OB验证联合≥95%、SS≥90%为保留约束，再比较四cohort联合均值、总体联合、NLL。所有选择只用40条原验证集；完成后冻结再校准、门控、回归。质量工作目标仍为90%根因/90%故障/85%联合，它们是此前工作假设，并非用户给出的明确数值。门控经验风险≤5%、至少接受30条的要求保持不变；不能将复用gate的经验结果宣传成独立安全确认。

公开数据调查另见 outputs/round6/source_investigation.json。官方 AIOps2025 的400条标签中仅42条能直接映射至当前单服务 CPU/内存压力；网络标签是链路，磁盘填满和读写错误不等同磁盘压力。本轮没有将其错误映射，也未拿这两类代替完整五分类确认。

来源：[RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval/tree/afeacb11bcc94dadfd1c8f483ee4377b2b8b614e)、[Jaeger单位定义](https://github.com/jaegertracing/jaeger/blob/v1.54.0/model/json/model.go)、[gRPC状态码](https://grpc.io/docs/guides/status-codes/)、[AIOps2025官方说明](https://www.aiops.cn/gitlab/aiops-live-benchmark/agenticopseval/-/blob/main/AIOps2025/README.md)。
