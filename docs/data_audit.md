# 数据审计：RCAEval RE1-OB

状态：数据审计已完成，独立于模型训练状态。

官方来源：https://huggingface.co/datasets/phamquiluan/RCAEval
固定 revision：afeacb11bcc94dadfd1c8f483ee4377b2b8b614e。
许可证：MIT（官方数据卡）。索引 SHA-256：c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb。
每例原始 metrics 和 onset 文件的 SHA-256、原始 case、服务、故障、重复编号均保存于 data/rcaeval/manifest.json。

实际检查了 125 个原始故障注入案例，125 个去重分组。每例准备一个固定决策窗口，共 125 个派生窗口；审计时训练增强为 0，训练时增强数量另在对应 artifact/augmentations.json 中记录，不能将增强当独立样本。

根因来自 cases.parquet.root_cause_service，fault 来自 cases.parquet.fault。标签来源为官方受控注入元数据，不是独立人工确认的生产事故标签。两个任务标签覆盖率均为 100%；本子集多根因、未知故障、缺失标签和候选遗漏均为 0。每 target 保存 raw_value、value、provenance.kind=gold、source_ref、mapping_version 与 collection_method。

映射：cpu→cpu_stress、mem→memory_stress、disk→disk_io_stress、delay→network_delay、loss→network_packet_loss；不将 memory stress 称作 memory leak。候选取自遥测列服务前缀，候选数为 13、14 或 19，candidate coverage=100%。不按 gold 增加、移除或过滤候选。

真实模态：只有 metrics。logs/traces 不存在；topology/deployment_context 未提供并保持 null。每文件行数、非空行数、时间范围和缺失比例在 manifest.entries.metrics 中。time 为 Unix 秒；指标单位无 schema 级声明，保持 source_unit_unspecified。原始缺失比例范围 0–0.0227544545；不做未来回填或猜测数据。

重复分组检查包括完整数值矩阵、300 秒基线及按列尺度归一/四舍五入的基线和观测窗口指纹，当前未发现重复。公开资料未提供完整采集谱系；不能由这些指纹推断所有 case 已被证明相互独立，高度相关但非副本的重复实验仍可能存在。

冻结分配：50 train、15 model_validation、15 calibration、30 gate_selection、15 test。split manifest 在增强和训练之前建立，绑定 source manifest hash。同一 run 的所有变体继承 split。校准和测试仅 15 例，因此均属探索性小样本证据。

因果边界：oracle_onset=true，使用官方注入时点定位正常基线 [t0-300,t0) 与观测 [t0,t0+60]；decision_time=t0+60。模型文本不含原始时点、case 名、路径、索引 fault_description 或 targets。绝对时点仅用于观测可见性校验。

真实 tokenizer 审计（outputs/serialization_audit.json）：125 例长度 1,679–2,048 tokens，74 例压缩了 evidence；平均保留指标比例 0.9618621554，全部候选保持完整。每例预测保存具体保留 metric keys、缺失模态与截断状态。
