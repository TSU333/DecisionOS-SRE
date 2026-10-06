# 工程选择记录

1. 新建本地 Git 仓库；机器 PATH 无 Git，通过 dulwich 初始化与提交。未创建远程仓库，因为用户未指定远程平台或组织。
2. 原 E 盘写操作返回 WinError 433。经目录级读写授权转到 D:\CODEX\DecisionOS-SRE；未修改任何 ACL 或关闭安全机制。
3. 系统 Python 3.13 无 ML 依赖。使用仓库内 venv，固定 torch 2.7.1/cu128、transformers 4.51.3 及 pyproject 版本；完整解析版本另存 lock。
4. 选择官方转为 Parquet 的 RE1-OB 125 例；先下载 29.5 KB 索引并检查一例，再下载最小完整单应用子集。排除 research/deployment 范围。
5. 不猜测单位；Parquet 字段无单位元数据时明确 source_unit_unspecified。原始列名和数值均保留。缺失值不做回填。
6. 遥测列名最后一个下划线分隔服务与指标；该规则经真实列验证。候选包含主机/聚合项等实际遥测实体，即便它们从未成为本数据集注入标签。不会依据 gold 删减候选。
7. 原始 case/绝对采集时间可能编码故障类型。模型文本只使用相对窗口定义与观测汇总；绝对 decision_time 仅用于输入因果校验，不进入 encoder。
8. 一个 run 一个固定窗口；原始/基线/归一化四舍五入窗口指纹用于保守重复分组。公开来源未给出更完整实验谱系，因此不承诺已证明所有重复采集独立。
9. 采用固定五路小规模 holdout，保持部署 checkpoint、校准器和策略对应。15 例校准只支持探索性拟合；阈值按至少 30 独立组和 joint error <=5% 搜索。无可行阈值即全 REVIEW，不降低门槛。
10. 使用 PyTorch SDPA，关闭 ModernBERT reference_compile，无 FlashAttention 依赖。训练允许 CUDA autocast BF16、梯度检查点；CPU 始终 FP32。
11. 初始 max_length=2048。完整保留候选，按观测变化绝对值排序压缩 metrics，记录每例保留率。没有训练拟合的预处理；基线来自本 incident 决策前已观测的正常窗口。
12. 仅两轮、固定 seed 42、相同数据与增强预算比较 frozen 与 SFT。训练样本不足以得出广泛质量结论。类别在 train 按 fault 分层均衡，不使用额外类权重。
13. scalars 温度拟合使用 scipy 有界优化 logT in [-3,3]。联合 routing score 为两个 confidence 的 min，不称为联合校准概率。
14. checkpoint SHA、tokenizer SHA、split hash、ontology/serializer/max_length hash 绑定校准器与策略。API 无有效接受条件时拒绝自动接受。
