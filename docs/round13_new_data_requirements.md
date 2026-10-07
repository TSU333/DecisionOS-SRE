# 新案例准入与恢复条件

本轮已准备OpenNetAI/Sock-Shop-Dataset固定提交的19个pod CSV、94条容器故障及30条节点故障标注。原数据在data/external/opennet_sockshop，逐事件清单candidate_events.json标记为quarantined_not_training_data。当前准入0条，未进行外部模型预测或调参。

| 条件 | 所需数据 | 本轮来源情况 |
|---|---|---|
| 独立运行谱系 | source/revision/collection_id/run_id，重复与派生归同组 | 有源提交、目标和时刻；更细运行谱系不足 |
| 时间与观察预算 | 已知注入时刻、明确时区/单位，[-300,0)基线及[0,60]观测 | 时区未注明，观察窗最多2行 |
| 真实采样 | 建议≤5秒原始点；新动态摘要至少6点、早晚都有、有效相邻对≥3 | 60秒采样，不满足；不得插值伪造 |
| 标签 | 公开或可核查注入记录，与Evidence分离 | 29 CPU、31内存、34延迟；没有磁盘/丢包 |
| 根因实体 | 可观测pod→service映射，不依赖gold生成候选 | 当前标签为pod，映射待明确 |
| 遥测语义 | CPU/内存/IO/延迟等的明确单位、counter/rate定义和处理脚本 | 不可仅凭列名直接映射RCAEval；_id属于元数据 |
| 质量 | 缺失、重启/计数器重置、重复时间、其它同期故障记录 | 保留原始缺失；需更细时间观测进一步核查 |
| 评估隔离 | 先按独立run/collection分组封存TRAIN/校准/门控/盲测 | 本轮来源全部隔离，未加入既有分区 |

采集矩阵应覆盖两个现有应用和五类既有故障，并变化负载、注入强度、目标服务与独立运行批次。数量是后续采集预算决定，不把切窗、重采样、同run重复或换导出格式算新案例。盲测在选择模型前封存，原始run及近重复不得跨分区。生产故障注入不在本轮执行范围内。

若获得该来源的高频原始数据，保留独立新revision、原始SHA256和单位说明；不得覆盖本轮CSV。先校验上述准入条件，再实现显式映射adapter，准备模型输入和单独gold、封存新分区，最后对一个预选模型进行一次新盲测。当前分钟级汇总无法可靠恢复缺失的细粒度变化。

当前可重跑来源审计：

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe scripts/audit_round13_external.py
```

这条命令下载/检查固定公开来源，不会产生新的真实运行，不会训练或改默认模型。新高频数据未取得前，准入数量仍为0。详细事件原因与文件校验见outputs/round13/external_data_audit.json及external_source_manifest.json。

来源：[固定提交](https://github.com/OpenNetAI/Sock-Shop-Dataset/tree/1037771bc3c143f95bc615ec8d5b436b33586704)、[RCAEval官方数据](https://huggingface.co/datasets/phamquiluan/RCAEval)。
