# DecisionOS-SRE 第十轮：应用专属故障分类头

execution_scope: mvp。未通过全部替换条件，保留第五轮默认模型。4组真实GPU训练共 **5269 次参数更新**，训练函数计时合计 258.5 秒。没有新增独立数据，也没有新一轮主干SFT。本轮结论仅适用于反复使用的验证和历史回归，不等于达到生产质量。

## 实际效果

以下每格依次为根因准确率 / 故障准确率 / 联合准确率。

| 历史回归集 | 原案例数 | 默认第五轮 | 本轮封存候选 |
|---|---:|---|---|
| Online Boutique RE1 | 15 | 86.7% / 100.0% / 86.7% | 86.7% / 100.0% / 86.7% |
| Online Boutique RE2 | 25 | 96.0% / 80.0% / 76.0% | 96.0% / 76.0% / 72.0% |
| Sock Shop | 30 | 96.7% / 96.7% / 93.3% | 96.7% / 93.3% / 90.0% |

选中 `artifacts\round10\app_drop15\frozen`，验证联合准确率 95.0%，sum NLL 0.255663；现任验证联合92.5%，sum NLL 0.259229。验证晋升条件=True，历史回归保护=False，最终替换=False。每组根因/故障/联合比较见 release_guard.json；失败后不测试次优模型或追加调参。

| 实验 | 学习率 | dropout | 实际更新 | 最佳epoch | 验证联合 | sum NLL |
|---|---:|---:|---:|---:|---:|---:|
| shared_control | 0.0001 | 0.0 | 891 | 1 | 92.5% | 0.265865 |
| app_lr100 | 0.0001 | 0.0 | 1540 | 60 | 92.5% | 0.258118 |
| app_lr30 | 3e-05 | 0.0 | 891 | 1 | 92.5% | 0.262573 |
| app_drop15 | 0.0001 | 0.15 | 1947 | 97 | 95.0% | 0.255663 |

四组均从第五轮初始化，seed47、batch16、weight_decay0.01、最多2200更新，80个epoch无改善早停；只训练故障头，根因和主干均冻结。shared_control是同预算共享头对照，另外三组为应用专属头。预先按OB验证联合至少95%、SS至少90%，再按四cohort联合均值、总体联合、NLL选一个模型；协议与配置的哈希在训练前封存。验证40例经过多轮使用，选择偏差不能忽略。

## 数据、架构和工程选择

沿用RCAEval固定版本的400例公开故障注入数据。原始run划分为165训练、40验证、40校准、85门控、15 RE1-OB回归、25 RE2-OB回归、30 SS回归。本轮新案例=0，增强视图=0。训练中一条无有效观测记录 `21a11a8fa96a215914feab22` 不参与抽样，但仍保留原始记录及父模型训练历史，因此有效TRAIN为164。manifest、splits、examples哈希均与前轮一致，无跨split原始run。

应用名称只从TRAIN的公开 `input.application` 确定，映射为 Online Boutique / Sock Shop。每个应用单独复制文本、全局数值、根因条件数值故障分类头；新增约13.4万参数。初始参数来自现任模型，根因加权只使用预测根因，不使用gold。独立应用头避免两个应用通过同一故障参数相互影响，但小样本仍可能过拟合；较低学习率和dropout为预先固定的两个约束方案。所有训练只更新所属故障头，未知应用保留旧共享头并强制REVIEW。映射顺序写入模型与校准绑定，不支持静默重排。

API字段不变；编码器仍共享，一次事故只编码一次，候选span/mask、2048 token预算和temporal-v1不变。CPU仅用8线程；本机Ryzen 9 7945HX、16GB RAM、RTX4060 Laptop 8GB，启动前可用RAM约3.6GiB，因此顺序运行、缓存冻结表示，无新增依赖。已知注入起点和60秒观测窗口仍是实验假设，未测试真实事故检测。

## 按原Definition of Done验收

原始MVP工程验收与本轮增量验收分开记录，历史状态保存在 status.json 的 historical_mvp_checks。历史baseline / frozen / SFT及首次holdout证据仍在此前产物，本轮仅重新训练分类头并跑历史回归，不把它们重新称为独立holdout。

| 原始DoD条目 | 本轮核验或历史证据 |
|---|---|
| 公开数据adapter、标签审计、manifest | 已有真实400例；本轮文件哈希、分组与应用输入复核通过 |
| Evidence/gold、路径和注入答案隔离 | schema/serializer测试通过；新路由只取公开application |
| 同run不跨split、增强继承、预处理分离 | 本轮交叉重叠0；无增强；校准与门控仍用指定分区 |
| 动态候选mask/span/ID、无效输入与截断 | 测试及真实HTTP边界验证通过 |
| 每incident一次共享编码 | 模型单测、真实predict和runtime调用断言通过 |
| 缺失标签、有效loss、无NaN | 单测通过；四组实际训练完成 |
| baseline、frozen、SFT路径 | 历史真实实验已完成；本轮不重复SFT |
| 至少一轮真实训练与holdout、注明范围 | 历史首次holdout已完成；本轮实际训练与历史回归完成，未有新独立holdout |
| checkpoint保存重载 | 本轮CPU最大logits差 0.0 |
| 独立校准、argmax保持、版本不匹配处理 | 校准40例；相关单测和绑定检查通过 |
| gate_selection、无可行阈值全REVIEW | 门控85例；策略状态 `selected`；零覆盖回退测试通过 |
| 可手算指标、空接受/缺标/候选缺失 | 指标边界测试通过 |
| CPU实测、可运行API、机器结果与复现 | 本轮benchmark.json、integration.json、results.json和下方命令 |
| 区分完成/实现/mock/未运行 | 69项测试含小fixture；正式指标来自真实RCAEval；未运行项如下 |

69项测试通过，失败0；pip check无损坏依赖。测试有一条Starlette依赖弃用提示，不影响结果。本轮候选 152 个冻结张量与父模型完全相等；缓存/完整GPU推理最大logits差 0.0。真实HTTP验证健康、诊断、拒绝gold、缺时间证据、单/空/超预算候选及未知应用；若选中应用分支模型，另对两个应用分别比对API与离线概率。所有服务验证进程在结束后停止。

CPU batch1、8线程，固定3个SS输入各预热3次、测量12次：P50 876.6ms，P95 964.9ms，吞吐 1.131/秒，进程生命周期峰值工作集 1.96GiB。不同轮次系统负载不一致，单次计时不能证明性能提高。

## 门控和局限

候选门控结果 `{'threshold': 0.8852766614821203, 'n': 40, 'errors': 2, 'risk': 0.05, 'wilson_95': [0.013820667386148344, 0.16503877369140962]}`；预设经验联合风险不超过5%、至少接受30组。以下是候选在历史回归上的实际选择性结果，不是安全承诺，也不能自动外推门控集风险。

| 历史回归集 | 接受数 | 接受后错误数 | 覆盖率 | 接受后错误率 |
|---|---:|---:|---:|---:|
| Online Boutique RE1 | 12 | 0 | 80.0% | 0.0% |
| Online Boutique RE2 | 8 | 1 | 32.0% | 12.5% |
| Sock Shop | 12 | 0 | 40.0% | 0.0% |

三个历史回归集是否全部达到工作假设90%根因/90%故障/85%联合：False。这些数值是之前的工程假设，不是原Prompt明确要求。工程验收通过也不等于独立泛化、自动接受风险或生产安全已确认。固定来源中与当前两个应用、五类故障兼容的400例已用完；独立确认仍需要新的、故障定义一致的运行案例。

本轮未运行：新主干SFT、KD、外部LLM、跨新应用测试、生产事故实验、ONNX、INT8量化。未执行任何运维操作。保留全部未晋升候选和失败证据，默认指针按预先固定的保护规则处理。

## 产物与复现

默认模型：`artifacts\round5\temporal_seed44\frozen`。
本轮候选 checkpoint SHA256：`8b265e5f975783400089612cb9f04ec354fd855f271d3e144589b58094b4473c`。
模型本地权重与数据由Git忽略，源码、配置、协议、逐案例指标和报告归档。训练时Git工作区标记为dirty：训练前提交只包含新文件及协议，漏暂存的已修改文件在归档时补齐。22个实际训练源文件均由训练时SHA256封存，事后验证与实现快照 `c507e175346e284ed32463d31eb46d651e3a56ef` 完全一致；未改写训练元数据或权重。仅检出训练元数据中的旧Git HEAD不足以复现，应使用该实现快照或当前完整仓库；详见 source_snapshot.json。验收证据包括 protocol.json、selection.json、training_integrity.json、training_draw_audit.json、weight_audit.json、artifact_manifest.json、release_guard.json、status.json、results.json。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round10_app_drop15.json -Mode frozen
```

复现依赖本地 data/round5、pinned backbone及第五轮父模型，通用脚本使用新输出目录。四组完整过程由 prepare_round10.py、run_round10_training.py、evaluate_round10.py、report_round10.py保存；原封存输出不覆盖。
