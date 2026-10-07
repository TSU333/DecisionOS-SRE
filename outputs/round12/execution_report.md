# DecisionOS-SRE 第十二轮：因果动态特征对照

客户日期：2026-10-08（Australia/Sydney）。execution_scope: mvp。**未通过全部替换条件，保留第五轮默认模型。** 本轮完成4组真实GPU冻结编码器分类头训练，共 **3608 次优化器更新**，训练函数计时合计 191.1 秒（包含缓存、优化、验证与导出，不含进程启动与最初模型加载）。全部候选保留，未进行新的主干SFT。

## 实际效果

仅按原40例验证选中 `artifacts\round12\legacy_control\frozen`。验证联合准确率由现任 92.5% 到候选 95.0%，sum NLL 0.259229 → 0.252290。验证胜出=True，历史回归执行=True，所有回归项不退步=False，默认模型替换=False。

| 实验 | 可训练参数 | 实际更新 | 最佳checkpoint更新步 | 验证联合 | sum NLL |
|---|---:|---:|---:|---:|---:|
| legacy_control | 66,831 | 902 | 22 | 95.0% | 0.252290 |
| dynamics_fault | 102,671 | 902 | 22 | 95.0% | 0.256788 |
| dynamics_drop15 | 102,671 | 902 | 22 | 92.5% | 0.243114 |
| dynamics_joint | 410,129 | 902 | 22 | 95.0% | 0.256763 |

新增特征最佳方案按预定验证排名是否优于匹配旧特征对照：**False**。不能把旧特征对照也出现的改善归因于新特征。仅一个seed、40例反复使用的验证集，不构成显著性或泛化证明。

每格为根因 / 故障 / 联合准确率。

| 历史回归集 | 第五轮现任 | 本轮候选 |
|---|---|---|
| Online Boutique RE1 | 86.7% / 100.0% / 86.7% | 86.7% / 100.0% / 86.7% |
| Online Boutique RE2 | 96.0% / 80.0% / 76.0% | 96.0% / 76.0% / 72.0% |
| Sock Shop | 96.7% / 96.7% / 93.3% | 96.7% / 93.3% / 90.0% |

候选三个历史回归集是否均达到既有90%根因/90%故障/85%联合工作假设：False（None为未评测）。这些数值不是用户原Prompt明确要求。历史数据多轮重复使用，本轮无新独立质量确认。

预定消融仅对选中的旧特征对照实施：Sock Shop候选反序和服务一致重命名均保持联合90.0%；屏蔽全部指标降至3.3%。RE2-OB移除旧temporal后联合76.0%，移除dynamics仍为72.0%；该候选本来不读取dynamics，所以后者只是兼容性对照，不能评判新增特征贡献。消融不用于重新选择模型。

## 这轮做了什么及原因

先复查现有仓库、固定依赖和资源。沿用Python3.13、PyTorch2.7.1/cu128、transformers4.51.3；Ryzen9 7945HX、16GB RAM、RTX4060 Laptop 8GB。未安装新包。冻结主干允许每组205次预计算（165原TRAIN+40验证），随后只优化小型分类头；每incident完整推理仍只共享编码一次。训练中一次nvidia-smi采样为GPU利用率94%、显存1769/8188MiB，属于瞬时整机采样，非峰值或整轮利用率。

TRAIN审计发现部分旧摘要触及±100截断，例如latency-90的变化z有150/1215项达到截断界限。提出保留幅度和局部波动形态的可证伪假设：新增q10/q90的signed-log幅度、平均相邻变化率、最大跳变、相邻相关和|z|>3比例，另加存在标识。原基线尺度为max(std,abs(mean)*.01,1e-6)，signed-log为sign(v)*log1p(min(abs(v),1e6))/5。

每项都来自本案例相对onset的[-300,0)基线、[0,decision_time-onset]观测，decision不超过onset+60。重复时间取均值；只用间隔≤5秒相邻对，不跨长缺测间隔。至少2个基线点、6个观测时间点、早晚窗口都有观测且至少3对有效相邻采样。缺少动态摘要时使用存在标识，并由新版本运行时强制REVIEW。公式在处理holdout波形前封存，无跨案例拟合统计和gold输入。

新候选数值维度120→190、全局360→570。新增列初始化为0，原文本、候选、span、排序和原数值前缀逐条保持一致。TRAIN6个真实输入的扩维初始完整CPU logits最大差 1.430511474609375e-06，argmax一致；旧默认checkpoint的CPU重载差 0.0。

四组共同seed49、batch16、lr1e-4、weight_decay.01、cohort均衡抽样、最多2200更新、80轮无改善早停。从第五轮重新初始化。前三组仅故障头，第四组根因/故障头同时训练；drop15组仅训练时隐藏层dropout=.15。主干张量全部冻结并核查相等，故障头组根因logits变化严格为0。实际最佳权重经还原导出，保存步数见表；不能把执行了更多更新解释为选中模型更强。

## 数据、选择和发布约束

400个固定原run，165 TRAIN、40验证、40校准、85门控、15 RE1-OB回归、25 RE2-OB回归、30 SS回归。原空观测TRAIN `21a11a8fa96a215914feab22` 保留记录，但从新梯度抽样中排除，有效164。400个原始Parquet哈希重验；manifest/splits逐字节保持，examples只增加动态摘要，删除新字段后与原字典相同，400例旧表示及数值前缀一致。无新增独立案例、无增强视图、无跨split抽样。

先完成全部训练，再只用40验证选一个候选：OB联合≥95%、SS≥90%，之后cohort宏观联合、整体联合、sum NLL，且必须胜过现任。验证失败跳过回归；验证胜出仅封存候选进入历史回归，各组根因/故障/联合必须不退步，失败不改选次优、不按本轮回归调参。校准仅用40，门控仅用85；95%验证不代表95%未知案例准确率。

训练前源码提交 `2efebde78dd7819d87eebb5bb4feafc95e742f94`，24个源文件哈希与提交及实际文件一致。新特征/训练配置/协议均提前封存；报告与审计脚本在训练期间补齐，不改变训练源码。源码、配置、数据、checkpoint、校准器及policy版本绑定验证通过。

## 原Definition of Done验收

| 条目 | 本轮结果与证据范围 |
|---|---|
| 真实公开adapter、标签审计、manifest | 原RCAEval400例固定版本；新摘要和原文件哈希核查通过 |
| Evidence/gold和路径/答案隔离 | 输入扩展只含遥测数值；原标签和分组不变；序列化隔离测试通过 |
| 同run不跨split、增强继承和预处理隔离 | 分组重叠0；无增强；无跨案例拟合预处理；校准/门控指定分区 |
| 动态候选mask/span/ID/target及截断 | 单测、400例兼容比较、真实HTTP边界通过 |
| 每incident一次共享编码 | 缓存205次与完整forward调用断言、runtime断言通过 |
| 缺标监督与有限loss | 既有边界单测及四组真实训练有限loss通过 |
| baseline、frozen、SFT真实路径 | baseline/完整SFT见历史验收；本轮实际训练冻结头，不冒充主干SFT |
| 真实训练及holdout评测 | 历史首次holdout已完成；本轮真实训练/验证/历史回归执行=True；无新holdout |
| checkpoint保存和重载 | 选中完整forward/缓存差 0.0；CPU重载差 0.0 |
| 校准分区、argmax和版本异常 | 标量温度/绑定测试通过；真实拟合40条指定校准例 |
| gate_selection与无阈值零接受 | 85条指定门控例；状态 `selected`；无可行阈值回退测试通过 |
| 可手算指标、缺标和候选遗漏 | 既有指标边界测试通过 |
| CPU实测、API、机器结果、复现 | benchmark/integration/results和下方命令齐全 |
| 实现、正式实验、fixture和未运行分开 | 91项单测包括fixture；正式训练/评测用真实模型与数据 |

**91项测试通过，失败0，pip check通过。** 唯一提示为既有Starlette弃用警告。候选实际HTTP涵盖正常诊断、拒绝gold、缺时间证据、单/空/超预算候选、未知应用。另对固定dynamics_fault工件使用验证输入运行新特征真实HTTP工程检查：原输入、缺dynamics、缺temporal、未来时间戳、拒绝gold均通过；该检查不拟合门控、不查看历史回归、不选模型，缺校准/策略时强制REVIEW。所有测试服务完成后停止。

CPU batch1、8线程、3条固定输入各预热3次并计时12次：端到端P50 826.5ms、P95 871.0ms、吞吐 1.206/秒；进程生命周期峰值工作集 1.98GiB。不同系统负载的单次基准不作为速度提升证明。

## 门控与适用范围

候选门控选择结果 `{'threshold': 0.8673454141081494, 'n': 41, 'errors': 2, 'risk': 0.04878048780487805, 'wilson_95': [0.013480945365169042, 0.16138978870370846]}`，约束为经验联合风险≤5%、至少30个接受run；不可行时全REVIEW。

| 历史回归集 | 接受数 | 错误数 | 接受后错误率 |
|---|---:|---:|---:|
| Online Boutique RE1 | 12 | 1 | 8.3% |
| Online Boutique RE2 | 9 | 1 | 11.1% |
| Sock Shop | 12 | 0 | 0.0% |

以上是重复使用的历史样本风险，不是自动接受的独立安全保证。本机这两应用/五故障的400例均已纳入既定分区，本轮没有取得新独立run；新摘要没有增加独立信息来源。原工程MVP流程已经完成；未知案例质量、生产安全和自动接受风险仍未确认。需要新增独立运行案例，保持已知注入起点/60秒观测与受控故障范围声明。不执行运维动作。

本轮未执行新主干SFT、KD、外部LLM、ONNX、量化或生产事故实验。当前没有已授权MVP工程步骤因缺算力而未完成；缺的是独立质量证据。

## 产物与复现

默认工件 `artifacts\round5\temporal_seed44\frozen`。本轮候选 `artifacts\round12\legacy_control\frozen`，checkpoint SHA256 `c79201b1de48bf21742e1142ba2116f5cfe05efb49104b26c5a39261a801bb68`。完整结果、训练抽样、冻结参数、发布保护、逐案例预测、CPU/API结果、校验清单在 outputs/round12；新输入在 data/round12，原始数据在 data/round4，权重与数据由Git忽略并留存本机。

实际执行 prepare_round12_data.py → run_round12_training.py → evaluate_round12.py → audit_round12.py → verify_round12_dynamics_api.py → report_round12.py；全部通过后提交最终报告。训练前执行pytest，另执行pip check。执行日志保存在outputs/round12。

单配置复现使用新输出目录：

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round12_legacy_control.json -Mode frozen
```

依赖本地data/round12、固定主干及第五轮父模型。该通用单配置脚本会执行历史回归；完整本轮四组封存和门槛逻辑保存在run_round12_training.py/evaluate_round12.py。重复训练旧案例只验证可复现性，不构成新的未见数据或重新选择候选的理由。prepare_round12_data.py拒绝覆盖本轮已经封存的数据和特征公式。若本机新摘要缺失，可从原数据按封存公式重建到新目录，逐项核对outputs/round12/data_audit.json中的SHA256，再将复现配置data_dir指向新目录：

```powershell
./work/.venv/Scripts/python.exe scripts/prepare_round12_data.py --data-output data/reproductions/round12 --audit-output outputs/reproductions/round12
```

数据准备脚本在训练完成后补充了可选输出路径和固定父工件参数，特征计算与模型源码未变；训练使用的版本仍由上述训练前Git提交保存。
