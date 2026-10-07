# DecisionOS-SRE 第五轮真实执行结果

execution_scope: mvp；experiment_completed。共完成 4 组训练、7397 次更新；训练函数计时合计 233.4 秒，不含数据审计、启动和完整 CPU 验收。

**原 Online Boutique RE2 历史回归仍未达到全部暂定数值目标。本轮没有新增独立案例，不能宣称新的泛化目标已被确认。** 工作质量目标沿用 90% 根因 / 90% 故障 / 85% 联合准确率，该数值来自前轮工作假设，并非用户提供的精确指标。

## 实际分数

| 历史回归集 | 案例数 | 上轮根因 / 故障 / 联合 | 本轮根因 / 故障 / 联合 |
|---|---:|---|---|
| Online Boutique RE1 | 15 | 86.7% / 100.0% / 86.7% | 86.7% / 100.0% / 86.7% |
| Online Boutique RE2 | 25 | 96.0% / 80.0% / 76.0% | 96.0% / 80.0% / 76.0% |
| Sock Shop | 30 | 96.7% / 90.0% / 86.7% | 96.7% / 96.7% / 93.3% |

所有表中数据都已在此前轮次打开。本轮冻结选择后才重新评估，不使用回归结果再次训练或选择。逐案例预测、修复与退化计数、Wilson 区间和配对 bootstrap 区间保存于 results.json；小样本、重复查看以及案例采集独立性未证实均限制结论。不能把提高历史分数等同于独立泛化。

## 本轮改变与选择理由

原表示仅给出均值变化，新数值表示 temporal-v1 补充 p10、p90、波动、趋势、前后半段差异和存在标记。只用保留指标在 [-300,0) 基线及 [0,60] 秒观测内的值，裁剪并固定变换，无标签、路径或 cohort 输入。仍依赖公开注入起点，未评估真实起点发现。候选预留和一次共享 ModernBERT 编码保持不变，新增列使用零权重初始化，原数字列与文本保持原样。

原有 400 个案例组全部保留：train165 / validation40 / calibration40 / gate85 / OB历史回归15+25 / SS历史回归30。源文件哈希复核通过，训练及验证 205 条旧表示指纹一致。三个时间模型使用同配置的不同 seed，均获得 92.5% 验证联合准确率；这是有限种子下的验证稳定性观察，没有进行独立多种子测试确认。

| 试验 | 实际更新 | 最佳 epoch | 验证联合 | 验证 sum NLL |
|---|---:|---:|---:|---:|
| mean_control | 1397 | 67 | 90.0% | 0.315020 |
| temporal_seed42 | 2000 | 127 | 92.5% | 0.262673 |
| temporal_seed43 | 2000 | 147 | 92.5% | 0.268785 |
| temporal_seed44 | 2000 | 124 | 92.5% | 0.259229 |

选择规则训练前封存：保留 OB 验证联合≥95%、SS≥90%，再按四 cohort 联合均值、总体联合、sum NLL 排序，包含上轮现任对照。选中 `artifacts\round5\temporal_seed44\frozen`，最佳 epoch 124 / 更新 1364；实际更新 2000。上一轮 sum NLL 为 0.316845，本轮 0.259229。验证准确率持平时，概率损失改善构成选中依据。

修复了离散准确率在 0.9 与 0.8999999999999999 之间的浮点平局问题；对准确率排序取 12 位精度，再比较 NLL。修复发生在校准、门控和回归前；四组完整训练历史复核表明最优 epoch 与已保存权重均不变，无须重训。原选择记录、代码前后哈希与核查记录保存在 selection_before_precision_fix.json 和 selection_precision_fix.json。

缓存只包含训练和验证输入；所有抽样 ID 均属于 TRAIN，重复抽样不是独立新案例。参数量 149,384,081，训练参数 369,809，缓存与完整模型的验证 logits 最大差 0.0。共享主干本轮冻结；是诊断头续训，不是重新训练整个基础模型。

## 校准与门控

40 个 calibration 案例单独拟合温度：根因 5.3597，故障 2.2701。85 个 gate 案例选择阈值：存在满足门槛的经验阈值。候选阈值中最小观测错误率 0/33=0.0%。原要求仍为经验联合错误率≤5%、至少接受30个案例组。

**原 OB RE2 历史回归中，接受 7 条、错误 1 条，错误率 14.3%，高于 5%。因此门控集经验达标并不等于跨案例自动接受风险达标；不能宣称自动接受已获验证。**

policy 状态 `selected`，选择详情 `{'threshold': 0.8919073952074755, 'n': 36, 'errors': 1, 'risk': 0.027777777777777776, 'wilson_95': [0.004920407139810443, 0.14169718653273855]}`。已有 calibration 和 gate 被复用；选出的经验风险及区间是描述性结果，没有生产安全保证。只接受诊断，不执行任何运维动作。缺少时间证据的旧格式输入会 REVIEW。

| 历史回归集 | 接受覆盖率 | 已接受案例联合错误率 |
|---|---:|---:|
| Online Boutique RE1 | 73.3% | 0.0% |
| Online Boutique RE2 | 28.0% | 14.3% |
| Sock Shop | 40.0% | 0.0% |

## 验收与性能

35 项测试通过；真实 HTTP health/ready/decide、拒绝标签字段、空候选、单候选、候选超预算、未知应用及缺失时间证据检查通过。CPU checkpoint 重载最大 logits 差 0.0。时间截断、常数/缺失指标、旧序列兼容、数值维度迁移、标签隔离、候选顺序与候选重命名均有检查。

CPU 8 线程、batch1：3个SS历史案例各预热3次、测量12次；端到端 P50 809.0 ms / P95 853.3 ms，吞吐 1.231 请求/秒，进程峰值工作集 1.96 GiB。测量受机器负载影响，不是严格配对的硬件速度实验。

候选倒序、统一重命名、删除所有指标以及删除 OB 时间特征的结果见 results.json 的 ablations；不据此再调参。原 MVP 工程 DoD 的数据审计、共享推理、损失/缺失标签、checkpoint重载、校准和门控绑定、CPU/API及机器结果要求已回归验收；本轮没有新的未打开 holdout，独立质量确认保持未完成。

## 产物与复现

仓库 D:/CODEX/DecisionOS-SRE。选中 checkpoint SHA256 `461b93c00f348cc8759060b08f577ea5a7cce25672b0f7de0768d033b16b6423`，split hash `7067334f3de6006195df4aa0d91cd72413aaa551712558e91f1d6d16c8435dd6`，pipeline hash `cb2b7651e54b21f5b315e0250eb80d7abb00ee8ef8f954d289b360b7f5f0a047`。旧模型保留，outputs/latest_model.json 按冻结的验证选择更新。模型、数据与训练日志均留在本机；大型文件不进入 Git。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe -m decisionos_sre serve --artifact artifacts/round5/temporal_seed44/frozen --port 8000
./scripts/reproduce_retraining.ps1 -Config configs/round5_temporal_seed44.json -Mode frozen
```

复现命令建立新的 artifact_root，依赖保留的 data/round5、固定主干及 round4 父 checkpoint；再次评估仍是历史回归。数据准备脚本 scripts/prepare_round5.py、训练 scripts/run_round5_training.py、评测 scripts/evaluate_round5.py 和报告 scripts/report_round5.py 保存于仓库，原封存结果不覆盖。

详细数据审计、选择协议、逐案例概率和错误、训练抽样审计、门控、机器状态、模型清单位于 outputs/round5。协议见 docs/round5_protocol.md。尚未运行新独立同应用确认、新独立 gate 安全确认、生产事故、KD、外部 LLM、量化及 ONNX；不将其写成成果。
