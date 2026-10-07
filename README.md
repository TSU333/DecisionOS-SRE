# DecisionOS-SRE

研究型结构化诊断 MVP。共享 ModernBERT encoder 输出动态候选根因与五类故障；独立校准与 policy 决定 ACCEPT_DIAGNOSIS / REVIEW。不执行运维操作。

最新第九轮结果见 `docs/round9_results.md`，机器报告 `outputs/round9/results.json`，验收 `outputs/round9/status.json`。增强候选未通过回归保护，默认仍保留第五轮模型，统一入口为 `outputs/latest_model.json`。原模型及全部实验保留，execution_scope 仍为 mvp。

本轮审计发现原165个训练案例中有1例没有故障后的有效观测，保留原始记录但不再抽入训练。其余164例生成328个因果缺测视图：末尾延迟15秒与中间缺测15秒；标签和分组不变，不算新增独立案例。共4组真实训练、7590次更新。验证、校准与回归输入保持原样，未进行新的主干SFT。

选中候选验证联合准确率为92.5%，sum NLL为0.243747；第五轮为92.5% / 0.259229。候选历史回归联合准确率：OB RE2 72.0%、Sock Shop 90.0%；第五轮为76%与93.3%。晋升需同时通过验证选择及各历史cohort不退步保护，不能仅凭验证损失降低宣称模型增强。

58项测试、候选真实HTTP和CPU重载通过。候选CPU端到端P95为1019.7ms。原MVP工程流程完成，但质量和自动接受风险仍缺新的独立确认；系统不执行运维动作。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./work/.venv/Scripts/python.exe -m decisionos_sre serve --artifact artifacts/round5/temporal_seed44/frozen --port 8000
```

复现选中方案：`./scripts/reproduce_retraining.ps1 -Config configs/round5_temporal_seed44.json -Mode frozen`。需要保留 data/round5、固定主干和第四轮父模型；脚本创建新的输出目录并执行训练、校准、门控、历史回归、CPU benchmark 与真实 HTTP。重跑已有案例是复现，不是新的未见确认。

协议与选择理由见 `docs/round5_protocol.md`。指标字段新增可选 `temporal` 对象，包含 q10_z / q90_z / std_ratio / trend_z / late_shift_z；使用 `decisionos_sre.temporal.temporal_summary` 从决策时刻之前的同一时间窗口提取，计算边界和尺度见实现。旧版模型继续兼容旧格式。

## 本地环境
本次工作目录为 D:\CODEX\DecisionOS-SRE。原 E 盘工作目录读操作可用，但写操作实际返回 WinError 433；用户已授权使用 D 盘指定目录。保留此事实以免误认为所有命令仍在原目录执行。

Windows PowerShell，Python 3.13。硬件为 AMD Ryzen 9 7945HX、16 核 / 32 线程、约 16 GB RAM、RTX 4060 Laptop 8 GB。CUDA 训练、CPU 推理。安装与下载都使用仓库内 `work/`；不占用低剩余空间的 C 盘缓存。

```powershell
Set-Location D:\CODEX\DecisionOS-SRE
.\scripts\setup.ps1
.\scripts\reproduce.ps1
```

完整环境锁定见 `requirements-lock.txt`（生成后）；顶层经过选择的版本在 pyproject.toml。PyTorch CUDA 轮子来自官方 cu128 索引。若没有 CUDA，请显式修改训练设备并评估计算预算，不会自动启动长时间 CPU 训练。

## 初始数据与实验（历史配置）
官方 RCAEval RE1-OB，固定 revision，125 个受控故障注入案例，metrics-only；不是线上生产事故，也不是多模态实验。下载与模型路径均不进入 Git。125 个观测组划分为 50 train / 15 model_validation / 15 calibration / 30 gate_selection / 15 test。这些 holdout 很小，结果仅为流程可行性与探索性质量评估。

标签来自官方索引中的注入元数据；候选来自遥测列，而非正确答案。时间窗口使用已知注入起点 `oracle_onset=true`，不测试故障检测。详见 data_audit 与 evaluation_protocol。

CLI:
```powershell
$env:PYTHONPATH = 'src'
$python = '.\work\.venv\Scripts\python.exe'
& $python -m decisionos_sre audit-data
& $python -m decisionos_sre make-splits
& $python -m decisionos_sre prepare-data
& $python -m decisionos_sre baseline
& $python -m decisionos_sre train --mode frozen
& $python -m decisionos_sre train --mode sft
& $python -m decisionos_sre calibrate --artifact artifacts/sft --device cuda
& $python -m decisionos_sre select-policy --artifact artifacts/sft --device cuda
& $python -m decisionos_sre evaluate --artifact artifacts/sft --device cpu
& $python -m decisionos_sre benchmark --artifact artifacts/sft
& $python -m decisionos_sre serve --artifact artifacts/sft --port 8000
```

已有 checkpoint 不会被 train 静默覆盖。重新实验应复制配置、换新的 artifact_root，并把 `--config 路径` 放在子命令之前。测试集一旦打开，禁止用其结果选模型或修改阈值；代码缺陷修正须记录后重新标记评测历史。

## API
只监听 127.0.0.1。GET /health 判断进程存活，GET /ready 检查模型加载。POST /v1/decide 仅接受 IncidentInput；训练标签、路径等额外字段为 422。缺 checkpoint 为 503，不回退随机模型。候选超预算返回 REVIEW 和明确原因，两个任务不伪造概率。API 支持指标摘要及可选 `evidence.traces` 服务摘要；调用链特征仅由显式启用的第六轮实验模型使用，默认第五轮模型仍只使用指标。logs 尚未接入。调用链原始数据需通过 `decisionos_sre.traces.summarize_traces` 按决策时间汇总；不能直接传入原始 span。

准备数据后可从 examples.json 取出某条 `input` 保存为请求；不发送整个 TrainingExample。每个 metric 包含服务、原始指标名、基线均值、观测均值、变化 z 值、缺失比例和观测截止时点。汇总公式和时间边界必须与 adapter 一致。

## 验证与产物
```powershell
& .\work\.venv\Scripts\python.exe -m pytest -q
```
单元测试不下载模型或数据；已有真实数据时自动附加 manifest 检查。微型随机 backbone 只用于张量和错误分支测试，不生成正式实验结果。

- data/rcaeval/: 原始 Parquet、每文件 checksum、审计 manifest、split 与输入样本。
- artifacts/backbone/: 固定版本的官方权重和 tokenizer。
- artifacts/frozen/ 与 artifacts/sft/: checkpoint、元数据、校准器、policy、逐 incident 预测、指标、消融、图表、benchmark。
- outputs/: 运行证据、验收状态与最终报告。
- docs/: 数据审计、协议、工程选择、结果与后续任务。

没有远程发布、付费 LLM、KD、ONNX、INT8、large 模型比较。此项目不能据当前小样本声称生产安全、开放集识别或普遍的自动接受保证。
