# DecisionOS-SRE 继续执行与成功判定

execution_scope: mvp。原 Prompt 的工程流程已通过历史验收；诊断质量与自动接受风险尚不能宣布全部成功。本轮完成新的真实实验，但没有选出优于第五轮的模型，默认入口仍为 `artifacts\round5\temporal_seed44\frozen`，旧权重不覆盖。

## 本次实际执行

核查了代码、依赖和资源；新增审计固定 RCAEval revision 的75个RE2-OB调用链文件，共 728,637,182 字节。原400个案例、所有split及group均保留，新增证据不算新增独立案例。其余325例缺少traces，继续仅使用指标。

实现 completed-span 服务摘要与24个数值通道，零填充迁移原数值头，单例仍只调用一次共享ModernBERT。训练前封存预算和选择规则。四组续训共 **5005 次更新**、训练函数计时 212.3 秒；另完成8组ExtraTrees轻量基线拟合，属于只使用训练/验证数据的诊断，不是已部署模型。

| 方案 | 实际更新 | 最佳 epoch | 验证联合准确率 | 验证 sum NLL |
|---|---:|---:|---:|---:|
| metrics_control | 1056 | 26 | 92.5% | 0.262878 |
| trace_seed42 | 1606 | 76 | 92.5% | 0.313495 |
| trace_seed43 | 1177 | 37 | 92.5% | 0.308697 |
| trace_seed44 | 1166 | 36 | 92.5% | 0.304489 |

第五轮现任验证联合准确率92.5%，sum NLL=0.259229。本轮所有神经候选准确率相同，NLL均更差；8个轻量基线均为87.5%。根据预先规定的验证规则拒绝替换。不能因训练次数增加、加入新模态或某个回归分数变化就称为模型增强成功。

## 冻结后的实验候选验收

为验证新调用链实现，额外验收验证集最好的调用链候选 `artifacts\round6\trace_seed44\frozen`。它是实验产物，未晋升；不是用回归结果重新选模型。下面比较的是第五轮默认模型与该实验候选。

| 历史回归集 | n | 第五轮根因 / 故障 / 联合 | 本轮调用链候选根因 / 故障 / 联合 |
|---|---:|---|---|
| Online Boutique RE1 | 15 | 86.7% / 100.0% / 86.7% | 86.7% / 100.0% / 86.7% |
| Online Boutique RE2 | 25 | 96.0% / 80.0% / 76.0% | 96.0% / 76.0% / 72.0% |
| Sock Shop | 30 | 96.7% / 96.7% / 93.3% | 96.7% / 96.7% / 93.3% |

全部都是此前已打开的案例。本轮没有新独立测试，重复验证选择也可能过拟合。逐案例概率、修复/退化计数和描述性区间在 results.json；failure_cases.json 保存错误，不能用它们继续拟合再称为未见测试。

候选门控状态 `selected`，只在85个gate案例上按经验错误率≤5%、至少30例选择，详情 `{'threshold': 0.867538699397492, 'n': 43, 'errors': 2, 'risk': 0.046511627906976744, 'wilson_95': [0.012849260094380227, 0.15455498004063967]}`。OB RE2历史回归接受 8 例、错误 1 例，错误率 12.5%。门控、校准均被复用，经验结果不构成独立自动接受安全验证。系统不执行任何运维动作。默认第五轮策略及其已记录的OB接受风险1/7=14.3%保持原记录，不拿实验候选结果替换它。

## 数据与工程正确性

- 使用span结束时间过滤：baseline开始≥onset−300且结束<onset；观测开始≥onset且结束≤onset+60。持续到决策之后的span不参与统计。
- Jaeger时间/持续时长以微秒读取，延迟转成ms；重复traceID/spanID去重；缺失状态码不当成成功。实际入库延迟未知，完成时间仅近似可见性。
- 全span与operation目标服务匹配自身的RPC子集分别汇总；后者是明确的名称匹配规则，源数据没有提供统一span-kind保证。
- 别名、排序和预算不依赖gold；traceID、spanID、operationName、标签及路径不进入模型文本。只用完整保留的调用链摘要作为数值输入。
- 205条旧训练/验证指标表示完全一致。含trace的训练/验证20例共保留 118/140 个服务摘要（84.3%），输入长度 1960–2043，不超过2048。
- 41项测试通过；缓存与完整推理最大logits差 0.0；CPU重载差 0.0。
- 真实HTTP检查包含指标输入、调用链输入、拒绝未来trace、标签泄漏、空/单一/超预算候选、缺少时间证据与未知应用。候选倒序、统一改名和去掉traces等实际消融见 results.json。

CPU benchmark本次改为3个含调用链的OB回归输入，各预热3次/测量12次：P50 1525.0 ms、P95 1583.2 ms、吞吐 0.657/秒，峰值进程工作集 1.97 GiB。样本与第五轮SS性能测试不同，不作直接速度因果比较。输入选择记录于 outputs/round6/benchmark_config.json。

## 为什么还不算完整成功

当前默认模型的OB RE2根因96%、故障80%、联合76%，未达到此前工作假设90%/90%/85%；该假设并非用户原Prompt中的明确数值。默认模型的自动接受错误率在该历史子集中为14.3%，不能宣称≤5%得到独立验证。

本轮4组续训和8组轻量基线均未提高验证表现。现有RE2-OB训练集每类只有3个案例，限制结论；不能保证继续加轮次会获得收益。公开AIOps2025的400个标签已核查，只有42个能严格映射CPU/内存两类；网络链路与磁盘填满/读写错误不能冒充当前单服务网络故障与磁盘压力。因此本轮未将它们混入或作为完整五分类确认。

下一步需要新增同应用、五类定义一致、按原始运行分组的真实故障数据；提前封存独立确认集和gate确认集。缺少这些证据时，可以说MVP工程流程完成，不能说诊断质量和自动接受已经全面达标。原MVP历史验收与本轮没有fresh holdout的状态在 status.json 分开记录。

## 产物与复现

默认模型：`artifacts\round5\temporal_seed44\frozen`。实验调用链模型：`artifacts\round6\trace_seed44\frozen`；checkpoint SHA256 `05062b636ea823ef10da0a8adeb9d883f00f07e6d10090525beceef458191b82`。实验代码、配置、逐案例输出与测试记录已保存，权重和原始数据留在本机并忽略Git。

```powershell
Set-Location D:/CODEX/DecisionOS-SRE
$env:PYTHONPATH='src'
./scripts/reproduce_retraining.ps1 -Config configs/round6_trace_seed44.json -Mode frozen
./work/.venv/Scripts/python.exe scripts/diagnose_round7.py
```

复现建立新artifact_root并依赖保留的data/round6与第五轮父模型；已打开案例只能作为回归复现。通用复现脚本默认测SS性能；复现本轮含trace的CPU测量，另传 `--config outputs/round6/benchmark_config.json` 执行benchmark。没有运行新的SFT主干微调、KD、外部LLM、量化或ONNX；本轮只训练诊断头，旧SFT实验仍保留。

来源与单位核验：[固定RCAEval数据](https://huggingface.co/datasets/phamquiluan/RCAEval/tree/afeacb11bcc94dadfd1c8f483ee4377b2b8b614e)、[Jaeger字段定义](https://github.com/jaegertracing/jaeger/blob/v1.54.0/model/json/model.go)、[gRPC状态码](https://grpc.io/docs/guides/status-codes/)、[AIOps2025官方说明](https://www.aiops.cn/gitlab/aiops-live-benchmark/agenticopseval/-/blob/main/AIOps2025/README.md)。
