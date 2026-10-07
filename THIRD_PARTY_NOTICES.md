# 第三方来源

本源码包不包含原始遥测、预训练权重或第三方依赖的安装副本。项目代码的许可不替代上游许可。

| 来源 | 项目中的用途 | 固定版本与上游 |
|---|---|---|
| RCAEval | 公开受控故障案例、数据来源和注入标签 | 数据revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`；[代码仓库](https://github.com/phamquiluan/RCAEval)、[数据卡](https://huggingface.co/datasets/phamquiluan/RCAEval) |
| ModernBERT-base | 共享预训练编码器 | revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`；[模型卡与Apache-2.0许可](https://huggingface.co/answerdotai/ModernBERT-base) |
| OpenNetAI Sock-Shop-Dataset | 外部数据准入审计；94事件均未纳入训练或评估 | commit `1037771bc3c143f95bc615ec8d5b436b33586704`；[MIT许可与来源](https://github.com/OpenNetAI/Sock-Shop-Dataset/tree/1037771bc3c143f95bc615ec8d5b436b33586704) |

Python依赖和版本见 `pyproject.toml` 与 `requirements-lock.txt`。依赖通过包管理器安装，各自遵循上游许可。`outputs/` 包含本项目生成的统计、预测和审计记录；数据源名称与案例标识仅用于追溯公开实验来源。

本项目独立于上述上游项目，不代表其维护者提供背书。后续若单独发布权重或原始数据，需要随相应产物保留其适用许可和来源说明。
