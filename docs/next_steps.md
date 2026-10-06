# 后续任务与前置条件

MVP 当前验收状态以 outputs/status.json 和 experiment_results 为准。若训练或测试失败，先按失败日志修复并用新 artifact_root 重跑；不以 fixture 替代真实实验。

1. 扩大真实独立采集数量、审查实验谱系并使用多个 seed。优先扩充独立 calibration / gate / test，评估置信区间与阈值稳定性。当前 15 例校准和测试极小。
2. 验证 RE2/RE3 实际存在的 logs/traces 后才新增 typed schema、serializer 和消融。当前 API 明确只接受 metrics summaries。
3. research：有真实 teacher 资源与预算后实现候选对齐目标及 KD；无收益同样如实报告。AnoMod 先审计真实文件、标签、application 来源，再定义跨来源/跨系统协议。
4. research：外部 LLM-only/cascade 必须实际执行、保存解析/重试/成本。没有凭据或预算时仍为 not_run。
5. deployment：用已验收 checkpoint 导出 ONNX、验证可变候选/长度、再做硬件支持的 INT8，最终 artifact 独立重校准。large 必须实际训练才比较。
6. 部署安全动作不在本项目范围。ACCEPT_DIAGNOSIS 只接受诊断，不授权重启/回滚/扩缩容。


第二轮已完成独立学习率、训练预算、身份规范化、均值池化、数值融合与轻量数值分类对照。下一步优先获取新的独立确认性案例，并扩大 calibration/gate 样本；在新协议开始前确定概率损失与联合准确率的模型选择优先级。不要继续对已查看的回归集合反复调参后称作未见测试。
