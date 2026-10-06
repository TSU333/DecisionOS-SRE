# 第四轮继续训练协议

execution_scope: mvp。延续用户“继续”训练请求。按固定 RCAEval revision 扩充 Sock Shop 同五类故障样本，不引入新模态、外部 LLM、KD 或量化。

## 数据审计和封存

首先检查 TORAI 官方包：75 条已支持 Online Boutique 案例中，73 条与已有 RE2 同列、同时间、同数值，另 2 条时间覆盖更多，但与已有时间交集的数值相同，注入时间也相同。因此全部排除为独立新增来源，详见 outputs/round4/torai_overlap_audit.json。官方包 DOI https://doi.org/10.6084/m9.figshare.31925976.v1；原始压缩包 SHA256 与许可在审计记录。

加入 200 条此前未使用的 RE1-SS / RE2-SS 案例，仍限 cpu/mem/disk/delay/loss。原始案例在下载指标前用 seed 20261008 分配；每服务×故障，RE1 取 3 条训练、RE2 取 1 条训练，其余按故障和来源分层划给独立 validation/calibration/gate/test。没有按结果选择候选或样本。

合计 400 案例、400 指纹组，train 165 / model_validation 40 / calibration 40 / gate_selection 85 / 原 RE1-OB regression 15 / 原 RE2-OB regression 25 / 新 Sock Shop test 30。旧 train/validation/calibration/gate 分配原样保留，旧两批 test 只能称 regression。每案例一个窗口，不用重复抽样冒充新增独立案例。全部候选覆盖率 100%。指纹检查不等于已证明采集谱系统计独立。

输入仍是已知注入起点前 300 秒和后 60 秒的因果汇总。数字词表从 TRAIN 得到，保持十个已有指标名及顺序，所以能正确复用上一轮数值头。仅元数据中的来源 cohort 用于采样和统计，不序列化到模型。

## 有限训练与选择

三个初始候选：常规 shuffle（seed42, lr3e-4），四来源均衡有放回采样（seed42, lr3e-4），均衡采样（seed43, lr1e-3）。第三个同时变化 seed/lr，不据此宣称纯随机种子稳定性。各最多 2000 更新，早停 60 轮，冻结编码器并复用精确特征缓存，温启动上一轮选中 checkpoint。训练 draw 的 run IDs 完整记录；所有 draw 仅来自 TRAIN。

每轮只用 model_validation 选择 checkpoint。先要求旧 Online Boutique 20 条验证的联合准确率至少 85%（上一轮为 90%，允许至多多错一条），再最大化 RE1-OB / RE2-OB / RE1-SS / RE2-SS 联合准确率的等权平均，平手依次比较整体联合和两任务 NLL。未满足保留约束者不能成为新的默认模型。原质量目标不因这一保留约束下调。

选定后保存 selection.json，再拟合独立温度、选择 gate 门限、打开新 test 一次。门控标准仍为经验联合风险 ≤5%、至少 30 条接受案例。没有可行门限就全部 REVIEW。旧应用回归和新应用测试分开报告，不能将新应用分数当作原应用新泛化目标达标的证明。暂定 90% / 90% / 85% 只作为逐数据集的探索性目标，不保证实现。

## 验收

检查候选排序/重命名/删除证据消融，CPU 单请求实测、真实 HTTP、checkpoint 重载、采样隔离和旧模型兼容。保留原权重与结果。负结果必须保存；新测试打开后不据此继续选参。
