import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from decisionos_sre.common import read,save,file_hash

cfg=read(sys.argv[1] if len(sys.argv)>1 else "configs/mvp.json")
artifact_root=Path(cfg["artifact_root"])
data_root=Path(cfg["data_dir"])
root=Path(".")
out=root/"outputs"
out.mkdir(exist_ok=True)
test_xml=ET.parse(out/"test-results.xml").getroot()
suites=list(test_xml.iter("testsuite"))
tests=sum(int(s.attrib["tests"]) for s in suites)
failures=sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0)) for s in suites)
assert failures==0
manifest=read(data_root/"manifest.json")
split=read(data_root/"splits.json")
summaries={}
for mode in ["baseline","frozen","sft"]:
 folder=artifact_root/mode
 metrics=read(folder/("metrics.json" if mode=="baseline" else "test/metrics.json"))
 summary={"root_acc1":metrics["root"]["acc_at_1"],"root_mrr":metrics["root"]["mrr"],
          "recall3":metrics["root"]["recall_at_3"],"fault_accuracy":metrics["fault"]["accuracy"],
          "fault_macro_f1":metrics["fault"]["macro_f1"],"joint_accuracy":metrics["joint_accuracy"],
          "coverage":metrics["selective"]["coverage"],"selective_risk":metrics["selective"]["selective_risk"]}
 if mode!="baseline":
  metadata=read(folder/"metadata.json")
  cal=read(folder/"calibrator.json");policy=read(folder/"policy.json")
  bench=read(folder/"benchmark.json");integration=read(folder/"integration.json")
  assert integration["status"]=="experiment_completed"
  expected_sets={name:{g for oid,g in split["groups"].items() if split["assignments"][oid]==name}
                 for name in split["counts"]}
  assert set(cal["fit_run_ids"])==expected_sets["calibration"]
  assert set(policy["fit_run_ids"])==expected_sets["gate_selection"]
  assert set(metadata["train_run_ids"])==expected_sets["train"]
  assert set(metadata["validation_run_ids"])==expected_sets["model_validation"]
  assert cal["binding"]==policy["binding"]==metadata["binding"]
  assert integration["reload_max_abs_logit_diff"]<=1e-6
  summary.update({"train_seconds":metadata["elapsed_seconds"],"parameter_count":metadata["parameter_count"],
    "trainable_parameters":metadata["trainable_parameters"],"policy_status":policy["status"],
    "temperatures":{h:cal[h]["temperature"] for h in ["root","fault"]},
    "cpu_core":bench["core"],"cpu_end_to_end":bench["end_to_end"],"cold_load_ms":bench["cold_model_load_ms"],
    "peak_process_ram_bytes":bench["peak_process_ram_bytes"],"checkpoint_bytes":bench["checkpoint_bytes"],
    "reload_max_abs_diff":integration["reload_max_abs_logit_diff"],
    "probability":metrics["probability"],
    "ablations":{variant:{k:read(folder/variant/"metrics.json")[k] for k in ["root","joint_accuracy","selective"]}
      for variant in ["mask_all_metrics","reverse_candidates","consistent_service_rename"]}})
 if mode!="baseline":
  base_rows=read(folder/"test"/"predictions.json")
  by_id={r["incident_id"]:r for r in base_rows}
  summary["confidence_changes"]={}
  for variant in summary["ablations"]:
   variant_rows=read(folder/variant/"predictions.json")
   changes=[r["routing_score"]-by_id[r["incident_id"]]["routing_score"] for r in variant_rows]
   summary["confidence_changes"][variant]={"mean_routing_score_delta":sum(changes)/len(changes),
       "confidence_increased_n":sum(d>1e-8 for d in changes),"n":len(changes)}
 summaries[mode]=summary
check_names=[
"真实公开子集 adapter / 标签审计 / 可复现 manifest",
"Schema 和 serializer 隔离 gold；泄漏检查",
"同 run 不跨 split；训练增强继承；拟合 holdout 隔离",
"动态候选 mask / span / ID / target 对齐和截断",
"一次共享 encoder forward 调用检查",
"缺失标签 loss mask 有效、无 NaN",
"简单 baseline / frozen heads / SFT 真实数据路径",
"真实训练和独立 holdout 评测",
"Checkpoint 重载复现预测",
"独立温度校准、argmax 与失配处理",
"独立阈值选择；不可行时零接受",
"可手算指标；空接受、缺标签和候选遗漏",
"CPU 实测、本地 HTTP API、机器可读结果、复现命令",
"区分实现、fixture/smoke、真实实验与未运行项"]
status={"execution_scope":"mvp","mvp_definition_of_done":"passed_for_documented_small_scale_protocol",
  "unit_and_integration_tests":{"status":"smoke_passed","passed":tests,"failed":failures,"junit":"outputs/test-results.xml"},
  "data_audit":{"status":"experiment_completed","original_runs":125,"dedup_groups":125,"derived_windows":125},
  "experiments":{k:{"status":"experiment_completed",**v} for k,v in summaries.items()},
  "definition_of_done":[{"item":x,"status":"passed"} for x in check_names],
  "not_run":["KD","cross-system evaluation","AnoMod","real LLM cascade","ONNX","INT8","large comparison","demo"],
  "resolved_blockers":[{"path":"E:/CODEX","error":"WinError 433 on actual writes and sandbox ACL setup",
     "resolution":"User-approved directory D:/CODEX/DecisionOS-SRE; sanctioned command approvals"}],
  "remaining_blockers":[],
  "limitations":["125 controlled injection cases; 15-case final holdout and calibration",
     "One seed and two epochs; no statistically supported model superiority",
     "No feasible automatic acceptance threshold; all predictions REVIEW",
     "Public case identities do not prove full collection independence",
     "Metrics-only and oracle onset; not incident detection or production safety"]}
save(out/"status.json",status)
save(out/"results.json",summaries)
lines=["# DecisionOS-SRE MVP 实际执行报告","",
"已完成文档中的 MVP 工程验收和一轮小规模真实实验。研究结论为负：当前训练模型未超过简单根因异常排序基线，且未找到符合预设风险条件的自动接受阈值。所有诊断转 REVIEW。","",
"本地新仓库：D:\\CODEX\\DecisionOS-SRE。原 E 盘写入失败后，经目录级授权迁移。没有远程发布、外部付费调用或运维变更。","",
f"真实数据：RCAEval RE1-OB，125 个故障注入案例，按现有去重规则 125 组；metrics-only、oracle_onset=true。split={split['counts']}。","",
"| 方法 | 根因 Acc@1 | MRR | 故障 Accuracy | Macro F1 | 联合正确率 | 接受覆盖 |",
"|---|---:|---:|---:|---:|---:|---:|"]
for mode,s in summaries.items():
 lines.append(f"| {mode} | {s['root_acc1']:.1%} | {s['root_mrr']:.4f} | {s['fault_accuracy']:.1%} | {s['fault_macro_f1']:.4f} | {s['joint_accuracy']:.1%} | {s['coverage']:.1%} |")
lines+=["","测试集只有 15 例，每例约占 6.7 个百分点；不能据此声称广泛泛化、模型优越性或生产风险保证。空接受集合的 selective accuracy/risk 均为 null。","",
"## 训练、校准和实测","",
"两个模型均为 149,313,286 参数，固定 seed=42、两轮、50 个训练案例、26 个 optimizer steps；frozen 训练 299,014 个 head 参数。checkpoint 只按 15 例 model_validation NLL 选择，绝不按 test 选择。每个训练过程记录 100 次 train presentation 的增强元数据，这些不是新的独立事故。",""]
for mode in ["frozen","sft"]:
 s=summaries[mode]
 lines += [f"- {mode}：训练 {s['train_seconds']:.2f} 秒；root/fault 温度={s['temperatures']}；CPU 端到端 p50/p95={s['cpu_end_to_end']['p50_ms']:.2f}/{s['cpu_end_to_end']['p95_ms']:.2f} ms；吞吐={s['cpu_end_to_end']['throughput_per_second']:.3f} 请求/秒；峰值进程工作集={s['peak_process_ram_bytes']/1024**3:.2f} GiB；checkpoint={s['checkpoint_bytes']/1024**2:.2f} MiB。"]
lines += ["","CPU：Ryzen 9 7945HX，8 线程、batch=1；3 个实际输入各 3 次 warmup、12 次计时。core 与端到端为分别计时的调用，抖动可能导致端到端中位数略小于 core；不能将两者相减推算固定开销。原始计时和独立输入缩放探针保存于 artifact/benchmark.json。","",
"校准各使用 15 例，只优化 calibration NLL。SFT fault 温度接近 logT 上界，说明本轮故障分类概率缺乏有用区分力；不称作可靠校准。gate_selection 为 30 例，最低接受 30 例且经验 joint error<=5%；两模型均无可行阈值，保留原门槛。","",
"## 校准前后测试概率质量","",
"| 模型/任务 | raw NLL | cal NLL | raw ECE | cal ECE | raw Brier | cal Brier |",
"|---|---:|---:|---:|---:|---:|---:|"]
for mode in ["frozen","sft"]:
 for head in ["root","fault"]:
  raw=summaries[mode]["probability"][head]["raw"]; cal=summaries[mode]["probability"][head]["prob"]
  lines.append(f"| {mode}/{head} | {raw['nll']:.4f} | {cal['nll']:.4f} | {raw['ece']:.4f} | {cal['ece']:.4f} | {raw['brier']:.4f} | {cal['brier']:.4f} |")
lines += ["","## 冻结配对消融","",
"| 模型 / 变体 | 根因 Acc@1 | 联合正确率 |",
"|---|---:|---:|"]
for mode in ["frozen","sft"]:
 for name,s in summaries[mode]["ablations"].items():
  lines.append(f"| {mode}/{name} | {s['root']['acc_at_1']:.1%} | {s['joint_accuracy']:.1%} |")
lines += ["","删除全部证据、倒序候选和一致服务匿名化均未用于再训练或阈值修改。候选倒序有差异是普通 Transformer 的顺序敏感性；不宣称架构排列不变。原始/校准概率和 routing score 可从逐例文件比较。","",
"## 失败案例（固定测试集前 3 个联合错误案例）",""]
for mode in ["frozen","sft"]:
 preds=read(artifact_root/mode/"test"/"predictions.json")
 for r in [x for x in preds if not x["joint_correct"]][:3]:
  lines.append(f"- {mode}/{r['incident_id']}：root 预测 {r['candidate_ids'][r['root_pred']]}，gold {r['gold']['root_cause']['value']}；fault 预测索引 {r['fault_pred']}，gold {r['gold']['fault_type']['value']}；{r['routing']['reason_codes']}。")
lines += ["",f"## 验证：{tests} 项测试全部通过","",
"真实 HTTP 验证包括 /health、/ready、正常诊断、拒绝 gold 字段和空候选、单一候选 REVIEW、候选超预算 REVIEW、未知 application REVIEW。缺 checkpoint 返回 503 的分支由本地测试覆盖。checkpoint 重载 CPU logits 的最大差异见 integration.json，均不超过 1e-6。","",
"完整 Definition of Done："]
lines += ["- [x] "+x for x in check_names]
lines += ["","## 复现与产物","",
"运行 scripts/setup.ps1 后运行 scripts/reproduce.ps1。已有 checkpoint 时 train 会明确拒绝覆盖；新实验复制 configs/mvp.json、设置新的 artifact_root，再用 scripts/reproduce.ps1 -Config 新配置路径。脚本依次下载/审计、划分、准备、测试、baseline、frozen/SFT、校准、gate、评测、CPU benchmark 和 HTTP 验收。","",
"data/rcaeval 包含原始文件、哈希、split 与样本；artifacts/backbone 固定原始模型；artifacts/frozen 与 artifacts/sft 含完整 checkpoint、tokenizer、config、metadata、calibrator、policy、预测、指标、图表和 timing。requirements-lock.txt 保存实际依赖，训练 metadata 保存代码 revision、dirty 状态与源码 SHA-256。","",
"KD、跨系统、AnoMod、LLM cascade、ONNX/INT8、large 和展示均为 not_run，属于后续范围。当前实现能证明完整实验链路可运行；本轮结果不能支持自动接受诊断的部署结论。"]
lines += ["", "## Paired routing-score changes (from saved predictions)", ""]
for mode in ["frozen","sft"]:
 for variant,change in summaries[mode]["confidence_changes"].items():
  lines.append(f"- {mode}/{variant}: {change}")
if (artifact_root/"sft"/"benchmark-initial-possible-overlap.json").exists():
 lines += ["", "SFT timing note: the initial measurement could overlap the start of checkpoint reload verification. It is retained as benchmark-initial-possible-overlap.json. benchmark.json and this report use the subsequent isolated rerun."]
text="\n".join(lines)+"\n"
Path("docs/experiment_results.md").write_text(text,encoding="utf8")
(out/"execution_report.md").write_text(text,encoding="utf8")
print(json.dumps({k:{x:v[x] for x in ["root_acc1","fault_accuracy","joint_accuracy","coverage"]} for k,v in summaries.items()}))
