from pathlib import Path
from decisionos_sre.common import read
r=read('outputs/round8/results.json');sel=r['selection'];guard=r['release_guard'];active=r['active_model'];s=Path('README.md').read_text(encoding='utf-8')
a=s.index('最新第');b=s.index('```powershell',a)
state='已更新默认模型' if guard['promote'] else '新候选触发回归保护，默认仍保留第五轮模型'
ob=r['regression_comparisons']['regression_re2_ob']['current'];ss=r['regression_comparisons']['regression_ss']['current']
intro=f"""最新第八轮结果见 `docs/round8_results.md`，机器报告 `outputs/round8/results.json`，验收 `outputs/round8/status.json`。{state}，统一入口为 `outputs/latest_model.json`。全部历史模型保留，execution_scope 仍为 mvp。

本轮固定共享主干和根因分支，只训练66,831个故障头参数，比较6组Dropout/标签平滑方案，实际完成 {r['training_updates']} 次更新。候选验证联合准确率从92.5%提高到{sel['selected']['validation']['joint_accuracy']*100:.1f}%，但历史回归必须同时满足预设的不退步保护。数据仍为原400个案例，没有新独立测试；未进行新的主干SFT。

本轮选中候选的历史回归联合准确率：Online Boutique RE2为{ob['joint_accuracy']*100:.1f}%，Sock Shop为{ss['joint_accuracy']*100:.1f}%。第五轮对应为76%和93.3%；本轮发布决定详见报告，不能把验证集上的提高直接称为泛化增强。原始MVP工程流程已完成，整体质量和自动接受风险仍缺独立确认。

{r['status_summary']['tests']['passed']}项测试及候选的真实HTTP、CPU重载通过，所有冻结权重与父模型精确相同。候选CPU端到端P95为{r['benchmark']['end_to_end']['p95_ms']:.1f}ms，输入为与第五轮相同的3个SS案例。系统只输出诊断与REVIEW/ACCEPT_DIAGNOSIS，不执行运维动作。

"""
s=s[:a]+intro+s[b:]
if guard['promote']:
 s=s.replace('serve --artifact artifacts/round5/temporal_seed44/frozen','serve --artifact '+active['artifact'].replace(chr(92),'/'))
Path('README.md').write_text(s,encoding='utf-8')
print('README updated; active',active['artifact'])
