from pathlib import Path
from decisionos_sre.common import read
r=read('outputs/round9/results.json');sel=r['selection'];guard=r['release_guard'];active=r['active_model'];s=Path('README.md').read_text(encoding='utf-8')
a=s.index('最新第');b=s.index('```powershell',a)
state='已通过预设规则并更新默认模型' if guard['promote'] else '增强候选未通过回归保护，默认仍保留第五轮模型'
ob=r['regression_comparisons']['regression_re2_ob']['current'];ss=r['regression_comparisons']['regression_ss']['current']
intro=f"""最新第九轮结果见 `docs/round9_results.md`，机器报告 `outputs/round9/results.json`，验收 `outputs/round9/status.json`。{state}，统一入口为 `outputs/latest_model.json`。原模型及全部实验保留，execution_scope 仍为 mvp。

本轮审计发现原165个训练案例中有1例没有故障后的有效观测，保留原始记录但不再抽入训练。其余164例生成328个因果缺测视图：末尾延迟15秒与中间缺测15秒；标签和分组不变，不算新增独立案例。共4组真实训练、{r['training_updates']}次更新。验证、校准与回归输入保持原样，未进行新的主干SFT。

选中候选验证联合准确率为{sel['selected']['validation']['joint_accuracy']*100:.1f}%，sum NLL为{sel['selected']['validation']['sum_nll']:.6f}；第五轮为92.5% / 0.259229。候选历史回归联合准确率：OB RE2 {ob['joint_accuracy']*100:.1f}%、Sock Shop {ss['joint_accuracy']*100:.1f}%；第五轮为76%与93.3%。晋升需同时通过验证选择及各历史cohort不退步保护，不能仅凭验证损失降低宣称模型增强。

58项测试、候选真实HTTP和CPU重载通过。候选CPU端到端P95为{r['benchmark']['end_to_end']['p95_ms']:.1f}ms。原MVP工程流程完成，但质量和自动接受风险仍缺新的独立确认；系统不执行运维动作。

"""
s=s[:a]+intro+s[b:]
if guard['promote']:
 s=s.replace('serve --artifact artifacts/round5/temporal_seed44/frozen','serve --artifact '+active['artifact'].replace(chr(92),'/'))
Path('README.md').write_text(s,encoding='utf-8')
print('README active',active['artifact'])
