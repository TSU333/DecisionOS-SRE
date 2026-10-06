"""Execute final calibration, regression, CPU benchmark and HTTP checks after freeze."""
import subprocess
import sys
from pathlib import Path
from decisionos_sre.common import read,save,file_hash
selection=read('outputs/retrain/selection.json')
selected=selection['selected']['artifact']
# Evaluate the selected model plus the frozen hybrid control; selection cannot change.
folders=[p for p in dict.fromkeys([selected,'artifacts/retrain/hybrid_frozen/frozen']) if p!='artifacts/sft']
commands=[]
for folder in folders:
    meta=read(Path(folder)/'metadata.json')
    cfg=Path(folder)/'resolved_config.json'
    assert file_hash(Path(folder)/'checkpoint.pt')==meta['binding']['checkpoint_sha256']
    for task in ('calibrate','select-policy','evaluate'):
        device='cpu'  # Match the actual deployment inference device for calibration and gate.
        commands.append([sys.executable,'-u','-m','decisionos_sre','--config',str(cfg),task,'--artifact',folder,'--device',device])
commands.append([sys.executable,'scripts/numeric_baseline.py','--split','test'])
if selected!='artifacts/sft': commands.extend([
 [sys.executable,'-u','-m','decisionos_sre','--config',str(Path(selected)/'resolved_config.json'),'benchmark','--artifact',selected],
 [sys.executable,'scripts/verify_integration.py',selected],
])
for i,cmd in enumerate(commands):
    print('RUN',i,cmd,flush=True)
    path=Path('outputs/retrain')/('evaluation-'+str(i)+'.log')
    with path.open('wb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise SystemExit('FAILED '+str(i)+'; see '+str(path))
    print('PASS',i,flush=True)
save('outputs/retrain/evaluation_commands.json',commands)
