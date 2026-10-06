"""Run a declared set of experiments sequentially; no test/calibration access."""
import subprocess
import sys
import json
from pathlib import Path

names=sys.argv[1:] or ['budget','canonical','hybrid']
for name in names:
    config=Path('configs')/('retrain_'+name+'.json')
    out=Path('outputs/retrain')/(name+'-training.log')
    with out.open('wb') as log:
        command=[sys.executable,'-u','-m','decisionos_sre','--config',str(config),'train','--mode','sft']
        print('START',name,flush=True)
        run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:raise SystemExit('FAILED '+name+'; see '+str(out))
    cfg=json.loads(config.read_text(encoding='utf-8'))
    metadata=json.loads((Path(cfg['artifact_root'])/'sft/metadata.json').read_text(encoding='utf-8'))
    best=min(metadata['history'],key=lambda h:h['validation_loss'])
    print('DONE',name,'steps',metadata['history'][-1]['optimizer_steps'],'best',best,flush=True)
