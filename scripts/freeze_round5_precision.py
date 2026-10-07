"""Resolve numerical accuracy ties before any heldout evaluation."""
from pathlib import Path
from datetime import datetime,timezone
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.training import selection_key
out=Path('outputs/round5');sel=read(out/'selection.json');protocol=read(out/'protocol.json');cfg=protocol['selection_config']
fix=read(out/'selection_precision_fix.json');audits=[]
for c in sel['candidates']:
    folder=Path(c['artifact']);assert not (folder/'calibrator.json').exists() and not (folder/'test_freeze.json').exists()
    meta=read(folder/'metadata.json');best=min(meta['history'],key=lambda h:selection_key(h['validation'],cfg))
    rank=list(selection_key(c['validation'],cfg));same=rank==list(selection_key(best['validation'],cfg)) and best['epoch']==c['best_epoch']
    audits.append({'artifact':str(folder),'unchanged':same,'best_epoch':best['epoch']})
    assert same,'Existing selected weights differ from correct tie rule; rerun this trial'
    c['raw_float_rank']=c['rank'];c['rank']=rank
fix['checkpoint_audit']=audits;save(out/'selection_precision_fix.json',fix)
inc=sel['incumbent'];inc['raw_float_rank']=inc['rank'];inc['rank']=list(selection_key(inc['validation'],cfg))
best=min(sel['candidates'],key=lambda c:c['rank']);sel['selected']=best
sel.update(promote_by_validation=best['rank'][0]==0 and best['rank']<inc['rank'],timestamp_utc=datetime.now(timezone.utc).isoformat(),precision_fix_sha256=file_hash(out/'selection_precision_fix.json'))
archive=out/'selection_before_precision_fix.json'
if archive.exists():raise FileExistsError('Precision correction already frozen')
(out/'selection.json').rename(archive);save(out/'selection.json',sel)
print('CHECKPOINTS UNCHANGED',audits,'PROMOTE',sel['promote_by_validation'],'SELECTED',best['artifact'],flush=True)
