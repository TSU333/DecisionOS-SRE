"""Evaluate the frozen challenger; all historical test cases are regression only."""
from pathlib import Path
import copy,gc,subprocess,sys
from collections import Counter
from decisionos_sre.common import read,save,file_hash
from decisionos_sre.pipeline import calibrate,policy,evaluate,benchmark,finish_rows,write_evaluation
from decisionos_sre.training import load_checkpoint,predict,prediction_summary
from decisionos_sre.data import load_split

out=Path('outputs/round6');selection=read(out/'selection.json');folder=Path(selection['selected']['artifact'])
assert file_hash(out/'protocol.json')==selection['protocol_sha256']
meta=read(folder/'metadata.json');cfg=meta['config']
assert file_hash(folder/'checkpoint.pt')==selection['selected']['binding']['checkpoint_sha256']
assert all(file_hash(p)==sha for p,sha in meta['code_state']['source_hashes'].items())
print('CALIBRATION CPU START',flush=True);cal=calibrate(folder,'cpu');print('CALIBRATION DONE',[(h,cal[h]['temperature']) for h in ['root','fault']],flush=True)
print('GATE CPU START',flush=True);pol=policy(folder,'cpu');print('GATE DONE',pol['status'],pol['selection'],flush=True)
gates=read(folder/'gate_predictions.json');thresholds=[]
for threshold in sorted({r['routing_score'] for r in gates}):
    rows=[r for r in gates if r['routing_score']>=threshold and r['evidence_usable'] and len(r['candidate_ids'])>1]
    n=len(rows);errors=sum(not r['joint_correct'] for r in rows)
    if n>=30:thresholds.append({'threshold':threshold,'n':n,'errors':errors,'risk':errors/n})
save(out/'gate_diagnostics.json',{'summary':prediction_summary(gates),'best_observed_risk_with_at_least_30':min(thresholds,key=lambda x:(x['risk'],-x['n'])) if thresholds else None,'constraint':{'risk':.05,'n':30},'policy':pol,'note':'Reused gate set, selected empirical risk only; not an independent safety confirmation.'})
print('SS REGRESSION CPU START',flush=True);m=evaluate(folder,'cpu');print('SS REGRESSION DONE',m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
model,tok,ser,meta=load_checkpoint(folder,'cpu')
for split in ['regression_re1_ob','regression_re2_ob']:
    examples=load_split(cfg['data_dir'],split);rows=predict(model,tok,ser,examples,'cpu',split)
    m=write_evaluation(finish_rows(rows,cal,pol,meta['binding']),out/split)
    print(split,m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
    if split=='regression_re2_ob' and cfg.get('trace_features'):
        altered=copy.deepcopy(examples)
        for e in altered:
            e.input.evidence.traces=None;e.input.modality_availability.traces=False
        masked=predict(model,tok,ser,altered,'cpu',split)
        m=write_evaluation(finish_rows(masked,cal,pol,meta['binding']),out/'mask_traces_ob')
        print('MASK TRACES OB',m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
# Check actual trace-containing OB examples, since SS has no source traces.
if cfg.get('trace_features'):
    originals=load_split(cfg['data_dir'],'regression_re2_ob')
    for variant in ['reverse_candidates','consistent_service_rename']:
        altered=copy.deepcopy(originals)
        for ex in altered:
            if variant=='reverse_candidates':ex.input.candidates.reverse()
            else:
                names=sorted({c.candidate_id for c in ex.input.candidates}|{m.service for m in ex.input.evidence.metrics}|{t.service for t in ex.input.evidence.traces or []})
                mapping={name:f'component_{i:02d}' for i,name in enumerate(names)}
                for c in ex.input.candidates:c.candidate_id=mapping[c.candidate_id];c.display_name=c.candidate_id
                for m in ex.input.evidence.metrics:m.service=mapping[m.service]
                for t in ex.input.evidence.traces or []:t.service=mapping[t.service]
                ex.targets.root_cause.value=mapping[ex.targets.root_cause.value]
        rr=predict(model,tok,ser,altered,'cpu','regression_re2_ob')
        m=write_evaluation(finish_rows(rr,cal,pol,meta['binding']),out/(variant+'_ob'))
        print(variant+'_ob',m['root']['acc_at_1'],m['fault']['accuracy'],m['joint_accuracy'],flush=True)
del model,tok,ser;gc.collect()
print('BENCHMARK CPU START',flush=True);bench=benchmark(folder,cfg);print('BENCHMARK DONE P95',bench['end_to_end']['p95_ms'],flush=True)
subprocess.run([sys.executable,'scripts/verify_integration.py',str(folder)],check=True)
audits=[]
train=load_split(cfg['data_dir'],'train');byid={e.original_run_id:e.source_metadata['dataset_suite'] for e in train}
for c in selection['candidates']:
    draws=read(Path(c['artifact'])/'training_draws.json');ids=[x for row in draws['draws'] for x in row['run_ids']]
    assert set(ids)<=set(byid)
    audits.append({'artifact':c['artifact'],'unique_train_cases':len(set(ids)),'total_draws':len(ids),'cohort_draws':dict(Counter(byid[x] for x in ids)),'outside_train':0,'checkpoint_draws':sum(len(row['run_ids']) for row in draws['draws'] if row['epoch']<=c['best_epoch'])})
save(out/'training_draw_audit.json',audits)
print('ALL EVALUATION COMPLETED',flush=True)
