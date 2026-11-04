"""AI (Codex): preserve completed q evidence and diagnose a mutable scratch buffer."""
from pathlib import Path
import ast
import hashlib
import json

OUT=Path(__file__).resolve().parent
assert not (OUT/'evaluate_after_q.py').exists()
assert not (OUT/'diagnostic_spectra.npz').exists()
q=json.loads((OUT/'q_ablation/summary.json').read_text(encoding='utf-8'))
decision=json.loads((OUT/'q_decision_before_evaluation.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OUT/'q_ablation/summary.json')==decision['q_summary_sha256']
assert q['retain_q']==decision['retain_q']==False
protected={str(p.relative_to(OUT)):sha(p) for p in (OUT/'q_ablation').rglob('*') if p.is_file()}
protected.update({name:sha(OUT/name) for name in ('q_decision_before_evaluation.json','resume_run.log','resume_process.log','resume_diagnose_b1.py')})
(OUT/'completed_q_before_evaluation.json').write_text(json.dumps(dict(
    author='AI（Codex）', protected_files=protected, evaluation_generated=False,
    reason='Prior strict state assertion stopped before evaluation; inspect GF._m2 batch scratch only, preserving all fitted inference parameters and complete q decision.'),ensure_ascii=False,indent=2),encoding='utf-8')
src=(OUT/'resume_diagnose_b1.py').read_text(encoding='utf-8')
prefix=src[:src.index('# q decision is made on 31438')]
prefix=prefix.replace('/resume_diagnose_b1.py --diag-seed 22658','/evaluate_after_q.py --diag-seed 22658',1)
prefix=prefix.replace("assert not (OUT / 'q_ablation').exists(), 'Full q ablation must start in a fresh directory.'", "assert (OUT / 'q_ablation/summary.json').exists(), 'Completed immutable q decision required.'",1)
prefix=prefix.replace("assert not (OUT / 'resume_run.log').exists(), 'Resume run already started.'", "assert not (OUT / 'evaluation_run.log').exists(), 'Evaluation process already started.'",1)
prefix=prefix.replace("path = OUT / 'resume_existing_files_before.json'", "path = OUT / 'evaluation_existing_files_before.json'",1)
prefix=prefix.replace("(OUT / 'resume_run.log').open", "(OUT / 'evaluation_run.log').open",1)
tail=src[src.index("PHASE='overlap_audit_only'"):]
tail=tail.replace("post=numeric_state(roots)", "assert_frozen_with_scratch_audit('after_22658')\npost=numeric_state(roots)",1)
tail=tail.replace("seconds=time.time()-start,peak_cuda_GB=", "seconds=time.time()-start,prior_q_process_seconds=796.1,peak_cuda_GB=",1)
middle='''# q is already complete and immutable. Recompute only its statistics, never its predictions.
QOUT=OUT/'q_ablation'
qdecision=json.loads((QOUT/'summary.json').read_text(encoding='utf-8'))
qr=list(csv.DictReader((QOUT/'family_metrics.csv').open(encoding='utf-8-sig')))
qs=list(csv.DictReader((QOUT/'per_reading_metrics.csv').open(encoding='utf-8-sig')))
assert len(qr)==26 and len(qs)==4160
for r in qr:
    for k in ('q_off','q_on'):r[k]=float(r[k])
recomputed=q_statistics(qr,qs)
assert all(qdecision[k]==v for k,v in recomputed.items())
previous_decision=json.loads((OUT/'q_decision_before_evaluation.json').read_text(encoding='utf-8'))
assert hashlib.sha256((QOUT/'summary.json').read_bytes()).hexdigest()==previous_decision['q_summary_sha256']
assert qdecision['retain_q']==previous_decision['retain_q']
log('Validation-only q decision: retain_q='+str(qdecision['retain_q'])+'; loaded immutable complete 31438 evidence, no re-selection.')

# GaussCal._solve writes _m2 on every call. It is a batch Mahalanobis-distance
# scratch output, not a prior/normalizer/weight. Retain strict checks on all other
# entries; record raw before/after evidence before restoring this scratch value.
gate_closure=dict(zip(closure['base'].__code__.co_freevars,(c.cell_contents for c in closure['base'].__closure__)))
gate_fn=gate_closure['gate_of']
gate_members=dict(zip(gate_fn.__code__.co_freevars,(c.cell_contents for c in gate_fn.__closure__)))
GF=gate_members['GF']
scratch_original=GF._m2.clone()
scratch_key='model.decoder.closure.base.closure.gate_of.closure.GF._m2'
assert scratch_key in full_freeze['entries']
scratch_audits=[]
def assert_frozen_with_scratch_audit(stage):
    actual=numeric_state(roots)
    differences={k:dict(before=full_freeze['entries'].get(k),after=actual['entries'].get(k))
        for k in full_freeze['entries'].keys()|actual['entries'].keys()
        if full_freeze['entries'].get(k)!=actual['entries'].get(k)}
    assert set(differences)<={scratch_key},differences
    scratch_audits.append(dict(stage=stage,raw_state_sha256=actual['sha256'],differences=differences,
        parameter_or_prior_changes=0,scratch_restored_after_comparison=True))
    dump('scratch_state_audit_'+stage+'.json',scratch_audits[-1])
    GF._m2=scratch_original.clone()
    restored=numeric_state(roots)
    assert restored==full_freeze,'Restoring scratch failed to recover the exact registered state.'
    return differences

# Probe only already-used validation data. Two identical decodes establish that
# restoring the scratch cannot change predictions. No 22658 spectrum exists yet.
PHASE='validation_scratch_probe_only'
probe_S=next(iter(qsets.values()))
probe_Y=AD.meas(probe_S.T,np.random.default_rng(9090),'real')
M._TR.clear(); probe1=decode(probe_Y)
first_difference=assert_frozen_with_scratch_audit('validation_probe_1')
assert set(first_difference)=={scratch_key},'Expected scratch-only update was not reproduced.'
M._TR.clear(); probe2=decode(probe_Y)
probe_max={k:float(np.max(np.abs(probe1[k]-probe2[k]))) for k in ('gate','t3','t3c','A11','w')}
assert all(v==0 for v in probe_max.values()),probe_max
assert_frozen_with_scratch_audit('validation_probe_2')
dump('scratch_probe_functional_check.json',dict(status='passed',allowed_data_seed=31438,
    maximum_absolute_output_difference=probe_max,excluded_parameter_paths=[],
    scratch_path=scratch_key,evaluation_generated=False,original_full_state_recovered=True))
log('Scratch probe passed: only GF._m2 changes; restored cache gives identical A/B/fusion/weight outputs; all registered parameters unchanged.')
'''
result=prefix+middle+tail
ast.parse(result)
(OUT/'evaluate_after_q.py').write_text(result,encoding='utf-8')
print('Prepared evaluation recovery: q evidence locked; strict scratch-only audit required before any 22658 generation.')
