"""AI (Codex): continue saved evaluation, never regenerate spectra or completed outputs."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil

OUT=Path(__file__).resolve().parent
assert not (OUT/'continue_evaluation.py').exists()
assert not (OUT/'summary.json').exists()
assert (OUT/'diagnostic_spectra.npz').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checkpoint=OUT/'checkpoint_after_progress_assertion'
checkpoint.mkdir()
names=('per_reading_metrics.csv','diagnostic_partial.json','evaluation_run.log','evaluation_process.log','evaluate_after_q.py')
for name in names:shutil.copyfile(OUT/name,checkpoint/name)
rows=list(csv.DictReader((checkpoint/'per_reading_metrics.csv').open(encoding='utf-8-sig')))
families=list(dict.fromkeys(r['family'] for r in rows))
assert len(rows)==1200 and len(families)==2
for family in families:
    assert len([r for r in rows if r['family']==family])==600
protected={name:sha(OUT/name) for name in ('diagnostic_spectra.npz','generation_provenance.json','model_freeze.json','full_model_freeze.json','q_decision_before_evaluation.json','scratch_probe_functional_check.json')}
protected.update({str(p.relative_to(OUT)):sha(p) for p in checkpoint.iterdir()})
(OUT/'continuation_before.json').write_text(json.dumps(dict(author='AI（Codex）',
    cause='Over-broad write protection incorrectly rejected the second streaming progress update; numerical output was saved first.',
    completed_families=families,completed_spectra=300,completed_readings=1200,
    saved_evaluation_seed=22658,generate_evaluation_again=False,
    reuse_completed_predictions=True,replay_completed_measurement_rng_only=True,
    protected_files=protected),ensure_ascii=False,indent=2),encoding='utf-8')
src=(OUT/'evaluate_after_q.py').read_text(encoding='utf-8')
src=src.replace('/evaluate_after_q.py --diag-seed 22658','/continue_evaluation.py --diag-seed 22658',1)
src=src.replace("if name == 'existing_files_before.json':", "if name == 'diagnostic_partial.json':\n        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')\n        return\n    if name == 'existing_files_before.json':",1)
src=src.replace("path = OUT / 'evaluation_existing_files_before.json'", "path = OUT / 'continuation_existing_files_before.json'",1)
src=src.replace("assert not (OUT / 'diagnostic_spectra.npz').exists(), 'Never regenerate the evaluation spectra.'", "assert (OUT / 'diagnostic_spectra.npz').exists(), 'Continuation requires the already-generated spectra.'",1)
src=src.replace("assert not (OUT / 'generation_provenance.json').exists(), 'Evaluation already started.'", "assert (OUT / 'generation_provenance.json').exists(), 'Original generation provenance required.'",1)
src=src.replace("assert not (OUT / 'evaluation_run.log').exists(), 'Evaluation process already started.'", "assert not (OUT / 'continue_run.log').exists(), 'Continuation already started.'",1)
src=src.replace("(OUT / 'evaluation_run.log').open", "(OUT / 'continue_run.log').open",1)
start=src.index('# Probe only already-used validation data.')
end=src.index("PHASE='overlap_audit_only'",start)
src=src[:start]+'''# Existing validation probe evidence was completed before original generation.
probe=json.loads((OUT/'scratch_probe_functional_check.json').read_text(encoding='utf-8'))
assert probe['status']=='passed' and not probe['evaluation_generated']
assert all(v==0 for v in probe['maximum_absolute_output_difference'].values())
assert numeric_state(roots)==full_freeze
log('Loaded previously passed validation-only scratch probe; all registered inference state matches again.')

'''+src[end:]
src=src.replace("log('Generating 22658 exactly once: 150 spectra per family, four real readings.')", "log('Continuing saved 22658 spectra; no evaluation generation call.')",1)
src=src.replace('dsets=AD.all_fams(DSEED,150,P,True)', "archive=np.load(OUT/'diagnostic_spectra.npz')\ndsets={k:archive[k] for k in archive.files}\nassert len(dsets)==26 and all(s.shape==(150,176) for s in dsets.values())",1)
start=src.index("np.savez_compressed(OUT/'diagnostic_spectra.npz'")
end=src.index('rows=[]; samples=[]',start)
src=src[:start]+'''assert not any(c['generator']=='all_fams' and c['seed']==DSEED for c in generation_calls)
dump('continuation_generation_calls.json',dict(calls=generation_calls,
    evaluation_loaded_from_saved_npz=True,new_evaluation_generation_calls=0))
'''+src[end:]
src=src.replace('rows=[]; samples=[]', '''checkpoint=OUT/'checkpoint_after_progress_assertion'
saved_samples=list(csv.DictReader((checkpoint/'per_reading_metrics.csv').open(encoding='utf-8-sig')))
completed_families=list(dict.fromkeys(s['family'] for s in saved_samples))
assert len(saved_samples)==1200 and list(dsets)[:2]==completed_families
metric_keys=('B_raw','B_consistent','A','fusion_pre_q','final','hard_raw','hard_consistent')
samples=[dict(family=r['family'],spectrum=int(r['spectrum']),draw=int(r['draw']),
    **{key:float(r[key]) for key in (*metric_keys,'wA')}) for r in saved_samples]
rows=[]
for family in completed_families:
    readings=[r for r in samples if r['family']==family]
    assert len(readings)==600
    choose=np.array([r['wA']>.5 for r in readings])
    ea=np.array([r['A'] for r in readings])
    row=dict(family=family,n_spectra=150,n_readings=600,
        **{key:float(np.mean([r[key] for r in readings])) for key in metric_keys},
        wA_mean=float(np.mean([r['wA'] for r in readings])),A_weight_gt_half=float(choose.mean()))
    for b in ('B_raw','B_consistent'):
        eb=np.array([r[b] for r in readings]); correct=choose==(ea<eb); trueA=choose&(ea<eb)
        row.update({f'route_vs_{b}_accuracy':float(correct.mean()),
            f'route_vs_{b}_correct_n':int(correct.sum()),
            f'route_vs_{b}_A_error_when_correct':float(ea[correct].mean()) if correct.any() else None,
            f'route_vs_{b}_correct_A_n':int(trueA.sum()),
            f'route_vs_{b}_A_error_when_correct_A':float(ea[trueA].mean()) if trueA.any() else None})
    rows.append(row)
dump('continuation_loaded_metrics.json',dict(completed_families=completed_families,
    readings=len(samples),source_sha256=hashlib.sha256((checkpoint/'per_reading_metrics.csv').read_bytes()).hexdigest()))''',1)
src=src.replace("family=k.replace('留出·',''); Y=AD.meas(S.T,g,'real'); M._TR.clear(); o=decode(Y)", """family=k.replace('留出·',''); Y=AD.meas(S.T,g,'real'); M._TR.clear()
    if family in completed_families:
        log('Preserved completed outputs; replayed measurement RNG only: '+family)
        continue
    o=decode(Y)""",1)
src=src.replace('prior_q_process_seconds=796.1,peak_cuda_GB=', 'prior_q_process_seconds=796.1,prior_evaluation_last_logged_seconds=498.8,continued_existing_spectra=True,completed_family_predictions_reused=completed_families,peak_cuda_GB=',1)
ast.parse(src)
(OUT/'continue_evaluation.py').write_text(src,encoding='utf-8')
print(json.dumps(dict(prepared=True,completed_families=families,completed_readings=len(rows),evaluation_regeneration=False),ensure_ascii=False))
