"""AI (Codex): resume the preregistered run without overwriting frozen evidence."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT / 'resume_before.json').exists(), 'Resume already prepared; inspect before retrying.'
for name in ('summary.json', 'diagnostic_spectra.npz', 'generation_provenance.json', 'q_ablation'):
    assert not (OUT / name).exists(), name
freeze = json.loads((OUT / 'full_model_freeze.json').read_text(encoding='utf-8'))
assert freeze['sha256'] == '59f19648abb6ceab81cec07ca40d68c5ce188b38988d923738405e4c749976ce'
raw = json.dumps(freeze['entries'], sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
assert hashlib.sha256(raw).hexdigest() == freeze['sha256']
before = dict(author='AI（Codex）', recorded=datetime.datetime.now().astimezone().isoformat(),
    full_inference_state_sha256=freeze['sha256'],
    local_existing_files={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()},
    outside_existing_files={str(p.relative_to(WORK)): sha(p)
        for root in (WORK/'A18', WORK/'算法对照', WORK/'A17')
        for p in root.rglob('*') if p.is_file() and not p.is_relative_to(OUT)},
    protected_freeze_files=['model_freeze.json', 'full_model_freeze.json'],
    partial_q_directory='q_ablation_partial_1218',
    decision='Restart complete q ablation on 31438, then generate 22658 once; no new B or model selection.')
(OUT / 'resume_before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding='utf-8')
src = (OUT / 'diagnose_b1.py').read_text(encoding='utf-8')
src = src.replace('Run: py -I -X utf8 ', 'Run: py -I -X utf8 ', 1)
src = src.replace('/diagnose_b1.py --diag-seed 22658', '/resume_diagnose_b1.py --diag-seed 22658', 1)
old = """def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')"""
new = """def dump(name, obj):
    path = OUT / name
    if name == 'existing_files_before.json':
        path = OUT / 'resume_existing_files_before.json'
    elif path.exists():
        # All pre-interruption evidence is immutable. Compare, never replace it.
        assert json.loads(path.read_text(encoding='utf-8')) == obj, 'Pre-existing artifact differs: ' + name
        return
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')"""
assert old in src
src = src.replace(old, new, 1)
src = src.replace("(OUT / 'run.log').open", "(OUT / 'resume_run.log').open")
src = src.replace("assert not (OUT / 'summary.json').exists(), 'Completed run exists; do not overwrite.'", """assert not (OUT / 'summary.json').exists(), 'Completed run exists; do not overwrite.'
assert not (OUT / 'diagnostic_spectra.npz').exists(), 'Never regenerate the evaluation spectra.'
assert not (OUT / 'generation_provenance.json').exists(), 'Evaluation already started.'
assert not (OUT / 'q_ablation').exists(), 'Full q ablation must start in a fresh directory.'
assert not (OUT / 'resume_run.log').exists(), 'Resume run already started.'""", 1)
src = src.replace("dump('full_model_freeze.json', full_freeze)", """dump('full_model_freeze.json', full_freeze)
assert full_freeze['sha256'] == '59f19648abb6ceab81cec07ca40d68c5ce188b38988d923738405e4c749976ce'
log('Resumed inference state equals the 15:16 registered full freeze; original freeze files preserved.')""", 1)
# The decision evidence is persisted and fingerprinted before evaluation generation.
src = src.replace("log('Validation-only q decision: retain_q='+str(qdecision['retain_q']))", """dump('q_decision_before_evaluation.json', dict(
    recorded_at=time.time(), retain_q=qdecision['retain_q'],
    q_summary_sha256=hashlib.sha256((QOUT/'summary.json').read_bytes()).hexdigest(),
    evaluation_spectra_exist=(OUT/'diagnostic_spectra.npz').exists(),
    evaluation_generation_called=any(c['generator']=='all_fams' and c['seed']==DSEED for c in generation_calls)))
assert not (OUT/'diagnostic_spectra.npz').exists()
log('Validation-only q decision: retain_q='+str(qdecision['retain_q']))""", 1)
ast.parse(src)
(OUT / 'resume_diagnose_b1.py').write_text(src, encoding='utf-8')
print(json.dumps(dict(status='prepared', full_state_sha256=freeze['sha256'],
    original_files=len(before['local_existing_files']), protected_external_files=len(before['outside_existing_files'])), ensure_ascii=False))
