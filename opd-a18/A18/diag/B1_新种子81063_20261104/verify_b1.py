"""Independent arithmetic/statistical checks, frozen model, UTF-8 and file scope."""
from pathlib import Path
import csv,json,hashlib,ast,datetime,re
import numpy as np

OUT=Path(__file__).resolve().parent; WORK=OUT.parents[2]; OLD=OUT.parent/'B1_诊断_20261104'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def truth(value):return value is True or value=='True'

j=load('summary.json'); rows=table('family_metrics.csv'); samples=table('per_reading_metrics.csv')
assert len(rows)==26 and len(samples)==3120
assert len({(r['family'],r['spectrum'],r['draw']) for r in samples})==3120
assert j['diagnostic']==dict(seed=81063,actual_five_family_seed=81064,actual_CIE_jitter_seed=81396,
    n_per_family=30,draws=4,jitter=True,mode='real',noise_seed=4321)
keys=('B_raw','B_consistent','A','fusion_pre_q','final')
recomputed={}; all_arrays={}
for row in rows:
    name=row['family']; assert '留出' not in name
    assert int(row['n_spectra'])==30 and int(row['n_readings'])==120
    ss=sorted([s for s in samples if s['family']==name],key=lambda s:(int(s['spectrum']),int(s['draw'])))
    assert len(ss)==120
    arrays={k:np.array([float(s[k]) for s in ss]).reshape(30,4) for k in keys}
    all_arrays[name]=arrays
    for k,a in arrays.items():
        assert np.all(np.isfinite(a)) and np.all(a>0)
        assert abs(float(a.mean())-float(row[k]))<1e-12
    assert abs(float(row['B_minus_A'])-(arrays['B_raw'].mean()-arrays['A'].mean()))<1e-12
    assert abs(float(row['wA_mean'])-np.mean([float(s['wA']) for s in ss]))<1e-12
    assert all(0<=float(s['wA'])<=1 for s in ss)
    for baseline in ('B_raw','B_consistent'):
        # Independently construct all resampled paired spectra in one array.
        lr=(np.log(arrays['A'])-np.log(arrays[baseline])).sum(axis=1)/4
        ids=np.random.default_rng(1901122).integers(0,30,size=(200000,30))
        bootstrap=np.mean(np.take(lr,ids),axis=1)
        lo,hi=np.percentile(bootstrap,[100*.05/26/2,100*(1-.05/26/2)])
        estimate=float(lr.sum()/30); ratio=np.exp(estimate)
        for field,expected in dict(log_ratio=estimate,ratio=ratio,log_ci_low=lo,log_ci_high=hi,
                                  ratio_ci_low=np.exp(lo),ratio_ci_high=np.exp(hi),
                                  ratio_of_arithmetic_means=arrays['A'].mean()/arrays[baseline].mean()).items():
            assert abs(float(row[f'A_vs_{baseline}_{field}'])-expected)<1e-12,(name,baseline,field)
        assert truth(row[f'A_vs_{baseline}_A_stable'])==bool(hi<0 and ratio<=.95)
        assert truth(row[f'A_vs_{baseline}_B_stable'])==bool(lo>0 and ratio>=1/.95)
        assert int(row['bootstrap_n'])==200000 and int(row['bootstrap_seed'])==1901122
        assert abs(float(row['bonferroni_alpha'])-.05/26)<1e-15
        recomputed[(name,baseline)]=(lo,hi,ratio)

for k in keys:
    expected=float(np.exp(np.mean(np.log([float(r[k]) for r in rows]))))
    assert abs(expected-j['geometric_means'][k])<1e-12
for baseline in ('B_raw','B_consistent'):
    for label,column in [('A_stable','A_stable'),('B_stable','B_stable')]:
        expected=[r['family'] for r in rows if truth(r[f'A_vs_{baseline}_{column}'])]
        assert expected==j['statistics'][f'{label}_vs_{baseline}']
improve=100*(1-j['geometric_means']['final']/j['geometric_means']['B_raw'])
assert abs(improve-j['final_improvement_over_old_B_raw_percent'])<1e-12
assert j['evaluation_only'] and j['old_seed']==31451
assert not load('overlap.json')['diagnostic_vs_used']
assert 'SS调参集31451' in load('overlap.json')['compared_sets']
with np.load(OUT/'diagnostic_spectra.npz') as z:
    assert len(z.files)==26
    assert all(z[name].shape==(30,176) and np.all(np.isfinite(z[name])) for name in z.files)

freeze=load('model_freeze.json'); post=load('model_post_evaluation.json')
assert freeze['model_state_sha256']==post['model_state_sha256'] and post['unchanged']
assert freeze['K']==32 and freeze['ss_alpha']==.75 and freeze['kappa']['t3']==.25
assert freeze['diagnostic_not_generated_yet'] and not freeze['evaluation_seed_used_for_fitting']
assert freeze['source_and_B_match_old_B1'] and freeze['old_validation_metrics_match_atol']==1e-9
assert sha(OLD/'q_validation_only.pt')==j['q_checkpoint_sha256']==freeze['q_checkpoint_sha256']
provenance=load('generation_provenance.json')
assert provenance['actual_all_fams_seed']==81063
assert all(c['seed'] not in (6161,4242) for c in provenance['calls'])
assert all(c['phase']=='evaluation_only' for c in provenance['calls'] if c['seed'] in (81063,81064))
assert sum(c['generator']=='all_fams' and c['seed']==81063 for c in provenance['calls'])==1
assert any(c['generator']=='gen5' and c['seed']==81064 and c['n']==30 for c in provenance['calls'])
log=(OUT/'run.log').read_text(encoding='utf-8')
assert log.index('Model frozen;')<log.index('同分布五类: B ')
scope=load('scope_review.json') if (OUT/'scope_review.json').exists() else None
assert 'Complete:' in log or (scope and scope['status']=='external_concurrent_change_reviewed' and '[post-run scope review]' in log)

old_hashes=load('old_artifacts_before.json')
assert len(old_hashes)>=25
for rel,expected in old_hashes.items():assert sha(WORK/rel)==expected,rel
for rel,expected in load('source_manifest.json')['sources'].items():assert sha(WORK.parent/rel)==expected,rel
for rel,expected in j['architecture']['source_sha256'].items():assert sha(WORK/rel)==expected,rel
b=WORK/'算法对照/45_Uddin2024_真实R矩阵/results/a15_nets/cnet0.pt'
assert sha(b)==j['architecture']['B_sha256']
assert load('existing_file_changes.json')==({} if scope is None else {rel:dict(before=v['before'],after=v['after']) for rel,v in scope['external_changes'].items()})

# Compare original and adapted numerical function ASTs, without importing modules.
script=ast.parse((OUT/'diagnose_b1.py').read_text(encoding='utf-8'))
san=next(n for n in script.body if isinstance(n,ast.ClassDef) and n.name=='Sanitize')
ns=dict(ast=ast,OUT=OUT,P0_FIXED=1.0542339586846165e-05,edits=[])
exec(compile(ast.Module(body=[san],type_ignores=[]),'<sanitizer>','exec'),ns)
function_counts={}
for name in ('a15_three','a11_peakfix','a11_device'):
    text=(WORK/'算法对照'/(name+'.py')).read_text(encoding='utf-8')
    original=ast.parse(text); adapted=ns['Sanitize'](name).visit(ast.parse(text))
    funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert funcs(original)==funcs(adapted),name
    function_counts[name]=len(funcs(original))

hand=WORK/'A18/HANDOFF_A18-B1.md'; before=(OUT/'HANDOFF_before.md').read_bytes()
report=(OUT/'report_81063.md').read_bytes()
assert hand.read_bytes()==before+b'\n\n'+report
txt=report.decode('utf-8')
assert all(word in txt for word in ('AI（Codex）','81063','31451','SS 调参谱','Bonferroni','仿真','新 B'))
assert sum('\u4e00'<=c<='\u9fff' for c in txt)>1500
assert '\ufffd' not in txt and '???' not in txt
assert not any(ord(c)<32 and c not in '\n\r\t' for c in txt)
assert re.search(r'(?m)^\|.*\|\n\n\|',txt) is None, 'Blank line splits a Markdown table.'
for p in OUT.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')

baseline=load('existing_files_before.json'); allowed_changes=[]
for rel,st in baseline.items():
    p=WORK/rel
    assert p.exists(),rel
    current=dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    if current!=st:
        if scope and rel in scope['external_changes']:
            assert sha(p)==scope['external_changes'][rel]['current_sha256'],rel
        else:
            assert p==hand,rel
        allowed_changes.append(rel)
new_outside=[]
for p in WORK.rglob('*'):
    if not p.is_file() or p.is_relative_to(OUT):continue
    rel=str(p.relative_to(WORK))
    if rel not in baseline:
        assert p.parent==WORK/'memory' and '81063' in p.name,rel
        new_outside.append(rel)

result=dict(status='PASS',verified_at=datetime.datetime.now().astimezone().isoformat(),
    family_count=26,spectrum_count=780,reading_count=3120,actual_diagnostic_seed=81063,
    paired_bootstrap_recomputed_comparisons=len(recomputed),bootstrap_replicates=200000,
    correction='Bonferroni, 26 families per baseline',old_B1_files_unchanged=len(old_hashes),
    source_files_unchanged=len(load('source_manifest.json')['sources']),
    original_handoff_bytes_preserved=len(before),handoff_sha256=sha(hand),
    model_states_unchanged=True,numerical_functions_preserved=function_counts,
    observed_existing_changes=allowed_changes,new_files_outside_diagnostic=new_outside,
    external_scope_review=scope,
    A_stable_vs_B_raw=j['statistics']['A_stable_vs_B_raw'],
    A_stable_vs_B_consistent=j['statistics']['A_stable_vs_B_consistent'],
    old_B_raw_reference_improvement_percent=improve,
    new_B_stopping_rule='not evaluated; no new B implemented')
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
