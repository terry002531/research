"""Verify artifacts, source function preservation, file scope and UTF-8."""
from pathlib import Path
import ast, json, csv, hashlib
import numpy as np

OUT=Path(__file__).resolve().parent; WORK=OUT.parents[2]
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
j=load('summary.json'); families=table('family_metrics.csv'); samples=table('per_reading_metrics.csv'); gaps=table('train_validation_gap.csv')
assert len(families)==26 and len(samples)==3120 and len(gaps)==26
assert len({(r['family'],r['spectrum'],r['draw']) for r in samples})==3120
keys=('B_raw','B_consistent','A','fusion_pre_q','final')
for r in families:
    assert int(r['n_spectra'])==30 and int(r['n_readings'])==120
    rr=[s for s in samples if s['family']==r['family']]
    assert len(rr)==120
    for k in keys:
        assert np.isfinite(float(r[k])) and float(r[k])>=0
        assert abs(np.mean([float(s[k]) for s in rr])-float(r[k]))<1e-12
    assert abs(float(r['B_minus_A'])-(float(r['B_raw'])-float(r['A'])))<1e-12
    assert abs(float(r['B_vs_A_percent'])-100*(float(r['B_raw'])/float(r['A'])-1))<1e-10
    assert 0<=float(r['wA_mean'])<=1
for k in keys:
    gm=float(np.exp(np.log([float(r[k]) for r in families]).mean()))
    assert abs(gm-j['geometric_means'][k])<1e-12
loss_component_residuals=[]
for r in gaps:
    for label in ('train','validation','diagnostic'):
        a=float(r[label+'_value_mse']); b=float(r[label+'_weighted_diff_mse']); c=float(r[label+'_training_objective'])
        loss_component_residuals.append(abs(a+b-c))
        assert np.isclose(a+b,c,rtol=5e-7,atol=1e-8), 'Allow float32 rounding of separate means (about a few ulps).'
assert not load('overlap.json')['diagnostic_vs_used']
assert not load('existing_file_changes.json')

checked=[]
for rel,sha in j['architecture']['source_sha256'].items():
    assert hashlib.sha256((WORK/rel).read_bytes()).hexdigest()==sha
    checked.append(rel)
bfile=WORK/'算法对照/45_Uddin2024_真实R矩阵/results/a15_nets/cnet0.pt'
assert hashlib.sha256(bfile.read_bytes()).hexdigest()==j['architecture']['B_sha256']
checked.append(str(bfile.relative_to(WORK)))

# Prove that memory-only adaptation preserved each numerical function body.
script=ast.parse((OUT/'diagnose_b1.py').read_text(encoding='utf-8'))
san=next(n for n in script.body if isinstance(n,ast.ClassDef) and n.name=='Sanitize')
namespace=dict(ast=ast,OUT=OUT,P0_FIXED=1.0542339586846165e-05,edits=[])
exec(compile(ast.Module(body=[san],type_ignores=[]),'<sanitizer>','exec'),namespace)
function_checks={}
for name in ('a15_three','a11_peakfix','a11_device'):
    src=(WORK/'算法对照'/(name+'.py')).read_text(encoding='utf-8')
    original=ast.parse(src); adapted=namespace['Sanitize'](name).visit(ast.parse(src))
    funcs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=funcs(original),funcs(adapted)
    assert a==b, f'Numerical function changed: {name}'
    function_checks[name]=len(a)
    if name=='a15_three':
        cls=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='CNet'),include_attributes=False)
        assert cls(original)==cls(adapted), 'CNet class changed during import adaptation'

before=load('existing_files_before.json'); changes={}
for rel,st in before.items():
    p=WORK/rel
    now=None if not p.exists() else dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    if st!=now:changes[rel]=dict(before=st,after=now)
assert not changes,changes
new=[]
for p in WORK.rglob('*'):
    if not p.is_file() or p.is_relative_to(OUT):continue
    rel=str(p.relative_to(WORK))
    if rel not in before:new.append(rel)
assert new==['A18\\HANDOFF_A18-B1.md'] or new==['A18/HANDOFF_A18-B1.md'],new
hand=WORK/'A18/HANDOFF_A18-B1.md'
txt=hand.read_text(encoding='utf-8')
assert txt.count('## ')>=8 and all(x in txt for x in ('AI（Codex）','仿真','已核实','推算','推测','方案一','方案二','方案三'))
assert '\ufffd' not in txt and '???' not in txt and sum('\u4e00'<=x<='\u9fff' for x in txt)>1500
assert not [(i,ord(c)) for i,c in enumerate(txt) if ord(c)<32 and c!='\n'], 'Unexpected control characters in Markdown/LaTeX.'
assert r'\tfrac12' in txt and r'\frac{' in txt and r'\rangle' in txt
for p in OUT.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
result=dict(status='PASS',family_count=26,reading_count=3120,existing_files_checked=len(before),
            existing_changes=changes,allowed_new_file_outside_diag=new,source_hashes_checked=checked,
            numerical_functions_preserved=function_checks,utf8_chinese_characters=sum('\u4e00'<=x<='\u9fff' for x in txt),
            loss_component_max_residual=max(loss_component_residuals),loss_component_rtol=5e-7,
            handoff_sha256=hashlib.sha256(hand.read_bytes()).hexdigest(),
            data_integrity='per-reading means, family differences, geometric means, loss components, overlap, weight bounds verified')
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
