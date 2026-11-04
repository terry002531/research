"""Independent output validation only; never imports the reconstruction code."""
from pathlib import Path
import csv, json, hashlib, math
import numpy as np
O=Path(__file__).resolve().parent
W=O.parents[2]
CHECKS={}
def check(name,condition):
    CHECKS[name]=bool(condition)
    if not condition: raise AssertionError(name)
def readj(name): return json.loads((O/name).read_text(encoding='utf-8-sig'))
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def close(a,b): return bool(np.allclose(a,b,rtol=1e-7,atol=1e-9))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tier(lo,hi,blo,bhi):
    return 'B 稳定胜' if bhi<0 else 'A 稳定胜' if blo>0 else '边界·未校正显著' if hi<0 or lo>0 else '跨零'
required=['summary.json','family_metrics.csv','per_reading_metrics.csv','per_spectrum_deltas.json','compare_with_B1.csv','overlap.json','q_selection.json','architecture.json','run_validated.log','报告.md','existing_file_changes.json']
check('all_eleven_required_deliverables_present',all((O/n).is_file() for n in required))
S=readj('summary.json'); F=readcsv(O/'family_metrics.csv'); R=readcsv(O/'per_reading_metrics.csv'); P=readj('per_spectrum_deltas.json')
E=readj('process_exit.json'); H=readj('preflight_hashes.json'); A=readj('architecture.json'); X=readj('overlap.json'); Q=readj('q_selection.json')
check('process_exit_zero_without_timeout',E['exit_code']==0 and E['timed_out'] is False)
check('task_and_simulation_only',S['task']=='A18-B3' and S['status']=='simulation_only')
D=S['diagnostic']
check('registered_diagnostic_parameters',(D['seed'],D['noise_seed']) in [(31600,8191),(31700,8293)] and D['n_per_family']==30 and D['bootstrap']==20000 and D['bonferroni_families']==26 and D['jitter'] is True and D['mode']=='real')
check('twenty_six_families',len(F)==26 and len({r['family'] for r in F})==26)
check('three_thousand_one_hundred_twenty_readings',len(R)==3120)
metrics=['B_raw','B_consistent','A','fusion_pre_q','final']
check('finite_nonnegative_reading_metrics',all(math.isfinite(float(r[k])) and float(r[k])>=0 for r in R for k in metrics))
old=readcsv(W/'A18/diag/B1_诊断_20261104/family_metrics.csv'); oldby={r['family']:r for r in old}
check('family_set_matches_original_B1',{r['family'] for r in F}==set(oldby))
readings={name:[] for name in oldby}
for r in R: readings[r['family']].append(r)
for f in F:
    name=f['family']; rr=readings[name]; groups={i:[] for i in range(30)}
    for r in rr: groups[int(r['spectrum'])].append(r)
    check(name+':sample_structure',int(f['n_spectra'])==30 and int(f['n_readings'])==120 and len(rr)==120 and all(len(g)==4 and {int(r['draw']) for r in g}=={0,1,2,3} for g in groups.values()))
    for k in metrics: check(name+':mean:'+k,close(np.mean([float(r[k]) for r in rr]),float(f[k])))
    means={k:np.array([np.mean([float(r[k]) for r in groups[i]]) for i in range(30)]) for k in metrics}
    delta=np.array([np.mean([float(r['B_raw'])-float(r['A']) for r in groups[i]]) for i in range(30)])
    check(name+':paired_delta',close(delta.mean(),float(f['B_minus_A'])))
    rng=np.random.default_rng(1901122); boot=delta[rng.integers(0,30,(20000,30))].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975]); blo,bhi=np.quantile(boot,[.05/26/2,1-.05/26/2])
    check(name+':bootstrap_95_and_bonferroni',close([lo,hi,blo,bhi],[float(f[k]) for k in ('paired_delta_ci_low','paired_delta_ci_high','bonf_ci_low','bonf_ci_high')]))
    check(name+':tier',tier(lo,hi,blo,bhi)==f['tier'])
    best=min(float(f['A']),float(f['B_consistent']))
    check(name+':best_single_and_fusion_percent',close([best,100*(float(f['final'])/best-1)],[float(f['best_single']),float(f['final_vs_best_single_percent'])]))
    check(name+':B_vs_A_percent',close(100*(float(f['B_raw'])/float(f['A'])-1),float(f['B_vs_A_percent'])))
    for dest,key in [('eB','B_raw'),('eA','A'),('eBc','B_consistent'),('eF','final')]: check(name+':per_spectrum:'+dest,close(means[key],P[name][dest]))
    check(name+':per_spectrum_delta',close(delta,P[name]['d_B_minus_A']))
for k in metrics: check('geometric_mean:'+k,close(np.exp(np.mean(np.log([float(f[k]) for f in F]))),S['geometric_means'][k]))
for t in ('B 稳定胜','A 稳定胜','边界·未校正显著','跨零'): check('tier_summary:'+t,{r['family'] for r in F if r['tier']==t}==set(S['tiers'].get(t,[])))
check('fusion_worse_summary',{r['family'] for r in F if float(r['final_vs_best_single_percent'])>0}==set(S['fusion_worse_than_best_single']))
C=readcsv(O/'compare_with_B1.csv'); newby={r['family']:r for r in F}
check('comparison_has_all_families',len(C)==26 and {r['family'] for r in C}==set(oldby))
for c in C:
    name=c['family']; oldrow=oldby[name]; newrow=newby[name]
    for stage,col in [('B','B_raw'),('A','A'),('final','final')]: check(name+':comparison:'+stage,close([float(c[stage+'_old']),float(c[stage+'_new'])],[float(oldrow[col]),float(newrow[col])]))
    check(name+':comparison_tier',c['tier_new']==newrow['tier'])
check('reported_overlap_empty',X['diagnostic_vs_used']=={} and X['diagnostic_vs_old_diag']=={})
check('overlap_seed_matches',X['diagnostic_seed']==D['seed'] and X['noise_seed']==D['noise_seed'])
check('selected_q_matches_summary',Q['selected']==S['q_lambda'])
ok=[float(k) for k,v in Q['candidates'].items() if v['error']<=Q['baseline']*1.005]
if not ok: ok=[float(min(Q['candidates'],key=lambda k:Q['candidates'][k]['error']))]
selected=min(ok,key=lambda k:Q['candidates'][str(k)]['peak_mae_nm'])
check('q_selected_by_validation_rule',selected==Q['selected'])
check('runtime_shared_blocks_unchanged',all(readj('source_review.json').values()))
check('script_hash_and_size',sha(O/'diagnose_b3_newseed.py')==H['script']['sha256'] and (O/'diagnose_b3_newseed.py').stat().st_size==21781)
post=[]
for r in H['inputs']+[H['a16_ss'],H['b1']]:
    actual=sha(W/r['file']); check('unchanged_hash:'+r['file'],actual==r['sha256']);post.append(dict(file=r['file'],sha256=actual,match=True))
for key,value in A['source_sha256'].items(): check('architecture_hash:'+key,sha(W/key)==value)
check('architecture_B_hash',A['B_sha256']==H['inputs'][-1]['sha256'])
check('original_run_change_report_empty',readj('existing_file_changes.json')=={})
before=readj('existing_files_before.json'); after={}
for p in W.rglob('*'):
    if not p.is_file() or p.is_relative_to(O): continue
    st=p.stat(); after[str(p.relative_to(W))]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
changes={k:dict(before=before.get(k),after=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
allowed=str(Path('A18')/'HANDOFF_A18-B3.md'); intentional=[]
if allowed in changes and allowed not in before:
    intentional.append(allowed);del changes[allowed]
check('independent_protected_file_metadata_unchanged',not changes)
text=(O/'报告.md').read_text(encoding='utf-8')
for f in F:
    line=f"| {f['family']} | {float(f['B_raw']):.4f} | {float(f['A']):.4f} | {float(f['final']):.4f} | {float(f['B_vs_A_percent']):+.1f} | [{float(f['bonf_ci_low']):+.4f}, {float(f['bonf_ci_high']):+.4f}] | {f['tier']} | {float(f['wA_mean']):.2f} | {float(f['final_vs_best_single_percent']):+.1f} |"
    check(f['family']+':report_row',line in text)
handoff=W/'A18/HANDOFF_A18-B3.md'
if handoff.exists():
    ht=handoff.read_text(encoding='utf-8')
    family_table=text.split('## 逐族\n\n',1)[1].split('\n\n## ',1)[0].rstrip()
    check('handoff_verbatim_family_table',family_table in ht)
    check('handoff_contains_all_input_hashes',all(r['sha256'] in ht for r in H['inputs']+[H['a16_ss'],H['b1']]) and H['script']['sha256'] in ht)
    check('handoff_all_geometric_means',all(f'{S["geometric_means"][k]:.6f}' in ht for k in metrics))
    check('handoff_all_four_tier_counts',all(f"{t} {len(S['tiers'].get(t,[]))} 族" in ht for t in ('B 稳定胜','A 稳定胜','边界·未校正显著','跨零')))
V=dict(passed=True,checks_passed=len(CHECKS),required_deliverables=required,missing=[],protected_file_changes=changes,intentional_new_files=intentional,postflight_hashes=post,checks=CHECKS,scope='CSV/JSON statistical consistency, hashes, unchanged protected metadata; no reconstruction imports, spectrum generation, test-data access or parameter fitting')
(O/'verification.json').write_text(json.dumps(V,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:V[k] for k in ('passed','checks_passed','missing','protected_file_changes','intentional_new_files')},ensure_ascii=False,indent=2))