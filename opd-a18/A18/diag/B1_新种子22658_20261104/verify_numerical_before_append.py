"""AI (Codex): independent file, pairing, arithmetic and statistical verification."""
from pathlib import Path
import ast
import csv
import datetime
import hashlib
import json
import sys
import numpy as np
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]
sys.path.insert(0,str(OUT))
from evaluation_support import arrays, paired, q_statistics, summarize
load=lambda n:json.loads((OUT/n).read_text(encoding='utf-8'))
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read_process_log(path):
    raw=path.read_bytes()
    return raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig')
checks=[]
def check(ok,description):
    assert ok,description
    checks.append(description)
def near(a,b):
    return bool(np.allclose(a,b,rtol=1e-11,atol=1e-12))

j=load('summary.json'); p=load('protocol.json'); rows=read(OUT/'family_metrics.csv')
samples=read(OUT/'per_reading_metrics.csv')
check(j['diagnostic']==dict(seed=22658,n_per_family=150,draws=4,jitter=True,mode='real',noise_seed=1022658),'Only preregistered 22658 setup')
check(len(rows)==26 and len(samples)==15600,'26 families, 3900 spectra, 15600 readings')
check(len({r['family'] for r in rows})==26,'Unique family keys')
keys=('A','B_raw','B_consistent','fusion_pre_q','final','hard_raw','hard_consistent','wA')
parsed=[]
for row in rows:
    family=row['family']; a=arrays(samples,family,keys)
    check(a['A'].shape==(150,4),family+': 150 independent spectrum blocks with four draws')
    check(all(np.isfinite(x).all() for x in a.values()),family+': finite metrics')
    check(all((a[k]>0).all() for k in keys if k!='wA'),family+': positive NRMSE')
    check(((a['wA']>=0)&(a['wA']<=1)).all(),family+': bounded gate weights')
    choose=a['wA']>.5
    out=dict(family=family)
    for key in keys:
        if key!='wA': check(near(a[key].mean(),float(row[key])),family+': arithmetic '+key)
    for b,h in (('B_raw','hard_raw'),('B_consistent','hard_consistent')):
        check(near(a[h],np.where(choose,a['A'],a[b])),family+': fixed hard route '+b)
        correct=choose==(a['A']<a[b]); trueA=choose&(a['A']<a[b])
        check(near(correct.mean(),float(row[f'route_vs_{b}_accuracy'])),family+': routing accuracy '+b)
        check(int(correct.sum())==int(row[f'route_vs_{b}_correct_n']),family+': correct count '+b)
        check(int(trueA.sum())==int(row[f'route_vs_{b}_correct_A_n']),family+': correct A count '+b)
        for mask,name in ((correct,'when_correct'),(trueA,'when_correct_A')):
            actual=row[f'route_vs_{b}_A_error_{name}']
            check(near(a['A'][mask].mean(),float(actual)) if mask.any() else actual=='',family+': conditional A error '+b+' '+name)
        st=paired(a['A'],a[b])
        for k,v in st.items():
            saved=row[f'A_vs_{b}_{k}']
            check(saved==str(v) if isinstance(v,bool) else near(v,float(saved)),family+': recomputed '+b+' '+k)
        out.update({f'A_vs_{b}_{k}':v for k,v in st.items()})
    parsed.append(out)
check(j['statistics']==summarize(parsed),'Stable family lists exactly match conservative inference')
for key,value in j['geometric_means'].items():
    check(near(value,np.exp(np.log([float(r[key]) for r in rows]).mean())),'Geometric mean '+key)

qrows=read(OUT/'q_ablation/family_metrics.csv')
qsamples=read(OUT/'q_ablation/per_reading_metrics.csv')
check(len(qrows)==26 and len(qsamples)==4160,'q ablation: 1040 spectra and 4160 readings')
for r in qrows:
    a=arrays(qsamples,r['family'],('q_off','q_on','q'))
    check(a['q_on'].shape==(40,4),'q paired 40 spectra: '+r['family'])
    check(near(a['q_on'].mean(),float(r['q_on'])) and near(a['q_off'].mean(),float(r['q_off'])),'q arithmetic means: '+r['family'])
    check(((a['q']>=0)&(a['q']<=1)).all(),'q bounded: '+r['family'])
    for k in ('q_off','q_on'):r[k]=float(r[k])
qcheck=q_statistics(qrows,qsamples)
q=load('q_ablation/summary.json')
check(q==j['q_decision'],'q decision copied without changes')
for k,v in qcheck.items():check(v==q[k],'Validation-only q rule: '+k)
check(q['validation_seed']==31438 and q['checkpoint_frozen'] and not q['independent_generalization_claim'],'q uses existing validation only; no false independence claim')

fr=load('full_model_freeze.json'); post=load('full_model_post_evaluation.json')
check(fr==post,'Entire captured inference state identical before/after evaluation')
raw=json.dumps(fr['entries'],sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
check(hashlib.sha256(raw).hexdigest()==fr['sha256']==j['full_inference_state_sha256'],'Complete inference content digest')
paths=list(fr['entries'])
coverage={name:any(needle in k for k in paths) for name,needle in dict(
    A_Gaussian_prior='.GF.',SS_dictionary='.GSS.SV.Phi.',
    consistency_covariance='.CS.t3.Ch',fusion_normalization='.mu2',
    q_normalization='.q_mu',response_matrix='.RN',
    SS_configuration='.ss.CFG').items()}
check(all(coverage.values()),'Freeze covers priors, SS, covariance, normalization and response matrix: '+str(coverage))
freeze=load('model_freeze.json')
old=json.loads((OUT.parent/'B1_新种子81063_20261104/model_freeze.json').read_text(encoding='utf-8'))
check(freeze['model_state_sha256']==old['model_state_sha256'],'B/fusion/q networks identical to 81063')
check(freeze['diagnostic_not_generated_yet'] and not freeze['evaluation_seed_used_for_fitting'],'Freeze precedes 22658 and excludes it from fitting')
check(p['q_ablation']['decide_before_22658'],'q decision protocol predates new seed generation')
runlog=(OUT/'evaluation_run.log').read_text(encoding='utf-8')+(OUT/'continue_run.log').read_text(encoding='utf-8')
check(runlog.index('Validation-only q decision:') < runlog.index('Generating 22658 exactly once:'),'Execution log confirms q decision before 22658 generation')
calls=load('generation_provenance.json')['calls']
eval_calls=[c for c in calls if c['generator']=='all_fams' and c['seed']==22658]
check(len(eval_calls)==1 and eval_calls[0]['n']==150 and eval_calls[0]['phase']=='evaluation_only','22658 generated exactly once, after freeze')
forbidden={6161,4242,81063,63862}
check(not any(c['seed'] in forbidden for c in calls),'No forbidden seed regenerated')
check(any(c.get('adaptation') and c['n']==40 and c['seed']==31439 for c in calls),'Previously omitted gen5 adapter provenance recorded at runtime')
check(not load('overlap.json')['diagnostic_vs_used'],'Zero exact rounded-row overlap with allowed development pools')
z=np.load(OUT/'diagnostic_spectra.npz')
check(len(z.files)==26 and all(z[k].shape==(150,176) and np.isfinite(z[k]).all() for k in z.files),'Saved evaluation spectra have expected dimensions and finite values')

# Confirm all original numerical functions other than runner-only audit/logging adapters.
oldsrc=(OUT.parent/'B1_新种子81063_20261104/diagnose_b1.py').read_text(encoding='utf-8')
newsrc=(OUT/'continue_evaluation.py').read_text(encoding='utf-8')
oldtree=ast.parse(oldsrc); newtree=ast.parse(newsrc)
function_map=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
of,nf=function_map(oldtree),function_map(newtree)
for name in ('Sanitize','Loader','Finder','fit_capture','errors','net_hash'):
    check(of[name]==nf[name],'Original numerical/source adapter function preserved: '+name)
check('M.fit_blend3=M.train_cnet=M.gf_a16a=PF.fit_q=forbidden_fit' in newsrc and 'P18.Consist.fit_cov=forbidden_fit' in newsrc,'Fitting entry points disabled after freeze')
for relative,expected in load('source_manifest.json').items():
    check(sha(WORK.parent/relative)==expected,'Original source content hash: '+relative)

# Content verification may read old files solely to hash, never to evaluate old seeds.
before=load('existing_files_before.json')
changes={}
protected=[]
for relative,expected in before.items():
    path=WORK/relative
    actual=sha(path) if path.exists() else None
    if actual!=expected:changes[relative]=dict(before=expected,after=actual)
    norm=relative.replace('\\','/').lower()
    if any(k in norm for k in ('a18_seed6161','a18_seed4242','b1_新种子81063','b2_新种子63862','b1_诊断_20261104')):
        protected.append(relative)
        check(actual==expected,'Protected old file content: '+relative)
# Old source and weights are mandatory even if unrelated agents change their own handoffs/results.
unexpected=[k for k in changes if k.startswith('算法对照') or k.startswith('A17')]
check(not unexpected,'All previous algorithm/source/weight files unchanged')
(OUT/'existing_file_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
# Preserve all pre-interruption evidence, including partial q and original logs.
resume=load('resume_before.json')
for relative,expected in resume['local_existing_files'].items():
    check(sha(OUT/relative)==expected,'Pre-interruption artifact preserved: '+relative)
check(fr['sha256']==resume['full_inference_state_sha256']=='59f19648abb6ceab81cec07ca40d68c5ce188b38988d923738405e4c749976ce','Full state matches the 15:16 preregistered digest')
preq=load('q_decision_before_evaluation.json')
check(preq['retain_q']==q['retain_q'] and preq['q_summary_sha256']==sha(OUT/'q_ablation/summary.json'),'q decision persisted and fingerprinted before evaluation')
check(not preq['evaluation_spectra_exist'] and not preq['evaluation_generation_called'],'No 22658 spectra existed or had been generated when q was decided')
check((OUT/'q_ablation/summary.json').stat().st_mtime <= (OUT/'diagnostic_spectra.npz').stat().st_mtime,'q decision file predates evaluation spectra file')
resume_changes={}
for relative,expected in resume['outside_existing_files'].items():
    path=WORK/relative; actual=sha(path) if path.exists() else None
    if actual!=expected:resume_changes[relative]=dict(before=expected,after=actual)
check(set(resume_changes) <= {'A18\\HANDOFF_A18-B1.md'},'Only the authorized B1 handoff changed outside the diagnostic directory during the resumed run')
(OUT/'resume_external_changes.json').write_text(json.dumps(resume_changes,ensure_ascii=False,indent=2),encoding='utf-8')
process=read_process_log(OUT/'continue_process.log')
check('Traceback (most recent call last)' not in process,'Resumed numerical process has no traceback')
check('Numerical evaluation finished; independent verification and report pending.' in runlog,'Resumed numerical process reached the end')
# The failed q-process remains preserved; completed evidence is immutable.
completed=load('completed_q_before_evaluation.json')
for relative,expected in completed['protected_files'].items():
    check(sha(OUT/relative)==expected,'Completed q/process evidence preserved: '+relative)
failed=read_process_log(OUT/'resume_process.log')
check('AssertionError' in failed,'Original strict scratch assertion remains in the failed process log')
check('Generating 22658 exactly once:' not in (OUT/'resume_run.log').read_text(encoding='utf-8'),'Failed q process did not begin 22658 generation')
scratch_key='model.decoder.closure.base.closure.gate_of.closure.GF._m2'
for stage in ('validation_probe_1','validation_probe_2','after_22658'):
    audit=load('scratch_state_audit_'+stage+'.json')
    check(set(audit['differences'])=={scratch_key},'Raw inference state changes only the declared batch scratch: '+stage)
    check(audit['parameter_or_prior_changes']==0 and audit['scratch_restored_after_comparison'],'All parameters/priors remain frozen: '+stage)
    check(audit['differences'][scratch_key]['before']==fr['entries'][scratch_key],'Scratch baseline matches the registered freeze: '+stage)
probe=load('scratch_probe_functional_check.json')
check(probe['status']=='passed' and probe['allowed_data_seed']==31438 and not probe['evaluation_generated'],'Cache probe used only existing validation before evaluation')
check(not probe['excluded_parameter_paths'] and probe['original_full_state_recovered'],'No model parameter paths excluded from strict verification')
check(all(v==0 for v in probe['maximum_absolute_output_difference'].values()),'Restoring batch scratch leaves A/B/consistency/fusion/weights exactly unchanged')
check(runlog.index('Scratch probe passed:') < runlog.index('Generating 22658 exactly once:'),'Scratch explanation was validated before 22658 was generated')
check('assert set(differences)<={scratch_key}' in newsrc and 'assert restored==full_freeze' in newsrc,'Runner enforces scratch-only differences and full-state restoration')
continuation=load('continuation_before.json')
for relative,expected in continuation['protected_files'].items():
    check(sha(OUT/relative)==expected,'Saved spectra/checkpoint/recovery evidence preserved: '+relative)
cg=load('continuation_generation_calls.json')
check(cg['evaluation_loaded_from_saved_npz'] and cg['new_evaluation_generation_calls']==0,'Continuation loads the unique saved evaluation spectra')
check(not any(c['generator']=='all_fams' and c['seed']==22658 for c in cg['calls']),'Continuation never calls the 22658 spectrum generator')
loaded=load('continuation_loaded_metrics.json')
check(loaded['completed_families']==continuation['completed_families'] and loaded['readings']==1200,'Exactly the completed two families reused')
checkpoint=read(OUT/'checkpoint_after_progress_assertion/per_reading_metrics.csv')
check(len(checkpoint)==1200 and len(samples[:1200])==1200,'Complete 1200-reading checkpoint retained')
for old,new in zip(checkpoint,samples[:1200]):
    check(old==new,'Completed prediction row preserved: '+old['family']+' '+old['spectrum']+'/'+old['draw'])
failed_progress=read_process_log(OUT/'evaluation_process.log')
check('Pre-existing artifact differs: diagnostic_partial.json' in failed_progress,'Original progress assertion remains documented')
aggregate=load('aggregate_uncertainty.json')
check(not aggregate['preregistered_primary_criterion'],'Aggregate uncertainty explicitly marked descriptive')
check(aggregate['bootstrap_n']==200000 and aggregate['bootstrap_seed']==1901122,'Aggregate inference resampling settings')
for name,expected in aggregate['input_sha256'].items():
    check(sha(OUT/name)==expected,'Aggregate inference input digest: '+name)
for r in aggregate['comparisons']:
    a,b=r['method'],r['reference']
    check(near(r['improvement_percent'],100*(1-j['geometric_means'][a]/j['geometric_means'][b])),'Aggregate point estimate: '+a+' vs '+b)
    lo,hi=r['percentile_95_ci_percent']
    check(np.isfinite([lo,hi]).all() and lo<=hi,'Finite ordered descriptive interval: '+a+' vs '+b)

result=dict(status='numerical_checks_passed_handoff_not_yet_appended',checks_passed=len(checks),checks=checks)
(OUT/'numerical_verification_before_append.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=result['status'],checks_passed=len(checks)),ensure_ascii=False))
