"""Make a separate evaluation runner from the immutable, validated B1 runner."""
from pathlib import Path
import ast

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'B1_诊断_20261104'
src = (OLD / 'diagnose_b1.py').read_text(encoding='utf-8')

def replace(a, b):
    global src
    assert src.count(a) == 1, (a[:100], src.count(a))
    src = src.replace(a, b)

replace('import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv',
        'import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom paired_stats import add_paired_statistics, summarize_statistics')
replace("OUT = Path(__file__).resolve().parent", """OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'B1_诊断_20261104'
import argparse
ap = argparse.ArgumentParser(description='Frozen B1 evaluation; seed 81063 only.')
ap.add_argument('--diag-seed', type=int, required=True)
RUN_ARGS = ap.parse_args()
assert RUN_ARGS.diag_seed == 81063, 'This preregistered runner only accepts 81063.'
DSEED = RUN_ARGS.diag_seed
PHASE = 'reconstruct_from_original_training_validation'
generation_calls = []
""")
replace("dump('existing_files_before.json', fingerprint())", """dump('existing_files_before.json', fingerprint())
dump('protocol.json', dict(task='A18-B1 seed rerun',author='AI（Codex）',
    diagnostic_seed=DSEED,n_per_family=30,draws=4,jitter=True,mode='real',noise_seed=4321,
    fitting_seed=31438,training_seed=31337,q='load old B1 validation-only checkpoint',
    old_ss_tuning_seed=31451,forbidden_test_seeds=[6161,4242],
    bootstrap=dict(unit='spectrum, retain 4 paired readings',replicates=200000,
        seed=1901122,statistic='mean over spectra of mean over draws log(A/B)',
        alpha_per_family=0.05/26,point_requirement='geometric paired ratio <= 0.95'),
    preregistered_A_useful_rule=dict(final_improvement_over_B_raw_at_least=0.05,
        A_stable_families_at_least=5,applies_to='future NEW B, not the current old B'),
    restrictions='No fitting, K selection, routing/weight selection or early stopping on 81063.'))
""")
replace("with (OUT / 'run_validated.log').open('a', encoding='utf-8') as f: f.write(msg+'\\n')", "with (OUT / 'run.log').open('a', encoding='utf-8') as f: f.write(msg+'\\n')")
replace("    assert seed in (31337, 31438, 31445, 31451, 31459), seed", """    assert seed in (31337, 31438, 31445, 31451, 31459, DSEED), seed
    if seed == DSEED:
        assert PHASE == 'evaluation_only', PHASE
    generation_calls.append(dict(generator='all_fams',seed=int(seed),n=int(n),
                                 jitter=bool(jitter),phase=PHASE))""")
replace("    assert seed != 3, 'Legacy seed-3 validation is forbidden in this task.'", """    assert seed not in (3,6161,4242), 'Forbidden legacy/test seed.'
    if seed == DSEED+1:
        assert PHASE == 'evaluation_only', PHASE
    generation_calls.append(dict(generator='gen5',seed=int(seed),n=int(n),phase=PHASE))""")
replace("log('Building existing A18 from training/validation only; no B retraining.')", """old_summary = json.loads((OLD/'summary.json').read_text(encoding='utf-8'))
for rel, expected in old_summary['architecture']['source_sha256'].items():
    assert hashlib.sha256((WORK/rel).read_bytes()).hexdigest() == expected, rel
assert hashlib.sha256((Path(M.NETDIR)/'cnet0.pt').read_bytes()).hexdigest() == old_summary['architecture']['B_sha256']
assert M.ARG.a16a == 32 and a16_ss.ALPHA == 0.75
log(f'PREREGISTERED diagnostic seed={DSEED}; no evaluation spectra generated before model freeze.')
log('Reconstructing old B1 from original training/validation only; no B retraining; q will be loaded.')""")
replace("    capture.update(Fv=Fv, Xs=Xs, St=St, va=va, W=M.blend3_w(Fv, mu, sd, net))", "    capture.update(Fv=Fv, Xs=Xs, St=St, va=va, W=M.blend3_w(Fv, mu, sd, net), net=net, mu=mu, sd=sd)")
replace("decode, P, used = M.build3()", """decode, P, used = M.build3()
closure = dict(zip(decode.__code__.co_freevars, (c.cell_contents for c in decode.__closure__)))
assert closure['kap']['t3'] == 0.25, closure['kap']
""")
replace("for p in paths.values()},", "for p in list(paths.values()) + [ALG/'a16_ss.py']},")
start = src.index('# Original q procedure, fitted afresh exclusively on VSEED/QSEED.')
end = src.index('rows=[]; sample_rows=[]', start)
src = src[:start] + '''# Restore the EXACT q checkpoint from the validated old B1 run; never refit q.
qpath = OLD/'q_validation_only.pt'
qcheckpoint = torch.load(qpath, map_location=M.DEV, weights_only=False)
qmu, qsd, qlam = qcheckpoint['mu'], qcheckpoint['sd'], qcheckpoint['lam']
qnet = torch.nn.Sequential(torch.nn.Linear(len(qmu),32),torch.nn.Tanh(),torch.nn.Linear(32,1)).double().to(M.DEV)
qnet.load_state_dict(qcheckpoint['state']); qnet.eval()
assert qlam == old_summary['q_lambda'] and qcheckpoint['dmax'] == PF.DMAX
qsets = AD.all_fams(31438, AD.NVF, P, True)

# Verify that reconstruction matches the existing B1 fusion validation outputs.
old_cv = list(csv.DictReader((OLD/'fusion_validation_holdout.csv').open(encoding='utf-8-sig')))
cvrows=[]; n_each=AD.NVF*AD.LD.NDRAW
for j,k in enumerate(qsets):
    va=capture['va']; ii=va[(va//n_each)==j]
    row=dict(family=k.replace('留出·',''),n_readings=len(ii),
        A=float(AD.LD.band_nrmse(capture['Xs'][0][ii].T,capture['St'][ii].T).mean()),
        B_consistent=float(AD.LD.band_nrmse(capture['Xs'][1][ii].T,capture['St'][ii].T).mean()),
        wA=float(capture['W'][ii,0].mean()))
    assert old_cv[j]['family'] == row['family']
    for key in ('A','B_consistent','wA'):
        assert abs(row[key]-float(old_cv[j][key])) < 1e-9, (row['family'],key,row[key],old_cv[j][key])
    cvrows.append(row)

def net_hash(net):
    h=hashlib.sha256()
    for key,value in net.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
frozen_hashes=dict(B=net_hash(M.XN),fusion=net_hash(capture['net']),q=net_hash(qnet))
dump('model_freeze.json',dict(phase=PHASE,diagnostic_not_generated_yet=True,
    source_and_B_match_old_B1=True,old_validation_metrics_match_atol=1e-9,
    kappa=closure['kap'],K=M.ARG.a16a,ss_alpha=a16_ss.ALPHA,
    model_state_sha256=frozen_hashes,q_checkpoint_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
    q_checkpoint=str(qpath.relative_to(WORK)),q_lambda=qlam,
    evaluation_seed_used_for_fitting=False))

# All fitting entry points are disabled before generating the new evaluation set.
def forbidden_fit(*args, **kwargs):
    raise RuntimeError('Fitting/early stopping is forbidden after the B1 model freeze.')
M.fit_blend3=M.train_cnet=M.gf_a16a=PF.fit_q=forbidden_fit
P18.Consist.fit_cov=forbidden_fit
PHASE='overlap_audit_only'
ss_tuning = AD.all_fams(31451,30,P,True)
used['SS调参集31451']=np.concatenate(list(ss_tuning.values()))
PHASE='evaluation_only'
log(f'Model frozen; q={qlam}; validation reproduction passed. Generating diagnostic seed={DSEED}.')
dsets = AD.all_fams(DSEED,30,P,True)
leak = P18.overlap({k:v for k,v in dsets.items()},used)
assert not leak, leak
dump('overlap.json',dict(diagnostic_vs_used=leak, method='round 9 decimals exact rows; tests never generated',
    diagnostic_seed=DSEED,n_per_family=30,jitter=True,
    compared_sets={k:len(v) for k,v in used.items()}))
np.savez_compressed(OUT/'diagnostic_spectra.npz',**{k.replace('留出·',''):v for k,v in dsets.items()})
dump('generation_provenance.json',dict(actual_all_fams_seed=DSEED,
    actual_five_family_gen5_seed=DSEED+1,actual_CIE_jitter_seed=DSEED+333,
    family_generator_source='Q.fams uses seed and derived seeds; see source_manifest.json',
    calls=generation_calls))
''' + src[end:]
start = src.index('# Match training and validation noise/power distributions')
end = src.index('gm=lambda a:', start)
src = src[:start] + '''# Paired inference is evaluation only; no decoder parameters are selected.
rows = add_paired_statistics(rows, sample_rows)
csvwrite('family_metrics.csv',rows)
csvwrite('fusion_validation_holdout.csv',cvrows)
assert frozen_hashes == dict(B=net_hash(M.XN),fusion=net_hash(capture['net']),q=net_hash(qnet))
dump('model_post_evaluation.json',dict(model_state_sha256=frozen_hashes,unchanged=True))
''' + src[end:]
replace("diagnostic=dict(seed=31451,n_per_family=30,jitter=True,mode='real',noise_seed=4321)", "diagnostic=dict(seed=DSEED,actual_five_family_seed=DSEED+1,actual_CIE_jitter_seed=DSEED+333,n_per_family=30,draws=4,jitter=True,mode='real',noise_seed=4321)")
replace("             training_objective_r=r,\n             generalization_means={l:{metric:float(np.mean([v[f'{l}_{metric}'] for v in gaps])) for metric in ('shape_nrmse','training_objective')} for l in ('train','validation','diagnostic')})", "             statistics=summarize_statistics(rows),evaluation_only=True,old_seed=31451,\n             old_B1_geometric_means=old_summary['geometric_means'],\n             q_checkpoint_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest())")
replace("dump('summary.json',summary)", """summary['comparison_with_31451']={k:dict(old=v,new=summary['geometric_means'][k],
    relative_change_percent=100*(summary['geometric_means'][k]/v-1))
    for k,v in old_summary['geometric_means'].items()}
summary['final_improvement_over_old_B_raw_percent']=100*(1-summary['geometric_means']['final']/summary['geometric_means']['B_raw'])
summary['A_useful_rule_old_B_reference_only']=dict(
    improvement_at_least_5_percent=summary['final_improvement_over_old_B_raw_percent']>=5,
    stable_A_families_at_least_5=len(summary['statistics']['A_stable_vs_B_raw'])>=5,
    new_B_not_implemented=True,conclusion='Cannot decide the preregistered NEW-B stopping rule from old B.')
dump('summary.json',summary)""")
src = src.replace('Run: py -I -X utf8 A18/diag/B1_诊断_20261104/diagnose_b1.py',
                  'Run: py -I -X utf8 A18/diag/B1_新种子81063_20261104/diagnose_b1.py --diag-seed 81063')
ast.parse(src)
(OUT/'diagnose_b1.py').write_text(src,encoding='utf-8')
print('Prepared isolated runner; old B1 scripts untouched.')
