"""AI (Codex): derive an isolated runner; never modify old B1 artifacts."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
OLD81063 = OUT.parent/'B1_新种子81063_20261104'
old = (OLD81063/'diagnose_b1.py').read_text(encoding='utf-8')
# Keep the reviewed initialization and reconstruction functions byte-for-byte.
prefix = old[:old.index('# All fitting entry points are disabled')]
prefix = prefix.replace('from paired_stats import add_paired_statistics, summarize_statistics',
    'from evaluation_support import annotate, summarize, csvwrite, q_statistics, numeric_state')
prefix = prefix.replace('seed 81063 only.', 'seed 22658 only.')
prefix = prefix.replace('RUN_ARGS.diag_seed == 81063', 'RUN_ARGS.diag_seed == 22658')
prefix = prefix.replace('only accepts 81063.', 'only accepts 22658.')
prefix = prefix.replace('Run: py -I -X utf8 A18/diag/B1_新种子81063_20261104/diagnose_b1.py --diag-seed 81063',
    'Run: py -I -X utf8 A18/diag/B1_新种子22658_20261104/diagnose_b1.py --diag-seed 22658')
s = prefix.index("dump('protocol.json'")
e = prefix.index('# Fail closed', s)
prefix = prefix[:s] + prefix[e:]
# Content hashes, not mtime/size. The manifest only audits contents; it does not evaluate old seeds.
s = prefix.index('def fingerprint():')
e = prefix.index("assert not (OUT / 'summary.json')", s)
prefix = prefix[:s] + '''def fingerprint():
    return {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in (WORK/'A18', WORK/'算法对照', WORK/'A17')
            for p in root.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}

''' + prefix[e:]
# All data-source adaptations are retained, but log the previously missing gen5 call.
prefix = prefix.replace('    if n == 60 and seed == 31439:\n        return gen5_original(AD.NVF, 31439, band)',
    "    if n == 60 and seed == 31439:\n        generation_calls.append(dict(generator='gen5', seed=31439, requested_n=60, n=AD.NVF, phase=PHASE, adaptation=True))\n        return gen5_original(AD.NVF, 31439, band)")
prefix = prefix.replace("'qnet.pt'))", "'qnet.pt', 'b1_新种子81063', 'b2_新种子63862'))")
# Allow read-only comparison of 81063 metadata; never open its weights/spectra/logs.
prefix = prefix.replace("if not write and any(x in norm", "if not write and p.name not in ('model_freeze.json', 'summary.json', 'family_metrics.csv') and any(x in norm")
# fingerprint happens before installing the read guard, later audit performed separately.
prefix += '''
old_freeze = json.loads((OUT.parent/'B1_新种子81063_20261104/model_freeze.json').read_text(encoding='utf-8'))
assert frozen_hashes == old_freeze['model_state_sha256'], (frozen_hashes, old_freeze)
roots = dict(decoder=decode, B=M.XN, B_scale=M.XN0, pool=M._POOL,
             q_net=qnet, q_mu=qmu, q_sd=qsd, q_lambda=qlam,
             RN=AD.RN, WL=AD.WL, BM=AD.BM,
             ss=dict(ALPHA=a16_ss.ALPHA, KMAX=a16_ss.KMAX, CFG=a16_ss.CFG, USE=a16_ss.USE, CG=a16_ss.CG))
full_freeze = numeric_state(roots)
dump('full_model_freeze.json', full_freeze)

def forbidden_fit(*args, **kwargs):
    raise RuntimeError('No fitting/selection after model freeze.')
M.fit_blend3=M.train_cnet=M.gf_a16a=PF.fit_q=forbidden_fit
P18.Consist.fit_cov=forbidden_fit

# q decision is made on 31438 before 22658 is generated. Existing q is never refitted.
PHASE='q_ablation_validation_only'
QOUT=OUT/'q_ablation'; QOUT.mkdir(exist_ok=True)
qr, qs = [], []
g = np.random.default_rng(9090)
for k, S in qsets.items():
    family=k.replace('留出·','')
    Y=AD.meas(S.T,g,'real'); M._TR.clear(); o=decode(Y)
    fx, xa, de=PF.featx(o); q=PF.q_of(fx,qmu,qsd,qnet)
    xp=P18.unflat(PF.shift_np(xa,q*de),Y.shape[0])
    eo, en=errors(o['A11'],S), errors(xp,S)
    qr.append(dict(family=family,n_spectra=len(S),n_readings=eo.size,
                   q_off=float(eo.mean()),q_on=float(en.mean()),
                   change_percent=float(100*(en.mean()/eo.mean()-1))))
    for i in range(len(S)):
        for d in range(4):
            qs.append(dict(family=family,spectrum=i,draw=d,q_off=float(eo[d,i]),q_on=float(en[d,i]),q=float(q.reshape(4,-1)[d,i])))
    csvwrite(QOUT/'per_reading_metrics.csv',qs)
    log('q ablation 31438: '+family)
qdecision=q_statistics(qr,qs)
qdecision.update(validation_seed=31438,n_per_family=40,draws=4,mode='real',noise_seed=9090,
                 checkpoint_frozen=True,validation_spectra_previously_used_for_q_fit=True,
                 independent_generalization_claim=False)
csvwrite(QOUT/'family_metrics.csv',qr)
(QOUT/'summary.json').write_text(json.dumps(qdecision,ensure_ascii=False,indent=2),encoding='utf-8')
log('Validation-only q decision: retain_q='+str(qdecision['retain_q']))
assert numeric_state(roots)==full_freeze

PHASE='overlap_audit_only'
ss_tuning=AD.all_fams(31451,30,P,True)
used['SS调参集31451']=np.concatenate(list(ss_tuning.values()))
PHASE='evaluation_only'
log('Generating 22658 exactly once: 150 spectra per family, four real readings.')
dsets=AD.all_fams(DSEED,150,P,True)
leak=P18.overlap(dsets,used)
assert not leak, leak
dump('overlap.json',dict(diagnostic_vs_used=leak,method='round 9 decimals exact rows; no forbidden sets generated',
    compared_sets={k:len(v) for k,v in used.items()}))
np.savez_compressed(OUT/'diagnostic_spectra.npz',**{k.replace('留出·',''):v for k,v in dsets.items()})
dump('generation_provenance.json',dict(actual_all_fams_seed=DSEED,
    derived_seeds=dict(five_family=DSEED+1,CIE_jitter=DSEED+333,ood4=DSEED+500,parts=DSEED+900),
    calls=generation_calls,provenance_recorded_during_execution=True))
rows=[]; samples=[]
g=np.random.default_rng(1022658)
for k,S in dsets.items():
    family=k.replace('留出·',''); Y=AD.meas(S.T,g,'real'); M._TR.clear(); o=decode(Y)
    fx,xa,de=PF.featx(o); q=PF.q_of(fx,qmu,qsd,qnet)
    o['final']=P18.unflat(PF.shift_np(xa,q*de),Y.shape[0])
    wA=1-o['w']; chooseA=wA>.5
    o['hard_raw']=np.where(chooseA[:,None,:],o['gate'],o['t3'])
    o['hard_consistent']=np.where(chooseA[:,None,:],o['gate'],o['t3c'])
    es={label:errors(o[key],S) for label,key in dict(B_raw='t3',B_consistent='t3c',A='gate',fusion_pre_q='A11',final='final',hard_raw='hard_raw',hard_consistent='hard_consistent').items()}
    row=dict(family=family,n_spectra=len(S),n_readings=es['A'].size,
             **{key:float(v.mean()) for key,v in es.items()},
             wA_mean=float(wA.mean()),A_weight_gt_half=float(chooseA.mean()))
    for b in ('B_raw','B_consistent'):
        eA,eB=es['A'],es[b]; correct=chooseA==(eA<eB); trueA=chooseA&(eA<eB)
        row.update({f'route_vs_{b}_accuracy':float(correct.mean()),
                    f'route_vs_{b}_correct_n':int(correct.sum()),
                    f'route_vs_{b}_A_error_when_correct':float(eA[correct].mean()) if correct.any() else None,
                    f'route_vs_{b}_correct_A_n':int(trueA.sum()),
                    f'route_vs_{b}_A_error_when_correct_A':float(eA[trueA].mean()) if trueA.any() else None})
    rows.append(row)
    for i in range(len(S)):
        for d in range(4):
            samples.append(dict(family=family,spectrum=i,draw=d,
                                **{key:float(v[d,i]) for key,v in es.items()},wA=float(wA[d,i])))
    csvwrite(OUT/'per_reading_metrics.csv',samples)
    dump('diagnostic_partial.json',rows)
    log(f'{family}: B={row["B_raw"]:.6f}, A={row["A"]:.6f}, hard={row["hard_raw"]:.6f}')
rows=annotate(rows,samples)
csvwrite(OUT/'family_metrics.csv',rows)
post=numeric_state(roots)
dump('full_model_post_evaluation.json',post)
assert full_freeze==post, 'Inference state changed'
assert frozen_hashes==dict(B=net_hash(M.XN),fusion=net_hash(capture['net']),q=net_hash(qnet))
gm=lambda values:float(np.exp(np.log(values).mean()))
summary=dict(task='A18-B1-22658',author='AI（Codex）',status='simulation_only',
    diagnostic=dict(seed=DSEED,n_per_family=150,draws=4,jitter=True,mode='real',noise_seed=1022658),
    geometric_means={key:gm([r[key] for r in rows]) for key in es},
    statistics=summarize(rows),q_decision=qdecision,
    evaluation_only=True,new_B_implemented=False,configuration='reconstructed frozen old-B1 adaptation; not official A18',
    matches_81063_network_hashes=True,full_inference_state_unchanged=True,
    full_inference_state_sha256=full_freeze['sha256'],
    seconds=time.time()-start,peak_cuda_GB=torch.cuda.max_memory_allocated()/2**30,
    chol_fallback=a16_ss.NBAD)
dump('summary.json',summary)
log('Numerical evaluation finished; independent verification and report pending.')
'''
ast.parse(prefix)
assert not (OUT/'diagnose_b1.py').exists(), 'Do not regenerate a prepared run.'
(OUT/'diagnose_b1.py').write_text(prefix,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'HANDOFF_before.md').write_bytes((WORK/'A18/HANDOFF_A18-B1.md').read_bytes())
protocol=dict(author='AI（Codex）',recorded=datetime.datetime.now().astimezone().isoformat(),
    source_preregistration='HANDOFF_A18-B1.md: 2026-11-04 12:00',
    evaluation=dict(seed=22658,n_per_family=150,draws=4,jitter=True,mode='real',noise_seed=1022658,generate_once=True),
    model='same old-B1 adaptation as 81063; reconstruct only on 31337/31438, verify hashes before 22658',
    frozen_B='old cnet0',no_new_B=True,forbidden_evaluation_seeds=[6161,4242,81063,63862],
    statistics=dict(unit='spectrum; mean of four log ratios',bootstrap_n=200000,bootstrap_seed=1901122,
        alpha_per_family=.05/26,interval='envelope of percentile bootstrap and Student t',minimum_improvement=.05),
    hard_route=dict(rule='wA > .5 selects A',primary_comparison='A vs B_raw',secondary='A vs B_consistent',
        accuracy='agreement with lower per-reading NRMSE; ties assigned B'),
    q_ablation=dict(seed=31438,n_per_family=40,draws=4,mode='real',noise_seed=9090,refit=False,
        overall='equal-family mean of spectrum mean log(q_on/q_off)',
        interval='95% envelope of stratified spectrum bootstrap and Satterthwaite Student t',
        family_limit='no family arithmetic mean NRMSE worsens >10%',
        retain='overall log-ratio upper CI <0 AND family limit',decide_before_22658=True,
        caveat='31438 was used to fit q; development decision, not an independent generalization estimate'))
(OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
sources={str(p.relative_to(WORK.parent)):sha(p) for root in (
    WORK/'算法对照',WORK.parent/'03_仿真/41_重建算法二_形状自适应解码/scripts') for p in root.rglob('*.py')}
(OUT/'source_manifest.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared isolated 22658 evaluation; old scripts unchanged.')
