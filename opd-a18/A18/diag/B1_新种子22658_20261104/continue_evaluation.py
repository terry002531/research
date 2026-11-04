"""A18-B1, AI (Codex). Read-only source execution; all writes stay beside this file.
Run: py -I -X utf8 A18/diag/B1_新种子22658_20261104/continue_evaluation.py --diag-seed 22658
No B training and no project/author test-spectrum or test-current evaluation.
"""
from pathlib import Path
import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluation_support import annotate, summarize, csvwrite, q_statistics, numeric_state

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'B1_诊断_20261104'
import argparse
ap = argparse.ArgumentParser(description='Frozen B1 evaluation; seed 22658 only.')
ap.add_argument('--diag-seed', type=int, required=True)
RUN_ARGS = ap.parse_args()
assert RUN_ARGS.diag_seed == 22658, 'This preregistered runner only accepts 22658.'
DSEED = RUN_ARGS.diag_seed
PHASE = 'reconstruct_from_original_training_validation'
generation_calls = []

WORK = OUT.parents[2]
ALG = WORK / '算法对照'
PROJ = WORK.parent
sys.dont_write_bytecode = True
for key in ('TMP', 'TEMP', 'MPLCONFIGDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
    p = OUT / 'runtime' / key
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def dump(name, obj):
    path = OUT / name
    if name == 'diagnostic_partial.json':
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        return
    if name == 'existing_files_before.json':
        path = OUT / 'continuation_existing_files_before.json'
    elif path.exists():
        # All pre-interruption evidence is immutable. Compare, never replace it.
        assert json.loads(path.read_text(encoding='utf-8')) == obj, 'Pre-existing artifact differs: ' + name
        return
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def fingerprint():
    return {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in (WORK/'A18', WORK/'算法对照', WORK/'A17')
            for p in root.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}

assert not (OUT / 'summary.json').exists(), 'Completed run exists; do not overwrite.'
assert (OUT / 'diagnostic_spectra.npz').exists(), 'Continuation requires the already-generated spectra.'
assert (OUT / 'generation_provenance.json').exists(), 'Original generation provenance required.'
assert (OUT / 'q_ablation/summary.json').exists(), 'Completed immutable q decision required.'
assert not (OUT / 'continue_run.log').exists(), 'Continuation already started.'
dump('existing_files_before.json', fingerprint())
# Fail closed on writes outside the diagnostic directory, including imported modules.
def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        if os.fsdecode(path).lower() in ('nul', os.devnull.lower()):
            return  # Windows null device; no filesystem artifact.
        p = Path(os.fsdecode(path)).absolute()
        write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)) if isinstance(flags, int) else False
        if write and not p.is_relative_to(OUT):
            raise PermissionError(f'B1 write outside diagnostic directory: {p}')
        norm = str(p).replace('\\', '/').lower()
        if not write and p.name not in ('model_freeze.json', 'summary.json', 'family_metrics.csv') and any(x in norm for x in ('a18_seed6161', 'a18_seed4242', 'testset_', 'peakfix.json', 'qnet.pt', 'b1_新种子81063', 'b2_新种子63862')) and not p.is_relative_to(OUT):
            raise PermissionError(f'B1 forbidden test-result read: {p}')
    elif event in ('os.remove', 'os.rename', 'os.rmdir'):
        ps = args[:2] if event == 'os.rename' else args[:1]
        if any(not Path(p).absolute().is_relative_to(OUT) for p in ps):
            raise PermissionError(f'B1 disallows destructive operation outside diagnostic directory: {event}')
    elif event == 'os.mkdir':
        p = Path(args[0]).absolute()
        if not p.is_relative_to(OUT) and not p.is_dir():
            raise PermissionError(f'B1 directory outside diagnostic directory: {p}')

sys.addaudithook(audit)
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
assert torch.cuda.is_available(), 'CPU runtime exceeds the estimated budget; stop.'

# Only in-memory initialization changes: keep original numerical functions unchanged.
# Remove author test-current loading and use the previously fixed simulator power.
P0_FIXED = 1.0542339586846165e-05
edits = []
class Sanitize(ast.NodeTransformer):
    def __init__(self, name): self.name = name
    def visit_Expr(self, node):
        if self.name == 'a11_device' and ast.unparse(node) == "sys.modules.pop('paper_check')":
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)
    def visit_Import(self, node):
        if self.name == 'a11_device' and any(a.name == 'paper_check' for a in node.names):
            edits.append([self.name, node.lineno, 'skip unused author test-check import'])
            return ast.copy_location(ast.Pass(), node)
        return node
    def visit_Assign(self, node):
        names = [x.id for x in node.targets if isinstance(x, ast.Name)]
        replacement = None
        if self.name == 'a11_device':
            if any(k in names for k in ('_pc', 'P_FIT')):
                return ast.copy_location(ast.Pass(), node)
            if 'P0' in names and node.lineno < 110: replacement = ast.Constant(P0_FIXED)
            if any(k in names for k in ('Y_REAL', 'LAM_TEST', 'TEST_ST', 'REC')): replacement = ast.Constant(None)
        if self.name in ('a11_peakfix', 'a15_three') and any(k in names for k in ('OUT',)):
            replacement = ast.Constant(str(OUT / 'runtime'))
        if self.name == 'a15_three' and any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == 'PF' and x.attr == 'OUT' for x in node.targets):
            replacement = ast.Constant(str(OUT / 'runtime'))
        if self.name == 'forward' and isinstance(node.targets[0], ast.Tuple):
            attrs = [x.attr for x in node.targets[0].elts if isinstance(x, ast.Attribute)]
            if attrs == ['lam_test', 'I_test']: replacement = ast.Tuple(elts=[ast.Constant(None), ast.Constant(None)], ctx=ast.Load())
        if replacement is not None:
            edits.append([self.name, node.lineno, ast.unparse(node.targets[0]) + ' initialization only'])
            node.value = replacement
        return self.generic_visit(node)

paths = {k: ALG / (k + '.py') for k in ('a15_three', 'a11_peakfix', 'a11_device')}
paths['forward'] = ALG / '45_Uddin2024_真实R矩阵/scripts/forward.py'
class Loader(importlib.abc.Loader):
    def __init__(self, name): self.name = name
    def create_module(self, spec): return None
    def exec_module(self, mod):
        p = paths[self.name]
        mod.__file__ = str(p)
        tree = Sanitize(self.name).visit(ast.parse(p.read_text(encoding='utf-8'), filename=str(p)))
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(p), 'exec'), mod.__dict__)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'forward' and Path(sys.path[0]).resolve() != paths['forward'].parent.resolve():
            return None
        if name in paths: return importlib.util.spec_from_loader(name, Loader(name))
sys.meta_path.insert(0, Finder())
sys.path.insert(0, str(ALG))
sys.argv = ['a15_three.py', '--third', 'cnet0', '--no-b', '--a16a', '32', '--a16a-ss', '--a16a-vscale', '--a16a-escale', '--test-seed', '31438']
import a15_three as M
import a16_ss
AD, PF, P18, BM = M.AD, M.PF, M.P18, M.BM
assert P18.VSEED == 31438 and M.ARG.third == 'cnet0' and M.A18
assert AD.Y_REAL is None and AD.LAM_TEST is None
assert AD.DV.I_test is None and AD.DV.lam_test is None

start = time.time()
def log(s):
    msg = f'[{time.time()-start:.1f}s] {s}'
    print(msg, flush=True)
    with (OUT / 'continue_run.log').open('a', encoding='utf-8') as f: f.write(msg+'\n')
M.log = PF.log = log
original_fams = AD.all_fams
def safe_fams(seed, n, P, jitter):
    assert seed in (31337, 31438, 31445, 31451, 31459, DSEED), seed
    if seed == DSEED:
        assert PHASE == 'evaluation_only', PHASE
    generation_calls.append(dict(generator='all_fams',seed=int(seed),n=int(n),
                                 jitter=bool(jitter),phase=PHASE))
    assert jitter, 'Fixed CIE spectra would overlap test spectra.'
    return original_fams(seed, n, P, jitter)
AD.all_fams = safe_fams
# Legacy build3 uses a separate seed-3 tuning validation set. Replace only that
# data source with the EXACT five-family member of VSEED=31438 (40 spectra).
# Keep noise seed 1004 and the existing parameter-selection algorithm.
old_seeds = P18.SEEDS
P18.SEEDS = [(31439, old_seeds[0][1], old_seeds[0][2])]
gen5_original = AD.D.gen5
def gen5_allowed(n, seed, band):
    if n == 60 and seed == 31439:
        generation_calls.append(dict(generator='gen5', seed=31439, requested_n=60, n=AD.NVF, phase=PHASE, adaptation=True))
        return gen5_original(AD.NVF, 31439, band)
    assert seed not in (3,6161,4242), 'Forbidden legacy/test seed.'
    if seed == DSEED+1:
        assert PHASE == 'evaluation_only', PHASE
    generation_calls.append(dict(generator='gen5',seed=int(seed),n=int(n),phase=PHASE))
    return gen5_original(n, seed, band)
AD.D.gen5 = gen5_allowed
# Preserve the existing A18 ParamE hyperparameter read from the allowed historical
# diagnostic logs, rather than tune a different A while diagnosing the current B.
AD.A.SIG_E[:] = [0.08]
dump('strict_validation_adapter.json', dict(legacy_seed_excluded=3,source='VSEED=31438 same-distribution five-family member',
     actual_gen5_seed=31439,actual_gen5_n=AD.NVF,fixed_sigma_e=0.08,
     fixed_sigma_source='A18/diag/scale_ev.out and fuse_diag2.out',
     reason='Use only explicitly allowed validation spectra and retain existing ParamE sigma; not an old-run bitwise reproduction'))

# Stored B and supporting networks must exist; never call a long network trainer.
assert (Path(M.NETDIR) / 'cnet0.pt').exists()
fit0 = M.fit_blend3
capture = {}
def fit_capture(Fv, Xs, St, b, **kw):
    result = fit0(Fv, Xs, St, b, **kw)
    mu, sd, net, va = result
    capture.update(Fv=Fv, Xs=Xs, St=St, va=va, W=M.blend3_w(Fv, mu, sd, net), net=net, mu=mu, sd=sd)
    return result
M.fit_blend3 = fit_capture

old_summary = json.loads((OLD/'summary.json').read_text(encoding='utf-8'))
for rel, expected in old_summary['architecture']['source_sha256'].items():
    assert hashlib.sha256((WORK/rel).read_bytes()).hexdigest() == expected, rel
assert hashlib.sha256((Path(M.NETDIR)/'cnet0.pt').read_bytes()).hexdigest() == old_summary['architecture']['B_sha256']
assert M.ARG.a16a == 32 and a16_ss.ALPHA == 0.75
log(f'PREREGISTERED diagnostic seed={DSEED}; no evaluation spectra generated before model freeze.')
log('Reconstructing old B1 from original training/validation only; no B retraining; q will be loaded.')
decode, P, used = M.build3()
closure = dict(zip(decode.__code__.co_freevars, (c.cell_contents for c in decode.__closure__)))
assert closure['kap']['t3'] == 0.25, closure['kap']

dump('runtime_initialization_changes.json', edits)
log(f'Build complete; captured {len(capture["St"])} validation readings.')
net = M.XN
architecture = dict(parameters=sum(p.numel() for p in net.parameters()), input=AD.N_RD,
                    output=int(BM.sum()), grid_points=len(BM), train_spectra=len(used['训练库']),
                    NVF=AD.NVF, noise_draws=AD.LD.NDRAW, VSEED=P18.VSEED,
                    P0_W=AD.P0, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    source_sha256={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(paths.values()) + [ALG/'a16_ss.py']},
                    B_sha256=hashlib.sha256((Path(M.NETDIR)/'cnet0.pt').read_bytes()).hexdigest())
dump('architecture.json', architecture)

def errors(X, S):
    return np.array([AD.LD.band_nrmse(x, S.T) for x in X])
def training_loss(Y, S, r):
    yr=Y.transpose(0,2,1).reshape(-1,Y.shape[1])
    with torch.no_grad():
        x=M.nrm_t(torch.nn.functional.softplus(M.XN(M.l1n(torch.tensor(yr,dtype=torch.float32,device=M.DEV))))).cpu().numpy()
    st = np.tile(S[:, BM].astype(np.float32), (Y.shape[0], 1))
    a = ((x-st)**2).mean(1)
    d = 0.5*r*((np.diff(x,axis=1)-np.diff(st,axis=1))**2).mean(1)
    return a, d

# Restore the EXACT q checkpoint from the validated old B1 run; never refit q.
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


old_freeze = json.loads((OUT.parent/'B1_新种子81063_20261104/model_freeze.json').read_text(encoding='utf-8'))
assert frozen_hashes == old_freeze['model_state_sha256'], (frozen_hashes, old_freeze)
roots = dict(decoder=decode, B=M.XN, B_scale=M.XN0, pool=M._POOL,
             q_net=qnet, q_mu=qmu, q_sd=qsd, q_lambda=qlam,
             RN=AD.RN, WL=AD.WL, BM=AD.BM,
             ss=dict(ALPHA=a16_ss.ALPHA, KMAX=a16_ss.KMAX, CFG=a16_ss.CFG, USE=a16_ss.USE, CG=a16_ss.CG))
full_freeze = numeric_state(roots)
dump('full_model_freeze.json', full_freeze)
assert full_freeze['sha256'] == '59f19648abb6ceab81cec07ca40d68c5ce188b38988d923738405e4c749976ce'
log('Resumed inference state equals the 15:16 registered full freeze; original freeze files preserved.')

def forbidden_fit(*args, **kwargs):
    raise RuntimeError('No fitting/selection after model freeze.')
M.fit_blend3=M.train_cnet=M.gf_a16a=PF.fit_q=forbidden_fit
P18.Consist.fit_cov=forbidden_fit

# q is already complete and immutable. Recompute only its statistics, never its predictions.
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

# Existing validation probe evidence was completed before original generation.
probe=json.loads((OUT/'scratch_probe_functional_check.json').read_text(encoding='utf-8'))
assert probe['status']=='passed' and not probe['evaluation_generated']
assert all(v==0 for v in probe['maximum_absolute_output_difference'].values())
assert numeric_state(roots)==full_freeze
log('Loaded previously passed validation-only scratch probe; all registered inference state matches again.')

PHASE='overlap_audit_only'
ss_tuning=AD.all_fams(31451,30,P,True)
used['SS调参集31451']=np.concatenate(list(ss_tuning.values()))
PHASE='evaluation_only'
log('Continuing saved 22658 spectra; no evaluation generation call.')
archive=np.load(OUT/'diagnostic_spectra.npz')
dsets={k:archive[k] for k in archive.files}
assert len(dsets)==26 and all(s.shape==(150,176) for s in dsets.values())
leak=P18.overlap(dsets,used)
assert not leak, leak
dump('overlap.json',dict(diagnostic_vs_used=leak,method='round 9 decimals exact rows; no forbidden sets generated',
    compared_sets={k:len(v) for k,v in used.items()}))
assert not any(c['generator']=='all_fams' and c['seed']==DSEED for c in generation_calls)
dump('continuation_generation_calls.json',dict(calls=generation_calls,
    evaluation_loaded_from_saved_npz=True,new_evaluation_generation_calls=0))
checkpoint=OUT/'checkpoint_after_progress_assertion'
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
    readings=len(samples),source_sha256=hashlib.sha256((checkpoint/'per_reading_metrics.csv').read_bytes()).hexdigest()))
g=np.random.default_rng(1022658)
for k,S in dsets.items():
    family=k.replace('留出·',''); Y=AD.meas(S.T,g,'real'); M._TR.clear()
    if family in completed_families:
        log('Preserved completed outputs; replayed measurement RNG only: '+family)
        continue
    o=decode(Y)
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
assert_frozen_with_scratch_audit('after_22658')
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
    seconds=time.time()-start,prior_q_process_seconds=796.1,prior_evaluation_last_logged_seconds=498.8,continued_existing_spectra=True,completed_family_predictions_reused=completed_families,peak_cuda_GB=torch.cuda.max_memory_allocated()/2**30,
    chol_fallback=a16_ss.NBAD)
dump('summary.json',summary)
log('Numerical evaluation finished; independent verification and report pending.')
