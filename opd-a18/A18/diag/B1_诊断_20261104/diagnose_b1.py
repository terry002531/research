"""A18-B1, AI (Codex). Read-only source execution; all writes stay beside this file.
Run: py -I -X utf8 A18/diag/B1_诊断_20261104/diagnose_b1.py
No B training and no project/author test-spectrum or test-current evaluation.
"""
from pathlib import Path
import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv

OUT = Path(__file__).resolve().parent
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
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def fingerprint():
    files = {}
    for p in WORK.rglob('*'):
        if not p.is_file() or p.is_relative_to(OUT):
            continue
        st = p.stat()
        files[str(p.relative_to(WORK))] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    return files

assert not (OUT / 'summary.json').exists(), 'Completed run exists; do not overwrite.'
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
        if not write and any(x in norm for x in ('a18_seed6161', 'a18_seed4242', 'testset_', 'peakfix.json', 'qnet.pt')) and not p.is_relative_to(OUT):
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
    with (OUT / 'run_validated.log').open('a', encoding='utf-8') as f: f.write(msg+'\n')
M.log = PF.log = log
original_fams = AD.all_fams
def safe_fams(seed, n, P, jitter):
    assert seed in (31337, 31438, 31445, 31451, 31459), seed
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
        return gen5_original(AD.NVF, 31439, band)
    assert seed != 3, 'Legacy seed-3 validation is forbidden in this task.'
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
    capture.update(Fv=Fv, Xs=Xs, St=St, va=va, W=M.blend3_w(Fv, mu, sd, net))
    return result
M.fit_blend3 = fit_capture

log('Building existing A18 from training/validation only; no B retraining.')
decode, P, used = M.build3()
dump('runtime_initialization_changes.json', edits)
log(f'Build complete; captured {len(capture["St"])} validation readings.')
net = M.XN
architecture = dict(parameters=sum(p.numel() for p in net.parameters()), input=AD.N_RD,
                    output=int(BM.sum()), grid_points=len(BM), train_spectra=len(used['训练库']),
                    NVF=AD.NVF, noise_draws=AD.LD.NDRAW, VSEED=P18.VSEED,
                    P0_W=AD.P0, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    source_sha256={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
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

# Original q procedure, fitted afresh exclusively on VSEED/QSEED.
qsets = AD.all_fams(31438, AD.NVF, P, True)
g = np.random.default_rng(PF.QSEED)
qF, qX, qD, qS = [], [], [], []
for k,S in qsets.items():
    Y = AD.meas(S.T, g)
    M._TR.clear()  # disable unused oracle computation
    o = decode(Y); fx, xa, de = PF.featx(o)
    qF.append(fx); qX.append(xa); qD.append(de); qS.append(np.tile(S,(AD.LD.NDRAW,1)))
    log('q validation features: '+k.replace('留出·',''))
qF,qX,qD,qS = [np.concatenate(v) for v in (qF,qX,qD,qS)]
rng = np.random.default_rng(5); idx=rng.permutation(len(qF)); split=int(.7*len(idx)); tr=(idx[:split],idx[split:]); va=tr[1]
pkm=PF.single_peak(qS[va]); pt=PF.peaks(qS[va]); e_old=float(PF.nrmse_t(torch.tensor(qX[va],device=PF.DEV),torch.tensor(qS[va],device=PF.DEV)).mean())
cand={}; q_scores={}
for lam in PF.LAMS:
    mu,sd,qn=PF.fit_q(qF,qX,qD,qS,lam,tr)
    q=PF.q_of(qF[va],mu,sd,qn); xp=PF.shift_np(qX[va],q*qD[va])
    e=float(AD.LD.band_nrmse(xp.T,qS[va].T).mean()); pe=float(np.abs(PF.peaks(xp)-pt)[pkm].mean())
    cand[lam]=(e,pe,mu,sd,qn); q_scores[str(lam)]=dict(error=e,peak_mae_nm=pe)
    log(f'q lambda {lam}: shape={e:.6f}, peak={pe:.6f}')
ok=[l for l in PF.LAMS if cand[l][0]<=e_old*1.005] or [min(PF.LAMS,key=lambda l:cand[l][0])]
qlam=min(ok,key=lambda l:cand[l][1]); _,_,qmu,qsd,qnet=cand[qlam]
torch.save(dict(state=qnet.state_dict(),mu=qmu,sd=qsd,lam=qlam,dmax=PF.DMAX),OUT/'q_validation_only.pt')
dump('q_selection.json',dict(selected=qlam,candidates=q_scores,baseline=e_old))
log(f'q selected {qlam}; diagnostic evaluation starts.')

# Diagnose one independent allowed set: VSEED+13, 30 spectra/family, jitter=True.
dsets = AD.all_fams(31451,30,P,True)
# Compare spectra to all used training/calibration sets, WITHOUT generating tests.
leak = P18.overlap({k:v for k,v in dsets.items()},used)
assert not leak, leak
dump('overlap.json',dict(diagnostic_vs_used=leak, method='round 9 decimals, exact rows; no test spectra generated',
                         diagnostic_seed=31451,n_per_family=30,jitter=True))
rows=[]; sample_rows=[]
g=np.random.default_rng(4321)
for k,S in dsets.items():
    name=k.replace('留出·',''); Y=AD.meas(S.T,g,'real'); M._TR.clear(); o=decode(Y)
    fx,xa,de=PF.featx(o); q=PF.q_of(fx,qmu,qsd,qnet)
    o['final']=P18.unflat(PF.shift_np(xa,q*de),Y.shape[0])
    es={key:errors(o[key],S) for key in ('t3','t3c','gate','A11','final')}
    wA=1-o['w']; eA=es['gate']; eB=es['t3']; ec=es['t3c']; ef=es['final']
    delta=eB.mean()-eA.mean()
    # Paired uncertainty across spectra, retaining all four noise draws per spectrum.
    db=(eB-eA).mean(0); rr=np.random.default_rng(1901122)
    boot=db[rr.integers(0,len(db),(2000,len(db)))].mean(1)
    ci=np.quantile(boot,[.025,.975])
    row=dict(family=name,n_spectra=len(S),n_readings=eB.size,
             B_raw=float(eB.mean()),B_consistent=float(ec.mean()),A=float(eA.mean()),
             fusion_pre_q=float(es['A11'].mean()),final=float(ef.mean()),
             B_minus_A=float(delta),B_vs_A_percent=float(100*(eB.mean()/eA.mean()-1)),
             paired_delta_ci_low=float(ci[0]),paired_delta_ci_high=float(ci[1]),
             wA_mean=float(wA.mean()),wA_median=float(np.median(wA)),A_weight_gt_half=float((wA>.5).mean()),
             A_win_vs_B_fraction=float((eA<eB).mean()),A_win_vs_Bc_fraction=float((eA<ec).mean()),
             final_minus_B=float((ef-eB).mean()),
             routing_better_branch_majority=float(((wA>.5)==(eA<ec)).mean()))
    rows.append(row)
    for j in range(len(S)):
        for z in range(Y.shape[0]):
            sample_rows.append(dict(family=name,spectrum=j,draw=z,B_raw=float(eB[z,j]),B_consistent=float(ec[z,j]),A=float(eA[z,j]),fusion_pre_q=float(es['A11'][z,j]),final=float(ef[z,j]),wA=float(wA[z,j])))
    log(f'{name}: B {row["B_raw"]:.5f} / A {row["A"]:.5f} / final {row["final"]:.5f}; wA {row["wA_mean"]:.3f}')
    dump('diagnostic_partial.json',rows)

def csvwrite(name,rs):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
csvwrite('family_metrics.csv',rows); csvwrite('per_reading_metrics.csv',sample_rows)

# Match training and validation noise/power distributions for a generalization gap.
lib=AD.all_fams(31337,AD.NTR,P,True); sg=used['训练库'][:,BM].astype(np.float32)
r=float((sg**2).mean()/(np.diff(sg,axis=1)**2).mean())
gaps=[]
for j,(k,Sv) in enumerate(qsets.items()):
    Si=lib[k]
    ii=np.random.default_rng(11).choice(len(Si),min(400,len(Si)),replace=False)
    St=Si[ii]
    sets={'train':St,'validation':Sv,'diagnostic':dsets[k]}; vals={}
    for label,S in sets.items():
        # Training-mode random power [0.5,2]P0 and identical device-noise model.
        Y=AD.meas(S.T,np.random.default_rng(810000+100*j),'train')
        X=M.t3_infer(AD.RN,Y); ee=errors(X,S); a,d=training_loss(Y,S,r)
        vals[label]=dict(shape_nrmse=float(ee.mean()),training_objective=float((a+d).mean()),value_mse=float(a.mean()),weighted_diff_mse=float(d.mean()),n_spectra=len(S))
    gaps.append(dict(family=k.replace('留出·',''),**{f'{l}_{v}':n for l,vs in vals.items() for v,n in vs.items()}))
csvwrite('train_validation_gap.csv',gaps)
dump('train_validation_gap.json',dict(r=r,families=gaps))

# Existing fusion-validation 70/30 results, no extra decoding.
cvrows=[]; n_each=AD.NVF*AD.LD.NDRAW
for j,k in enumerate(qsets):
    va=capture['va']; m=(va//n_each)==j; ii=va[m]
    cvrows.append(dict(family=k.replace('留出·',''),n_readings=len(ii),
                       A=float(AD.LD.band_nrmse(capture['Xs'][0][ii].T,capture['St'][ii].T).mean()),
                       B_consistent=float(AD.LD.band_nrmse(capture['Xs'][1][ii].T,capture['St'][ii].T).mean()),
                       wA=float(capture['W'][ii,0].mean())))
csvwrite('fusion_validation_holdout.csv',cvrows)
gm=lambda a:float(np.exp(np.log(a).mean()))
summary=dict(task='A18-B1',author='AI（Codex）',status='simulation_only',architecture=architecture,
             diagnostic=dict(seed=31451,n_per_family=30,jitter=True,mode='real',noise_seed=4321),
             geometric_means={k:gm([v[k] for v in rows]) for k in ('B_raw','B_consistent','A','fusion_pre_q','final')},
             B_better_families=[v['family'] for v in rows if v['B_minus_A']<0],
             A_better_families=[v['family'] for v in rows if v['B_minus_A']>0],
             prefer_A=[v['family'] for v in rows if v['wA_mean']>.5],
             q_lambda=qlam,chol_fallback=a16_ss.NBAD,seconds=time.time()-start,
             peak_cuda_GB=torch.cuda.max_memory_allocated()/2**30,
             training_objective_r=r,
             generalization_means={l:{metric:float(np.mean([v[f'{l}_{metric}'] for v in gaps])) for metric in ('shape_nrmse','training_objective')} for l in ('train','validation','diagnostic')})
dump('summary.json',summary)
after=fingerprint(); before=json.loads((OUT/'existing_files_before.json').read_text(encoding='utf-8'))
changes={k:dict(before=before.get(k),after=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
dump('existing_file_changes.json',changes)
assert not changes, changes
log(f'Complete: {summary["seconds"]:.1f}s, existing files unchanged, all results are simulations.')
