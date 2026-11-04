"""A18-B1, AI (Codex). Read-only source execution; all writes stay beside this file.
Run: py -I -X utf8 A18/diag/B1_新种子81063_20261104/diagnose_b1.py --diag-seed 81063
No B training and no project/author test-spectrum or test-current evaluation.
"""
from pathlib import Path
import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv
sys.path.insert(0,str(Path(__file__).resolve().parent))
from paired_stats import add_paired_statistics, summarize_statistics

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'B1_诊断_20261104'
import argparse
ap = argparse.ArgumentParser(description='Frozen B1 evaluation; seed 81063 only.')
ap.add_argument('--diag-seed', type=int, required=True)
RUN_ARGS = ap.parse_args()
assert RUN_ARGS.diag_seed == 81063, 'This preregistered runner only accepts 81063.'
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
    with (OUT / 'run.log').open('a', encoding='utf-8') as f: f.write(msg+'\n')
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

# Paired inference is evaluation only; no decoder parameters are selected.
rows = add_paired_statistics(rows, sample_rows)
csvwrite('family_metrics.csv',rows)
csvwrite('fusion_validation_holdout.csv',cvrows)
assert frozen_hashes == dict(B=net_hash(M.XN),fusion=net_hash(capture['net']),q=net_hash(qnet))
dump('model_post_evaluation.json',dict(model_state_sha256=frozen_hashes,unchanged=True))
gm=lambda a:float(np.exp(np.log(a).mean()))
summary=dict(task='A18-B1',author='AI（Codex）',status='simulation_only',architecture=architecture,
             diagnostic=dict(seed=DSEED,actual_five_family_seed=DSEED+1,actual_CIE_jitter_seed=DSEED+333,n_per_family=30,draws=4,jitter=True,mode='real',noise_seed=4321),
             geometric_means={k:gm([v[k] for v in rows]) for k in ('B_raw','B_consistent','A','fusion_pre_q','final')},
             B_better_families=[v['family'] for v in rows if v['B_minus_A']<0],
             A_better_families=[v['family'] for v in rows if v['B_minus_A']>0],
             prefer_A=[v['family'] for v in rows if v['wA_mean']>.5],
             q_lambda=qlam,chol_fallback=a16_ss.NBAD,seconds=time.time()-start,
             peak_cuda_GB=torch.cuda.max_memory_allocated()/2**30,
             statistics=summarize_statistics(rows),evaluation_only=True,old_seed=31451,
             old_B1_geometric_means=old_summary['geometric_means'],
             q_checkpoint_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest())
summary['comparison_with_31451']={k:dict(old=v,new=summary['geometric_means'][k],
    relative_change_percent=100*(summary['geometric_means'][k]/v-1))
    for k,v in old_summary['geometric_means'].items()}
summary['final_improvement_over_old_B_raw_percent']=100*(1-summary['geometric_means']['final']/summary['geometric_means']['B_raw'])
summary['A_useful_rule_old_B_reference_only']=dict(
    improvement_at_least_5_percent=summary['final_improvement_over_old_B_raw_percent']>=5,
    stable_A_families_at_least_5=len(summary['statistics']['A_stable_vs_B_raw'])>=5,
    new_B_not_implemented=True,conclusion='Cannot decide the preregistered NEW-B stopping rule from old B.')
dump('summary.json',summary)
after=fingerprint(); before=json.loads((OUT/'existing_files_before.json').read_text(encoding='utf-8'))
changes={k:dict(before=before.get(k),after=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
dump('existing_file_changes.json',changes)
assert not changes, changes
log(f'Complete: {summary["seconds"]:.1f}s, existing files unchanged, all results are simulations.')
