"""A18-B3：换一个从未用过的诊断种子，重定 A / B 的分工边界（AI，Claude Code 云端，2026-11-04）。
以 A18/diag/B1_诊断_20261104/diagnose_b1.py 为底，只改四处：诊断集种子和噪声种子可配置、去掉训练-验证差距段、
加多重比较校正（Bonferroni，26 族）后的三档分类、和 B1 旧种子结果并排比较。建模、q 拟合、只读防护与 B1 逐字相同。
运行（在 06_重建算法 下）：
  py -I -X utf8 A18/diag/B3_新种子_20261104/diagnose_b3_newseed.py [--dseed 31600] [--noise-seed 8191] [--n 30]
默认种子 31600（= VSEED + 162）和 8191 都不在 A18/HANDOFF_A18-R1.md 的种子表里，也不等于表中任何种子 ± 333（cie_jitter 偏移）。
脚本仍会把新诊断谱和训练库、各验证集、VSEED+7 / +13 / +21（jitter=True）逐条查重，有重合就停。
不训练 B，不生成、不读取任何测试谱和测试结果。全部为仿真。
"""
from pathlib import Path
import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv, argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--dseed', type=int, default=31600)
_ap.add_argument('--noise-seed', type=int, default=8191)
_ap.add_argument('--n', type=int, default=30)
ARG3 = _ap.parse_args()
DSEED, NOISE_SEED, NDIAG = ARG3.dseed, ARG3.noise_seed, ARG3.n
# R1 种子表（HANDOFF_A18-R1.md）里的值；新种子及其 ±333 都不能撞上
_USED = {20261102,20261103,20261104,20261105,31337,31344,31438,31445,31451,31459,9090,4242,6161,4243,4575,4742,5142,6162,6494,6661,7061,
         3,2,1004,11,5,777,77,1,991,779,4321,0,1901122,8675309,7340117,54321,18}
for _k in range(1,5):
    _USED |= {31337+1000*_k, 31337+1000*_k+7}
_USED |= set(range(20261102,20261160)) | set(range(24503102,24503155)) | set(range(26422102,26422155))
for _v in (DSEED, DSEED+333, NOISE_SEED):
    assert _v not in _USED and not (10_000_000 <= _v <= 10_002_047) and not (20_000_000 <= _v <= 20_000_255), f'种子 {_v} 和已用种子撞车'

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

assert not (OUT / 'summary.json').exists(), 'B3 已有完成结果，不覆盖；要重跑请新建目录。'
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
    assert seed in (31337, 31438, 31445, 31451, 31459, DSEED), seed
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

# B3：新诊断集 DSEED，每族 NDIAG 条，jitter=True。
dsets = AD.all_fams(DSEED,NDIAG,P,True)
# 查重：对训练库 / 各验证集（used），再对 VSEED+7、+13、+21 三个旧诊断集（都按 jitter=True 生成，避免 9 条固定 CIE 谱）。
# 不生成测试谱（和 B1 同一规矩）。
xd = {}
for nm, off, n_, _jit, _src in PF.XDIAG:
    xd[nm] = np.concatenate(list(original_fams(P18.VSEED + off, n_, P, True).values()))
leak = P18.overlap({k:v for k,v in dsets.items()},used)
leak_x = P18.overlap({k:v for k,v in dsets.items()},xd)
assert not leak and not leak_x, (leak, leak_x)
dump('overlap.json',dict(diagnostic_vs_used=leak, diagnostic_vs_old_diag=leak_x,
                         method='round 9 decimals, exact rows; no test spectra generated',
                         diagnostic_seed=DSEED,n_per_family=NDIAG,jitter=True,noise_seed=NOISE_SEED,
                         old_diag_sets=[nm for nm,*_ in PF.XDIAG]))
rows=[]; sample_rows=[]; per_spec={}
g=np.random.default_rng(NOISE_SEED)
for k,S in dsets.items():
    name=k.replace('留出·',''); Y=AD.meas(S.T,g,'real'); M._TR.clear(); o=decode(Y)
    fx,xa,de=PF.featx(o); q=PF.q_of(fx,qmu,qsd,qnet)
    o['final']=P18.unflat(PF.shift_np(xa,q*de),Y.shape[0])
    es={key:errors(o[key],S) for key in ('t3','t3c','gate','A11','final')}
    wA=1-o['w']; eA=es['gate']; eB=es['t3']; ec=es['t3c']; ef=es['final']
    delta=eB.mean()-eA.mean()
    # Paired uncertainty across spectra, retaining all four noise draws per spectrum.
    db=(eB-eA).mean(0); rr=np.random.default_rng(1901122)
    boot=db[rr.integers(0,len(db),(20000,len(db)))].mean(1)
    ci=np.quantile(boot,[.025,.975])
    ALPHA_B=0.05/26; cib=np.quantile(boot,[ALPHA_B/2,1-ALPHA_B/2])
    per_spec[name]=dict(d_B_minus_A=db.tolist(), eB=eB.mean(0).tolist(), eA=eA.mean(0).tolist(), eBc=ec.mean(0).tolist(), eF=ef.mean(0).tolist())
    row=dict(family=name,n_spectra=len(S),n_readings=eB.size,
             B_raw=float(eB.mean()),B_consistent=float(ec.mean()),A=float(eA.mean()),
             fusion_pre_q=float(es['A11'].mean()),final=float(ef.mean()),
             B_minus_A=float(delta),B_vs_A_percent=float(100*(eB.mean()/eA.mean()-1)),
             paired_delta_ci_low=float(ci[0]),paired_delta_ci_high=float(ci[1]),
             bonf_ci_low=float(cib[0]),bonf_ci_high=float(cib[1]),
             tier=('B 稳定胜' if cib[1]<0 else 'A 稳定胜' if cib[0]>0 else '边界·未校正显著' if (ci[1]<0 or ci[0]>0) else '跨零'),
             best_single=float(min(eA.mean(),ec.mean())),final_vs_best_single_percent=float(100*(ef.mean()/min(eA.mean(),ec.mean())-1)),
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

dump('per_spectrum_deltas.json', per_spec)

# ---- 分档汇总与 B1（VSEED+13）并排
tiers={}
for r_ in rows: tiers.setdefault(r_['tier'],[]).append(r_['family'])
b1csv = OUT.parent/'B1_诊断_20261104'/'family_metrics.csv'
cmp_rows=[]
if b1csv.exists():
    with b1csv.open(encoding='utf-8-sig') as f:
        b1={r_['family']:r_ for r_ in csv.DictReader(f)}
    for r_ in rows:
        o=b1.get(r_['family'])
        if o is None: continue
        same=(float(o['B_minus_A'])<0)==(r_['B_minus_A']<0)
        cmp_rows.append(dict(family=r_['family'],B_old=float(o['B_raw']),B_new=r_['B_raw'],A_old=float(o['A']),A_new=r_['A'],
                             final_old=float(o['final']),final_new=r_['final'],
                             B_vs_A_percent_old=float(o['B_vs_A_percent']),B_vs_A_percent_new=r_['B_vs_A_percent'],
                             winner_same=same,tier_new=r_['tier']))
    csvwrite('compare_with_B1.csv',cmp_rows)
    log(f'与 B1 旧种子比较：胜负方向一致 {sum(c["winner_same"] for c in cmp_rows)} / {len(cmp_rows)} 族')

gm=lambda a:float(np.exp(np.log(a).mean()))
summary=dict(task='A18-B3',author='AI（Claude Code 云端）',status='simulation_only',architecture=architecture,
             diagnostic=dict(seed=DSEED,n_per_family=NDIAG,jitter=True,mode='real',noise_seed=NOISE_SEED,bootstrap=20000,bonferroni_families=26),
             geometric_means={k:gm([v[k] for v in rows]) for k in ('B_raw','B_consistent','A','fusion_pre_q','final')},
             B_better_families=[v['family'] for v in rows if v['B_minus_A']<0],
             A_better_families=[v['family'] for v in rows if v['B_minus_A']>0],
             prefer_A=[v['family'] for v in rows if v['wA_mean']>.5],
             tiers=tiers,
             fusion_worse_than_best_single=[v['family'] for v in rows if v['final_vs_best_single_percent']>0],
             q_lambda=qlam,chol_fallback=a16_ss.NBAD,seconds=time.time()-start,
             peak_cuda_GB=torch.cuda.max_memory_allocated()/2**30,
             )
dump('summary.json',summary)
after=fingerprint(); before=json.loads((OUT/'existing_files_before.json').read_text(encoding='utf-8'))
changes={k:dict(before=before.get(k),after=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
dump('existing_file_changes.json',changes)
assert not changes, changes
log(f'Complete: {summary["seconds"]:.1f}s, existing files unchanged, all results are simulations.')

# ---- 报告（Markdown）
L=[f'# A18-B3：新诊断种子 {DSEED}（噪声种子 {NOISE_SEED}，每族 {NDIAG} 条 × {AD.LD.NDRAW} 次读数，real 模式）','',
   '全部为仿真（Uddin 2024 实测 R + 项目既定噪声模型）。B = cnet0 原始输出，A = A18 门控支（含 SS、escale），最终 = 融合 + q。',
   '成对 bootstrap 以谱为单位、保留 4 次读数，20000 次；Bonferroni 按 26 族校正（双侧 0.05/26）。','',
   '## 几何平均','',
   '| B 原始 | B 一致化 | A | 融合（q 前） | 最终 |','|---|---|---|---|---|',
   '| '+' | '.join(f'{summary["geometric_means"][k]:.4f}' for k in ('B_raw','B_consistent','A','fusion_pre_q','final'))+' |','',
   '## 分档','']
for t_ in ('B 稳定胜','A 稳定胜','边界·未校正显著','跨零'):
    L.append(f'- **{t_}**（{len(tiers.get(t_,[]))} 族）：'+('、'.join(tiers.get(t_,[])) or '无'))
L += ['','## 逐族','','| 谱族 | B | A | 最终 | B/A−1 % | 校正后区间（B−A） | 档 | 平均 w_A | 最终 vs 较好单支 % |','|---|---|---|---|---|---|---|---|---|']
for r_ in sorted(rows,key=lambda x:x['B_vs_A_percent']):
    L.append(f"| {r_['family']} | {r_['B_raw']:.4f} | {r_['A']:.4f} | {r_['final']:.4f} | {r_['B_vs_A_percent']:+.1f} | [{r_['bonf_ci_low']:+.4f}, {r_['bonf_ci_high']:+.4f}] | {r_['tier']} | {r_['wA_mean']:.2f} | {r_['final_vs_best_single_percent']:+.1f} |")
if cmp_rows:
    L += ['','## 与 B1（VSEED+13，A 的 SS 调参集）比较','','| 谱族 | B 旧 → 新 | A 旧 → 新 | B/A−1 % 旧 → 新 | 方向一致 |','|---|---|---|---|---|']
    for c_ in cmp_rows:
        L.append(f"| {c_['family']} | {c_['B_old']:.4f} → {c_['B_new']:.4f} | {c_['A_old']:.4f} → {c_['A_new']:.4f} | {c_['B_vs_A_percent_old']:+.1f} → {c_['B_vs_A_percent_new']:+.1f} | {'是' if c_['winner_same'] else '否'} |")
L += ['','## 注意','','- 新种子只避开了 R1 种子表里登记过的值；生成器 algo18 内部派生规则未核对，查重结果（overlap.json）是实际保障。',
      '- 本脚本没有生成测试谱，所以没有和 4242 / 6161 的测试谱逐条查重；诊断集不参与任何拟合，泄漏方向只可能是"诊断谱进了训练"，这一项已查。',
      '- 融合小网、κ、一致化协方差、q 仍按 B1 的适配在 VSEED=31438 上现拟，不是 6161 测试运行的逐位复现。']
(OUT/'报告.md').write_text('\n'.join(L),encoding='utf-8')
log('报告.md written')
