"""AI (Codex): preregistered B2 evaluation, rebuild/freeze before new data.

No network optimizer or training step is executed. Read-only historical sources;
all generated files stay here, except explicitly authorized HANDOFF appends.
"""
from pathlib import Path
import sys, os, json, ast, time, csv, hashlib, types
from datetime import datetime
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
OLD = OUT.parent / 'B2_第六类_20261104'
HANDOFF = WORK / 'A18/HANDOFF_A18-B2.md'
START = time.perf_counter()
PHASE = 'historical_reconstruction'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    if name == 'execution_protocol.json' and (OUT / name).exists():
        assert not (OUT / 'model_freeze.json').exists()
        assert not (OUT / 'diagnostic_spectra.npz').exists()
        return
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def writecsv(name, rows):
    with (OUT / name).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def note(s):
    msg = f'[{time.perf_counter()-START:.1f}s] {s}'
    print(msg, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf-8') as f:
        f.write(msg + '\n')


assert not (OUT / 'model_freeze.json').exists() and not (OUT / 'diagnostic_spectra.npz').exists(), 'Do not repeat a frozen/evaluated run.'
HANDOFF_BEFORE = HANDOFF.read_bytes()
if not (OUT / 'HANDOFF_before_execution.md').exists():
    (OUT / 'HANDOFF_before_execution.md').write_bytes(HANDOFF_BEFORE)
save('execution_protocol.json', dict(author='AI（Codex）', timestamp=datetime.now().astimezone().isoformat(),
    seed=63862, n_per_family=30, draws=4, noise_seed=4321, jitter=True, mode='real',
    target_families=['阶梯谱','零件组合','随机光滑','同分布五类'],
    control_families=['单峰 60–200 nm','宽带通'], K=[64,256],
    K_semantics='Total requested spectral-kmeans components per family. Small clusters <3 dropped as original code. Both K reported; no selection.',
    covariance_shrinkage='Fixed to each family\'s final historical selected shrinkage for both K; no new validation selection.',
    original_selected_oracle='Also report six original selected configurations, including five-subtype K=320, separately from fixed spectral K=64/256.',
    oracle_candidate_universe='Original run_oracles.py last version; no new prior form or candidate selection.',
    unknown_family=dict(family_universe='Six listed families, equal family weights 1/6, all within-family weights retained.',
        K='Same frozen K=64/256 spectral prior files as known-family oracle.',
        noise='Estimate covariance using only archived VSEED=31438 repeated real readings and known calibration spectra. No new evaluation labels, amplitudes, powers, or DEVCOV passed.',
        estimator='For each old calibration spectrum regress unit-area readings onto [R*x, dark/||dark||]. Estimate white variance from residual RSS/(n-2). Estimate dark-direction variance as mean squared fitted dark coefficient minus white coefficient variance; clip at zero. Freeze before evaluation.',
        boundary='Calibration is at historical nominal operating conditions. This does not establish robustness to a new power distribution or arbitrary 26-family inputs.'),
    bootstrap=dict(replicates=200000, seed=1901122, unit='spectrum, all four readings paired',
        statistic='mean_spectrum mean_draw log(NRMSE_method/NRMSE_B)',
        uncorrected_quantiles=[0.025,0.975], bonferroni_quantiles=[0.00625,0.99375],
        scope='Four target families separately per method comparison; not simultaneous over all methods and K.',
        stable_improvement='upper corrected log CI <0 and paired geometric ratio <=0.95',
        controls='Uncorrected intervals only; exploratory controls.'),
    restrictions='No fitting/choice using 31451 or 63862; 31451 only a locked reproduction gate and retrospective statistics. Never load/generate 6161/4242 or load 81063 results. No B1/B3 decision.',
    reproduction_gate='All six families, all five old main-table methods: absolute mean NRMSE difference <1e-6; otherwise stop without generating 63862.',
    authorization='HANDOFF_A18-B2.md 2026-11-04 11:42'))

# Reuse the historical import sanitizer, never its training/evaluation drivers.
boot = (OLD / 'bootstrap_b2.py').read_text(encoding='utf-8')
boot = boot.replace('run_validated.log', 'run.log')
boot = boot.replace('strict_validation_adapter.json', 'strict_validation_adapter_execution.json')
# The HANDOFF is the only authorized append outside OUT.
boot = boot.replace("if write and not p.is_relative_to(OUT):",
                    "if write and not p.is_relative_to(OUT) and p != HANDOFF:")
exec(compile(boot, str(OLD / 'bootstrap_b2.py'), 'exec'), globals())
# Historical A's Gaussian components use the original seed 11 fitting library.
# This seed is recorded in B1's original used_() audit; no diagnostic fitting.
ALLOWED_GEN5.add(11)
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
# bootstrap installs a numerical logger; preserve one run.log and wall timer.
RUN_START = START
def log(s):
    msg = f'[{time.perf_counter()-RUN_START:.1f}s] {s}'
    print(msg, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf-8') as f: f.write(msg+'\n')
M.log = PF.log = log
tree = ast.parse((OLD / 'run_oracles.py').read_text(encoding='utf-8'))
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('fit_prior','posterior','metric')]
exec(compile(ast.Module(body=funcs, type_ignores=[]), str(OLD / 'run_oracles.py'), 'exec'), globals())

inputs = json.loads((OUT / 'input_inventory_before.json').read_text(encoding='utf-8'))
for rel, item in inputs['old_B2'].items(): assert sha(WORK / rel) == item['sha256'], rel
for rel, item in inputs['source'].items(): assert sha(WORK / rel) == item['sha256'], rel
assert sha(WORK / inputs['B']['path']) == inputs['B']['sha256']
z = np.load(OLD / 'allowed_data.npz', allow_pickle=False)
R = z['RN'][:, BM]; NOISE = z['DEVCOV']
oldrows = readcsv(OLD / 'oracle_metrics_complete_resume_v2.csv')
oldtable = {r['family']:r for r in readcsv(OLD / 'family_metrics_complete_resume_v2.csv')}

# Load networks directly, bypassing legacy train_* initializers entirely.
def no_train(*a, **kw): raise RuntimeError('No network training is authorized.')
M.train_cnet = M.fit_blend3 = PF.fit_q = no_train
P = AD.H.Parts(AD.RN, AD.BAND)
lib = AD.all_fams(31337, AD.NTR, P, True)
assert np.array_equal(np.concatenate(list(lib.values()) + [AD.D.gen5(9000,31344,AD.BAND)[0]]), z['full_train'])
ntpath = Path(AD.OUT) / 'net8.pt'
NT8 = dict(net=torch.nn.Sequential(torch.nn.Linear(AD.N_RD,512),torch.nn.ReLU(),
    torch.nn.Linear(512,512),torch.nn.ReLU(),torch.nn.Linear(512,512),torch.nn.ReLU(),
    torch.nn.Linear(512,P.na)).to(M.DEV), dev=M.DEV, torch=torch)
NT8['net'].load_state_dict(torch.load(ntpath, weights_only=True)); NT8['net'].eval()
musd = torch.load(str(ntpath)+'.musd', weights_only=False)
NT8.update(mu=musd['mu'], sd=musd['sd'])
M.XN = M.CNet(AD.N_RD, int(BM.sum())).to(M.DEV)
M.XN.load_state_dict(torch.load(Path(M.NETDIR)/'cnet0.pt', weights_only=True)); M.XN.eval()
specialists = {}
for name in ('specialist_512x4','specialist_768x6'):
    ck = torch.load(OLD/(name+'.pt'), map_location=M.DEV, weights_only=False)
    net = M.CNet(AD.N_RD,int(BM.sum()),w=ck['width'],nb=ck['blocks']).to(M.DEV)
    net.load_state_dict(ck['state']); net.eval(); specialists[name] = net

# Exactly the historical A construction inputs and RNG advancement.
g = np.random.default_rng(P18.SEEDS[0][2])
_ = AD.meas(AD.D.gen5(40,31439,AD.BAND)[0].T,g)
vsets = AD.all_fams(31438,40,P,True)
Yv = {k:AD.meas(s.T,g) for k,s in vsets.items()}
E = lambda X,S: np.array([AD.LD.band_nrmse(x,S.T) for x in X])
famg = [k for k in lib if k not in M.P17.PEAKY and k != '同分布五类']
GF = M.gf_a16a(AD.RN,lib,famg,vsets,Yv,E)
GM = AD.S7.GaussMR(AD.RN,AD.BAND,AD.F.C_TRUE)
PM = AD.D.ParamE(AD.RN,AD.BAND)
GSS = a16_ss.GateSS(AD.RN,AD.WL,AD.BAND,GM,GF.classes,dev=M.DEV)
def decode_A(Y):
    a = M.t3_infer(AD.RN,Y)[:,BM,:].max(1)
    yy = Y/np.maximum(a,1e-30)[:,None,:]
    c = a16_ss.escale(GF,yy)[:,None,:]; yy = yy*c
    xp = PM.solve(yy,0.08)[0]
    x8 = AD.H.solve_pick(P,yy,AD.H.infer(NT8,yy))[0]
    xf,df,wf = GF.solve_u(yy)
    gp = AD.DS.fuse_parts(AD.RN,xf,wf,df,[xp,x8],yy,GF,GF.classes)
    gate = np.clip(AD.DS.gate_mean(gp),0,None)
    gate = GSS.apply(yy,gate,wf,np.clip(M.t3_infer(AD.RN,yy),0,None))
    return gate/c

def predict(net,Y):
    yy = Y.transpose(0,2,1).reshape(-1,Y.shape[1])
    with torch.no_grad():
        xx = M.nrm_t(torch.nn.functional.softplus(net(M.l1n(torch.tensor(yy,dtype=torch.float32,device=M.DEV))))).cpu().numpy()
    out = np.zeros((len(xx),len(BM))); out[:,BM] = xx
    return out.reshape(Y.shape[0],Y.shape[2],len(BM)).transpose(0,2,1)

def fit_config(j,K,kind,shrink):
    a = z[f'tr{j}'][:,BM]; a = a/a.sum(1,keepdims=True)
    if kind == 'training_subtype_mixture':
        assert j == 3 and K % 5 == 0
        s0,l0,*_ = AD.D.gen5(3000,31338,AD.BAND)
        s1,l1,*_ = AD.D.gen5(9000,31344,AD.BAND)
        assert np.array_equal(np.concatenate([s0,s1]),z[f'tr{j}'])
        subtype = np.concatenate([l0,l1]); labels = np.zeros(len(a),dtype=int); offset=0
        for typ in np.unique(subtype):
            ii = np.flatnonzero(subtype==typ)
            labels[ii] = offset+KMeans(n_clusters=K//5,random_state=18,n_init=3,max_iter=150).fit_predict(a[ii])
            offset += K//5
    else:
        labels = KMeans(n_clusters=K,random_state=18,n_init=3,max_iter=150).fit_predict(a)
    return fit_prior(a,K,shrink,labels)

priors = {}; prior_rows=[]
for j,row in enumerate(oldrows):
    K = int(float(row['selected_K'])); shrink=float(row['shrink'])
    p = fit_config(j,K,row['prior_kind'],shrink)
    with np.load(OLD/f'prior_{j}.npz') as ref:
        diff={key:float(np.max(np.abs(p[key]-ref[key]))) for key in ('mu','cov','weight')}
        assert max(diff.values()) < 1e-12, (j,diff)
    priors['selected',j]=p
    np.savez_compressed(OUT/f'frozen_prior_selected_{j}.npz',**{k:p[k] for k in ('mu','cov','weight')})
    for fixedK in (64,256):
        pp=p if K==fixedK and row['prior_kind']=='spectral_kmeans' else fit_config(j,fixedK,'spectral_kmeans',shrink)
        priors[fixedK,j]=pp
        fname=f'frozen_prior_K{fixedK}_{j}.npz'
        np.savez_compressed(OUT/fname,**{k:pp[k] for k in ('mu','cov','weight')})
        prior_rows.append(dict(family=FAMILIES[j],requested_K=fixedK,actual_K=len(pp['mu']),
            prior_kind='spectral_kmeans',shrink=shrink,file=fname,sha256=sha(OUT/fname)))
    log(f'Reconstructed priors: {FAMILIES[j]}, selected K={K}, fixed K=64/256.')
writecsv('frozen_prior_manifest.csv',prior_rows)

# Calibration-only estimate, independent of truth power/DEVCOV in new inference.
d = AD.DARK/np.linalg.norm(AD.DARK); n=R.shape[0]
rss=[]; beta=[]; coeff_var=[]
for j in range(6):
    S=z[f'va{j}'][:,BM]; area=S/S.sum(1,keepdims=True)
    yy=z[f'yv{j}'].transpose(0,2,1)/S.sum(1)[None,:,None]
    for sid,ss in enumerate(area):
        B=np.column_stack([R@ss,d]); inv=np.linalg.inv(B.T@B)
        for obs in yy[:,sid]:
            coef=np.linalg.lstsq(B,obs,rcond=None)[0]; res=obs-B@coef
            rss.append(float(res@res)/(n-2)); beta.append(float(coef[1])); coeff_var.append(float(inv[1,1]))
white=float(np.mean(rss)); dark=float(max(0,np.mean(np.array(beta)**2)-white*np.mean(coeff_var)))
EST_NOISE=white*np.eye(n)+dark*np.outer(d,d)
np.savez_compressed(OUT/'frozen_estimated_noise.npz',cov=EST_NOISE,dark_direction=d,white_variance=white,dark_variance=dark)
save('noise_estimation.json',dict(calibration_seed=31438,calibration_readings=len(rss),white_variance=white,
    dark_direction_variance=dark,P_est_W=float(AD.ER['sigma_n']/(AD.SCALE*np.sqrt(white))),
    inference_uses_true_P0=False,inference_uses_true_DEVCOV=False,
    estimator='OLS signal and dark-direction removal on old unit-area calibration readings; fixed before new data.',
    caveat='Historical nominal-power calibration; no power-shift validation.'))

# Save the reconstructed A numerical state, including all supporting tensors.
def state(obj):
    return {k:v for k,v in vars(obj).items() if not isinstance(v,types.ModuleType)}
torch.save(dict(GF=state(GF),GM=state(GM),PM=state(PM),SS=state(GSS.SV),ss_idx=GSS.idx,
    NT8=NT8['net'].state_dict(),NT8_mu=NT8['mu'],NT8_sd=NT8['sd'],B=M.XN.state_dict(),
    sigma_e=0.08,ss_alpha=a16_ss.ALPHA,ss_CFG=a16_ss.CFG,escale_grid=a16_ss.CG),OUT/'frozen_A_state.pt')

log('All reconstruction finished. Reproducing old 31451 main table; no parameters can change.')
repro=[]; olderrors={}
for j,k in enumerate(FAMILIES):
    S=z[f'dg{j}']; Y=z[f'yd{j}']
    err={'B':E(predict(M.XN,Y),S),'A':E(decode_A(Y),S)}
    yy=Y.transpose(0,2,1).reshape(-1,n)
    err['oracle']=metric(posterior(priors['selected',j],yy,R,NOISE),np.tile(S,(4,1))).reshape(4,30)
    for name,net in specialists.items():err[name]=E(predict(net,Y),S)
    olderrors[k]=err
    for name,value in err.items():
        delta=float(value.mean())-float(oldtable[k][name])
        repro.append(dict(family=k,method=name,old=float(oldtable[k][name]),reconstructed=float(value.mean()),difference=delta,passed=bool(abs(delta)<1e-6)))
    log(f'31451 reproduction: {k}, maximum mean error difference {max(abs(r["difference"]) for r in repro if r["family"]==k):.3g}')
writecsv('reproduction_31451.csv',repro)
if not all(row['passed'] for row in repro):
    save('reproduction_failed.json',dict(status='STOPPED',new_seed_generated=False,failed=[r for r in repro if not r['passed']]))
    raise RuntimeError('Historical reproduction failed; 63862 stays unused.')

frozen_files=list(OUT.glob('frozen_*'))+[OUT/'reevaluate.py',OUT/'execution_protocol.json',OUT/'noise_estimation.json']
freeze={str(p.relative_to(WORK)):sha(p) for p in frozen_files if p.is_file()}
save('model_freeze.json',dict(phase=PHASE,at=datetime.now().astimezone().isoformat(),
    new_seed_generated=False,reproduction_passed=True,max_mean_NRMSE_difference=max(abs(r['difference']) for r in repro),
    files=freeze,old_candidate_code_sha256=sha(OLD/'run_oracles.py'),old_candidates_csv_sha256=sha(OLD/'oracle_candidates.csv'),
    no_network_training=True,unknown_family_count=6))
append='\n\n### 63862 开跑前冻结（AI：Codex）\n\n'
append+='按 11:42 授权仅用旧训练／验证数据确定性重建；31451 六族、五种方法全部复现，最大均值 NRMSE 差值 '+f'{max(abs(r["difference"]) for r in repro):.3g}'+'，均小于 1e-6。新谱尚未生成。\n\n'
append+='原候选代码 `diag/B2_第六类_20261104/run_oracles.py` SHA256：`'+sha(OLD/'run_oracles.py')+'`；原候选表 `oracle_candidates.csv` SHA256：`'+sha(OLD/'oracle_candidates.csv')+'`。\n\n'
append+='固定 K=64／256 表示每族 spectral-kmeans 请求总分量数，小簇仍按原代码剔除；协方差收缩逐族固定为旧最终配置。不挑 K。原最终选择（含五子类 K=320）另列，不混充 K=256。\n\n'
append+='不知族估计器只混合本轮六族，族权重各 1/6；使用同一两档 prior。噪声协方差只从旧 31438 校准读数估计：逐谱回归去除信号和暗电流方向，以残差估白噪声，再扣除系数白噪声方差估暗漂；不输入新评估的真谱、亮度、真功率或真 DEVCOV。该标定不能证明任意功率变化下有效。估计等效功率 '+f'{AD.ER["sigma_n"]/(AD.SCALE*np.sqrt(white))*1e6:.6f}'+' µW。\n\n'
append+='全部逐文件 SHA256 见 `diag/B2_新种子63862_20261104/model_freeze.json`，逐族两档 prior 见 `frozen_prior_manifest.csv`。关键文件：\n\n'
for rel,h in freeze.items():append+=f'- `{rel}`：`{h}`\n'
assert HANDOFF.read_bytes()==HANDOFF_BEFORE,'Concurrent HANDOFF edit; stop before new spectra.'
with HANDOFF.open('ab') as f:f.write(append.encode('utf-8'))
(OUT/'HANDOFF_after_freeze.md').write_bytes(HANDOFF.read_bytes())

# Hard phase boundary. No fitting entry point is callable from this point.
PHASE='evaluation_only'
M.gf_a16a=fit_prior=fit_config=no_train
KMeans.fit=KMeans.fit_predict=no_train
orig=original_fams
calls=[]
def eval_fams(seed,count,p,jitter):
    assert seed==63862 and count==30 and jitter and PHASE=='evaluation_only'
    assert not calls,'Generate the new evaluation set once.'
    calls.append(dict(seed=seed,n=count,jitter=jitter))
    return orig(seed,count,p,jitter)
AD.all_fams=eval_fams
AD.D.gen5=gen5_original
log('REPRODUCTION PASSED; hashes appended and frozen. Generating 63862 exactly once.')
dsets=AD.all_fams(63862,30,P,True)
def hashes(S):return {hashlib.sha256(row.tobytes()).hexdigest() for row in np.round(S,9)}
newhash=hashes(np.concatenate(list(dsets.values())))
references={'training':z['full_train'],'validation':np.concatenate(list(vsets.values())),
    'old_31451':np.concatenate([z[f'dg{j}'] for j in range(6)])}
overlap={name:len(newhash & hashes(S)) for name,S in references.items()}
assert not any(overlap.values()),overlap
save('overlap.json',dict(overlaps=overlap,method='Exact rows after 9-decimal rounding',test_data_generated=False))
g=np.random.default_rng(4321); yyall={}
for k,S in dsets.items(): yyall[k.replace('留出·','')]=AD.meas(S.T,g,'real')
Snew={k.replace('留出·',''):S for k,S in dsets.items()}
np.savez_compressed(OUT/'diagnostic_spectra.npz',**Snew)
np.savez_compressed(OUT/'diagnostic_readings.npz',**yyall)
save('generation_provenance.json',dict(calls=calls,actual_five_family_seed=63863,
    actual_CIE_jitter_seed=64195,noise_seed=4321,full_family_order=list(dsets),
    selected_reporting_families=FAMILIES,mode='real',draws=4,
    model_freeze_sha256=sha(OUT/'model_freeze.json')))

# Stream the exact historical quadrature to return evidence without a huge stack.
def posterior_evidence(prior,Y,noise):
    U=Y/np.linalg.norm(Y,axis=1,keepdims=True)
    hg,wg=np.polynomial.hermite.hermgauss(15)
    gains=np.exp(np.sqrt(2)*AD.ER['sigma_g']*hg); lg=np.log(wg)-.5*np.log(np.pi)
    hr,wr=np.polynomial.hermite.hermgauss(48)
    lm=np.full(len(Y),-np.inf); den=np.zeros(len(Y)); num=np.zeros((len(Y),R.shape[1]))
    for mu,C,pi in zip(prior['mu'],prior['cov'],prior['weight']):
        rm=R@mu; rc=R@C
        for gain,lp in zip(gains,lg):
            cov=gain*gain*rc@R.T+noise;cov=(cov+cov.T)/2
            cf=cho_factor(cov,lower=True,check_finite=False)
            ciu=cho_solve(cf,U.T,check_finite=False).T
            m=gain*rm;cim=cho_solve(cf,m,check_finite=False)
            aa=np.einsum('ij,ij->i',U,ciu);bb=U@cim;cc=float(m@cim)
            disc=np.sqrt(bb*bb+4*aa*(n-1))
            mode=np.where(bb>=0,(bb+disc)/(2*aa),2*(n-1)/(disc-bb))
            sd=1/np.sqrt(aa+(n-1)/mode**2)
            a=mode[:,None]+np.sqrt(2)*sd[:,None]*hr[None,:]
            valid=a>0;ap=np.maximum(a,np.finfo(float).tiny)
            expo=(n-1)*np.log(ap)-.5*(aa[:,None]*ap**2-2*bb[:,None]*ap+cc)
            lw=expo+hr[None,:]**2+np.log(wr)[None,:]+np.log(np.sqrt(2)*sd)[:,None]
            lw=np.where(valid,lw,-np.inf);lz=logsumexp(lw,axis=1)
            ea=np.sum(np.exp(lw-lz[:,None])*np.where(valid,a,0),axis=1)
            norm=-.5*(n*np.log(2*np.pi)+2*np.log(np.diag(cf[0])).sum())
            val=np.log(pi)+lp+norm+lz
            mean=mu+(ea[:,None]*U-m)@cho_solve(cf,gain*rc,check_finite=False)
            nm=np.maximum(lm,val);oldw=np.exp(lm-nm);neww=np.exp(val-nm)
            num=num*oldw[:,None]+mean*neww[:,None];den=den*oldw+neww;lm=nm
    return num/den[:,None],lm+np.log(den)

# Verify streaming implementation using historical validation, before its outputs.
calY=z['yv0'].transpose(0,2,1).reshape(-1,n)[:8]
cal1=posterior(priors[64,0],calY,R,EST_NOISE)
cal2,_=posterior_evidence(priors[64,0],calY,EST_NOISE)
streamdiff=float(np.max(np.abs(cal1-cal2)));assert streamdiff<1e-10,streamdiff
save('streaming_numerical_check.json',dict(old_validation_only=True,max_abs_difference=streamdiff))

errors={};predictions={}
for j,k in enumerate(FAMILIES):
    S=Snew[k];Y=yyall[k];flatY=Y.transpose(0,2,1).reshape(-1,n)
    err={'B':E(predict(M.XN,Y),S),'A':E(decode_A(Y),S)}
    for name,net in specialists.items():err[name]=E(predict(net,Y),S)
    for K in ('selected',64,256):
        x=posterior(priors[K,j],flatY,R,NOISE)
        label='oracle_selected' if K=='selected' else f'oracle_K{K}'
        err[label]=metric(x,np.tile(S,(4,1))).reshape(4,30)
        predictions[f'{label}_{j}']=x
    errors[k]=err
    np.savez_compressed(OUT/f'errors_family_{j}.npz',**err)
    log(f'63862 {k}: B={err["B"].mean():.6f}, A={err["A"].mean():.6f}, oracle64={err["oracle_K64"].mean():.6f}, oracle256={err["oracle_K256"].mean():.6f}')

allY=np.concatenate([yyall[k].transpose(0,2,1).reshape(-1,n) for k in FAMILIES])
for K in (64,256):
    xx=[];ll=[]
    for j in range(6):
        xs=[];ls=[]
        for starti in range(0,len(allY),120):
            x,l=posterior_evidence(priors[K,j],allY[starti:starti+120],EST_NOISE)
            xs.append(x);ls.append(l)
        xx.append(np.concatenate(xs));ll.append(np.concatenate(ls))
        log(f'Unknown-family K={K}: integrated candidate {j+1}/6; no true family supplied.')
    ll=np.stack(ll,1);weights=np.exp(ll-logsumexp(ll,axis=1,keepdims=True))
    mix=np.einsum('nk,knd->nd',weights,np.stack(xx))
    predictions[f'unknown_K{K}']=mix;predictions[f'unknown_K{K}_family_weights']=weights
    for j,k in enumerate(FAMILIES):
        x=mix[j*120:(j+1)*120]
        errors[k][f'unknown_oracle_K{K}']=metric(x,np.tile(Snew[k],(4,1))).reshape(4,30)
np.savez_compressed(OUT/'predictions.npz',**predictions)
np.savez_compressed(OUT/'all_errors.npz',**{f'{j}__{name}':v for j,k in enumerate(FAMILIES) for name,v in errors[k].items()})

def paired(a,b,target=True):
    assert a.shape==b.shape==(4,30)
    val=np.log(a/b).mean(0);rng=np.random.default_rng(1901122);boot=np.empty(200000)
    for starti in range(0,200000,10000):
        ii=rng.integers(0,30,(min(10000,200000-starti),30));boot[starti:starti+len(ii)]=val[ii].mean(1)
    lo,hi=np.quantile(boot,[.025,.975]);bl,bh=np.quantile(boot,[.00625,.99375])
    ratio=float(np.exp(val.mean()))
    return dict(log_ratio=float(val.mean()),geometric_ratio=ratio,improvement_percent=100*(1-ratio),
        ratio_of_arithmetic_means=float(a.mean()/b.mean()),
        log_ci95_low=float(lo),log_ci95_high=float(hi),ratio_ci95_low=float(np.exp(lo)),ratio_ci95_high=float(np.exp(hi)),
        log_ci_bonf_low=float(bl) if target else '',log_ci_bonf_high=float(bh) if target else '',
        ratio_ci_bonf_low=float(np.exp(bl)) if target else '',ratio_ci_bonf_high=float(np.exp(bh)) if target else '',
        corrected_improvement=bool(bh<0) if target else '',stable_5pct_improvement=bool(bh<0 and ratio<=.95) if target else '',
        correction_scope='4 target families per comparison' if target else 'control: uncorrected exploratory')

table=[];samples=[];stats=[];oldstats=[]
oldper=readcsv(OLD/'per_reading_complete_resume_v2.csv')
oldarr={}
for k in FAMILIES:
    oldarr[k]={name:np.empty((4,30)) for name in ('B','A','oracle','specialist_512x4','specialist_768x6')}
for row in oldper:
    for name in oldarr[row['family']]:oldarr[row['family']][name][int(row['draw']),int(row['spectrum'])]=float(row[name])
for k in FAMILIES:
    er=errors[k];table.append(dict(family=k,role='target' if k in TARGETS else 'control',**{name:float(v.mean()) for name,v in er.items()}))
    for sid in range(30):
        for draw in range(4):samples.append(dict(family=k,spectrum=sid,draw=draw,**{name:float(v[draw,sid]) for name,v in er.items()}))
    for name,v in er.items():
        if name!='B':stats.append(dict(family=k,comparison=name+'/B',**paired(v,er['B'],k in TARGETS)))
    for name,v in oldarr[k].items():
        if name!='B':oldstats.append(dict(family=k,comparison=name+'/B',**paired(v,oldarr[k]['B'],k in TARGETS)))
writecsv('family_metrics.csv',table);writecsv('per_reading_metrics.csv',samples)
writecsv('paired_comparisons.csv',stats);writecsv('paired_31451_reanalysis.csv',oldstats)
changes={rel:dict(before=item['sha256'],after=sha(WORK/rel)) for rel,item in inputs['old_B2'].items() if sha(WORK/rel)!=item['sha256']}
assert not changes,changes
assert all(sha(WORK/rel)==h for rel,h in freeze.items())
save('summary.json',dict(task='A18-B2 seed 63862',author='AI（Codex）',status='NUMERICAL_EVALUATION_COMPLETE_PENDING_VERIFICATION',
    simulation_only=True,diagnostic_generated_once=True,seed=63862,family_count=6,spectra=180,readings=720,
    K=[64,256],unknown_candidate_families=6,unknown_noise_estimated_from='old 31438 calibration only',
    no_network_training=True,no_parameter_choice_on_63862=True,reproduction_31451_passed=True,
    maximum_reproduction_difference=max(abs(r['difference']) for r in repro),
    bootstrap_replicates=200000,correction='Bonferroni 4 target families separately per comparison',
    results=table,old_B2_hash_changes=changes,frozen_artifacts_unchanged=True,
    seconds=time.perf_counter()-RUN_START,
    limitations=['Fixed noise model and R; simulation only.','Unknown-family estimator covers six families only and uses historical nominal-condition calibration.',
        'No strict information-theoretic lower bound.','Single diagnostic spectral seed; four repeats are not independent spectra.',
        'Corrected over four families per comparison, not over every method/K comparison.','No B1/B3 model decision.']))
log('Numerical evaluation and paired statistics saved; final independent verification still required.')
