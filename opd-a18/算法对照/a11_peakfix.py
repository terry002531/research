# -*- coding: utf-8 -*-
"""A11 最后一步加峰位校正 → A11p（只 Uddin；a11_device.py 不改，导入它的 build()，网络读已训练的 net8.pt / mlp.pt）。

用户 2026-11-02："当我们在最后一步的时候出问题了，那不是前功尽弃了吗？…更新一下这个算法，最后的选择部分"。
问题（results/a11 的模拟 5 束，真实模式各 200 次）：融合小网只按形状误差（nrmse）学权重 w，92–100 % 跟网络支；
  门控支峰位更准（平均 |峰位误差| 0.75 nm，网络支 1.10 nm）却被丢掉，A11 输出 1.07 nm。
修法：A11 原融合不动，后面加一步——
  p_a = A11 输出峰位，p_g = 门控支峰位，d = p_g − p_a（|d| > DMAX 认为两支挑了不同的峰，不校正）；
  小网给 q ∈ [0, 1]，A11p = A11 输出整体平移 q·d（线性插值）：形状仍是网络支的，峰位往门控支靠，q = 0 就是旧 A11。
  q 的小网：特征 = A11 融合的 8 个特征 + 门控各类对数后验 + d、|d|、p_a、融合权重 w、两支半高宽；32 个隐单元。
  只用验证集训练：融合验证集的谱（训练族、种子 VSEED，测试谱一条不看）新抽一份读数，70 / 30 分；
  损失 = 形状误差 + λ·峰位 Huber（δ = 1 nm，只算单峰样本；峰位 = 平移后谱的抛物线峰位，对 q 可导）；
  λ 在 30 % 留出部分挑：形状误差不比旧 A11 差 0.5 % 以上的前提下，单峰样本平均 |峰位误差| 最小。
测试：重放 a11_device.main 的读数（同种子），比 门控支 / 网络支（一致化后）/ 旧 A11 / A11p，Xie MLP 从 all_algos.json 读。
  py -I -X utf8 a11_peakfix.py [--rerr] [--quick] [--noise-from-test-seed]
  （A18-R1 加：--noise-from-test-seed 见 noise_seed()，默认关；测试谱另和诊断集、其他测试种子扩查重，见 xcheck()，只报告。）
输出：45_Uddin2024_真实R矩阵/results/a11_peakfix(_R)/{peakfix.json, log.txt, qnet.pt}
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--rerr', action='store_true'); ap.add_argument('--quick', action='store_true')
ap.add_argument('--noise-from-test-seed', action='store_true',
                help='A18-R1：测试噪声和 5 束模拟的种子由测试种子派生（默认关 = 旧种子，逐位同改动前）；见 noise_seed()')
ARG = ap.parse_args()
sys.argv = [os.path.join(ROOT, 'a11_device.py'), 'uddin'] + (['--rerr'] if ARG.rerr else []) + (['--quick'] if ARG.quick else [])
sys.path.insert(0, ROOT)
import a11_device as AD
import torch

P18, LD = AD.P18, AD.LD
WL, BM = AD.WL, AD.BM; NW = len(WL); DW = float(WL[1] - WL[0])
OUT = os.path.join(AD.DEVDIR, 'results', 'a11_peakfix' + ('_R' if ARG.rerr else '') + ('_quick' if ARG.quick else '')
                   + ('_nts' if ARG.noise_from_test_seed else '')); os.makedirs(OUT, exist_ok=True)   # _nts：开关打开时另起目录，不覆盖旧结果（A18-R1）
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DMAX = 12.0                                             # |p_g − p_a| 超过它：两支挑了不同的峰，不校正
LAMS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)
QSEED = 9090                                            # q 训练读数的种子（谱 = 融合验证集）
METH = ['gate', 'cons', 'A11', 'A11p']
NAME = {'gate': '门控支', 'cons': '网络支', 'A11': '旧 A11', 'A11p': 'A11p', 'XieMLP': 'Xie MLP'}
PKF = ['窄线 4–10 nm', '单峰 15–60 nm']
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


# ------------------------------------------------------------------ A18-R1：噪声种子开关、扩查重
NTS = ARG.noise_from_test_seed


def noise_seed(off):
    """测试读数噪声（off = 模式序号 mi）和作者 5 束模拟读数（off = 50 + si）的种子。
    开关关（默认）：= AD.SEED_SIM + off，和 R1 之前逐位相同。默认不改的理由：A12 起所有 peakfix.json（含 A16–A18 的消融、
      4242 / 6161 两套结果）都用这组种子，"同种子同读数"的逐族对照全靠它；默认改了，旧结果就都不能再直接对照。
      代价（202144.txt 第 4 条）：噪声流和测试种子无关，4242 和 6161 的各族噪声是同一条流，5 束模拟读数从 A12 到 A18 是同一份。
    开关开：= AD.SEED_SIM + 1000·TEST_ST + off（off < 1000），不同测试种子的噪声各不相同。
      核对过（A18/HANDOFF_A18-R1.md 种子表）：TEST_ST = 4242 → 24 503 102–24 503 154，6161 → 26 422 102–26 422 154，
      和现有所有 default_rng 种子都不撞；不同测试种子的 1000·TEST_ST 至少差 1000 > off 上限，派生区间也不互撞；
      最小可能值 SEED_SIM + 1000 = 20 262 102 > SEED_SIM 一族（20 261 102–20 261 154）和 Wang 器件编号种子（≤ 20 000 255）。"""
    if not NTS:
        return AD.SEED_SIM + off
    assert AD.TEST_ST > 0 and 0 <= off < 1000, (AD.TEST_ST, off)
    return AD.SEED_SIM + 1000 * AD.TEST_ST + off


# 扩查重用的诊断集：生成参数必须和各诊断脚本一致（种子偏移、每族条数、jitter 都影响生成出的谱）
XDIAG = (('诊断集 VSEED+7', 7, 60, True, 'A17/diag/ss_int.py:65-66（ss_cls.py:65-66、ss_eval.py:72-73、ss_eval2.py:72-73 同参数）'),
         ('调参集 VSEED+13', 13, 30, True, 'A17/diag/ss_int.py:65-66（同上四个文件）'),
         ('诊断集 VSEED+21', 21, 30, False, 'A18/diag/scale_ev.py:53、A18/diag/fuse_diag2.py:38、A17/diag/fuse_diag.py:37'))
XTEST = (4242, 6161)                                    # 用过的测试种子，互相比对


def xcheck(sets, P):
    """测试谱再和诊断集 VSEED+7 / +13 / +21、其他测试种子的谱逐条比对（同 P18.overlap，四舍五入到 1e-9 完全相同才算）。
    只打印警告行（测试族 ∩ 集合、条数），不断言、不改任何结果；原有对训练库 / 验证集的断言不变（A18-R1 第 6 条）。
    生成这些谱只用各自显式的种子；保险起见前后保存、恢复 numpy / torch 全局随机状态，保证后面的读数逐位不变。"""
    st_np = np.random.get_state(); st_t = torch.get_rng_state(); st_c = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        oth = {nm: np.concatenate(list(AD.all_fams(P18.VSEED + off, n, P, jit).values())) for nm, off, n, jit, _ in XDIAG}
        for s in XTEST:
            if s != AD.TEST_ST:
                oth[f'测试种子 {s}'] = np.concatenate(list(AD.all_fams(s, AD.NTE, P, False).values()))
        hit = P18.overlap(sets, oth)
    finally:
        np.random.set_state(st_np); torch.set_rng_state(st_t)
        if st_c is not None:
            torch.cuda.set_rng_state_all(st_c)
    log(f'扩查重（只报告）：测试种子 {AD.TEST_ST} 的测试谱 对 ' + '、'.join(oth) + ('：无完全相同的谱' if not hit else f'：{len(hit)} 处重合'))
    for k, c in hit.items():
        log(f'   警告 扩查重：{k} 完全相同 {c} 条')
    return hit


_F = {}
_feats0 = P18.feats


def _feats(*a):
    """decode 里每次算的融合特征 (f8, lw) 存下来，峰位校正接着用。"""
    r = _feats0(*a); _F['last'] = r
    return r


P18.feats = _feats


# ------------------------------------------------------------------ 峰位、平移（numpy 与 torch 两版，算法相同）
def peaks(X):
    """(N, NW) → (N,) 峰位，= AD.peak_grid 逐行。"""
    xb = np.where(BM, np.clip(X, 0, None), 0); i = xb.argmax(1); r = np.arange(len(X)); ii = np.clip(i, 1, NW - 2)
    ok = (i > 0) & (i < NW - 1) & BM[ii - 1] & BM[ii + 1]
    y0, y1, y2 = xb[r, ii - 1], xb[r, ii], xb[r, ii + 1]; den = y0 - 2 * y1 + y2
    return WL[i] + np.where(ok & (den != 0), 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0) * DW


def width(X):
    xb = np.where(BM, np.clip(X, 0, None), 0)
    return (xb >= 0.5 * np.maximum(xb.max(1, keepdims=True), 1e-30)).sum(1) * DW


def single_peak(S):
    """真值只有一个高于 0.3 倍最大值的局部极大（单峰样本，峰位误差才有意义）。"""
    x = np.where(BM, S, 0); m = x.max(1, keepdims=True)
    lm = (x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] >= x[:, 2:]) & (x[:, 1:-1] >= 0.3 * m)
    return lm.sum(1) == 1


def shift_np(X, s):
    """x'(λ) = x(λ − s)，线性插值，波段外置零。"""
    u = np.arange(NW)[None] - s[:, None] / DW; i0 = np.floor(u).astype(int); f = u - i0
    g = lambda i: np.where((i >= 0) & (i < NW), np.take_along_axis(X, np.clip(i, 0, NW - 1), 1), 0)
    out = (1 - f) * g(i0) + f * g(i0 + 1); out[:, ~BM] = 0
    return out


BMt = torch.tensor(BM, device=DEV); WLt = torch.tensor(WL, dtype=torch.float64, device=DEV); AR = torch.arange(NW, device=DEV, dtype=torch.float64)


def shift_t(X, s):
    u = AR[None] - s[:, None] / DW; i0 = torch.floor(u).long(); f = u - i0

    def g(i):
        ok = (i >= 0) & (i < NW)
        return torch.where(ok, X.gather(1, i.clamp(0, NW - 1)), torch.zeros((), dtype=X.dtype, device=DEV))
    return torch.where(BMt[None], (1 - f) * g(i0) + f * g(i0 + 1), torch.zeros((), dtype=X.dtype, device=DEV))


def peak_t(X):
    xb = torch.where(BMt[None], X.clamp(min=0), torch.zeros((), dtype=X.dtype, device=DEV)); i = xb.detach().argmax(1); ii = i.clamp(1, NW - 2)
    ok = (i > 0) & (i < NW - 1) & BMt[ii - 1] & BMt[ii + 1]
    y0, y1, y2 = (xb.gather(1, (ii + k)[:, None])[:, 0] for k in (-1, 0, 1)); den = y0 - 2 * y1 + y2
    good = ok & (den.abs() > 1e-12)
    return WLt[i] + torch.where(good, 0.5 * (y0 - y2) / torch.where(good, den, torch.ones_like(den)), torch.zeros_like(den)) * DW


def nrmse_t(X, S):
    """同 algo18.fit_blend_net 的损失：峰值归一后均方根 / 真值极差（波段内）。"""
    x, s = X[:, BMt], S[:, BMt]
    xn = x / (x.amax(1, keepdim=True) + 1e-30); sn = s / s.amax(1, keepdim=True)
    return torch.sqrt(((xn - sn) ** 2).mean(1)) / (sn.amax(1) - sn.amin(1))


# ------------------------------------------------------------------ 特征与 q 小网
def featx(o):
    """decode 输出 → 特征 (N, nf)、A11 输出 (N, NW)、校正方向 d（超 DMAX 置零）。要紧跟在 decode 之后调用。"""
    f8, lw = _F['last']
    Xa, Xg = P18.flat(o['A11']), P18.flat(o['gate'])
    pa, pg = peaks(Xa), peaks(Xg); d = pg - pa; de = np.where(np.abs(d) <= DMAX, d, 0.0)
    Fx = np.c_[f8, lw, de, np.abs(de), (pa - WL[BM].mean()) / np.ptp(WL[BM]), o['w'].reshape(-1), width(Xa), width(Xg)]
    return Fx, Xa, de


def fit_q(Fv, Xa, de, St, lam, tr, steps=3000, seed=5):
    torch.manual_seed(seed)
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-9
    T = lambda a: torch.tensor(a, dtype=torch.float64, device=DEV)
    f, xa, d, s = T((Fv - mu) / sd), T(Xa), T(de), T(St)
    pt = T(peaks(St)); pk = T(single_peak(St).astype(float))
    net = torch.nn.Sequential(torch.nn.Linear(f.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double().to(DEV)
    with torch.no_grad():
        net[2].bias.fill_(-3.0)                                                   # 起点 q ≈ 0.05，接近旧 A11
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2)
    i_tr = torch.tensor(tr[0], device=DEV); i_va = torch.tensor(tr[1], device=DEV)

    def loss(i):
        q = torch.sigmoid(net(f[i]))[:, 0]; x = shift_t(xa[i], q * d[i])
        e = nrmse_t(x, s[i]); r = (peak_t(x) - pt[i]).abs()
        h = torch.where(r < 1, 0.5 * r ** 2, r - 0.5) * pk[i]
        return e.mean() + lam * h.sum() / pk[i].sum().clamp(min=1)
    best = (np.inf, None)
    for it in range(steps):
        l = loss(i_tr); opt.zero_grad(); l.backward(); opt.step()
        if it % 50 == 0:
            with torch.no_grad():
                lv = float(loss(i_va))
            if lv < best[0]:
                best = (lv, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    return mu, sd, net


def q_of(Fx, mu, sd, net):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor((Fx - mu) / sd, dtype=torch.float64, device=DEV)))[:, 0].cpu().numpy()


# ------------------------------------------------------------------ 主程序
def main():
    t0 = time.time()
    decode, P, used = AD.build()
    log(f'A11 建好 {time.time() - t0:.0f} s')

    # 1. q 训练数据：融合验证集的谱，新抽一份读数（训练模式），过一遍完整 A11
    vsets = AD.all_fams(P18.VSEED, AD.NVF, P, True); g = np.random.default_rng(QSEED); nd = LD.NDRAW
    Fv, Xa, de, St = [], [], [], []
    for k, S in vsets.items():
        o = decode(AD.meas(S.T, g)); Fx, xa, d = featx(o)
        Fv.append(Fx); Xa.append(xa); de.append(d); St.append(np.tile(S, (nd, 1)))
    Fv, Xa, de, St = (np.concatenate(v) for v in (Fv, Xa, de, St))
    assert np.allclose(peaks(Xa[:200]), [AD.peak_grid(x) for x in Xa[:200]])
    rng = np.random.default_rng(5); idx = rng.permutation(len(Fv)); tr = (idx[:int(0.7 * len(idx))], idx[int(0.7 * len(idx)):])
    va = tr[1]; pkm = single_peak(St[va]); pt = peaks(St[va])
    e_old = float(np.mean([AD.nrmse1(Xa[i], St[i]) for i in va])); p_all = np.abs(peaks(Xa[va]) - pt)[pkm]; p_old = float(p_all.mean())
    log(f'\nq 训练：验证谱 {len(vsets)} 族 {len(Fv)} 个样本（单峰 {single_peak(St).mean() * 100:.0f} %），|d| ≤ {DMAX:g} nm 的占 {(de != 0).mean() * 100:.0f} %；'
        f'30 % 留出部分旧 A11：形状误差 {e_old:.4f}，单峰 |峰位误差| 平均 {p_old:.3f}、中位 {np.median(p_all):.3f} nm')
    cand = {}
    for lam in LAMS:
        mu, sd, net = fit_q(Fv, Xa, de, St, lam, tr); q = q_of(Fv[va], mu, sd, net); Xp = shift_np(Xa[va], q * de[va])
        e = float(np.mean([AD.nrmse1(Xp[i], St[va][i]) for i in range(len(va))])); pa_ = np.abs(peaks(Xp) - pt)[pkm]; p = float(pa_.mean())
        cand[lam] = (e, p, mu, sd, net)
        log(f'   λ = {lam:<6g} 留出：形状误差 {e:.4f}（{(e / e_old - 1) * 100:+.1f} %），单峰 |峰位误差| 平均 {p:.3f}、中位 {np.median(pa_):.3f} nm；q 均值 {q.mean():.2f}')
    ok = [l for l in LAMS if cand[l][0] <= e_old * 1.005] or [min(LAMS, key=lambda l: cand[l][0])]
    lam = min(ok, key=lambda l: cand[l][1]); _, _, mu, sd, qnet = cand[lam]
    log(f'   选 λ = {lam:g}（形状不差于旧 A11 0.5 % 以内、峰位最小）')
    torch.save(dict(state=qnet.state_dict(), mu=mu, sd=sd, lam=lam, dmax=DMAX), os.path.join(OUT, 'qnet.pt'))

    def dec2(Y):
        o = decode(Y); Fx, xa, d = featx(o); q = q_of(Fx, mu, sd, qnet)
        o['A11p'] = P18.unflat(shift_np(xa, q * d), Y.shape[0]); o['q'] = q; o['d'] = d
        return o

    # 2. 生成谱族：重放 a11_device.main 的读数
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    leak = P18.overlap(sets, used); assert not leak, leak
    xcheck(sets, P)                                     # A18-R1：扩查重，只报告不断言
    if NTS:
        log(f'噪声种子由测试种子派生（--noise-from-test-seed）：测试读数 {noise_seed(0)} 起，5 束模拟 {noise_seed(50)} 起')
    XJ = json.load(open(os.path.join(AD.DEVDIR, 'results', 'all_algos', 'all_algos.json'), encoding='utf-8'))
    A0 = json.load(open(os.path.join(AD.OUT, 'a11.json'), encoding='utf-8'))
    A0['fam'] = {mo: {k.replace('留出·', ''): v for k, v in F.items()} for mo, F in A0['fam'].items()}
    modes = ['ideal', 'real'] + (['realR'] if ARG.rerr else [])
    E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
    J = dict(lam=lam, dmax=DMAX, fam={}, fampk={}, sim={}, qstat={})
    for mi, mo in enumerate(modes):
        g = np.random.default_rng(noise_seed(mi)); J['fam'][mo], J['fampk'][mo], J['qstat'][mo] = {}, {}, {}
        for nm, S in sets.items():
            o = dec2(AD.meas(S.T, g, mo))
            J['fam'][mo][nm] = {m: float(E(o[m], S).mean()) for m in METH}
            J['qstat'][mo][nm] = dict(q=float(o['q'].mean()), moved=float((np.abs(o['q'] * o['d']) > 0.5).mean()))
            if nm in PKF:
                tp = peaks(S)
                J['fampk'][mo][nm] = {m: float(np.median(np.abs(peaks(P18.flat(o[m])) - np.tile(tp, nd)))) for m in METH}
        dif = max(abs(J['fam'][mo][k]['A11'] - A0['fam'][mo][k]['A11']) for k in sets if k in A0['fam'].get(mo, {})) if mo in A0['fam'] else float('nan')
        log(f'\n== 生成谱族 · {mo}（每族 {AD.NTE} 条 × {nd} 次；形状误差 = 峰值归一均方根 / 真值极差）  重算旧 A11 与 a11.json 最大差 {dif:.2e}')
        xie = XJ['fam'].get(mo, {})
        log('   族                    ' + ''.join(f'{NAME[m]:>10s}' for m in METH + ['XieMLP']) + '   q 均值  挪动>0.5nm')
        for nm in sets:
            r = J['fam'][mo][nm]; xv = xie.get(nm, {}).get('XieMLP', np.nan)
            log(f'   {nm:<20s}' + ''.join(f'{r[m]:10.4f}' for m in METH) + f'{xv:10.4f}   {J["qstat"][mo][nm]["q"]:.2f}   {J["qstat"][mo][nm]["moved"] * 100:4.0f} %')
        med = {m: float(np.median([J['fam'][mo][k][m] for k in sets])) for m in METH}
        if xie:
            med['XieMLP'] = float(np.median([xie[k]['XieMLP'] for k in sets if k in xie]))
        J['fam'][mo]['_中位'] = med
        log('   各族中位              ' + ''.join(f'{med.get(m, np.nan):10.4f}' for m in METH + ['XieMLP']))
        nb = sum(J['fam'][mo][k]['A11p'] < J['fam'][mo][k]['A11'] - 1e-4 for k in sets); nw = sum(J['fam'][mo][k]['A11p'] > J['fam'][mo][k]['A11'] + 1e-4 for k in sets)
        log(f'   A11p 对旧 A11：形状误差变好 {nb} 族、变差 {nw} 族（差 > 1e-4）')
        for nm in PKF:
            r = J['fampk'][mo][nm]; xv = XJ.get('fampk', {}).get(mo, {}).get(nm, {}).get('XieMLP', np.nan)
            log(f'   {nm} 峰位 |误差| 中位（nm）：' + '，'.join(f'{NAME[m]} {r[m]:.3f}' for m in METH) + f'，Xie MLP {xv:.3f}')
        log(f'   用时 {time.time() - t0:.0f} s')

    # 3. 作者 5 束测试光的模拟读数（同 a11_device.main 的种子：真实模式 +50，realR +51，理想接着）
    import openpyxl
    fig = AD.PC.read_fig(openpyxl.load_workbook(AD.FW.XLSX, read_only=True, data_only=True), 'vis')
    refs = [fig[float(lt)] for lt in AD.LAM_TEST]
    truth = np.array([AD.to_grid(fl, ff) for _, _, fl, ff in refs]); ref_pk = np.array([AD.PC.peak(fl, ff) for _, _, fl, ff in refs])
    sim_modes = ['real'] + (['realR'] if ARG.rerr else []) + ['ideal']
    for si, mo in enumerate(sim_modes):
        rng = np.random.default_rng(noise_seed(50 + si)); J['sim'][mo] = {m: dict(dpk=[], err=[]) for m in METH}; qs = []
        for j, lt in enumerate(AD.LAM_TEST):
            nn = 1 if mo == 'ideal' else AD.NSIM
            if mo == 'realR':
                Yb = AD.meas(np.tile(truth[j][:, None], (1, nn)), rng, 'realR', nd=1, P=AD.P_FIT[j]).transpose(2, 1, 0)
            else:
                fl, ff = refs[j][2], refs[j][3]; p = np.clip(ff, 0, None); p = p / np.trapezoid(p, fl) * AD.P_FIT[j]
                Yb = np.array([AD.DV.measure(fl, p, dark=False, warn=False, real=mo != 'ideal', rng=rng) for _ in range(nn)])[:, :, None] / AD.SCALE
            o = dec2(Yb); qs.append(float(o['q'].mean()))
            for m in METH:
                X = o[m][:, :, 0]
                J['sim'][mo][m]['dpk'].append(list(peaks(X) - ref_pk[j])); J['sim'][mo][m]['err'].append([AD.nrmse1(x, truth[j]) for x in X])
        log(f'\n== 作者 5 束测试光（{", ".join(f"{v:.0f}" for v in AD.LAM_TEST)} nm）的模拟读数 · {mo}（各 {1 if mo == "ideal" else AD.NSIM} 次）：每束平均 |峰位误差| nm ｜ 形状误差平均')
        xs = XJ['sim'].get(mo, {}).get('XieMLP')
        for m in METH + (['XieMLP'] if xs else []):
            dp = np.abs(np.array(J['sim'][mo][m]['dpk'] if m != 'XieMLP' else xs['dpk'])); er = np.array(J['sim'][mo][m]['err'] if m != 'XieMLP' else xs['err'])
            ref = A0['sim'].get(mo, {}).get(m) if m in ('gate', 'A11') else None
            chk = f'（a11.json {np.abs(np.array(ref["dpk"])).mean():.3f}）' if ref else ''
            log(f'   {NAME[m]:<8s}' + ' '.join(f'{v:5.2f}' for v in dp.mean(1)) + f'  平均 {dp.mean():.3f}{chk} ｜ {er.mean():.4f}')
        log(f'   各束 q 均值：' + ' '.join(f'{v:.2f}' for v in qs))
    log(f'\n用时 {time.time() - t0:.0f} s')
    json.dump(J, open(os.path.join(OUT, 'peakfix.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
