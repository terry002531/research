# -*- coding: utf-8 -*-
"""Cui 2025 器件 · M3：A11 最后一步加峰位校正 → A12（= a11_peakfix.py 的 A11p，算法照搬，只换器件、读数误差和测试口径）。

用户 2026-11-02："你试一下 A 十二呢？A 十二的修正基于这个东西（M3），然后来重新弄一下，看一下性能有没有提升。"
a11_peakfix.py 写死了 Uddin（参考谱从 Uddin 的 xlsx 读），不能直接跑 Cui，所以另写本文件；它的函数整段照抄（import 它会把器件设成 Uddin）。
A12 做法（同 a11_peakfix）：A11 原融合不动；p_a = A11 峰位，p_g = 门控支峰位，d = p_g − p_a（|d| > DMAX 不校正）；
  小网给 q ∈ [0, 1]，A12 = A11 输出整体平移 q·d。q 小网只用融合验证集的谱（训练族、种子 VSEED）新抽一份 M3 训练口径读数来训，70 / 30 分，
  λ 在 30 % 留出部分挑：形状误差不比 A11 差 0.5 % 以上的前提下，单峰样本平均 |峰位误差| 最小。
测试（全部重放 a11_device.py cui --m3 的读数，同种子）：
  a. 生成谱族 · M3 读数（a = 1），种子 SEED_SIM + 3（那次模式顺序 ideal / recon / reading / m3）
  b. 作者 6 束真实读数（参考 = 标称波长）
  c. 同一批光的模拟 M3 读数 × NSIM，种子 SEED_SIM + 52（那次 sim 顺序 recon / reading / m3 / ideal）
  Xie MLP 从 all_algos_M3/all_algos.json 读，作对照。
  py -I -X utf8 a12_cui_m3.py
输出：46_Cui2025_真实R矩阵/results/a12_M3/{a12.json, log.txt, qnet.pt}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(ROOT, 'a11_device.py'), 'cui', '--m3']
sys.path.insert(0, ROOT)
import a11_device as AD
import torch

P18, LD = AD.P18, AD.LD
WL, BM = AD.WL, AD.BM; NW = len(WL); DW = float(WL[1] - WL[0])
OUT = os.path.join(AD.DEVDIR, 'results', 'a12_M3'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DMAX = 12.0                                             # |p_g − p_a| 超过它：两支挑了不同的峰，不校正（同 a11_peakfix）
LAMS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)
QSEED = 9090                                            # q 训练读数的种子（谱 = 融合验证集）
METH = ['gate', 'cons', 'A11', 'A12']
NAME = {'gate': '门控支', 'cons': '网络支', 'A11': 'A11', 'A12': 'A12', 'XieMLP': 'Xie MLP'}
PKF = ['窄线 4–10 nm', '单峰 15–60 nm']
NB5 = 5                                                 # 720 nm 在波段边缘，汇总只统计前 5 束
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


_F = {}
_feats0 = P18.feats


def _feats(*a):
    """decode 里每次算的融合特征 (f8, lw) 存下来，峰位校正接着用。"""
    r = _feats0(*a); _F['last'] = r
    return r


P18.feats = _feats


# ------------------------------------------------------------------ 峰位、平移（照抄 a11_peakfix）
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
    x, s = X[:, BMt], S[:, BMt]
    xn = x / (x.amax(1, keepdim=True) + 1e-30); sn = s / s.amax(1, keepdim=True)
    return torch.sqrt(((xn - sn) ** 2).mean(1)) / (sn.amax(1) - sn.amin(1))


# ------------------------------------------------------------------ 特征与 q 小网（照抄 a11_peakfix）
def featx(o):
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
        net[2].bias.fill_(-3.0)
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
    log(f'A11（M3 训练）建好 {time.time() - t0:.0f} s')

    # 1. q 训练数据：融合验证集的谱，新抽一份 M3 训练口径读数（a 对数均匀），过一遍完整 A11
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
        f'30 % 留出部分 A11：形状误差 {e_old:.4f}，单峰 |峰位误差| 平均 {p_old:.3f}、中位 {np.median(p_all):.3f} nm')
    cand = {}
    for lam in LAMS:
        mu, sd, net = fit_q(Fv, Xa, de, St, lam, tr); q = q_of(Fv[va], mu, sd, net); Xp = shift_np(Xa[va], q * de[va])
        e = float(np.mean([AD.nrmse1(Xp[i], St[va][i]) for i in range(len(va))])); pa_ = np.abs(peaks(Xp) - pt)[pkm]; p = float(pa_.mean())
        cand[lam] = (e, p, mu, sd, net)
        log(f'   λ = {lam:<6g} 留出：形状误差 {e:.4f}（{(e / e_old - 1) * 100:+.1f} %），单峰 |峰位误差| 平均 {p:.3f}、中位 {np.median(pa_):.3f} nm；q 均值 {q.mean():.2f}')
    ok = [l for l in LAMS if cand[l][0] <= e_old * 1.005] or [min(LAMS, key=lambda l: cand[l][0])]
    lam = min(ok, key=lambda l: cand[l][1]); _, _, mu, sd, qnet = cand[lam]
    log(f'   选 λ = {lam:g}（形状不差于 A11 0.5 % 以内、峰位最小）')
    torch.save(dict(state=qnet.state_dict(), mu=mu, sd=sd, lam=lam, dmax=DMAX), os.path.join(OUT, 'qnet.pt'))

    def dec2(Y):
        o = decode(Y); Fx, xa, d = featx(o); q = q_of(Fx, mu, sd, qnet)
        o['A12'] = P18.unflat(shift_np(xa, q * d), Y.shape[0]); o['q'] = q; o['d'] = d
        return o

    A0 = json.load(open(os.path.join(AD.OUT, 'a11.json'), encoding='utf-8'))                 # a11_M3/a11.json：核对重放
    XJ = json.load(open(os.path.join(AD.DEVDIR, 'results', 'all_algos_M3', 'all_algos.json'), encoding='utf-8'))
    J = dict(lam=lam, dmax=DMAX, fam={}, fampk={}, qstat={}, real={}, sim={})

    # 2a. 生成谱族 · M3 读数
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    leak = P18.overlap(sets, used); assert not leak, leak
    E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
    g = np.random.default_rng(AD.SEED_SIM + 3)
    for nm, S in sets.items():
        o = dec2(AD.meas(S.T, g, 'm3'))
        J['fam'][nm] = {m: float(E(o[m], S).mean()) for m in METH}
        J['qstat'][nm] = dict(q=float(o['q'].mean()), moved=float((np.abs(o['q'] * o['d']) > 0.5).mean()))
        if nm in PKF:
            tp = peaks(S)
            J['fampk'][nm] = {m: float(np.median(np.abs(peaks(P18.flat(o[m])) - np.tile(tp, nd)))) for m in METH}
    A0f = {k.replace('留出·', ''): v for k, v in A0['fam']['m3'].items()}
    dif = max(abs(J['fam'][k]['A11'] - A0f[k]['A11']) for k in sets)
    log(f'\n== 生成谱族 · M3 读数（a = 1；每族 {AD.NTE} 条 × {nd} 次；形状误差 = 峰值归一均方根 / 真值极差）  重算 A11 与 a11.json 最大差 {dif:.2e}')
    log('   族                    ' + ''.join(f'{NAME[m]:>10s}' for m in METH + ['XieMLP']) + '   q 均值  挪动>0.5nm')
    for nm in sets:
        r = J['fam'][nm]; xv = XJ['fam'][nm]['XieMLP']
        log(f'   {nm:<20s}' + ''.join(f'{r[m]:10.4f}' for m in METH) + f'{xv:10.4f}   {J["qstat"][nm]["q"]:.2f}   {J["qstat"][nm]["moved"] * 100:4.0f} %')
    med = {m: float(np.median([J['fam'][k][m] for k in sets])) for m in METH}; med['XieMLP'] = float(np.median([XJ['fam'][k]['XieMLP'] for k in sets]))
    J['med'] = med
    log('   各族中位              ' + ''.join(f'{med[m]:10.4f}' for m in METH + ['XieMLP']))
    nb = sum(J['fam'][k]['A12'] < J['fam'][k]['A11'] - 1e-4 for k in sets); nw = sum(J['fam'][k]['A12'] > J['fam'][k]['A11'] + 1e-4 for k in sets)
    log(f'   A12 对 A11：形状误差变好 {nb} 族、变差 {nw} 族（差 > 1e-4）')
    for nm in PKF:
        r = J['fampk'][nm]; xv = XJ.get('fampk', {}).get(nm, {}).get('XieMLP', np.nan)
        log(f'   {nm} 峰位 |误差| 中位（nm）：' + '，'.join(f'{NAME[m]} {r[m]:.3f}' for m in METH) + f'，Xie MLP {xv:.3f}')
    log(f'   用时 {time.time() - t0:.0f} s')

    # 2b. 作者 6 束真实读数（参考 = 标称波长）
    LT = AD.LAM_TEST.astype(float); rm = lambda a: float(np.sqrt(np.nanmean(np.square(a))))
    o = dec2((AD.Y_REAL / AD.SCALE)[None])
    for m in METH:
        J['real'][m] = dict(dpk=list(peaks(o[m][0].T) - LT))
    J['real']['q'] = list(o['q']); J['real']['d'] = list(o['d'])
    rchk = max(abs(x - y) for x, y in zip(A0['real']['A11']['dpk'], J['real']['A11']['dpk']))
    log(f'\n== 作者 6 束真实读数（{", ".join(f"{v:.0f}" for v in LT)} nm，参考 = 标称波长）：峰位误差 nm   RMSD 只算前 {NB5} 束   （重算 A11 与 a11.json 最大差 {rchk:.2e}）')
    for m in METH + ['XieMLP']:
        dp = np.array(J['real'][m]['dpk'] if m != 'XieMLP' else XJ['real']['XieMLP']['dpk'], float)
        log(f'   {NAME[m]:<8s}' + ' '.join(f'{v:+7.2f}' for v in dp) + f'   RMSD(5) {rm(dp[:NB5]):.2f}')
    log('   q：' + ' '.join(f'{v:.2f}' for v in o['q']) + '   d：' + ' '.join(f'{v:+.2f}' for v in o['d']))

    # 2c. 同一批光的模拟 M3 读数（重放 a11_device 的 'm3' 测试光）
    rng = np.random.default_rng(AD.SEED_SIM + 52); KB = AD.M3_B.shape[1]; J['sim'] = {m: dict(dpk=[]) for m in METH}; qs = []
    for j, lt in enumerate(LT):
        I0 = AD.DV.measure([lt], [AD.P0], lines=True, warn=False).ravel()
        Yb = np.array([AD.U.T @ (np.tile(1 + AD.M3_B @ rng.standard_normal(KB), AD._nV) * I0 + AD.FW.REAL['recon'] * rng.standard_normal(I0.size))
                       for _ in range(AD.NSIM)])[:, :, None] / AD.SCALE
        o = dec2(Yb); qs.append(float(o['q'].mean()))
        for m in METH:
            J['sim'][m]['dpk'].append(list(peaks(o[m][:, :, 0]) - lt))
    schk = np.nanmax(np.abs(np.array(A0['sim']['m3']['A11']['dpk'], float) - np.array(J['sim']['A11']['dpk'])))
    log(f'\n== 同一批光的模拟 M3 读数（各 {AD.NSIM} 次）：每束 |峰位误差| 平均 nm；RMSD / 中位只算前 {NB5} 束   （重算 A11 与 a11.json 最大差 {schk:.2e}）')
    for m in METH + ['XieMLP']:
        dp = np.array(J['sim'][m]['dpk'] if m != 'XieMLP' else XJ['sim']['XieMLP']['dpk'], float)
        log(f'   {NAME[m]:<8s}' + ' '.join(f'{v:6.2f}' for v in np.nanmean(np.abs(dp), 1)) + f'   RMSD(5) {rm(dp[:NB5]):6.2f}   中位 {np.nanmedian(np.abs(dp[:NB5])):.3f}')
    log('   各束 q 均值：' + ' '.join(f'{v:.2f}' for v in qs))
    log(f'\n用时 {time.time() - t0:.0f} s')
    json.dump(J, open(os.path.join(OUT, 'a12.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
