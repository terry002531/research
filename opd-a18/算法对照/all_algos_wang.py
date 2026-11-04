# -*- coding: utf-8 -*-
"""文献里的光谱重建 / 波长识别方法，在 Wang 2024 器件（47 号 forward.py）真实器件模式读数上重训、重测。

用户 2026-11-02："使用 A15 算法对比 A12、Xie MLP 还有常规算法，进行光谱重建。使用这个的数据进行重新的训练。"
方法和实现同 all_algos_cui_m3.py（方法清单 = 01_文献调研/已有重建算法/OPD光谱重建_已有算法与效果.md），只换器件和读数：
  器件 = Wang 2024，101 个偏压读数（0 到 −10 V），波段 400–900 nm，4 nm 网格；
  读数误差 = 47 号真实器件模式（a11_device.py wang：R 测量误差、定标、波长偏差、器件间差异、温度、光源、偏压、暗电流、散粒、读出），
  训练用训练器件（编号 1e7 起 2048 只），测试用测试器件（2e7 起 256 只），两批不重叠。
  · 要训练 / 调参的方法全部在训练口径读数上训练、调参：XieMLP、Wang NSF 用 a11_device.meas_t，随机森林、CS 的 α、A1 的 λ
    用 a11_device.meas 'train' 档；GCV 类经典方法每条读数自己挑 γ。
  · Wang NSF 就是这台器件的论文算法（结构同 44 号复现），这里同训练库、同读数误差重训。
  · 测试 = wang_eval（共用测试读数，同 A11 / A12 / A15）；本脚本重算的 A1 逐族误差和 results/a11/a11.json 比对，核对读数一致。
  py -I -X utf8 all_algos_wang.py          （SMOKE=1 缩小规模试跑）
输出：47_Wang2024_真实R矩阵/results/all_algos/{log.txt, all_algos.json, raw.pkl, xie_samelib.pt, nsf_samelib.pt}
"""
import os, sys, time, json, pickle
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = ROOT
while not os.path.isdir(os.path.join(PROJ, '03_仿真')) and os.path.dirname(PROJ) != PROJ:
    PROJ = os.path.dirname(PROJ)
sys.path.insert(0, os.path.join(PROJ, '03_仿真', '44_Wang2024_真实R矩阵', 'scripts'))
import _recon_workers as RW                                                      # nnls / lasso / atom：44 号的多进程子任务
sys.path.pop(0)

SMOKE = os.environ.get('SMOKE') == '1'
WDIR = os.path.join(ROOT, '47_Wang2024_真实R矩阵')
NAME = {'A1': '经典平滑非负（41 号）', 'T0NN': '高斯基 Tikhonov + 非负', 'T0': '高斯基 Tikhonov', 'Ridge': '岭回归', 'LapNN': 'Laplacian + 非负',
        'Hao': 'Hao 两步自适应', 'CS': '压缩感知', 'Gauss1': '单高斯 LM', 'Pinv': '直接伪逆', 'XieMLP': 'Xie MLP', 'NSF': 'Wang NSF（论文算法）',
        'LUT': '查表', 'RF': '随机森林'}
SRC = {'A1': '41 号', 'T0NN': 'He 2024 / Liu 2026 / Schrickx 2025 / Zhu 2025', 'T0': 'Han 2022', 'Ridge': 'Guo 2026（Zhang 2025、Qin 2026 未写细节）',
       'LapNN': 'Hao 2025 第一步', 'Hao': 'Hao 2025', 'CS': 'Zhu 2024', 'Gauss1': 'Dyson 2020', 'Pinv': 'Altaqui 2021', 'XieMLP': 'Xie 2026b',
       'NSF': 'Wang 2024', 'LUT': 'Shen 2021 / Song 2024 / Kirch 2022', 'RF': 'Xie 2026a'}
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


def chunks(x, k):
    return [x[i:i + k] for i in range(0, len(x), k)]


def lm_chunk(args):
    """Dyson 2020 单高斯 LM（同 44 号 RW.lm_chunk，起点改成本波段内）。"""
    Rb, wl, h, starts, Ys = args; out = []
    for y in Ys:
        def res(p):
            a = Rb @ RW.atom(wl, h, p[0], np.exp(np.clip(p[1], -1.2, 6.0))); A_ = max(a @ y / max(a @ a, 1e-30), 0)
            return A_ * a - y
        best = None
        for c0 in starts:
            for s0 in (5.0, 20.0):
                try:
                    r = least_squares(res, [c0, np.log(s0)], method='lm', max_nfev=400)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
        if best is None:
            out.append(np.zeros(len(wl))); continue
        c, s = best.x[0], np.exp(np.clip(best.x[1], -1.2, 6.0)); a = RW.atom(wl, h, c, s); ra = Rb @ a
        out.append(max(ra @ y / max(ra @ ra, 1e-30), 0) * a)
    return np.array(out)


class Tik:
    """同 44 号：min ||K c − y||² + γ ||L c||²，谱 = G c；γ 网格 σ1² × 10^[−12, 1]，每条读数用 GCV 挑（只看读数）。"""
    def __init__(s, K, L, G, ngam=66):
        s.K, s.L, s.G = K, L, G; n = K.shape[0]
        s.gam = np.linalg.svd(K, compute_uv=False)[0] ** 2 * np.logspace(-12, 1, ngam)
        B = np.vstack([np.eye(n), np.zeros((L.shape[0], n))])
        s.M = np.array([np.linalg.lstsq(np.vstack([K, np.sqrt(g) * L]), B, rcond=None)[0] for g in s.gam])
        s.H = np.einsum('ij,gjk->gik', K, s.M); s.tr = n - np.einsum('gii->g', s.H)

    def gcv(s, Yf):
        I, S = [], []
        for Yc in chunks(Yf, 1500):
            R = Yc[None] - np.einsum('gij,nj->gni', s.H, Yc); sc = (R ** 2).sum(2) / s.tr[:, None] ** 2
            i = sc.argmin(0); I.append(i); S.append(sc[i, np.arange(len(i))])
        return np.concatenate(I), np.concatenate(S)

    def free(s, Yf, i):
        C = np.zeros((len(Yf), s.M.shape[1]))
        for g in np.unique(i):
            C[i == g] = Yf[i == g] @ s.M[g].T
        return C @ s.G.T

    def nonneg(s, Yf, i, pool):
        items = [(np.sqrt(s.gam[g]), y) for g, y in zip(i, Yf)]
        return np.concatenate(pool.map(RW.nnls_chunk, [(s.K, s.L, it) for it in chunks(items, 60)])) @ s.G.T


def main():
    sys.argv = ['a11_device.py', 'wang'] + (['--quick'] if SMOKE else []); sys.path.insert(0, ROOT)
    import a11_device as AD                                                    # 只 import，不跑它的 main；器件、真实器件模式 meas / meas_t 随之就位
    import wang_eval as WE
    import torch, torch.nn as tnn
    from multiprocessing import Pool
    from sklearn.ensemble import RandomForestRegressor
    LD, A, D, H, P17, P18 = AD.LD, AD.A, AD.D, AD.H, AD.P17, AD.P18
    OUT = os.path.join(WDIR, 'results', 'all_algos' + ('_smoke' if SMOKE else '')); os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); WL, b = AD.WL, AD.BM; wl = WL[b]; h = (WL[1] - WL[0]) / 2; NWb = int(b.sum()); lo, hi = AD.BAND
    Rn = AD.RN; Rb = Rn[:, b]; n = AD.N_RD; nd = LD.NDRAW
    pool = Pool(15); DEV = 'cuda'
    log(f'器件 Wang 2024：{n} 个偏压读数，{lo:.0f}–{hi:.0f} nm，4 nm 网格 {NWb} 点；读数误差 = 47 号真实器件模式（训练器件 / 测试器件分开）')

    # ---------------- 经典方法的矩阵
    gb = lambda sg, cs: np.array([RW.atom(wl, h, c, sg) for c in cs]).T
    cen = np.arange(lo, hi + 0.1, 2.0); G5 = gb(5.0, cen); I_ = np.eye(NWb); D2 = np.diff(np.eye(NWb), 2, axis=0)
    TK = {'T0': Tik(Rb @ G5, np.eye(len(cen)), G5), 'Ridge': Tik(Rb, I_, I_), 'LapNN': Tik(Rb, D2, I_)}
    HW = [6, 10, 15, 20, 30, 50]; TH = {w: Tik(Rb @ gb(w / 2.3548, cen), np.eye(len(cen)), gb(w / 2.3548, cen)) for w in HW}
    cd = np.arange(lo, hi + 0.1, 4.0); Gd = np.concatenate([gb(sg, cd) for sg in (2.0, 5.0, 10.0, 20.0, 40.0)], 1)
    Dd = Rb @ Gd; dn = np.linalg.norm(Dd, axis=0); Dd = Dd / dn; Gdn = Gd / dn
    Pi = np.linalg.pinv(Rb, rcond=1e-15); starts = np.arange(lo + 20, hi, 40.0)

    def full(Xb):
        X = np.zeros((len(Xb), len(WL))); X[:, b] = Xb; return X

    def peaks_local(x, thr=0.3):
        x = np.clip(x, 0, None); m = x.max()
        return [wl[i] for i in range(1, len(x) - 1) if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= thr * m] if m > 0 else []

    def classic(Yf, alpha):
        Yn = Yf / np.linalg.norm(Yf, axis=1, keepdims=True); o = {}
        i0, _ = TK['T0'].gcv(Yn); o['T0'] = TK['T0'].free(Yn, i0); o['T0NN'] = TK['T0'].nonneg(Yn, i0, pool)
        ir, _ = TK['Ridge'].gcv(Yn); o['Ridge'] = TK['Ridge'].free(Yn, ir)
        il, _ = TK['LapNN'].gcv(Yn); o['LapNN'] = TK['LapNN'].nonneg(Yn, il, pool)
        cand = {}
        for w in HW:
            iw, sw = TH[w].gcv(Yn); cand[w] = (TH[w].nonneg(Yn, iw, pool), sw)
        Xh = np.zeros_like(o['LapNN'])
        for k in range(len(Yn)):
            anc = peaks_local(o['LapNN'][k]); best = None
            for w in HW:
                x, sc = cand[w][0][k], cand[w][1][k]; ok = (not anc) or min(abs(wl[np.argmax(x)] - a) for a in anc) <= 8
                if ok and (best is None or sc < best[0]):
                    best = (sc, x)
            Xh[k] = best[1] if best is not None else cand[min(HW, key=lambda w: cand[w][1][k])][0][k]
        o['Hao'] = Xh
        o['CS'] = np.concatenate(pool.map(RW.lasso_chunk, [(Dd, alpha, c) for c in chunks(list(Yn), 40)])) @ Gdn.T
        o['Gauss1'] = np.concatenate(pool.map(lm_chunk, [(Rb, wl, h, starts, c) for c in chunks(list(Yn), 20)]))
        o['Pinv'] = Yn @ Pi.T
        return o

    # ---------------- 训练库、验证集（照 a11_device.build 的种子重放；'train' 档 = 训练器件、真实器件模式、功率随机）
    sv, _, se = P18.SEEDS[0]; gv = np.random.default_rng(se)
    Sva = D.gen5(60, sv, AD.BAND)[0]; Yva = AD.meas(Sva.T, gv)
    E = lambda X, S: np.array([LD.band_nrmse(X[d], S.T) for d in range(X.shape[0])])
    lam = min(LD.NN_LAMS, key=lambda l: E(A.dec_snn(Rn, Yva, l), Sva).mean())
    P = H.Parts(Rn, AD.BAND); lib = AD.all_fams(P17.TRSEED, AD.NTR, P, True)
    Slib = np.concatenate([lib[k] for k in lib] + [D.gen5(len(lib['同分布五类']) * 3, P17.TRSEED + 7, AD.BAND)[0]])
    log(f'A1 的 λ（验证集挑）= {lam:g}；训练库 {len(Slib)} 条')

    Yvf = Yva.transpose(0, 2, 1).reshape(-1, n); Svf = np.tile(Sva, (nd, 1)); Yvn = Yvf / np.linalg.norm(Yvf, axis=1, keepdims=True)
    if SMOKE:
        Yvn, Svf = Yvn[:60], Svf[:60]
    ev = {}
    for al in np.logspace(-7, -3, 9):
        Xc = full(np.concatenate(pool.map(RW.lasso_chunk, [(Dd, al, cc) for cc in chunks(list(Yvn), 40)])) @ Gdn.T)
        ev[al] = float(LD.band_nrmse(Xc.T, Svf.T).mean())
    alpha = min(ev, key=ev.get); log(f'CS 的 α（验证集挑）= {alpha:.1e}；验证误差 ' + '，'.join(f'{k:.0e}:{v:.3f}' for k, v in ev.items()))

    # ---------------- 只认波长：查表、随机森林（单色光：单高斯，中心 U[lo, hi]，FWHM U[2, 10] nm；读数误差 = 训练口径）
    lc = np.arange(lo, hi + 0.01, 0.25); Lut = np.array([RW.atom(wl, h, c, 2 / 2.3548) for c in lc]) @ Rb.T
    Lut /= np.linalg.norm(Lut, axis=1, keepdims=True)
    gr = np.random.default_rng(77); Nm = 500 if SMOKE else 10000
    cm = lo + (hi - lo) * gr.random(Nm); fm = 2 + 8 * gr.random(Nm)
    Sm = np.zeros((Nm, len(WL))); Sm[:, b] = np.array([RW.atom(wl, h, c, f / 2.3548) for c, f in zip(cm, fm)])
    Sm /= Sm.max(1, keepdims=True); Ym = AD.meas(Sm.T, gr)
    Xrf = np.concatenate([y.T for y in Ym]); Xrf = Xrf / Xrf.sum(1, keepdims=True)
    rf = RandomForestRegressor(200, n_jobs=15, random_state=0).fit(Xrf, np.tile(cm, len(Ym)))
    log(f'查表 / 随机森林就绪 {time.time() - t0:.0f} s')

    # ---------------- XieMLP、NSF：和 A11 全族网络同一训练库、同一读数误差模型、同样步数
    torch.backends.cuda.matmul.allow_tf32 = True
    NSTEP, NBS = (300, 256) if SMOKE else (30000, 2048)
    bt = torch.tensor(b, device=DEV); St_ = torch.tensor(Slib, dtype=torch.float32, device=DEV)
    PE = lambda u: torch.cat([f(2 ** k * np.pi * u)[:, None] for k in range(6) for f in (torch.sin, torch.cos)], 1)
    LAM = np.arange(lo, hi + 1.0)                                                # NSF 解码查询：1 nm
    pe_grid = PE(torch.tensor((wl - lo) / (hi - lo), dtype=torch.float32, device=DEV))
    pe_1nm = PE(torch.tensor((LAM - lo) / (hi - lo), dtype=torch.float32, device=DEV))
    Bm = np.zeros((len(WL), len(LAM)))                                           # 1 nm → 4 nm 网格区间平均（同 44 号 bin_matrix）
    for i, l in enumerate(WL):
        w_ = np.clip(np.minimum(LAM + 0.5, l + h) - np.maximum(LAM - 0.5, l - h), 0, None)
        if w_.sum() > 0:
            Bm[i] = w_ / w_.sum()

    class XieMLP(tnn.Module):
        def __init__(s, n_, no, w=512):
            super().__init__(); s.i = tnn.Linear(n_, w); s.o = tnn.Linear(w, no)
            s.B = tnn.ModuleList([tnn.Sequential(tnn.Linear(w, w), tnn.ReLU(), tnn.Linear(w, w)) for _ in range(3)])

        def forward(s, y):
            z = torch.relu(s.i(y))
            for blk in s.B:
                z = torch.relu(z + blk(z))
            return s.o(z)

    class FE(tnn.Module):                                                       # Wang 2024 特征网络：12 层 × 1024，残差 + BN
        def __init__(s, n_, w=1024, L=12, nf=501):
            super().__init__(); s.i = tnn.Linear(n_, w); s.a = tnn.LeakyReLU(0.01); s.o = tnn.Linear(w, nf)
            s.L = tnn.ModuleList([tnn.Sequential(tnn.Linear(w, w), tnn.BatchNorm1d(w), tnn.LeakyReLU(0.01)) for _ in range(L)])

        def forward(s, y):
            z = s.a(s.i(y))
            for blk in s.L:
                z = z + blk(z)
            return s.o(z)

    class NSFq(tnn.Module):                                                     # 特征 + 位置编码 → 3 × 256 → 该波长强度
        def __init__(s, nf=501, w=256):
            super().__init__(); s.fe = FE(n); s.f = tnn.Linear(nf, w); s.p = tnn.Linear(12, w, bias=False)
            s.m = tnn.Sequential(tnn.LeakyReLU(0.01), tnn.Linear(w, w), tnn.LeakyReLU(0.01), tnn.Linear(w, w), tnn.LeakyReLU(0.01), tnn.Linear(w, 1))

        def forward(s, y, pe=None):
            return s.m(s.f(s.fe(y))[:, None, :] + s.p(pe_grid if pe is None else pe)[None]).squeeze(-1)

    def train_same(name, net, inorm, lr=1e-3, clip=None):
        f_ = os.path.join(OUT, f'{name}_samelib.pt')
        if os.path.exists(f_):
            net.load_state_dict(torch.load(f_)); log(f'  读缓存 {os.path.basename(f_)}'); return net.eval()
        torch.manual_seed(11); N_ = len(St_)
        opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=NSTEP)
        for it in range(NSTEP):
            if it % 50 == 0:
                gb_ = np.random.default_rng(11 * 100000 + it)
            s_ = St_[torch.randint(0, N_, (NBS,), device=DEV)]; y = inorm(H.meas_t(torch, Rn, s_, gb_, DEV))
            x = torch.nn.functional.softplus(net(y)); loss = ((x / (x.amax(1, keepdim=True) + 1e-6) - s_[:, bt]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step(); sch.step()
            if it % 5000 == 0 or it == NSTEP - 1:
                log(f'    {name} step {it} loss {loss.item():.5f}  {time.time() - t0:.0f} s')
        if not SMOKE:
            torch.save(net.state_dict(), f_)
        return net.eval()
    l2 = lambda y: y / (y.norm(dim=1, keepdim=True) + 1e-30); l1 = lambda y: y / (y.abs().sum(1, keepdim=True) + 1e-30)
    log(f'== 训练 Xie MLP / Wang NSF：训练库 {len(St_)} 条，{NSTEP} 步 × batch {NBS}（同 A11 全族网络，读数误差 = 真实器件模式）')
    # NSF 同 Uddin / Cui 版：峰值 lr 3e-4 + 梯度裁剪 1.0（1e-3 会发散）
    xs = train_same('xie', XieMLP(n, NWb).to(DEV), l1); ns = train_same('nsf', NSFq().to(DEV), l2, lr=3e-4, clip=1.0)

    def xie_same(Yf):
        with torch.no_grad():
            y = torch.tensor(Yf, dtype=torch.float32, device=DEV); P_ = torch.nn.functional.softplus(xs(l1(y)))
        return full(P_.double().cpu().numpy())

    RB = torch.tensor(Rn @ Bm, dtype=torch.float32, device=DEV)

    def nsf_decode(Yf, seed=0):
        """同 44 号 nsf_decode：查 1 nm 出谱 → 200 次随机 ±1 nm 平移的物理修正（留读数残差小的）→ 区间平均到 4 nm。"""
        Yr = torch.tensor(Yf, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            Pp = torch.cat([torch.nn.functional.softplus(ns(l2(y), pe_1nm)) for y in Yr.split(2000)])

            def err(Pc):
                p = Pc @ RB.T; a = (p * Yr).sum(1) / (p * p).sum(1).clamp_min(1e-30)
                return ((a[:, None] * p - Yr) ** 2).sum(1)
            g = torch.Generator(device=DEV).manual_seed(seed); e = err(Pp)
            for _ in range(200):
                up = torch.zeros_like(Pp); up[:, 1:] = Pp[:, :-1]; dn_ = torch.zeros_like(Pp); dn_[:, :-1] = Pp[:, 1:]
                cand = torch.where(torch.rand(len(Pp), 1, device=DEV, generator=g) < 0.5, up, dn_)
                ec = err(cand); acc = ec < e; Pp = torch.where(acc[:, None], cand, Pp); e = torch.where(acc, ec, e)
        return Pp.double().cpu().numpy() @ Bm.T

    def run_all(Y):
        """Y (nd, n, M) → {方法: (nd, NW, M)}，另返回只认波长的两种 {方法: (nd, M) 波长}。"""
        nd_, _, M = Y.shape; Yf = Y.transpose(0, 2, 1).reshape(nd_ * M, n)
        unf = lambda X: X.reshape(nd_, M, -1).transpose(0, 2, 1)
        r = {k: unf(full(v)) for k, v in classic(Yf, alpha).items()}
        r['XieMLP'] = unf(xie_same(Yf)); r['NSF'] = unf(nsf_decode(Yf))
        r['A1'] = np.clip(A.dec_snn(Rn, Y, lam), 0, None)
        Yn = Yf / np.linalg.norm(Yf, axis=1, keepdims=True)
        pk = {'LUT': lc[np.argmax(Yn @ Lut.T, 1)].reshape(nd_, M), 'RF': rf.predict(Yf / Yf.sum(1, keepdims=True)).reshape(nd_, M)}
        return r, pk

    # ---------------- 测试（共用测试读数）
    used = {'训练库': Slib, '调参验证集': Sva, '单色光训练集': Sm}
    sets, TR, F3 = WE.test_data(P, used)
    log('测试谱与训练库 / 验证集 / 单色光训练集逐条比对：无相同')
    if SMOKE:
        sets = {k: v for k, v in list(sets.items())[:3] + [(k, sets[k]) for k in WE.PKF]}
        F3 = dict(F3, Y=F3['Y'][:3])
    J, raw = WE.evaluate(run_all, sets, TR, F3, log)
    pool.close()
    fa = os.path.join(AD.OUT, 'a11.json')
    if os.path.exists(fa):
        A0 = json.load(open(fa, encoding='utf-8'))
        chk = max(abs(J['fam'][k]['A1'] - A0['fam']['device'][k]['A1']) for k in sets)
        log(f'读数一致核对：本脚本重算的 A1 逐族误差和 a11/a11.json 的最大差 {chk:.2e}（≈ 0 说明是同一份测试读数）')
    else:
        log('a11/a11.json 还没有，读数一致核对留给汇总脚本')

    METH = [m for m in NAME if m in J['fig3']]
    J['med'] = WE.table(J, METH, NAME, log)
    log(f'\n用时 {time.time() - t0:.0f} s')
    J.update(alpha=alpha, lam=lam, nte=AD.NTE, nsim=AD.NSIM, name=NAME, src=SRC)
    json.dump(J, open(os.path.join(OUT, 'all_algos.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    pickle.dump(raw, open(os.path.join(OUT, 'raw.pkl'), 'wb'))
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
