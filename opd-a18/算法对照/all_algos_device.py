# -*- coding: utf-8 -*-
"""文献里所有光谱重建 / 波长识别方法，在 Uddin 2024 器件模拟器（45 号 forward.py）上、和 A11 吃同一份读数比。

用户 2026-11-02："就用这个论文里面的这个模型的建模，本地应该有一个文件，它里面记载了所有有机半导体的光谱重建的算法。
  你把它全部都应用到这个上面去……我要来对比哪个算法最牛逼……A11 的算法你也就不需要重新训练了"。

方法清单 = 01_文献调研/已有重建算法/OPD光谱重建_已有算法与效果.md；各方法的实现照搬 44 号 all_methods.py（Wang 2024 R 上那一版），
只换器件和波段（500–800 nm），设置见那份文件的说明，这里不重复：
  T0NN 高斯基 Tikhonov + 非负 + GCV（He / Liu / Schrickx / Zhu 2025）  T0 同上不加非负（Han 2022）  Ridge 像素基岭回归 + GCV（Guo 2026）
  LapNN Laplacian Tikhonov + 非负 + GCV（Hao 第一步）  Hao 两步自适应  CS 压缩感知 + 最小二乘（Zhu 2024）  Gauss1 单高斯 LM（Dyson 2020）
  Pinv 直接伪逆（Altaqui 2021）  XieMLP 残差 MLP（Xie 2026b）  NSF 特征网络 + 神经光谱场 + 物理修正（Wang 2024）
  只认波长：LUT 查表（Shen 2021 / Song 2024 / Kirch 2022）、RF 随机森林（Xie 2026a）
  不适用：Jovanov 2018 傅里叶变换（要驻波扫描器件）、Koh 2026 卷积 + LSTM（LED 时间序列分类）、窄带像素直接读数（要像素阵列）
  另有：paper = Uddin 论文自己的算法（paper_check.Recon）；A1 = 41 号经典平滑非负；A11 / A11_R 读 a11_device.py 已有结果，不重训、不重跑。
控制变量：
  读数 = a11_device.py main() 的同一份（同种子、同抽样顺序重放），所以 A11 两版的分数直接读 results/a11{,_R}/a11.json；
  重放是否一致，用本脚本重算的 paper / A1 分数和 a11.json 里的逐族比对（日志里写最大差）。
  要训练的 XieMLP / NSF：和 A11 全族网络同一个训练库（87000 条）、同一个读数误差模型（a11_device.meas_t，真实模式误差每步重抽）、
  30000 步 × batch 2048、AdamW + OneCycle、同一个损失；只有网络结构不同。CS 的 α、A1 的 λ 用同一套生成验证集挑。
  py -I -X utf8 all_algos_device.py          （SMOKE=1 缩小规模试跑）
输出：45_Uddin2024_真实R矩阵/results/all_algos/{log.txt, all_algos.json, xie_samelib.pt, nsf_samelib.pt}
"""
import os, sys, time, json
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
NAME = {'A11': 'A11', 'A11_R': 'A11（训练含 R 误差）', 'paper': 'Uddin 论文算法', 'A1': '经典平滑非负（41 号）',
        'T0NN': '高斯基 Tikhonov + 非负', 'T0': '高斯基 Tikhonov', 'Ridge': '岭回归', 'LapNN': 'Laplacian + 非负', 'Hao': 'Hao 两步自适应',
        'CS': '压缩感知', 'Gauss1': '单高斯 LM', 'Pinv': '直接伪逆', 'XieMLP': 'Xie MLP', 'NSF': 'Wang NSF', 'LUT': '查表', 'RF': '随机森林'}
SRC = {'A11': '本项目', 'A11_R': '本项目', 'paper': 'Uddin 2024', 'A1': '41 号', 'T0NN': 'He 2024 / Liu 2026 / Schrickx 2025 / Zhu 2025',
       'T0': 'Han 2022', 'Ridge': 'Guo 2026（Zhang 2025、Qin 2026 未写细节）', 'LapNN': 'Hao 2025 第一步', 'Hao': 'Hao 2025', 'CS': 'Zhu 2024',
       'Gauss1': 'Dyson 2020', 'Pinv': 'Altaqui 2021', 'XieMLP': 'Xie 2026b', 'NSF': 'Wang 2024', 'LUT': 'Shen 2021 / Song 2024 / Kirch 2022', 'RF': 'Xie 2026a'}
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


def main(build_only=False):
    """build_only=True：只建好所有方法就返回（给 algos_metrics.py 用），不跑评测。"""
    sys.argv = ['a11_device.py', 'uddin']; sys.path.insert(0, ROOT)
    import a11_device as AD                                                    # 只 import，不跑它的 main；器件、读数误差、meas_t 补丁随之就位
    import torch, torch.nn as tnn
    from multiprocessing import Pool
    from sklearn.ensemble import RandomForestRegressor
    LD, A, D, H, P17, P18 = AD.LD, AD.A, AD.D, AD.H, AD.P17, AD.P18
    AD.LEV_R = A.LEVELS['L5']                                                   # 只影响 'realR' 读数（'train' 只在 --rerr 时乘 R 误差）
    OUT = os.path.join(AD.DEVDIR, 'results', 'all_algos' + ('_smoke' if SMOKE else '')); os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); WL, b = AD.WL, AD.BM; wl = WL[b]; h = (WL[1] - WL[0]) / 2; NWb = int(b.sum()); lo, hi = AD.BAND
    Rn = AD.RN; Rb = Rn[:, b]; n = AD.N_RD; nd = LD.NDRAW
    pool = Pool(15); DEV = 'cuda'
    log(f'器件 Uddin 2024 可见段：{n} 个 Vds 读数，{lo:.0f}–{hi:.0f} nm，4 nm 网格 {NWb} 点；读数 = a11_device.meas 同一份')

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

    # ---------------- 训练库、验证集、测试谱（全部照 a11_device.build / main 的种子重放）
    sv, _, se = P18.SEEDS[0]; gv = np.random.default_rng(se)
    Sva = D.gen5(60, sv, AD.BAND)[0]; Yva = AD.meas(Sva.T, gv)
    E = lambda X, S: np.array([LD.band_nrmse(X[d], S.T) for d in range(X.shape[0])])
    lam = min(LD.NN_LAMS, key=lambda l: E(A.dec_snn(Rn, Yva, l), Sva).mean())
    P = H.Parts(Rn, AD.BAND); lib = AD.all_fams(P17.TRSEED, AD.NTR, P, True)
    Slib = np.concatenate([lib[k] for k in lib] + [D.gen5(len(lib['同分布五类']) * 3, P17.TRSEED + 7, AD.BAND)[0]])
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    if SMOKE:
        sets = {k: v[:4] for k, v in list(sets.items())[:4]}
    leak = P18.overlap(sets, {'训练库': Slib, '调参验证集': Sva})
    log(f'A1 的 λ = {lam:g}（a11_device 日志 17.7828）；训练库 {len(Slib)} 条；测试谱与训练库 / 验证集完全相同的条数：' + ('无' if not leak else str(leak)))
    assert not leak, leak

    Yvf = Yva.transpose(0, 2, 1).reshape(-1, n); Svf = np.tile(Sva, (nd, 1)); Yvn = Yvf / np.linalg.norm(Yvf, axis=1, keepdims=True)
    if SMOKE:
        Yvn, Svf = Yvn[:60], Svf[:60]
    ev = {}
    for al in np.logspace(-7, -3, 9):
        Xc = full(np.concatenate(pool.map(RW.lasso_chunk, [(Dd, al, cc) for cc in chunks(list(Yvn), 40)])) @ Gdn.T)
        ev[al] = float(LD.band_nrmse(Xc.T, Svf.T).mean())
    alpha = min(ev, key=ev.get); log(f'CS 的 α（生成验证集挑）= {alpha:.1e}；验证误差 ' + '，'.join(f'{k:.0e}:{v:.3f}' for k, v in ev.items()))

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
    log(f'== 训练 Xie MLP / Wang NSF：训练库 {len(St_)} 条，{NSTEP} 步 × batch {NBS}（同 A11 全族网络）')
    # NSF（12 × 1024 深网）在 lr 1e-3 峰值处发散（第 1 次全量：step 5000 loss 0.015 → step 10000 起卡在 0.25），改峰值 3e-4 + 梯度裁剪 1.0
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
        r['A1'] = np.clip(A.dec_snn(Rn, Y, lam), 0, None); r['paper'] = AD.paper_decode(Y)
        Yn = Yf / np.linalg.norm(Yf, axis=1, keepdims=True)
        pk = {'LUT': lc[np.argmax(Yn @ Lut.T, 1)].reshape(nd_, M), 'RF': rf.predict(Yf / Yf.sum(1, keepdims=True)).reshape(nd_, M)}
        return r, pk

    if build_only:
        return dict(AD=AD, LD=LD, run_all=run_all, pool=pool, E=E, used={'训练库': Slib, '调参验证集': Sva, '单色光训练集': Sm}, t0=t0)

    # ---------------- a. 生成谱族：重放 a11_device.main 的读数（--rerr 那次的模式顺序 ideal / real / realR，种子 SEED_SIM + 序号）
    modes = ['ideal', 'real', 'realR']; fam, fampk = {}, {}
    PKF = ['窄线 4–10 nm', '单峰 15–60 nm']                                       # 单峰族另算峰位误差（只认波长的方法也能比）
    for mi, mo in enumerate(modes):
        g = np.random.default_rng(AD.SEED_SIM + mi); fam[mo], fampk[mo] = {}, {}
        for nm, S in sets.items():
            Y = AD.meas(S.T, g, mo); r, pk = run_all(Y)
            fam[mo][nm] = {m: float(E(X, S).mean()) for m, X in r.items()}
            if nm in PKF:
                tp = np.array([AD.peak_grid(s) for s in S])
                d = {m: [abs(AD.peak_grid(X[k][:, i]) - tp[i]) for k in range(nd) for i in range(len(S))] for m, X in r.items()}
                d.update({m: list(np.abs(v - tp[None]).ravel()) for m, v in pk.items()})
                fampk[mo][nm] = {m: float(np.median(v)) for m, v in d.items()}
        log(f'谱族 · {mo} 完 {time.time() - t0:.0f} s')

    # ---------------- b. 作者 5 束真实读数 + 同一批测试光的模拟读数（种子同 a11_device：real +50、realR +51）
    import openpyxl
    fig = AD.PC.read_fig(openpyxl.load_workbook(AD.FW.XLSX, read_only=True, data_only=True), 'vis')
    refs = [fig[float(lt)] for lt in AD.LAM_TEST]
    truth = np.array([AD.to_grid(fl, ff) for _, _, fl, ff in refs]); ref_pk = np.array([AD.PC.peak(fl, ff) for _, _, fl, ff in refs])
    r, pk = run_all((AD.Y_REAL / AD.SCALE)[None]); real = {}
    for m, X in r.items():
        real[m] = dict(dpk=[AD.peak_grid(X[0][:, j]) - ref_pk[j] for j in range(len(ref_pk))],
                       err=[AD.nrmse1(X[0][:, j], truth[j]) for j in range(len(ref_pk))])
    for m, v in pk.items():
        real[m] = dict(dpk=list(v[0] - ref_pk))
    NSIM = 5 if SMOKE else AD.NSIM; sim = {}
    for si, mo in enumerate(['real', 'realR', 'ideal']):
        rng = np.random.default_rng(AD.SEED_SIM + 50 + si); sim[mo] = {}
        for j, lt in enumerate(AD.LAM_TEST):
            nn = 1 if mo == 'ideal' else NSIM
            if mo == 'realR':
                Yb = AD.meas(np.tile(truth[j][:, None], (1, nn)), rng, 'realR', nd=1, P=AD.P_FIT[j]).transpose(2, 1, 0)
            else:
                fl, ff = refs[j][2], refs[j][3]; p = np.clip(ff, 0, None); p = p / np.trapezoid(p, fl) * AD.P_FIT[j]
                Yb = np.array([AD.DV.measure(fl, p, dark=False, warn=False, real=mo != 'ideal', rng=rng) for _ in range(nn)])[:, :, None] / AD.SCALE
            r, pk = run_all(Yb)
            for m, X in r.items():
                sm = sim[mo].setdefault(m, dict(dpk=[], err=[]))
                sm['dpk'].append([AD.peak_grid(x) - ref_pk[j] for x in X[:, :, 0]]); sm['err'].append([AD.nrmse1(x, truth[j]) for x in X[:, :, 0]])
            for m, v in pk.items():
                sim[mo].setdefault(m, dict(dpk=[]))['dpk'].append(list(v[:, 0] - ref_pk[j]))
        log(f'测试光 · 模拟 {mo} 完 {time.time() - t0:.0f} s')
    pool.close()

    # ---------------- 并入 A11 两版（读 a11_device 已有结果），并核对读数重放是否一致
    chk = []
    for tag, d in (('A11', 'a11'), ('A11_R', 'a11_R')):
        J = json.load(open(os.path.join(AD.DEVDIR, 'results', d, 'a11.json'), encoding='utf-8'))
        for mo in modes:
            if mo not in J['fam']:
                continue
            for nm in fam[mo]:
                jf = J['fam'][mo].get(nm, J['fam'][mo].get('留出·' + nm))
                fam[mo][nm][tag] = jf['A11']
                chk += [abs(jf['paper'] - fam[mo][nm]['paper']), abs(jf['A1'] - fam[mo][nm]['A1'])]
        real[tag] = dict(dpk=J['real']['A11']['dpk'], err=J['real']['A11']['err'])
        for mo in J['sim']:
            sim[mo][tag] = dict(dpk=J['sim'][mo]['A11']['dpk'], err=J['sim'][mo]['A11']['err'])
    log(f'读数重放核对：本脚本重算的论文算法 / A1 逐族误差和 a11.json 的最大差 {max(chk) if chk else float("nan"):.2e}（≈ 0 说明读数完全一致）')

    # ---------------- 汇总
    METH = [m for m in NAME if m in next(iter(fam['real'].values()))]
    for mo in modes:
        F = fam[mo]; ms = [m for m in METH if m in next(iter(F.values()))]
        med = {m: float(np.median([F[k][m] for k in F])) for m in ms}
        win = {m: sum(min(ms, key=lambda q: F[k][q]) == m for k in F) for m in ms}
        log(f'\n== 生成谱族 · {mo}：各族误差的中位（{len(F)} 族；误差 = 峰值归一均方根 / 真值极差），按中位排序')
        for m in sorted(ms, key=med.get):
            log(f'   {NAME[m]:<22s}{med[m]:8.3f}   第一 {win[m]:2d} 族   {SRC[m]}')
        for nm in PKF:
            if nm in fampk[mo]:
                log(f'   {nm} 峰位误差中位（nm）：' + '，'.join(f'{NAME[m]} {v:.2f}' for m, v in sorted(fampk[mo][nm].items(), key=lambda x: x[1])))
    log('\n== 作者 5 束真实读数（参考 = 商用光谱仪参考谱），按平均 |峰位误差| 排序；括号里是同一批光模拟真实模式 200 次的平均 |峰位误差|')
    for m in sorted(real, key=lambda m: np.mean(np.abs(real[m]['dpk']))):
        dp = np.array(real[m]['dpk']); s_ = np.abs(np.array(sim['real'][m]['dpk'])).mean() if m in sim['real'] else np.nan
        log(f'   {NAME[m]:<22s} 峰位误差 ' + ' '.join(f'{v:+7.2f}' for v in dp) + f'  平均 |·| {np.abs(dp).mean():6.2f} nm（模拟 {s_:5.2f}）'
            + (f'  误差 平均 {np.mean(real[m]["err"]):.3f}' if 'err' in real[m] else ''))
    log(f'\n用时 {time.time() - t0:.0f} s')
    json.dump(dict(fam=fam, fampk=fampk, real=real, sim=sim, ref_pk=ref_pk.tolist(), lam_test=[float(x) for x in AD.LAM_TEST], alpha=alpha, lam=lam,
                   nte=AD.NTE, name=NAME, src=SRC), open(os.path.join(OUT, 'all_algos.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
