# -*- coding: utf-8 -*-
"""A12（= A11p）的光谱仪指标（只 Uddin，真实模式读数，全模拟；读已训练的 qnet.pt，不重训）。

用户 2026-11-02："算全谱重建误差、双峰分辨率、峰位误差、抗噪音能力、有效重建波段、响应矩阵信息量，先算 A12 的这几项"。
  1. 全谱重建误差：生成谱族（26 族 × 40 条 × 4 次）真实模式读数，读 peakfix.json 的族平均，另从第 4 项 k = 1 那一档取逐样本误差。
  2. 双峰分辨率：两条高斯线（半高宽 fw、强度比 r，弱的一条随机放左或右），间隔 Δ；中心随机（520 – 780−Δ nm，不落格点）。
     判"分开"：重建谱在两条真值线各自 tol = min(Δ/2, 8) nm 内各有一个局部极大（高于本条最大值 0.2），两者不同，
     且两峰之间的最低点 ≤ 较矮那个峰的 0.8（凹陷 ≥ 20 %，接近瑞利判据的 26 %）。同一判据也用在真值上（网格采样后真值本身能不能分开）。
     分辨率 = 分开比例 ≥ 50 %（另报 ≥ 90 %）且所有更大间隔也都达到时的最小 Δ。
  3. 峰位误差 / 有效重建波段 / 线宽：单条高斯线中心从 504 扫到 796 nm（步长 2 nm，一半不落格点），半高宽 8、30 nm，每个中心 NDS 次读数；
     报每个中心的 |峰位误差| 中位、形状误差中位、重建半高宽 / 真值半高宽。有效波段 = |峰位误差| 中位 ≤ 1 nm（另报 2 nm）的最长连续区间。
  4. 抗噪声：生成谱族读数的光功率乘 k（参考功率 P0 ≈ 11 µW；A12 训练时只见过 0.5–2 倍），白噪声和暗电流漂移随之相对变大 / 变小。
     每档都用同一份随机数（种子 SEED_SIM + 1，同 a11_peakfix 的真实模式），k = 1 与 peakfix.json 逐族一致（断言）。
     报各族形状误差的中位、窄线 / 单峰 15–60 族的峰位误差中位、读数信噪比。弱光下限 = 族中位误差不比 k = 1 差 20 % 以上的最低功率。
  5. 响应矩阵信息量：R 的奇异值（相对最大值）、通道相关、两条 8 nm 线相隔 Δ 的读数相关度（器件指纹，和算法无关），
     以及按器件噪声白化后每个模式的信噪比（测试谱族在该模式上的均方根信号 / 1），信噪比 > 1 的模式数 = 噪声下的有效自由度。
  所有合成谱和训练库、各验证集逐条比对，完全相同的条数断言为零。
  py -I -X utf8 a12_metrics.py [--quick]      输出 45_Uddin2024_真实R矩阵/results/a11_peakfix/metrics(_quick)/{metrics.json, log.txt, fig_metrics.png, fig_doublets.png}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
QUICK = '--quick' in sys.argv
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
sys.path.insert(0, ROOT)
import a11_peakfix as PF
import torch, torch.nn as tnn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

AD, P18, LD = PF.AD, PF.P18, PF.LD
WL, BM, DEV = AD.WL, AD.BM, PF.DEV; NW = len(WL)
OUT = os.path.join(PF.OUT, 'metrics' + ('_quick' if QUICK else '')); os.makedirs(OUT, exist_ok=True)
ALG = 'A12'                                               # 图和日志里的算法名（a15_three.py 设成 A15）
SEED_M = 20261103                                         # 合成谱和它们的读数（新种子，前面都没用过）
KS = (1.0, 10.0) if QUICK else (0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
SEPS = (12, 24, 48) if QUICK else (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
RES_CFG = [(8.0, 1.0), (8.0, 0.5), (16.0, 1.0)]           # (半高宽 nm, 强度比)
NPAIR, NDP = (6, 2) if QUICK else (40, 4)
SCAN_FW = (8.0, 30.0); CEN = np.arange(504.0, 797.0, 20.0 if QUICK else 2.0); NDS = 4 if QUICK else 20
PKF = ['窄线 4–10 nm', '单峰 15–60 nm']
INK, INK2, GRID = '#1f1f1e', '#6b6a65', '#e4e3df'
C1, C2, C3 = '#2a78d6', '#d97a2b', '#3a9a5b'
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


def gau(c, fw):
    """(M,) 中心 → (M, NW) 网格点采样的高斯（同 algo13 的 _g），波段外置零。"""
    c = np.atleast_1d(c)[:, None]
    return np.where(BM, np.exp(-0.5 * ((WL[None] - c) / (fw / 2.3548)) ** 2), 0.0)


def pnorm(S):
    return S / S.max(1, keepdims=True)


def resolved(x, c1, c2):
    """两条线算不算分开（判据见文件头）。"""
    dl = abs(c2 - c1); tol = min(dl / 2, 8.0)
    xb = np.where(BM, np.clip(x, 0, None), 0); m = xb.max()
    if m <= 0:
        return False
    i = np.arange(1, NW - 1); lm = i[(xb[i] >= xb[i - 1]) & (xb[i] > xb[i + 1]) & (xb[i] >= 0.2 * m)]
    if len(lm) < 2:
        return False
    j1 = lm[np.argmin(np.abs(WL[lm] - c1))]; j2 = lm[np.argmin(np.abs(WL[lm] - c2))]
    if j1 == j2 or abs(WL[j1] - c1) > tol or abs(WL[j2] - c2) > tol:
        return False
    a, b = sorted((j1, j2))
    return xb[a:b + 1].min() <= 0.8 * min(xb[j1], xb[j2])


def longest(ok):
    """布尔数组里最长的连续 True 段 → (起, 止) 下标；没有返回 None。"""
    best, cur = None, None
    for i, v in enumerate(ok):
        if v:
            cur = (cur[0], i) if cur else (i, i)
            if best is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    return best


def main():
    t0 = time.time()
    decode, P, used = AD.build()
    Q = torch.load(os.path.join(PF.OUT, 'qnet.pt'), weights_only=False); mu, sd = Q['mu'], Q['sd']
    qnet = tnn.Sequential(tnn.Linear(len(mu), 32), tnn.Tanh(), tnn.Linear(32, 1)).double().to(DEV); qnet.load_state_dict(Q['state']); qnet.eval()
    log(f'{ALG} 建好 {time.time() - t0:.0f} s')

    def a12(Y, ch=64):
        """Y (nd, n, M) → (nd, M, NW)；按谱分块解码（每条谱独立）。"""
        out = []
        for i in range(0, Y.shape[2], ch):
            y = Y[:, :, i:i + ch]; o = decode(y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, mu, sd, qnet)
            out.append(PF.shift_np(xa, q * d).reshape(y.shape[0], y.shape[2], -1))
        return np.concatenate(out, 1)

    PK = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8'))
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    J = dict(P0_uW=AD.P0 * 1e6, ks=list(KS), train_k=[0.5, 2.0])

    # ---------------- 5. 响应矩阵信息量（不用解码）
    Rb = AD.RN[:, BM]; s = np.linalg.svd(Rb, compute_uv=False); sr = s / s[0]
    C = np.corrcoef(Rb); corr_med = float(np.median(np.abs(C[~np.eye(len(C), dtype=bool)])))
    Lc = np.linalg.cholesky(AD.DEVCOV); Rw = np.linalg.solve(Lc, Rb); _, sw, Vt = np.linalg.svd(Rw, full_matrices=False)
    Xall = np.concatenate([S for S in sets.values()])[:, BM]; Xall = Xall / Xall.sum(1, keepdims=True)
    msnr = np.sqrt(np.mean((sw[:, None] * (Vt @ Xall.T)) ** 2, 1))                 # 每个白化模式的均方根信号（噪声 = 1）
    fp = {}
    for dl in range(4, 121, 4):
        cs = np.arange(520.0, 780.0 - dl + 1e-9, 2.0)
        fp[dl] = float(np.mean([np.corrcoef(AD.RN @ gau(c, 8)[0], AD.RN @ gau(c + dl, 8)[0])[0, 1] for c in cs]))
    d99 = next((d for d in fp if fp[d] < 0.99), None); d90 = next((d for d in fp if fp[d] < 0.90), None)
    sens = np.linalg.norm(AD.RN, axis=0); sens = sens / sens[BM].max()
    J['R'] = dict(n_read=int(AD.N_RD), n_wl=int(BM.sum()), sv_rel=sr.tolist(), n_sv={str(t): int((sr > t).sum()) for t in (0.1, 0.01, 0.001)},
                  cond=float(s[0] / s[-1]), corr_med=corr_med, mode_snr=msnr.tolist(), n_mode={str(k): int((msnr * k > 1).sum()) for k in KS},
                  n_mode3=int((msnr > 3).sum()), fingerprint=fp, d99=d99, d90=d90, sens=sens[BM].tolist())
    log(f'\n== 5. 响应矩阵信息量：{AD.N_RD} 个读数 × {BM.sum()} 个波长（{WL[BM][0]:.0f}–{WL[BM][-1]:.0f} nm，4 nm 网格）')
    log(f'   奇异值 / 最大值 > 10 %：{(sr > 0.1).sum()} 个，> 1 %：{(sr > 0.01).sum()} 个，> 0.1 %：{(sr > 0.001).sum()} 个；条件数 {s[0] / s[-1]:.3g}；通道两两相关系数 |r| 中位 {corr_med:.3f}')
    log(f'   按器件噪声白化后（P0），各模式信噪比前 12 个：' + ' '.join(f'{v:.3g}' for v in msnr[:12]))
    log(f'   信噪比 > 1 的模式数（噪声下的有效自由度）：' + '，'.join(f'{k:g}×P0 {int((msnr * k > 1).sum())}' for k in KS) + f'；P0 下 > 3 的 {int((msnr > 3).sum())} 个')
    log(f'   器件指纹（两条 8 nm 线相隔 Δ 的读数相关度）：Δ = 8 nm {fp[8]:.4f}，16 nm {fp[16]:.4f}，32 nm {fp[32]:.4f}，64 nm {fp[64]:.4f}；'
        f'降到 0.99 以下要 Δ ≥ {d99} nm，0.90 以下要 Δ ≥ {d90} nm')

    # ---------------- 合成谱 + 防泄漏
    g = np.random.default_rng(SEED_M)
    scan = {fw: pnorm(gau(CEN, fw)) for fw in SCAN_FW}
    pairs = {}
    for fw, r in RES_CFG:
        for dl in SEPS:
            c1 = g.uniform(520.0, 780.0 - dl, NPAIR); sw_ = g.random(NPAIR) < 0.5
            a1 = np.where(sw_, r, 1.0); a2 = np.where(sw_, 1.0, r)
            pairs[(fw, r, dl)] = (pnorm(a1[:, None] * gau(c1, fw) + a2[:, None] * gau(c1 + dl, fw)), c1, c1 + dl)
    chk = {f'单线 {fw:g} nm': S for fw, S in scan.items()}
    chk.update({f'双线 {k}': v[0] for k, v in pairs.items()})
    leak = P18.overlap(chk, used); log('\n合成谱与训练库 / 验证集 / 先验用谱 / 零件训练池逐条比对，完全相同的条数：' + ('无' if not leak else str(leak)))
    assert not leak, leak

    # 分块解码与整块一致（门控 / 网络 / 融合只差 ~2e-6：GPU 单精度网络换批大小的浮点差；同一批重复解码完全相同）
    S0 = next(iter(sets.values()))[:12]; Y0 = AD.meas(S0.T, np.random.default_rng(1), 'real')
    dch = float(np.abs(a12(Y0, ch=5) - a12(Y0, ch=64)).max()); assert dch < 1e-4, f'分块解码不一致 {dch:.2e}'
    log(f'分块解码与整块解码最大差 {dch:.1e}（浮点级）')

    # ---------------- 4. 抗噪声（含 1. 全谱误差的逐样本分布）
    nd = LD.NDRAW; J['noise'] = {}
    sig = {k: float(np.median([np.linalg.norm(AD.RN @ x) / (x[BM].sum() * np.sqrt(np.trace(AD.DEVCOV))) for x in np.concatenate(list(sets.values()))])) * k for k in KS}
    log(f'\n== 4. 抗噪声：生成谱族（{len(sets)} 族 × {AD.NTE} 条 × {nd} 次）真实模式读数，光功率 = k × P0（P0 = {AD.P0 * 1e6:.3g} µW；{ALG} 训练时 k ∈ [0.5, 2]）')
    log('   k        读数信噪比     族中位形状误差   族 25 / 75 %        窄线峰位中位   单峰 15–60 峰位中位   用时')
    ek1 = None
    for k in KS:
        gk = np.random.default_rng(AD.SEED_SIM + 1); fam, pk, es = {}, {}, []
        for nm, S in sets.items():
            X = a12(AD.meas(S.T, gk, 'real', P=k * AD.P0))
            e = np.array([LD.band_nrmse(X[r].T, S.T) for r in range(nd)]); fam[nm] = float(e.mean()); es.append(e.ravel())
            if nm in PKF:
                pk[nm] = float(np.median(np.abs(PF.peaks(X.reshape(-1, NW)) - np.tile(PF.peaks(S), nd))))
        if k == 1.0:
            dmax = max(abs(fam[nm] - PK['fam']['real'][nm]['A11p']) for nm in sets)
            dpk = max(abs(pk[nm] - PK['fampk']['real'][nm]['A11p']) for nm in PKF)
            assert dmax < 1e-9 and dpk < 1e-9, (dmax, dpk)
            ek1 = np.concatenate(es)
        v = np.array(list(fam.values()))
        J['noise'][str(k)] = dict(snr=sig[k], fam=fam, med=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), pk=pk)
        log(f'   {k:<7g}  {sig[k]:8.3g}（{20 * np.log10(sig[k]):5.1f} dB）  {np.median(v):.4f}         {np.percentile(v, 25):.4f} / {np.percentile(v, 75):.4f}     '
            f'{pk[PKF[0]]:.3f} nm         {pk[PKF[1]]:.3f} nm            {time.time() - t0:.0f} s')
    log('   k = 1 与 peakfix.json 逐族形状误差、峰位误差一致（差 < 1e-9）')
    m1 = J['noise']['1.0']['med']; okk = [k for k in KS if J['noise'][str(k)]['med'] <= 1.2 * m1]
    J['weak_k'] = min(okk); log(f'   弱光下限（族中位误差 ≤ 1.2 × k = 1 时的 {m1:.4f}）：k = {min(okk):g}（{min(okk) * AD.P0 * 1e6:.3g} µW）')

    # 1. 全谱误差
    fv = np.array([PK['fam']['real'][nm]['A11p'] for nm in sets]); order = np.argsort(fv); names = list(sets)
    J['full'] = dict(fam={nm: float(PK['fam']['real'][nm]['A11p']) for nm in sets}, med=float(np.median(fv)), mean=float(fv.mean()),
                     sample_pct={p: float(np.percentile(ek1, p)) for p in (10, 25, 50, 75, 90)})
    log(f'\n== 1. 全谱重建误差（形状误差 = 峰值归一后均方根 / 真值极差；真实模式 P0）：族平均的中位 {np.median(fv):.4f}、平均 {fv.mean():.4f}')
    log(f'   逐样本（{len(ek1)} 个）分位：' + '，'.join(f'{p} % {np.percentile(ek1, p):.4f}' for p in (10, 25, 50, 75, 90)))
    log(f'   最好 3 族：' + '，'.join(f'{names[i]} {fv[i]:.4f}' for i in order[:3]) + f'；最差 3 族：' + '，'.join(f'{names[i]} {fv[i]:.4f}' for i in order[-3:]))

    # ---------------- 3. 单线扫描：峰位、有效波段、线宽
    J['scan'] = {}; log(f'\n== 3. 单线扫描：高斯线中心 {CEN[0]:.0f}–{CEN[-1]:.0f} nm、步长 {CEN[1] - CEN[0]:g} nm，每个中心 {NDS} 次真实模式读数（P0）')
    for fw in SCAN_FW:
        S = scan[fw]; X = a12(AD.meas(S.T, g, 'real', nd=NDS))
        pe = np.abs(PF.peaks(X.reshape(-1, NW)).reshape(NDS, -1) - CEN[None]); pm = np.median(pe, 0)
        em = np.median([LD.band_nrmse(X[r].T, S.T) for r in range(NDS)], 0)
        fwt = np.array([AD.fwhm_grid(x) for x in S]); fwr = np.median([[AD.fwhm_grid(x) for x in X[r]] for r in range(NDS)], 0)
        band = {}
        for th in (1.0, 2.0):
            b = longest(pm <= th); band[str(th)] = None if b is None else [float(CEN[b[0]]), float(CEN[b[1]])]
        inner = (CEN >= 520) & (CEN <= 780)
        J['scan'][str(fw)] = dict(cen=CEN.tolist(), pk_med=pm.tolist(), err_med=em.tolist(), fwhm_true=fwt.tolist(), fwhm_rec=fwr.tolist(), band=band,
                                  pk_all=float(np.median(pe[:, inner])), pk_mean=float(pe[:, inner].mean()), err_inner=float(np.median(em[inner])),
                                  fwhm_ratio=float(np.median(fwr[inner] / fwt[inner])))
        bs = lambda th: '无' if band[str(th)] is None else f'{band[str(th)][0]:.0f}–{band[str(th)][1]:.0f} nm'
        log(f'   半高宽 {fw:g} nm：520–780 nm 内 |峰位误差| 中位 {np.median(pe[:, inner]):.3f} nm、平均 {pe[:, inner].mean():.3f} nm；形状误差中位 {np.median(em[inner]):.4f}；'
            f'重建 / 真值半高宽中位 {np.median(fwr[inner] / fwt[inner]):.2f}')
        log(f'      有效波段（|峰位误差| 中位 ≤ 1 nm 的最长连续段）{bs(1.0)}；≤ 2 nm：{bs(2.0)}')
        worst = np.argsort(-pm)[:5]
        log(f'      峰位误差最大的 5 个中心：' + '，'.join(f'{CEN[i]:.0f} nm {pm[i]:.2f}' for i in worst) + f'   用时 {time.time() - t0:.0f} s')

    # ---------------- 2. 双峰分辨率
    J['res'] = {}; ex = {}
    log(f'\n== 2. 双峰分辨率：两条高斯线，每个间隔 {NPAIR} 对 × {NDP} 次真实模式读数（P0）；判据：各自 ±min(Δ/2, 8) nm 内有峰、中间凹陷 ≥ 20 %')
    for fw, r in RES_CFG:
        fr, ft = {}, {}
        for dl in SEPS:
            S, c1, c2 = pairs[(fw, r, dl)]; X = a12(AD.meas(S.T, g, 'real', nd=NDP))
            fr[dl] = float(np.mean([resolved(X[q, i], c1[i], c2[i]) for q in range(NDP) for i in range(NPAIR)]))
            ft[dl] = float(np.mean([resolved(S[i], c1[i], c2[i]) for i in range(NPAIR)]))
            if (fw, r) == RES_CFG[0]:
                ex[dl] = (S[0], X[0, 0], c1[0], c2[0])
        res = {}
        for th in (0.5, 0.9):
            ok = [dl for dl in SEPS if all(fr[d] >= th for d in SEPS if d >= dl)]; res[str(th)] = min(ok) if ok else None
        okt = [dl for dl in SEPS if all(ft[d] >= 0.9 for d in SEPS if d >= dl)]
        J['res'][f'{fw:g}|{r:g}'] = dict(frac=fr, truth=ft, res=res, res_truth=min(okt) if okt else None)
        log(f'   半高宽 {fw:g} nm、强度比 {r:g}：' + '  '.join(f'Δ{dl} {fr[dl] * 100:.0f}%' for dl in SEPS))
        log(f'      真值本身（网格采样后）能分开：' + '  '.join(f'Δ{dl} {ft[dl] * 100:.0f}%' for dl in SEPS))
        log(f'      分辨率（≥ 50 %）{res["0.5"]} nm，（≥ 90 %）{res["0.9"]} nm；真值 ≥ 90 % 起点 {min(okt) if okt else None} nm   用时 {time.time() - t0:.0f} s')

    json.dump(J, open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))

    # ---------------- 图
    def sty(ax, t):
        ax.set_title(t, fontsize=9.5, color=INK, loc='left'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8, colors=INK2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    fg, axs = plt.subplots(2, 3, figsize=(16, 9))
    kk = np.array(KS); med = [J['noise'][str(k)]['med'] for k in KS]
    ax = axs[0, 0]; ax.axvspan(0.5, 2, color=C1, alpha=0.08, lw=0); ax.text(1, 0.98, '训练功率范围', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.5, color=INK2)
    ax.fill_between(kk, [J['noise'][str(k)]['q25'] for k in KS], [J['noise'][str(k)]['q75'] for k in KS], color=C1, alpha=0.18, lw=0, label='各族 25–75 %')
    ax.plot(kk, med, 'o-', color=C1, lw=2, ms=5, label='各族中位'); ax.set_xscale('log'); ax.set_xlabel('光功率 / P0（P0 = 10.5 µW）', fontsize=8.5)
    ax.set_ylabel('形状误差', fontsize=8.5); ax.legend(fontsize=7.5, frameon=False); sty(ax, '(a) 抗噪声：全谱误差 vs 光功率')
    ax2 = ax.twiny(); ax2.set_xscale('log'); ax2.set_xlim(ax.get_xlim()); dbs = [d for d in range(-20, 61, 10) if ax.get_xlim()[0] <= 10 ** (d / 20) / sig[1.0] <= ax.get_xlim()[1]]
    ax2.set_xticks([10 ** (d / 20) / sig[1.0] for d in dbs]); ax2.set_xticklabels([str(d) for d in dbs]); ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.set_xlabel('读数信噪比 (dB)', fontsize=8); ax2.tick_params(labelsize=7.5, colors=INK2); ax2.spines['right'].set_visible(False)
    ax = axs[0, 1]; ax.axvspan(0.5, 2, color=C1, alpha=0.08, lw=0)
    for nm, c in zip(PKF, (C1, C2)):
        ax.plot(kk, [J['noise'][str(k)]['pk'][nm] for k in KS], 'o-', color=c, lw=2, ms=5, label=nm)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('光功率 / P0', fontsize=8.5); ax.set_ylabel('|峰位误差| 中位 (nm)', fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False); sty(ax, '(b) 抗噪声：峰位误差 vs 光功率')
    ax = axs[0, 2]
    for (fw, r), c in zip(RES_CFG, (C1, C2, C3)):
        R_ = J['res'][f'{fw:g}|{r:g}']; ax.plot(SEPS, [R_['frac'][d] * 100 for d in SEPS], 'o-', color=c, lw=2, ms=5, label=f'半高宽 {fw:g} nm、强度比 {r:g}')
        ax.plot(SEPS, [R_['truth'][d] * 100 for d in SEPS], ':', color=c, lw=1.2)
    ax.axhline(50, color=INK2, lw=0.8, ls='--'); ax.set_xlabel('两线间隔 Δ (nm)', fontsize=8.5); ax.set_ylabel('判为分开的比例 (%)', fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, title=f'实线 {ALG}，点线 = 真值本身', title_fontsize=7); sty(ax, '(c) 双峰分辨率')
    ax = axs[1, 0]
    for fw, c in zip(SCAN_FW, (C1, C2)):
        Sx = J['scan'][str(fw)]; ax.plot(CEN, Sx['pk_med'], color=c, lw=1.6, label=f'半高宽 {fw:g} nm')
    ax.axhline(1, color=INK2, lw=0.8, ls='--'); ax.set_yscale('log'); ax.set_xlabel('线中心 (nm)', fontsize=8.5); ax.set_ylabel('|峰位误差| 中位 (nm)', fontsize=8.5)
    b = J['scan'][str(SCAN_FW[0])]['band']['1.0']
    if b:
        ax.axvspan(b[0], b[1], color=C1, alpha=0.08, lw=0)
    ax.legend(fontsize=7.5, frameon=False); sty(ax, '(d) 峰位误差与有效波段（阴影：8 nm 线 ≤ 1 nm）')
    ax = axs[1, 1]
    for fw, c in zip(SCAN_FW, (C1, C2)):
        ax.plot(CEN, J['scan'][str(fw)]['err_med'], color=c, lw=1.6, label=f'形状误差，半高宽 {fw:g} nm')
    ax.set_xlabel('线中心 (nm)', fontsize=8.5); ax.set_ylabel('形状误差中位', fontsize=8.5)
    ax3 = ax.twinx(); ax3.plot(WL[BM], sens[BM], color=INK2, lw=1, ls='--'); ax3.set_ylim(0, 1.05); ax3.tick_params(labelsize=7.5, colors=INK2)
    ax3.set_ylabel('器件灵敏度 ‖R 列‖（相对，虚线）', fontsize=8, color=INK2); ax3.spines['top'].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc='upper center'); sty(ax, '(e) 逐波长形状误差')
    ax = axs[1, 2]; i = np.arange(1, len(msnr) + 1)
    ax.semilogy(i, msnr, 'o-', color=C1, ms=4, lw=1.5, label='白化后各模式信噪比（P0）'); ax.semilogy(i, sr * msnr[0], 's', color=INK2, ms=3, label='奇异值（相对，对齐第 1 个）')
    ax.axhline(1, color=INK2, lw=0.8, ls='--'); ax.set_xlabel('模式序号', fontsize=8.5); ax.set_ylabel('信噪比', fontsize=8.5)
    ax.text(0.97, 0.95, f'信噪比 > 1：{int((msnr > 1).sum())} 个模式\n奇异值 > 1 %：{(sr > 0.01).sum()} 个\n通道相关 |r| 中位 {corr_med:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left'); sty(ax, '(f) 响应矩阵信息量')
    fg.suptitle(f'{ALG} 光谱仪指标（Uddin 模拟器真实模式读数，全模拟）', fontsize=11.5, color=INK, x=0.01, ha='left')
    fg.tight_layout(rect=(0, 0, 1, 0.97)); f = os.path.join(OUT, 'fig_metrics.png'); fg.savefig(f, dpi=140); log(f)

    fg, axs = plt.subplots(1, len(ex), figsize=(2.9 * len(ex), 2.6), squeeze=False)
    for ax, (dl, (S, X, c1, c2)) in zip(axs[0], ex.items()):
        ax.plot(WL[BM], S[BM] / S[BM].max(), color=INK, lw=2, label='真值'); ax.plot(WL[BM], X[BM] / max(X[BM].max(), 1e-30), color=C1, lw=1.5, label=ALG)
        ax.set_xlim(c1 - 40, c2 + 40); ax.set_ylim(-0.05, 1.2); ok = resolved(X, c1, c2)
        sty(ax, f'Δ = {dl} nm（{"分开" if ok else "未分开"}）')
    axs[0, 0].legend(fontsize=7.5, frameon=False)
    fg.suptitle('双峰示例：两条 8 nm 等高线，每个间隔第 1 对、第 1 次读数', fontsize=10, color=INK, x=0.01, ha='left')
    fg.tight_layout(rect=(0, 0, 1, 0.92)); f = os.path.join(OUT, 'fig_doublets.png'); fg.savefig(f, dpi=140); log(f)
    log(f'用时 {time.time() - t0:.0f} s')
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
