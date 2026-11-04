# -*- coding: utf-8 -*-
"""Cui 2025 器件 · M3 读数：各重建算法的光谱仪指标（蛛网图用），定义照 a12_metrics.py / algos_metrics.py 的 1–4 节，只换器件、读数误差和波段。

用户 2026-11-02："画蛛网图"（A15 在 Cui M3 上和其他算法比，同 Uddin 的 fig_algos_radar.png）。
三组方法分三次跑（各自一个进程，互不 import 对方的器件补丁）：
  a15    A15（a15_cui_m3.py 已训好的三支 + qnet.pt，不重训）；另出两支融合 = A11，作读数一致的核对
  a12    A11、A12（a12_cui_m3.py 的 qnet.pt）
  algos  all_algos_cui_m3.py 的全部方法（main(build_only=True)，Xie MLP / NSF 读缓存；LUT / RF 只出波长）
三组的读数完全同一份：合成谱种子 SEED_M、各节随机数消耗顺序都一样，和解码方法无关。
和 Uddin 版的不同（器件不同，只改这些）：
  · 波段 500–720 nm：单线中心 504–716 nm（2 nm 步长），"内区" 520–700 nm；双峰第一条线中心 U[520, 700 − Δ]。
  · 读数 = 'm3' 档（recon 档白噪声 + 逐 Vgs 帧增益漂移，a = 1）。抗噪扫描乘光功率 k 只缩放白噪声，帧增益漂移是乘性的、不随功率变
    （所以强光下误差有底）；种子 SEED_SIM + 3 = a11_device 'm3' 档谱族读数，k = 1 与 a15.json / a12.json / all_algos.json 逐族一致（断言）。
  · 读数信噪比的噪声用 a11_device 的 DEVCOV（白噪声 + M3 协方差按典型谱平均），同 Uddin 版的定义。
  py -I -X utf8 cui_m3_metrics.py {a15|a12|algos} [--quick]
输出：46_Cui2025_真实R矩阵/results/metrics_M3(_quick)/{metrics_<组>.json, log_<组>.txt}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
PROJ = ROOT
while not os.path.isdir(os.path.join(PROJ, '03_仿真')) and os.path.dirname(PROJ) != PROJ:
    PROJ = os.path.dirname(PROJ)
sys.path.append(os.path.join(PROJ, '03_仿真', '44_Wang2024_真实R矩阵', 'scripts'))   # algos 组的子进程反序列化 _recon_workers 的任务要能找到它
QUICK = '--quick' in sys.argv
SEED_M = 20261103                                         # 同 a12_metrics.py
KS = (1.0, 10.0) if QUICK else (0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
SEPS = (12, 24, 48) if QUICK else (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
RES_CFG = [(8.0, 1.0), (8.0, 0.5), (16.0, 1.0)]
NPAIR, NDP = (6, 2) if QUICK else (40, 4)
SCAN_FW = (8.0, 30.0); CEN = np.arange(504.0, 717.0, 20.0 if QUICK else 2.0); NDS = 4 if QUICK else 20
LO_IN, HI_IN = 520.0, 700.0                               # 内区（波段两头各让 20 nm，同 Uddin 的 520–780）
PKF = ['窄线 4–10 nm', '单峰 15–60 nm']
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


def longest(ok):
    best, cur = None, None
    for i, v in enumerate(ok):
        if v:
            cur = (cur[0], i) if cur else (i, i)
            if best is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    return best


def setup(grp):
    """→ AD, LD, P, used, go(Y) → ({方法: (nd, M, NW)}, {方法: (nd, M) 波长}), ref(nm, 方法) = 已存的 k = 1 逐族误差, close()。"""
    import torch
    RES = None
    if grp == 'a15':
        import a15_cui_m3 as C                                # import 即把器件设成 Cui · M3、a15_three 的 NETDIR 指到 a15_M3
        AD, PF, P18 = C.AD, C.PF, C.P18
        decode, P, used = AD.build()                          # = a15_three.build3，三个网络都读缓存
        Q = torch.load(os.path.join(C.OUT, 'qnet.pt'), weights_only=False); mu, sd = Q['mu'], Q['sd']
        qnet = torch.nn.Sequential(torch.nn.Linear(len(mu), 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double().to(PF.DEV)
        qnet.load_state_dict(Q['state']); qnet.eval()
        J0 = json.load(open(os.path.join(C.OUT, 'a15.json'), encoding='utf-8'))

        def go(Y, ch=64):
            out = {'A15': [], 'A11': []}
            for i in range(0, Y.shape[2], ch):
                y = Y[:, :, i:i + ch]; o = decode(y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, mu, sd, qnet)
                out['A15'].append(PF.shift_np(xa, q * d).reshape(y.shape[0], y.shape[2], -1))
                out['A11'].append(P18.flat(o['A2']).reshape(y.shape[0], y.shape[2], -1))    # 两支融合 = A11（核对用）
            return {m: np.concatenate(v, 1) for m, v in out.items()}, {}
        ref = lambda nm, m: J0['fam'][nm]['A15' if m == 'A15' else 'two']
        return AD, AD.LD, P, used, go, ref, lambda: None
    if grp == 'a12':
        import a12_cui_m3 as C
        AD, P18 = C.AD, C.P18
        decode, P, used = AD.build()
        Q = torch.load(os.path.join(C.OUT, 'qnet.pt'), weights_only=False); mu, sd = Q['mu'], Q['sd']
        qnet = torch.nn.Sequential(torch.nn.Linear(len(mu), 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double().to(C.DEV)
        qnet.load_state_dict(Q['state']); qnet.eval()
        J0 = json.load(open(os.path.join(C.OUT, 'a12.json'), encoding='utf-8'))

        def go(Y, ch=64):
            out = {'A12': [], 'A11': []}
            for i in range(0, Y.shape[2], ch):
                y = Y[:, :, i:i + ch]; o = decode(y); Fx, xa, d = C.featx(o); q = C.q_of(Fx, mu, sd, qnet)
                out['A12'].append(C.shift_np(xa, q * d).reshape(y.shape[0], y.shape[2], -1))
                out['A11'].append(xa.reshape(y.shape[0], y.shape[2], -1))
            return {m: np.concatenate(v, 1) for m, v in out.items()}, {}
        ref = lambda nm, m: J0['fam'][nm][m]
        return AD, AD.LD, P, used, go, ref, lambda: None
    import all_algos_cui_m3 as AA
    B = AA.main(build_only=True); AD, run_all, pool = B['AD'], B['run_all'], B['pool']
    J0 = json.load(open(os.path.join(AD.DEVDIR, 'results', 'all_algos_M3', 'all_algos.json'), encoding='utf-8'))

    def go(Y):
        r, pk = run_all(Y)
        return {m: X.transpose(0, 2, 1) for m, X in r.items()}, pk
    ref = lambda nm, m: J0['fam'][nm][m]
    return AD, B['LD'], B['P'], B['used'], go, ref, pool.close


def main(grp):
    t0 = time.time()
    AD, LD, P, used, go, ref, close = setup(grp)
    OUT = os.path.join(AD.DEVDIR, 'results', 'metrics_M3' + ('_quick' if QUICK else '')); os.makedirs(OUT, exist_ok=True)
    assert AD.ARG.device == 'cui' and AD.M3
    WL, BM = AD.WL, AD.BM; NW = len(WL); nd = LD.NDRAW
    log(f'组 {grp}：建好 {time.time() - t0:.0f} s；Cui 2025 · M3，波段 {WL[BM][0]:.0f}–{WL[BM][-1]:.0f} nm，P0 = {AD.P0 * 1e6:.3g} µW')
    peaks = lambda X: np.array([AD.peak_grid(x) for x in X])
    gau = lambda c, fw: np.where(BM, np.exp(-0.5 * ((WL[None] - np.atleast_1d(c)[:, None]) / (fw / 2.3548)) ** 2), 0.0)
    pnorm = lambda S: S / S.max(1, keepdims=True)

    def resolved(x, c1, c2):
        """同 a12_metrics.resolved。"""
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

    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}

    # ---------------- 合成谱（随机数消耗顺序同 a12_metrics.py）+ 防泄漏
    g = np.random.default_rng(SEED_M)
    scan = {fw: pnorm(gau(CEN, fw)) for fw in SCAN_FW}
    pairs = {}
    for fw, r in RES_CFG:
        for dl in SEPS:
            c1 = g.uniform(LO_IN, HI_IN - dl, NPAIR); sw_ = g.random(NPAIR) < 0.5
            a1 = np.where(sw_, r, 1.0); a2 = np.where(sw_, 1.0, r)
            pairs[(fw, r, dl)] = (pnorm(a1[:, None] * gau(c1, fw) + a2[:, None] * gau(c1 + dl, fw)), c1, c1 + dl)
    chk = {f'单线 {fw:g} nm': S for fw, S in scan.items()}; chk.update({f'双线 {k}': v[0] for k, v in pairs.items()})
    leak = AD.P18.overlap(chk, used); log('合成谱与训练库 / 各验证集逐条比对，完全相同的条数：' + ('无' if not leak else str(leak)))
    assert not leak, leak
    J = dict(group=grp, P0_uW=AD.P0 * 1e6, ks=list(KS), band=[float(WL[BM][0]), float(WL[BM][-1])], inner=[LO_IN, HI_IN], M={})
    M = J['M']

    # ---------------- 抗噪声 + 全谱误差逐样本分布（k = 1 = 已有 M3 谱族结果的读数）
    sig = {k: float(np.median([np.linalg.norm(AD.RN @ x) / (x[BM].sum() * np.sqrt(np.trace(AD.DEVCOV))) for x in np.concatenate(list(sets.values()))])) * k for k in KS}
    J['snr'] = {str(k): sig[k] for k in KS}
    log(f'\n== 抗噪声：生成谱族（{len(sets)} 族 × {AD.NTE} 条 × {nd} 次）M3 读数，光功率 = k × P0')
    for k in KS:
        gk = np.random.default_rng(AD.SEED_SIM + 3); fam, pk, es = {}, {}, {}
        for nm, S in sets.items():
            X, W = go(AD.meas(S.T, gk, 'm3', P=k * AD.P0))
            for m, Xm in X.items():
                e = np.array([LD.band_nrmse(Xm[q].T, S.T) for q in range(nd)])
                fam.setdefault(m, {})[nm] = float(e.mean()); es.setdefault(m, []).append(e.ravel())
            if nm in PKF:
                tp = np.tile(peaks(S), nd)
                for m, Xm in X.items():
                    pk.setdefault(m, {})[nm] = float(np.median(np.abs(peaks(Xm.reshape(-1, NW)) - tp)))
                for m, v in W.items():
                    pk.setdefault(m, {})[nm] = float(np.median(np.abs(v.ravel() - tp)))
        for m in pk:
            d = M.setdefault(m, dict(noise={}))
            v = np.array(list(fam[m].values())) if m in fam else None
            d['noise'][str(k)] = dict(snr=sig[k], pk=pk[m], **({} if v is None else dict(fam=fam[m], med=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))))
            if k == 1.0 and m in fam:
                ek = np.concatenate(es[m]); d['full'] = dict(fam=fam[m], med=float(np.median(v)), mean=float(v.mean()), sample_pct={str(p): float(np.percentile(ek, p)) for p in (10, 25, 50, 75, 90)})
        if k == 1.0:
            dmax = {m: max(abs(fam[m][nm] - ref(nm, m)) for nm in sets) for m in fam}
            log('   k = 1 与已存 M3 谱族结果逐族形状误差的最大差：' + '，'.join(f'{m} {v:.1e}' for m, v in dmax.items()))
            assert max(dmax.values()) < 1e-6, dmax
        log(f'   k = {k:<6g}信噪比 {sig[k]:7.3g}  族中位：' + '，'.join(f'{m} {M[m]["noise"][str(k)]["med"]:.3f}' for m in fam) + f'   {time.time() - t0:.0f} s')
        log('      窄线峰位中位（nm）：' + '，'.join(f'{m} {pk[m][PKF[0]]:.2f}' for m in pk))

    # ---------------- 单线扫描
    log(f'\n== 单线扫描：中心 {CEN[0]:.0f}–{CEN[-1]:.0f} nm、步长 {CEN[1] - CEN[0]:g} nm，每个中心 {NDS} 次（M3，P0）')
    inner = (CEN >= LO_IN) & (CEN <= HI_IN)
    for fw in SCAN_FW:
        S = scan[fw]; X, W = go(AD.meas(S.T, g, 'm3', nd=NDS)); fwt = np.array([AD.fwhm_grid(x) for x in S])
        allp = {m: peaks(Xm.reshape(-1, NW)).reshape(NDS, -1) for m, Xm in X.items()}; allp.update(W)
        for m, P_ in allp.items():
            pe = np.abs(P_ - CEN[None]); pm = np.median(pe, 0); band = {}
            for th in (1.0, 2.0):
                b = longest(pm <= th); band[str(th)] = None if b is None else [float(CEN[b[0]]), float(CEN[b[1]])]
            d = dict(cen=CEN.tolist(), pk_med=pm.tolist(), band=band, pk_all=float(np.median(pe[:, inner])), pk_mean=float(pe[:, inner].mean()))
            if m in X:
                Xm = X[m]; em = np.median([LD.band_nrmse(Xm[q].T, S.T) for q in range(NDS)], 0)
                fwr = np.nanmedian([[AD.fwhm_grid(x) for x in Xm[q]] for q in range(NDS)], 0)
                d.update(err_med=em.tolist(), fwhm_true=fwt.tolist(), fwhm_rec=fwr.tolist(), err_inner=float(np.median(em[inner])),
                         fwhm_ratio=float(np.nanmedian(fwr[inner] / fwt[inner])))
            M[m].setdefault('scan', {})[str(fw)] = d
        log(f'   半高宽 {fw:g} nm，{LO_IN:.0f}–{HI_IN:.0f} nm 内逐中心 |峰位误差| 中位的平均：' + '，'.join(f'{m} {np.mean(np.array(M[m]["scan"][str(fw)]["pk_med"])[inner]):.2f}' for m in allp)
            + f'   {time.time() - t0:.0f} s')

    # ---------------- 双峰分辨率
    log(f'\n== 双峰分辨率：每个间隔 {NPAIR} 对 × {NDP} 次（M3，P0）')
    for fw, r in RES_CFG:
        fr = {}
        for dl in SEPS:
            S, c1, c2 = pairs[(fw, r, dl)]; X, _ = go(AD.meas(S.T, g, 'm3', nd=NDP))
            for m, Xm in X.items():
                fr.setdefault(m, {})[dl] = float(np.mean([resolved(Xm[q, i], c1[i], c2[i]) for q in range(NDP) for i in range(NPAIR)]))
        for m in fr:
            res = {}
            for th in (0.5, 0.9):
                ok = [dl for dl in SEPS if all(fr[m][d] >= th for d in SEPS if d >= dl)]; res[str(th)] = min(ok) if ok else None
            M[m].setdefault('res', {})[f'{fw:g}|{r:g}'] = dict(frac=fr[m], res=res)
        log(f'   半高宽 {fw:g} nm、强度比 {r:g}，分辨率（≥ 50 % / ≥ 90 %）：' + '，'.join(f'{m} {M[m]["res"][f"{fw:g}|{r:g}"]["res"]["0.5"]}/{M[m]["res"][f"{fw:g}|{r:g}"]["res"]["0.9"]}' for m in fr)
            + f'   {time.time() - t0:.0f} s')
    close()
    json.dump(J, open(os.path.join(OUT, f'metrics_{grp}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log(f'用时 {time.time() - t0:.0f} s')
    open(os.path.join(OUT, f'log_{grp}.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    grp = next(a for a in sys.argv[1:] if not a.startswith('--'))
    assert grp in ('a15', 'a12', 'algos'), grp
    main(grp)
