# -*- coding: utf-8 -*-
"""文献里其他重建算法在 Uddin 模拟器上的光谱仪指标，和 A12 同一套定义、同一份读数（只 Uddin，真实模式，全模拟）。

用户 2026-11-02："对于这个同一个模型，用其他的算法来标定它的这个多种性质……只是为了横向对比不同的算法，跨文章的那就不用了"。
  方法 = all_algos_device.py 的全部（main(build_only=True) 建好，XieMLP / NSF 读已训练的缓存，不重训）；A11 / A11_R 已搁置，不算。
  指标和取法完全照 a12_metrics.py 的 1–4 节（第 5 节响应矩阵和算法无关，不重复）：
    抗噪声 k × P0 十档（种子 SEED_SIM + 1）、单线扫描 8 / 30 nm（504–796 nm，2 nm 步长，每个中心 20 次）、双峰三组配置。
  读数和 A12 完全同一份：合成谱种子 SEED_M、随机数消耗顺序都照 a12_metrics.py，所以 A12 的数直接读它的 metrics.json。
  核对：k = 1 的逐族误差和 all_algos.json 的 fam['real'] 比（同种子同读数），日志写最大差。
  LUT / RF 只出波长，只算峰位类指标。
  py -I -X utf8 algos_metrics.py [--quick]      输出 45_Uddin2024_真实R矩阵/results/all_algos/metrics(_quick)/{metrics.json, log.txt}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
QUICK = '--quick' in sys.argv
PROJ = ROOT
while not os.path.isdir(os.path.join(PROJ, '03_仿真')) and os.path.dirname(PROJ) != PROJ:
    PROJ = os.path.dirname(PROJ)
sys.path.append(os.path.join(PROJ, '03_仿真', '44_Wang2024_真实R矩阵', 'scripts'))   # 子进程反序列化 _recon_workers 的任务要能找到它
SEED_M = 20261103                                         # 同 a12_metrics.py
KS = (1.0, 10.0) if QUICK else (0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
SEPS = (12, 24, 48) if QUICK else (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
RES_CFG = [(8.0, 1.0), (8.0, 0.5), (16.0, 1.0)]
NPAIR, NDP = (6, 2) if QUICK else (40, 4)
SCAN_FW = (8.0, 30.0); CEN = np.arange(504.0, 797.0, 20.0 if QUICK else 2.0); NDS = 4 if QUICK else 20
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


def main():
    sys.path.insert(0, ROOT)
    import all_algos_device as AA
    B = AA.main(build_only=True); AD, LD, run_all, pool, E = B['AD'], B['LD'], B['run_all'], B['pool'], B['E']
    t0 = time.time(); WL, BM = AD.WL, AD.BM; NW = len(WL); nd = LD.NDRAW
    OUT = os.path.join(AD.DEVDIR, 'results', 'all_algos', 'metrics' + ('_quick' if QUICK else '')); os.makedirs(OUT, exist_ok=True)
    peaks = lambda X: np.array([AD.peak_grid(x) for x in X])                      # = a11_peakfix.peaks
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

    def go(Y):
        """Y (nd, n, M) → ({方法: (nd, M, NW)}, {方法: (nd, M) 波长})。"""
        r, pk = run_all(Y)
        return {m: X.transpose(0, 2, 1) for m, X in r.items()}, pk

    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, AD.H.Parts(AD.RN, AD.BAND), False).items()}
    AAJ = json.load(open(os.path.join(AD.DEVDIR, 'results', 'all_algos', 'all_algos.json'), encoding='utf-8'))

    # ---------------- 合成谱（随机数消耗顺序同 a12_metrics.py）+ 防泄漏
    g = np.random.default_rng(SEED_M)
    scan = {fw: pnorm(gau(CEN, fw)) for fw in SCAN_FW}
    pairs = {}
    for fw, r in RES_CFG:
        for dl in SEPS:
            c1 = g.uniform(520.0, 780.0 - dl, NPAIR); sw_ = g.random(NPAIR) < 0.5
            a1 = np.where(sw_, r, 1.0); a2 = np.where(sw_, 1.0, r)
            pairs[(fw, r, dl)] = (pnorm(a1[:, None] * gau(c1, fw) + a2[:, None] * gau(c1 + dl, fw)), c1, c1 + dl)
    chk = {f'单线 {fw:g} nm': S for fw, S in scan.items()}; chk.update({f'双线 {k}': v[0] for k, v in pairs.items()})
    leak = AD.P18.overlap(chk, B['used']); log('合成谱与训练库 / 调参验证集 / 单色光训练集逐条比对，完全相同的条数：' + ('无' if not leak else str(leak)))
    assert not leak, leak
    J = dict(P0_uW=AD.P0 * 1e6, ks=list(KS), name=AA.NAME, src=AA.SRC, M={})
    M = J['M']

    # ---------------- 抗噪声 + 全谱误差逐样本分布
    sig = {k: float(np.median([np.linalg.norm(AD.RN @ x) / (x[BM].sum() * np.sqrt(np.trace(AD.DEVCOV))) for x in np.concatenate(list(sets.values()))])) * k for k in KS}
    J['snr'] = {str(k): sig[k] for k in KS}
    log(f'\n== 抗噪声：生成谱族（{len(sets)} 族 × {AD.NTE} 条 × {nd} 次）真实模式读数，光功率 = k × P0')
    for k in KS:
        gk = np.random.default_rng(AD.SEED_SIM + 1); fam, pk, es = {}, {}, {}
        for nm, S in sets.items():
            X, W = go(AD.meas(S.T, gk, 'real', P=k * AD.P0))
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
            dmax = {m: max(abs(fam[m][nm] - AAJ['fam']['real'][nm][m]) for nm in sets) for m in fam}
            log('   k = 1 与 all_algos.json 逐族形状误差的最大差：' + '，'.join(f'{m} {v:.1e}' for m, v in dmax.items()))
        log(f'   k = {k:<6g}信噪比 {sig[k]:7.3g}  族中位：' + '，'.join(f'{m} {M[m]["noise"][str(k)]["med"]:.3f}' for m in fam) + f'   {time.time() - t0:.0f} s')
        log('      窄线峰位中位（nm）：' + '，'.join(f'{m} {pk[m][PKF[0]]:.2f}' for m in pk))

    # ---------------- 单线扫描
    log(f'\n== 单线扫描：中心 {CEN[0]:.0f}–{CEN[-1]:.0f} nm、步长 {CEN[1] - CEN[0]:g} nm，每个中心 {NDS} 次（P0）')
    inner = (CEN >= 520) & (CEN <= 780)
    for fw in SCAN_FW:
        S = scan[fw]; X, W = go(AD.meas(S.T, g, 'real', nd=NDS)); fwt = np.array([AD.fwhm_grid(x) for x in S])
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
        log(f'   半高宽 {fw:g} nm，520–780 nm 内逐中心 |峰位误差| 中位的平均：' + '，'.join(f'{m} {np.mean(np.array(M[m]["scan"][str(fw)]["pk_med"])[inner]):.2f}' for m in allp)
            + f'   {time.time() - t0:.0f} s')

    # ---------------- 双峰分辨率
    log(f'\n== 双峰分辨率：每个间隔 {NPAIR} 对 × {NDP} 次（P0）')
    for fw, r in RES_CFG:
        fr = {}
        for dl in SEPS:
            S, c1, c2 = pairs[(fw, r, dl)]; X, _ = go(AD.meas(S.T, g, 'real', nd=NDP))
            for m, Xm in X.items():
                fr.setdefault(m, {})[dl] = float(np.mean([resolved(Xm[q, i], c1[i], c2[i]) for q in range(NDP) for i in range(NPAIR)]))
        for m in fr:
            res = {}
            for th in (0.5, 0.9):
                ok = [dl for dl in SEPS if all(fr[m][d] >= th for d in SEPS if d >= dl)]; res[str(th)] = min(ok) if ok else None
            M[m].setdefault('res', {})[f'{fw:g}|{r:g}'] = dict(frac=fr[m], res=res)
        log(f'   半高宽 {fw:g} nm、强度比 {r:g}，分辨率（≥ 50 % / ≥ 90 %）：' + '，'.join(f'{m} {M[m]["res"][f"{fw:g}|{r:g}"]["res"]["0.5"]}/{M[m]["res"][f"{fw:g}|{r:g}"]["res"]["0.9"]}' for m in fr)
            + f'   {time.time() - t0:.0f} s')
    pool.close()
    json.dump(J, open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log(f'用时 {time.time() - t0:.0f} s')
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
