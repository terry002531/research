# -*- coding: utf-8 -*-
"""Wang 2024 器件（47 号）各算法共用的测试口径：同一批测试谱、同一份测试读数、同一套误差算法，只有重建算法在变。

用户 2026-11-02："使用 A15 算法对比 A12、Xie MLP 还有常规算法，进行光谱重建。使用这个（47 号真实器件模式）的数据进行重新的训练。"
先 import a11_device（wang），再 import 本模块。测试集：
  a. 生成谱族：all_fams(TEST_ST = 4242, NTE) 全部 26 族，读数 = a11_device.test_readings（真实器件模式 'device' 档，测试器件，存盘共用）；
  b. 论文 Fig 3 的 25 条测试光：a11_device.fig3_set（47 号 forward 细网格完整正向、真实器件模式，NSIM 只测试器件各测 1 次，存盘共用）。
误差：形状误差 = 4 nm 网格上峰值归一均方根 / 真值极差（LD.band_nrmse）；单峰族、Fig 3 单峰组另算峰位误差。
"""
import numpy as np
import a11_device as AD

LD, P18 = AD.LD, AD.P18
assert AD.WANG
PKF = ['窄线 4–10 nm', '单峰 15–60 nm']                                       # 生成谱族里另算峰位误差的单峰族


def test_data(P, used):
    """→ (sets {族: (NTE, NW)}, TR {族: (nd, n, NTE)}, F3)；测试谱与 used 里所有训练 / 调参用谱逐条比对，有相同的就报错。"""
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    leak = P18.overlap(sets, used); assert not leak, leak
    return sets, AD.test_readings(sets), AD.fig3_set()


def evaluate(run, sets, TR, F3, log=print):
    """run(Y (nd, n, M)) → ({方法: (nd, NW, M) 谱}, {只认波长的方法: (nd, M) 波长 nm})。
    返回 J（可 json）和 raw（逐样本误差 + 示例重建，pickle）。"""
    J = dict(fam={}, fampk={}, fig3={}); raw = dict(famerr={}, f3err={}, f3dpk={}, demo={}, f3demo={})
    for nm, S in sets.items():
        r, pk = run(TR[nm]); nd = TR[nm].shape[0]
        e = {m: np.array([LD.band_nrmse(X[d], S.T) for d in range(nd)]) for m, X in r.items()}       # (nd, NTE)
        J['fam'][nm] = {m: float(v.mean()) for m, v in e.items()}
        for m, v in e.items():
            raw['famerr'].setdefault(m, {})[nm] = v
        raw['demo'][nm] = dict(truth=S[0], **{m: X[0, :, 0] for m, X in r.items()})
        if nm in PKF:
            tp = np.array([AD.peak_grid(s) for s in S])
            d = {m: np.abs([[AD.peak_grid(X[k][:, i]) - tp[i] for i in range(len(S))] for k in range(nd)]) for m, X in r.items()}
            d.update({m: np.abs(v - tp[None]) for m, v in pk.items()})
            J['fampk'][nm] = {m: float(np.median(v)) for m, v in d.items()}
    log(f'   生成谱族 {len(sets)} 族完')
    r, pk = run(F3['Y']); grp = np.array(F3['group']); single = np.isin(grp, AD.F3_PKG)
    for m, X in r.items():
        R = AD.fig3_eval(X, F3); s = AD.fig3_summary(R, F3)
        J['fig3'][m] = dict(summary=s, err_mean=R['err'].mean(0).tolist(), dpk_mean=np.nanmean(R['dpk'], 0).tolist())
        raw['f3err'][m] = R['err']; raw['f3dpk'][m] = R['dpk']; raw['f3demo'][m] = X[0]
    for m, v in pk.items():                                                    # 只认波长：只比单峰组峰位
        dpk = np.where(single[None], v - F3['pk'][None], np.nan); a = np.abs(dpk[:, single])
        J['fig3'][m] = dict(summary={'_峰位中位': float(np.median(a)), '_峰位RMS': float(np.sqrt(np.mean(a ** 2)))},
                            dpk_mean=np.nanmean(dpk, 0).tolist())
        raw['f3dpk'][m] = dpk
    log('   Fig 3 测试光完')
    return J, raw


def table(J, meths, name, log=print):
    """日志：各族误差表 + 中位 + 单峰族峰位 + Fig 3 各组。"""
    fam = list(J['fam'])
    log('   族                    ' + ''.join(f'{name.get(m, m):>10s}' for m in meths))
    for nm in fam:
        log(f'   {nm:<20s}' + ''.join(f'{J["fam"][nm][m]:10.4f}' if m in J['fam'][nm] else f'{"—":>10s}' for m in meths))
    med = {m: float(np.median([J['fam'][k][m] for k in fam])) for m in meths if m in J['fam'][fam[0]]}
    log('   各族中位              ' + ''.join(f'{med[m]:10.4f}' if m in med else f'{"—":>10s}' for m in meths))
    for nm in PKF:
        log(f'   {nm} 峰位 |误差| 中位 nm：' + '，'.join(f'{name.get(m, m)} {J["fampk"][nm][m]:.3f}' for m in meths if m in J['fampk'][nm]))
    log(f'\n   Fig 3 测试光（{AD.NSIM} 只测试器件各 1 次）  ' + ''.join(f'{G:>9s}' for G in AD.F3_GROUPS) + '   4组平均 |峰位|中位 峰位RMS')
    for m in meths:
        s = J['fig3'][m]['summary']; f = lambda k: f'{s[k]:9.3f}' if k in s else f'{"—":>9s}'
        log(f'   {name.get(m, m):<28s}' + ''.join(f(G) for G in AD.F3_GROUPS) + f('_4组平均') + f'{s["_峰位中位"]:9.2f}{s["_峰位RMS"]:8.2f}')
    return med
