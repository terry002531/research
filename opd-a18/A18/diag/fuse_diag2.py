# -*- coding: utf-8 -*-
"""查融合：融合验证集（train 模式、读数不缩放）和测试（real 模式、读数按 C 峰值缩放）差在哪。
新的非测试集 all_fams(VSEED + 21, 30, jitter=False)，2 种读数模式 × 2 种缩放，逐族报 门控（SS 前 / 后）、C 原始、平均缩放系数。"""
import os, sys, time
ROOT = r'D:\OneDrive\Terry\01_Research\09_OPD端到端设计_光谱重建\06_重建算法\算法对照'
sys.argv = [os.path.join(ROOT, 'a15_three.py'), '--third', 'cnet0', '--no-b', '--a16a', '32', '--a16a-ss', '--test-seed', '6161']
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import a15_three as M
import a16_ss

AD, P17, P18, LD, BM = M.AD, M.P17, M.P18, M.LD, M.BM
D, H, A, S7, F, DS = AD.D, AD.H, AD.A, AD.S7, AD.F, AD.DS
t0 = time.time(); Rn = AD.RN; BAND = AD.BAND
P, lib, Slib, NT8, NN = M.prep()
sv, _, se = P18.SEEDS[0]; g = np.random.default_rng(se)
Sva = D.gen5(60, sv, BAND)[0]; Yva = AD.meas(Sva.T, g)
vsets = AD.all_fams(P18.VSEED, AD.NVF, P, True); Yv = {k: AD.meas(S.T, g) for k, S in vsets.items()}
nd = LD.NDRAW; E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
sig = min(A.SIG_E, key=lambda s: E(D.Gauss5(Rn, BAND).solve(Yva, s)[0], Sva).mean())
GM = S7.GaussMR(Rn, BAND, F.C_TRUE); PM = D.ParamE(Rn, BAND)
famg = [k for k in lib if k not in P17.PEAKY and k != '同分布五类']
GF = M.gf_a16a(Rn, lib, famg, vsets, Yv, E)
GSS = a16_ss.GateSS(Rn, M.WL, BAND, GM, GF.classes, dev=M.DEV)
print(f'σ_e = {sig}；建好 {time.time() - t0:.0f} s', flush=True)


def gate_of(Y):
    Xp = PM.solve(Y, sig)[0]; X8 = H.solve_pick(P, Y, H.infer(NT8, Y))[0]
    Xf, dPf, Wf = GF.solve_u(Y); gp = DS.fuse_parts(Rn, Xf, Wf, dPf, [Xp, X8], Y, GF, GF.classes)
    g0 = np.clip(DS.gate_mean(gp), 0, None); XB = np.clip(M.t3_infer(Rn, Y), 0, None)
    return g0, GSS.apply(Y, g0, Wf, XB), XB



S_ = {k.replace('留出·', ''): v for k, v in AD.all_fams(P18.VSEED + 21, 30, P, False).items()}
pk = lambda X: X[:, BM, :].max(1)
gm = lambda v: float(np.exp(np.mean(np.log(v))))
EST = ('C峰值', 'C→门控峰值1轮', 'C→门控峰值2轮', 'max(C,门控1轮)')
R = {}
gr = np.random.default_rng(4321)
for k, S in S_.items():
    Y = AD.meas(S.T, gr, 'real')
    a0 = pk(M.t3_infer(Rn, Y)); sc = lambda a: Y / np.maximum(a, 1e-30)[:, None, :]
    _, g1, _ = gate_of(sc(a0)); a1 = a0 * pk(g1)
    _, g2, _ = gate_of(sc(a1)); a2 = a1 * pk(g2)
    am = np.maximum(a0, a1)
    _, g3, _ = gate_of(sc(a2)); _, g4, _ = gate_of(sc(am))
    _, gt, _ = gate_of(Y)
    R[k] = {'真缩放': (1.0, E(gt, S).mean()), EST[0]: (float(np.median(a0)), E(g1, S).mean()), EST[1]: (float(np.median(a1)), E(g2, S).mean()),
            EST[2]: (float(np.median(a2)), E(g3, S).mean()), EST[3]: (float(np.median(am)), E(g4, S).mean())}
print(f'完 {time.time() - t0:.0f} s', flush=True)
cols = ['真缩放', *EST]
print('\nreal 模式；各族：幅度估计 a 中位（真值 1） / 门控(SS后) 形状误差')
print('| 族 | ' + ' | '.join(cols) + ' |'); print('|---' * (len(cols) + 1) + '|')
for k in S_:
    print(f'| {k} | ' + ' | '.join(f'{R[k][c][0]:.2f} / {R[k][c][1]:.3f}' for c in cols) + ' |')
print('| **26 族几何** | ' + ' | '.join(f'{gm([R[k][c][1] for k in S_]):.4f}' for c in cols) + ' |')
