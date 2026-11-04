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



import torch as t
CG = np.exp(np.linspace(np.log(0.45), np.log(1.3), 11))


def evid(Y):
    """GaussX 两遍（同 solve_u）后的对数边缘似然 log Σ_k π_k p(y|k) (nd, M)。"""
    nd_, n, M_ = Y.shape; Yf_all = t.tensor(Y.transpose(0, 2, 1).reshape(nd_ * M_, n), dtype=t.float64, device=GF.dev); out = []
    ch = int(256 * 5 / len(GF.classes))
    for i in range(0, Yf_all.shape[0], ch):
        Yf = Yf_all[i:i + ch]; Xk, ll, P = GF._solve(Yf, GF.Sy_prior)
        Sy = GF.RCR[:, None] + GF.cal_cov(t.einsum('kbw,kbv->kbwv', Xk, Xk) + P[:, None]); _, ll, _ = GF._solve(Yf, Sy)
        out.append(t.logsumexp(ll + GF.logpi[None], -1))
    return t.cat(out).cpu().numpy().reshape(nd_, M_)


S_ = {k.replace('留出·', ''): v for k, v in AD.all_fams(P18.VSEED + 21, 30, P, False).items()}
pk = lambda X: X[:, BM, :].max(1)
gm = lambda v: float(np.exp(np.mean(np.log(v))))
R = {}; n = Rn.shape[0]
gr = np.random.default_rng(4321)
for k, S in S_.items():
    Y = AD.meas(S.T, gr, 'real')
    a0 = pk(M.t3_infer(Rn, Y)); Ys = Y / np.maximum(a0, 1e-30)[:, None, :]
    L = np.stack([evid(c * Ys) + n * np.log(c) for c in CG])                 # (nc, nd, M)
    ci = CG[L.argmax(0)]                                                    # 逐样本挑
    w = np.exp(L - L.max(0)); w /= w.sum(0); cs = np.exp((w * np.log(CG)[:, None, None]).sum(0))   # 后验平均（对数）
    _, gC, _ = gate_of(Ys); _, gP, _ = gate_of(Ys * ci[:, None, :]); _, gS, _ = gate_of(Ys * cs[:, None, :]); _, gt, _ = gate_of(Y)
    a_pick = a0 / ci
    R[k] = dict(真缩放=(1.0, E(gt, S).mean()), C峰值=(float(np.median(a0)), E(gC, S).mean()),
                证据挑=(float(np.median(a_pick)), E(gP, S).mean()), 证据平均=(float(np.median(a0 / cs)), E(gS, S).mean()))
    print(k, {c: tuple(round(x, 3) for x in v) for c, v in R[k].items()}, f'{time.time() - t0:.0f} s', flush=True)
cols = list(R[next(iter(R))])
print('\nreal 模式；各族：幅度估计 a 中位（真值 1） / 门控(SS后) 形状误差')
print('| 族 | ' + ' | '.join(cols) + ' |'); print('|---' * (len(cols) + 1) + '|')
for k in S_:
    print(f'| {k} | ' + ' | '.join(f'{R[k][c][0]:.2f} / {R[k][c][1]:.3f}' for c in cols) + ' |')
print('| **26 族几何** | ' + ' | '.join(f'{gm([R[k][c][1] for k in S_]):.4f}' for c in cols) + ' |')
