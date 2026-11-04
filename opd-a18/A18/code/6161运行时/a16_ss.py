# -*- coding: utf-8 -*-
"""A16S 用的结构解法（用户 2026-11-03："按这条路推进来优化这个算法"）：基本形状字典 + 贝叶斯贪心选形状（按证据增量）+ 回扫，闭式后验均值。
谱 s = Φ_bg a_bg + Σ_k Φ_k a_k + r，a ~ N(0, σa² I)，r ~ GP(0, K_r)（波段内），读数 y = R s + n，n ~ N(0, Σ_n)，Σ_n 取 B 原始输出处的噪声协方差。
加一组形状 U = R Φ_k 后证据增量 Δ = −½ log det(I + σa² Uᵀ S⁻¹ U) + ½ σa² zᵀ (I + σa² Uᵀ S⁻¹ U)⁻¹ z，z = Uᵀ S⁻¹ y；全字典 GPU 批量算，Δ < τ 停。
并入门控支 A（A 判族、结构解法重建）：A′ = A·(1 − α Σ_{h∈U} w_h) + α Σ_{h∈U} w_h X_SS(类型 h)，w_h = A 给 h 族各簇的后验之和。
U、α、各类型 (σa, τ, 背景阶数) 都在诊断用调参集（VSEED + 13）上选定（scratchpad ss_int / ss_cls，11-03），诊断集（VSEED + 7）上 8 难族几何 0.1641 → 0.1499。"""
import numpy as np, torch as t

TYPES = ('lines', 'peaks', 'steps', 'bp', 'dips', 'ripple', 'general')
KMAX = dict(lines=6, peaks=7, steps=6, bp=2, dips=3, ripple=1, general=8)


class SS:
    def __init__(self, R, WL, band, sa=3.0, sr=0.01, ell=30.0, dev='cuda'):
        self.dev = dev; self.sa = sa; dt = t.float64
        b = (WL >= band[0]) & (WL <= band[1]); self.b = b; self.WL = WL; NW = WL.size
        x = np.zeros(NW); x[b] = (WL[b] - WL[b].mean()) / (WL[b].max() - WL[b].mean())
        leg = np.stack([np.where(b, np.polynomial.legendre.legval(x, np.eye(3)[j]), 0.0) for j in range(3)], 1)   # (NW, 3)
        T = lambda a: t.tensor(a, dtype=dt, device=dev)
        self.R = T(R); self.BG = T(leg)
        dd = WL[:, None] - WL[None]; self.KR = T(np.where(np.outer(b, b), sr ** 2 * np.exp(-0.5 * (dd / ell) ** 2), 0.0))
        self.RKR = self.R @ self.KR @ self.R.T; RB = self.R @ self.BG; self.S_bg = sa ** 2 * RB @ RB.T
        g = lambda c, fw: np.exp(-0.5 * ((WL - c) / (fw / 2.3548)) ** 2)
        sg = lambda e, w: 1.0 / (1.0 + np.exp(-(WL - e) / (w / 4)))
        b0, b1 = band; c1 = np.arange(b0, b1 + 1e-9, 1.0); c2 = np.arange(b0, b1 + 1e-9, 2.0)
        D = {}
        D['lines'] = np.stack([g(c, w) for w in (2, 3, 4, 6, 8, 10) for c in c1])[:, :, None]
        D['peaks'] = np.stack([g(c, w) for w in (15, 20, 27, 35, 45, 60) for c in c2])[:, :, None]
        D['steps'] = np.stack([sg(e, w) for w in (2, 4, 8) for e in c1])[:, :, None]
        D['bp'] = np.stack([sg(e, w) * (1 - sg(e + wd, w)) for w in (3, 8) for e in c2 for wd in np.arange(20, 151, 2) if e + wd <= b1])[:, :, None]
        D['dips'] = np.stack([leg * g(c, w)[:, None] for w in (10, 15, 22, 33, 50) for c in c2])
        per = np.exp(np.linspace(np.log(20), np.log(100), 161))
        D['ripple'] = np.stack([np.stack([leg[:, j] * f for j in (0, 1) for f in (np.sin(2 * np.pi * WL / p), np.cos(2 * np.pi * WL / p))], 1) for p in per])
        D['general'] = np.concatenate([D['lines'], D['peaks'], D['steps']])
        self.Phi = {k: T(np.where(b[None, :, None], v, 0.0)) for k, v in D.items()}               # (Ng, NW, p)
        self.U = {k: t.einsum('nw,gwp->gnp', self.R, v) for k, v in self.Phi.items()}            # (Ng, n, p)

    def _delta(self, U, Si, Siy, sa):
        """U (Ng, n, p)，Si (N, n, n)，Siy (N, n) → Δ (N, Ng)。"""
        sa2 = sa ** 2; p = U.shape[-1]
        z = t.einsum('gnp,bn->bgp', U, Siy)
        if p == 1:
            q = t.einsum('gn,bnm,gm->bg', U[..., 0], Si, U[..., 0]); a = 1 + sa2 * q
            return -0.5 * t.log(a) + 0.5 * sa2 * z[..., 0] ** 2 / a
        W = t.einsum('bnm,gmq->bgnq', Si, U); A = t.eye(p, dtype=U.dtype, device=self.dev) + sa2 * t.einsum('gnp,bgnq->bgpq', U, W)
        L = t.linalg.cholesky(A); v = t.cholesky_solve(z[..., None], L)[..., 0]
        return -t.log(t.diagonal(L, dim1=-2, dim2=-1)).sum(-1) + 0.5 * sa2 * (z * v).sum(-1)

    def _scan(self, U, Si, Siy, sa, chunk=4096):
        best = []; idx = []
        for i in range(0, U.shape[0], chunk):
            d = self._delta(U[i:i + chunk], Si, Siy, sa); m, j = d.max(1); best.append(m); idx.append(j + i)
        best = t.stack(best, 1); idx = t.stack(idx, 1); m, k = best.max(1)
        return m, idx.gather(1, k[:, None])[:, 0]

    def solve(self, typ, Y, Sn, tau, sa=None, nback=2, bg=3):
        """bg = 背景 Legendre 列数（0–3）。"""
        sa = self.sa if sa is None else sa; BG = self.BG[:, :bg]; RB = self.R @ BG
        """Y (N, n)、Sn (N, n, n) numpy → 谱 (N, NW)、对数证据 (N,)、每条选的组数 (N,)。"""
        dt = t.float64; Yt = t.tensor(Y, dtype=dt, device=self.dev); N = Yt.shape[0]
        U = self.U[typ]; sa2 = sa ** 2; K = KMAX[typ]
        S = (self.sa ** 2 * RB @ RB.T)[None] + self.RKR[None] + t.tensor(Sn, dtype=dt, device=self.dev)
        sel = t.full((N, K), -1, dtype=t.long, device=self.dev)
        outer = lambda j: t.einsum('bnp,bmp->bnm', U[j], U[j])                                     # j (N,) → (N, n, n)
        inv = lambda S: t.cholesky_inverse(t.linalg.cholesky(S))
        for k in range(K):                                                                          # 贪心
            Si = inv(S); m, j = self._scan(U, Si, (Si @ Yt[..., None])[..., 0], sa)
            ok = m > tau; sel[ok, k] = j[ok]
            S = S + sa2 * outer(j) * ok[:, None, None]
            if not ok.any():
                break
        for _ in range(nback):                                                                      # 回扫
            for k in range(K):
                has = sel[:, k] >= 0
                if not has.any():
                    continue
                jk = sel[:, k].clamp_min(0); S0 = S - sa2 * outer(jk) * has[:, None, None]
                Si = inv(S0); m, j = self._scan(U, Si, (Si @ Yt[..., None])[..., 0], sa)
                keep = has & (m > tau); sel[has, k] = t.where(keep[has], j[has], t.full_like(j[has], -1))
                S = S0 + sa2 * outer(j) * keep[:, None, None]
        Si = inv(S); Siy = (Si @ Yt[..., None])[..., 0]
        L = t.linalg.cholesky(S); ll = -0.5 * (Yt * Siy).sum(1) - t.log(t.diagonal(L, dim1=-2, dim2=-1)).sum(1)
        # 后验均值 x = σa² Φ (RΦ)ᵀ S⁻¹ y + K_r Rᵀ S⁻¹ y，Φ = 背景 + 已选组
        x = self.sa ** 2 * BG @ (BG.T @ (self.R.T @ Siy.T)) + self.KR @ (self.R.T @ Siy.T)         # (NW, N)
        x = x.T
        for k in range(K):
            has = sel[:, k] >= 0
            if has.any():
                jk = sel[has, k]; Ph = self.Phi[typ][jk]; Uk = U[jk]                                 # (h, NW, p)、(h, n, p)
                x[has] += sa2 * t.einsum('hwp,hnp,hn->hw', Ph, Uk, Siy[has])
        return x.clamp_min(0).cpu().numpy(), ll.cpu().numpy(), (sel >= 0).sum(1).cpu().numpy()


CFG = dict(lines=(1.0, -1.0, 3), peaks=(0.3, -1.0, 0), steps=(0.3, 2.0, 1), general=(0.3, 1.0, 3))     # 类型 → (σa, τ, 背景列数)
USE = {'激光线': 'lines', '荧光灯': 'lines', '4–6': 'peaks', '阶梯': 'steps', '零件组合': 'general'}         # 族名片段 → 类型
ALPHA = 0.75


class GateSS:
    def __init__(self, R, WL, band, GM, classes, dev='cuda'):
        self.SV = SS(R, WL, band, dev=dev); self.GM = GM; self.R = R
        self.idx = {}
        for frag, typ in USE.items():
            ii = [i for i, c in enumerate(classes) if frag in c.split('#')[0]]
            if ii:
                self.idx[frag] = (np.array(ii), typ)

    def apply(self, Y, gate, Wf, XB):
        """Y (nd, n, M)、门控输出 gate (nd, NW, M)、Wf (nd, M, K)、B 原始输出 XB (nd, NW, M) → A′。"""
        nd, n, M = Y.shape; NW = gate.shape[1]
        Yf = Y.transpose(0, 2, 1).reshape(-1, n)
        Sn = np.stack([self.GM.noise_cov(XB[d][:, m]) for d in range(nd) for m in range(M)])
        Xs = {}
        for typ in set(t_ for _, t_ in self.idx.values()):
            sa, tau, bg = CFG[typ]
            X = np.concatenate([self.SV.solve(typ, Yf[i:i + 256], Sn[i:i + 256], tau, sa=sa, bg=bg)[0] for i in range(0, len(Yf), 256)])   # 分块，防显存爆
            Xs[typ] = X.reshape(nd, M, NW).transpose(0, 2, 1)
        tot = np.zeros((nd, M)); mix = np.zeros_like(gate)
        for frag, (ii, typ) in self.idx.items():
            w = Wf[:, :, ii].sum(-1); tot += w; mix += w[:, None, :] * Xs[typ]
        return np.clip(gate * (1 - ALPHA * tot[:, None, :]) + ALPHA * mix, 0, None)


CG = np.exp(np.linspace(np.log(0.45), np.log(1.3), 11))                  # 证据挑缩放的倍数网格（诊断集 2026-11-03 定）


def escale(GF, Y):
    """门控先验按证据挑读数缩放（用户 2026-11-03"做证据挑缩放"）：逐条读数在 c ∈ CG 里挑
    log Σ_k π_k p(c·y | k) + n log c 最大的 c（n log c 是换元雅可比），GaussX 两遍同 solve_u。Y (nd, n, M) → c (nd, M)。"""
    nd, n, M = Y.shape; L = []
    for c in CG:
        Yf_all = t.tensor((c * Y).transpose(0, 2, 1).reshape(nd * M, n), dtype=t.float64, device=GF.dev); out = []
        ch = int(256 * 5 / len(GF.classes))
        for i in range(0, Yf_all.shape[0], ch):
            Yf = Yf_all[i:i + ch]; Xk, ll, P = GF._solve(Yf, GF.Sy_prior)
            Sy = GF.RCR[:, None] + GF.cal_cov(t.einsum('kbw,kbv->kbwv', Xk, Xk) + P[:, None]); _, ll, _ = GF._solve(Yf, Sy)
            out.append(t.logsumexp(ll + GF.logpi[None], -1))
        L.append(t.cat(out).cpu().numpy().reshape(nd, M) + n * np.log(c))
    return CG[np.stack(L).argmax(0)]
