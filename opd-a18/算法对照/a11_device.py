# -*- coding: utf-8 -*-
"""用器件模拟器（45 号 Uddin 2024、46 号 Cui 2025 的 forward.py 真实模式）训练 A11，再拿模拟读数和作者的真实读数测。

用户 2026-11-02："基于这个来训练 A11 算法，训练出来的东西和模拟读数算法放到同一个文件夹里"；
  "不用特别留出来一些种类不训练，全部训练，只要确保训练的东西和测试的东西不是完全一致的就行"。

和 41 号原版 A11（diag_spike.run / 44 号 a11_vs_nsf.build）的区别只有三处：
  1. 读数：41 号用"R 乘 2 % 共模 + 2 % 每行标定误差、1 % 读数噪声"；这里换成器件模拟器的真实模式误差
       Uddin 可见段：I = g·R p + (b − 1)·I_dark + n（暗电流整体漂 1.5 %、增益 ln 标准差 0.15、白噪声 0.3 nA）
       Cui：I = R p + n（白噪声，训练时在 'recon' 1.3e-10 A 和 'reading' 6.3e-10 A 两档之间按对数均匀抽）
     训练时光功率在参考功率（Uddin ≈ 实测测试光拟合功率 11 µW，Cui 200 nW）的 0.5–2 倍之间随机。
     网络（A8 零件网络、全族网络）每步重新抽这些误差；闭式部件（门控、一致化、χ²）的噪声协方差在原标定误差项上
     加一项器件误差协方差（Uddin：σ_b² I_dark I_darkᵀ + σ_n² I；Cui：σ_n² I，σ_n 取两档几何平均），原标定误差项缩到 0.1 %。
  2. 训练库：41 号全部谱族都训练，不留出（原来的 6 个留出族、CIE LED 实谱也进训练库和验证集）。
     CIE LED 实谱是一组固定的真实 LED 谱，不随种子变；训练和验证用的是随机平移 4–20 nm、加 ±20 % 斜坡后的版本，测试用原谱。
     测试谱用新种子 TEST_ST 生成，和训练库、各验证集、高斯先验用谱、零件网络训练池逐条比对，断言完全相同的条数为零。
  3. 读数个数：Cui 有 101 × 41 = 4141 个读数，A11 的协方差矩阵放不下；R 只有 23 个学习波长，插值到 4 nm 网格后
     秩仍是 23，所以把读数投影到 R 的 23 个左奇异向量上（信号全在这 23 维里；白噪声投影后仍是同样大小的白噪声）。
波段：Uddin 500–800 nm，Cui 500–720 nm（各自 R 的范围），41 号的 4 nm 网格。Uddin 近红外段（1550–1560 nm）
  不在 41 号谱库的范围里，不做。

测试：
  a. 生成谱族（每族 NTE 条 × 4 次抽样）：理想读数、真实模式读数（Cui 分 recon / reading 两档）。
     对照：经典平滑非负 A1、A11 两个支路；Uddin 再加论文算法（paper_check.Recon，作者 MATLAB 代码的 Python 版）。
  b. 作者的真实读数：Uddin 5 束（560–750 nm，和商用光谱仪参考谱比误差和峰位）、Cui 6 束（520–720 nm，只有标称波长，比峰位）；
     同一批测试光再用模拟器真实模式各抽 NSIM 次，看模拟读数和真实读数给出的误差是否一致。
  真实读数的亮度未知：所有方法解码前先用全族网络估一次亮度，把读数缩放到"峰值 ≈ 1"的尺度（scale_y）。

  py -I scripts 不需要；在本文件夹运行：
  py -I -X utf8 a11_device.py uddin|cui|wang [--steps 30000] [--quick] [--rerr] [--m3]
输出：<器件文件夹>/results/a11/{log.txt, a11.json, examples.pkl, fig_real.png, fig_families.png, net8.pt, mlp.pt}
  --rerr 输出到 a11_R/；--m3（只 Cui，用户 2026-11-02"用 M3 重训 A11"）输出到 a11_M3/：训练读数再乘逐 Vgs 帧增益漂移，多一个 'm3' 测试档。

wang（47 号 Wang 2024，用户 2026-11-02"使用这个的数据进行重新的训练"）：101 个偏压读数，波段 400–900 nm，参考功率 1 µW。
  读数误差 = 47 号 forward.Device.measure(real=True) 的真实器件模式，逐项搬到 4 nm 网格、GPU 上按样本抽（dev_read）：
    器件系统误差（ε(λ)、R 测量误差、波长对不齐）每只器件固定：训练读数随机取 2048 只"训练器件"之一，测试读数取另外 256 只"测试器件"；
    每次测量另抽温度平移、光功率增益、扫描漂移、偏压偏差 / 抖动、暗电流、散粒 + 读出噪声。训练时功率 0.5–2 倍 P0 对数均匀。
  闭式部件的器件误差协方差 DEVCOV = 训练器件上"真实 − 理想"读数的二阶矩（去掉整体增益，Σx = 1 的 gen5 谱平均）。
  作者没公开实测读数，测试 = 生成谱族（'device' 档）+ 论文 Fig 3 的 25 条测试光（47 号 forward 细网格完整正向模型，NSIM 只测试器件各测 1 次）；
  两份测试读数第一次生成后存 results/testset_device.npz、testset_fig3.npz，A12 / A15 / 其他算法的脚本都读同一份。
"""
import os, sys, json, time, pickle, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = ROOT                                            # 往上找到含「03_仿真」的项目根目录
while not os.path.isdir(os.path.join(PROJ, '03_仿真')) and os.path.dirname(PROJ) != PROJ:
    PROJ = os.path.dirname(PROJ)
S41 = os.path.join(PROJ, '03_仿真', '41_重建算法二_形状自适应解码', 'scripts')
DEVS = {'uddin': ('45_Uddin2024_真实R矩阵', (500.0, 800.0)), 'cui': ('46_Cui2025_真实R矩阵', (500.0, 720.0)),
        'wang': ('47_Wang2024_真实R矩阵', (400.0, 900.0))}
WANG_UNITS = {'train': (10_000_000, 2048), 'test': (20_000_000, 256)}   # Wang：(器件编号起点, 只数)，训练器件和测试器件不重叠

ap = argparse.ArgumentParser(); ap.add_argument('device', choices=list(DEVS)); ap.add_argument('--steps', type=int, default=30000)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--rerr', action='store_true', help='训练时 R 乘 41 号 L5 标定误差（2 %% 共模 + 2 %% 每行），门控协方差同步恢复 c = 0.02')
ap.add_argument('--m3', action='store_true', help='只 Cui：训练读数再乘 M3 逐 Vgs 帧增益漂移（见 M3_* 常量），门控协方差同步加这一项')
ARG = ap.parse_args(); RERR = ARG.rerr; M3 = ARG.m3
UDDIN = ARG.device == 'uddin'; WANG = ARG.device == 'wang'
DEVDIR = os.path.join(ROOT, DEVS[ARG.device][0]); BAND = DEVS[ARG.device][1]
sys.path.insert(0, os.path.join(DEVDIR, 'scripts'))
import forward as FW
if not WANG:                                                                   # 47 号的 paper_check 会拉进 44 号整条链和 NSF，这里用不到
    import paper_check as PC
    sys.modules.pop('paper_check')
sys.path.pop(0); sys.modules.pop('forward')   # 41 号链条里 24 号也有个 forward.py，别让器件版顶替它
sys.path.insert(0, S41)
import algo18 as P18
import diag_spike as DS
P17, Q = P18.P17, P18.Q
A, D, F, S7, H, LD, T = P18.A, P18.D, P18.F, P18.S7, P18.H, P18.LD, P18.T

assert not (M3 and (UDDIN or RERR)), '--m3 只给 Cui，且不和 --rerr 同用'
OUT = os.path.join(DEVDIR, 'results', 'a11' + ('_R' if RERR else '') + ('_M3' if M3 else '') + ('_quick' if ARG.quick else '')); os.makedirs(OUT, exist_ok=True)
TEST_ST = 4242                                         # 新测试种子（41 号开发和 44 号都没用过）
NTR = 300 if ARG.quick else P17.NTR; NTE = 10 if ARG.quick else P17.NTE; NVF = 10 if ARG.quick else P18.NVF
STEPS = 300 if ARG.quick else ARG.steps; STEPS8 = 300 if ARG.quick else 6000
NSIM = 20 if ARG.quick else 200; SEED_SIM = 20261102
METH = ['A1', 'gate', 'net', 'A11'] + (['paper'] if UDDIN else [])
NAME = {'A1': '经典平滑非负', 'gate': 'A11 门控支路', 'net': 'A11 网络支路', 'A11': 'A11', 'paper': '论文算法'}
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


# ------------------------------------------------------------------ 器件：R、误差、真实读数
LD.BAND = BAND
WL = LD.WLg(); BM = (WL >= BAND[0]) & (WL <= BAND[1])
if UDDIN:
    DV = FW.Device('vis'); RDEV = DV._R_at(WL)[0]; RDEV[:, ~BM] = 0
    ER = FW.REAL['vis']; DARK = DV.dark
    _pc = json.load(open(os.path.join(DEVDIR, 'results', 'paper_check.json'), encoding='utf-8'))['vis']
    P_FIT = np.array([r['forward']['c_参考谱整条']['P_fit_uW'] * 1e-6 for r in _pc]); P0 = float(np.median(P_FIT))
    Y_REAL = DV.I_test - DARK[:, None]; U = None
    DEVCOV = ER['sigma_b'] ** 2 * np.outer(DARK, DARK) + ER['sigma_n'] ** 2 * np.eye(len(DARK))
elif WANG:
    import torch
    TDEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    DV = FW.Device(unit=None); P0 = FW.P_NOM; U = None; Y_REAL = None; DEVCOV = None   # DEVCOV 等 SCALE 定了再算
    WI = DV._R_at(WL, R=np.eye(len(DV.lam)))[0]; WI[:, ~BM] = 0               # (101 个 5 nm 节点, NW)：R(WL) = R @ WI（同 forward 的线性插值）
    RDEV = DV.R @ WI
    _DU = FW.Device(unit=None)

    def unit_R(u):
        """= FW.Device(unit=u).R_true()（同样的抽数顺序，省掉每只器件重读一次 npz）。"""
        g = np.random.default_rng(u); nV, nL = len(_DU.V), len(_DU.lam)
        _DU._z_cal = g.standard_normal(nL); _DU._z_off = float(g.standard_normal())
        _DU._z_unit = g.standard_normal((nV, nL)); _DU._z_rm = g.standard_normal((nV, nL))
        return _DU.R_true()
    assert np.allclose(unit_R(WANG_UNITS['test'][0] + 3), FW.Device(unit=WANG_UNITS['test'][0] + 3).R_true())
    RB = {k: torch.tensor(np.stack([unit_R(u0 + i) for i in range(nu)]), dtype=torch.float32, device=TDEV) for k, (u0, nu) in WANG_UNITS.items()}
else:
    DV = FW.Device(); _Rraw = DV._R_at(WL)[0].reshape(-1, len(WL)); _Rraw[:, ~BM] = 0
    _u, _s, _ = np.linalg.svd(_Rraw[:, BM], full_matrices=False); KP = int((_s > 1e-9 * _s[0]).sum()); U = _u[:, :KP]
    RDEV = U.T @ _Rraw; P0 = DV.P_learn
    SIG_LO, SIG_HI = FW.REAL['recon'], FW.REAL['reading']; SIG_CF = float(np.sqrt(SIG_LO * SIG_HI))
    Y_REAL = np.stack([U.T @ I.ravel() for I in DV.I_test], 1)
    DEVCOV = SIG_CF ** 2 * np.eye(KP)
SCALE = float(np.abs(RDEV).max()); RN = RDEV / SCALE; N_RD = RN.shape[0]
LAM_TEST = None if WANG else DV.lam_test
if WANG:
    _E = FW.REAL; _nV = len(DV.V); assert np.allclose(np.diff(DV.V), -0.1)
    _ar = _E['drift_rho'] ** np.abs(np.arange(_nV)[:, None] - np.arange(_nV)[None])          # AR(1) 平稳协方差 ρ^|i−j|
    _T = dict(WI=torch.tensor(WI.T, dtype=torch.float32, device=TDEV), LAM=torch.tensor(DV.lam, dtype=torch.float64, device=TDEV),
              LDR=torch.tensor(np.linalg.cholesky(_ar).T, dtype=torch.float32, device=TDEV), AV=torch.tensor(-DV.V, dtype=torch.float32, device=TDEV),
              DK=torch.tensor(DV.I_dark * 1e9, dtype=torch.float32, device=TDEV), RN=torch.tensor(RN, dtype=torch.float32, device=TDEV),
              BM=torch.tensor(BM, device=TDEV))

    def dev_read(St, Pw, RBu, gen=None, gain=True):
        """真实器件模式读数 = 47 号 forward.Device.measure(real=True) 的逐项误差，同样的顺序，搬到 4 nm 网格、GPU 上按样本抽。
        St (B, NW) 谱（任意尺度）；Pw (B,) 总光功率 W；RBu (B, nV, nλ) 每个样本那只器件的 R_true（5 nm 节点）。
        返回 (B, n) 读数，RN 单位（无误差时 = St @ RNᵀ）。gain=False 去掉整体功率增益（算 DEVCOV 用）。"""
        B = St.shape[0]; rn = lambda *s: torch.randn(*s, device=St.device, generator=gen)
        q = St @ _T['WI']                                                            # (B, nλ)：谱分到 5 nm 节点
        # 温度：R 沿能量平移，新 R(λ) = R(λ − δλ)，5 nm 节点间线性插值；等价于把 q 按插值权重挪到 R 的列上
        L = _T['LAM']; dE = -0.45e-3 * _E['temp_sigma_K'] * rn(B, 1).double()
        src = (L[None] - (L[None] - FW.HC_EV_NM / (FW.HC_EV_NM / L[None] + dE))).clamp(L[0], L[-1]).contiguous()
        i = (torch.searchsorted(L, src) - 1).clamp(0, len(L) - 2); w = ((src - L[i]) / (L[i + 1] - L[i])).float()
        q2 = torch.zeros_like(q).scatter_add_(1, i, q * (1 - w)).scatter_add_(1, i + 1, q * w)
        y = torch.einsum('bvl,bl->bv', RBu, q2) / SCALE
        # 光功率：整体增益 × 扫描漂移（AR(1)，按 0 → −10 V 顺序）
        if gain:
            y = y * torch.exp(_E['gain_sigma'] * rn(B, 1))
        y = y * (1 + _E['drift_sigma'] * (rn(B, _nV) @ _T['LDR']))
        # 偏压：读数其实在 V + δV 处测（同 np.interp，两端截平）
        u = ((_T['AV'][None] - (_E['bias_off_V'] * rn(B, 1) + _E['bias_jit_V'] * rn(B, _nV))) / 0.1).clamp(0, _nV - 1)
        i0 = u.floor().long().clamp(max=_nV - 2); f = u - i0
        y = y.gather(1, i0) * (1 - f) + y.gather(1, i0 + 1) * f
        # 换成 nA：暗电流（含温漂）+ 散粒 + 读出噪声，再换回读数单位
        k = St[:, _T['BM']].sum(1, keepdim=True) / (SCALE * Pw[:, None])             # 读数 = 电流 (A) × k
        I = y / k * 1e9 + _T['DK'][None] * (1 + _E['dark_sigma'] * rn(B, 1))
        I = I + torch.sqrt(FW.Q_E * 1e9 * I.abs() / _E['t_int'] + (_E['read_floor'] * 1e9) ** 2) * rn(B, _nV)
        return I * 1e-9 * k

    def dev_meas(St, Pw, bank, gen=None, gain=True, chunk=4096):
        """St (B, NW) GPU 谱 → (B, n) 读数；每个样本随机取 bank（'train' / 'test'）里的一只器件。分块免得显存不够。"""
        out = []
        for c in range(0, St.shape[0], chunk):
            s = St[c:c + chunk]; u = torch.randint(len(RB[bank]), (len(s),), device=St.device, generator=gen)
            out.append(dev_read(s, Pw[c:c + chunk], RB[bank][u], gen, gain))
        return torch.cat(out)

    _Sx = D.gen5(400, 777, BAND)[0]; _Sx = _Sx / _Sx[:, BM].sum(1, keepdims=True)              # Σ_band x = 1
    _g = torch.Generator(device=TDEV).manual_seed(777); _St = torch.tensor(np.tile(_Sx, (16, 1)), dtype=torch.float32, device=TDEV)
    _r = (dev_meas(_St, torch.full((len(_St),), P0, device=TDEV), 'train', _g, gain=False) - _St @ _T['RN'].T).double().cpu().numpy()
    DEVCOV = _r.T @ _r / len(_r)                       # 读数单位下、Σx = 1 时的器件误差二阶矩（整体增益除外：亮度由 scale_y 处理）
    del _St, _r
else:
    DEVCOV = DEVCOV / (SCALE * P0) ** 2                # 读数单位下、光谱 Σx = 1 时的器件误差协方差
LEV_R = A.LEVELS['L5'] if RERR else dict(cc=0.0, ci=0.0)   # R 标定误差：同 41 号 L5（algo2.meas / ladder.err_spec）
LEG = LD.lib().E.dl.LEG                                     # (3, NW) 1–3 阶勒让德

# M3（只 Cui，--m3）：测试那次相对学习矩阵，每个 Vgs 帧乘一个增益 g(Vgs)，所有 Vds、所有波长共用：
#   g − 1 = a · σ_g · s(Vgs) · GP(ℓ)，s ∝ 1/√(该帧信号 + 0.05)（弱信号帧漂得多，信号 = 6 束标称单色光模型读数的帧范数平均），ℓ = 2.5 V，
#   σ_g = 2.91 %（使模拟读数残差 ≈ 实测 0.8 %）。训练时 a 在 0.5–2 之间按对数均匀抽，'m3' 测试档 a = 1。
#   ℓ、σ_g 由作者 6 束实测读数（同一次会话，等于 g 的一个样本）估出：scratchpad try_struct/m3_gen.py、m3_het.py。
#   g 对读数是线性的：投影读数 = RN x + Σ_k z_k AK_k x，AK_k = Uᵀ diag(b_k) R_raw / SCALE，b_k 是 g 协方差的本征向量（含 √本征值）。
M3_ELL, M3_SG, M3_A = 2.5, 0.0291, (0.5, 2.0)
if M3:
    _nV, _nG = DV.R.shape[:2]; _vg = np.asarray(DV.Vgs, float)
    _w = np.mean([np.sqrt((DV.measure([l], [P0], lines=True, warn=False).reshape(_nV, _nG) ** 2).sum(0)) for l in LAM_TEST], 0); _w /= _w.max()
    _s = 1 / np.sqrt(_w + 0.05); _s /= np.sqrt(np.mean(_s ** 2))
    _C = M3_SG ** 2 * _s[:, None] * _s[None] * np.exp(-(_vg[:, None] - _vg[None]) ** 2 / (2 * M3_ELL ** 2))
    _ev, _eb = np.linalg.eigh(_C); _k = _ev > 1e-3 * _ev.max(); M3_B = _eb[:, _k] * np.sqrt(_ev[_k])   # (nG, K)
    AK = np.stack([U.T @ (np.tile(b, _nV)[:, None] * _Rraw) / SCALE for b in M3_B.T])                     # (K, 23, NW)；行序 = Vds 主序、Vgs 次序
    M3_EA2 = (M3_A[1] ** 2 - M3_A[0] ** 2) / (2 * np.log(M3_A[1] / M3_A[0]))                               # E[a²]，a 对数均匀
    _Sx = D.gen5(400, 777, BAND)[0]; _Sx = _Sx / _Sx[:, BM].sum(1, keepdims=True)                           # Σ_band x = 1
    _Ax = np.einsum('kiw,mw->mki', AK, _Sx)
    DEVCOV = DEVCOV + M3_EA2 * np.einsum('mki,mkj->ij', _Ax, _Ax) / len(_Sx)   # 门控等闭式部件：M3 协方差按典型谱平均，同白噪声一样随 (Σx)² 缩放
    log(f'M3：g 协方差保留 {_k.sum()} 个本征分量（占 {100 * _ev[_k].sum() / _ev.sum():.2f} %）；弱帧 / 强帧幅度 {_s.max():.2f} / {_s.min():.2f}')


def m3_z(g, M, a=None):
    """M3 系数 (K, M)：每条谱一份 z ~ N(0, 1)，乘幅度 a（训练时对数均匀抽，测试 a = 1）。"""
    a = np.exp(g.uniform(*np.log(M3_A), M)) if a is None else np.full(M, a)
    return g.standard_normal((AK.shape[0], M)) * a[None]


def rerr_np(g, M):
    """每条谱各抽一份 R 乘性误差 (M, n, NW)：共模（所有行同一份）× 每行，各为 (1 + 整行增益)(1 + 平滑起伏)(1 + 逐点)。"""
    def one(m, c):
        if c <= 0:
            return np.ones((M, 1, 1))
        a = g.normal(0, c, (M, m, 1)); gl = (g.normal(0, 1, (M, m, 3)) / np.sqrt(3) * (c / 2)) @ LEG
        return (1 + a) * (1 + gl) * (1 + g.normal(0, c, (M, m, LEG.shape[1])))
    return one(1, LEV_R['cc']) * one(N_RD, LEV_R['ci'])


def err_np(g, M, sig=None):
    """一次抽样：增益 (M,)、误差电流 (n, M)，单位 A。"""
    if UDDIN:
        gg = np.exp(ER['sigma_g'] * g.standard_normal(M)); bb = ER['sigma_b'] * g.standard_normal(M)
        return gg, DARK[:, None] * bb[None] + ER['sigma_n'] * g.standard_normal((N_RD, M))
    s = np.exp(g.uniform(np.log(SIG_LO), np.log(SIG_HI), M)) if sig is None else np.full(M, sig)
    return np.ones(M), s[None] * g.standard_normal((N_RD, M))


def meas(ST, g, mode='train', nd=None, P=None):
    """ST (NW, M) 峰值归一谱 → (nd, n, M) 读数（RN 单位：理想时 = RN @ ST）。
    mode：'train' 功率随机（Cui 噪声档也随机；--rerr 时 R 乘标定误差）；'ideal' 无误差；'real' Uddin 真实模式；
    'realR' 真实模式 + R 标定误差；'recon' / 'reading' Cui 两档；'m3' Cui recon 档白噪声 + M3 帧增益（a = 1）。
    --m3 时 'train' 也乘 M3 帧增益（a 随机）。功率除 train 外取 P（默认 P0）。"""
    nd = nd or LD.NDRAW; y0 = RN @ ST; Ss = ST[BM].sum(0); M = ST.shape[1]; out = []
    if WANG:                                                # 'train'：训练器件、功率随机；'device'：测试器件、功率 P（默认 P0）
        if mode == 'ideal':
            return np.repeat(y0[None], nd, 0)
        assert mode in ('train', 'device'), mode
        tg = torch.Generator(device=TDEV).manual_seed(int(g.integers(2 ** 62)))
        St = torch.tensor(np.tile(ST.T, (nd, 1)), dtype=torch.float32, device=TDEV)
        Pw = (P0 * torch.exp((torch.rand(len(St), device=TDEV, generator=tg) * 2 - 1) * np.log(2))) if mode == 'train' \
            else torch.full((len(St),), P or P0, device=TDEV)
        Y = dev_meas(St, Pw, 'train' if mode == 'train' else 'test', tg)
        return Y.double().cpu().numpy().reshape(nd, M, N_RD).transpose(0, 2, 1)
    m3 = mode == 'm3' or (mode == 'train' and M3)
    if m3:
        AX = np.einsum('kiw,wm->kim', AK, ST)
    for _ in range(nd):
        if mode == 'ideal':
            out.append(y0.copy()); continue
        Pm = P0 * np.exp(g.uniform(np.log(0.5), np.log(2), M)) if mode == 'train' else np.full(M, P or P0)
        gg, e = err_np(g, M, None if mode in ('train', 'real', 'realR') else FW.REAL['recon' if mode == 'm3' else mode])
        y = np.einsum('mnw,nw,wm->nm', rerr_np(g, M), RN, ST) if (mode == 'realR' or (mode == 'train' and RERR)) else y0
        if m3:
            y = y + np.einsum('kim,km->im', AX, m3_z(g, M, 1.0 if mode == 'm3' else None))
        out.append(gg[None] * y + e * (Ss / (SCALE * Pm))[None])
    return np.array(out)


_TC = {}


def meas_t(torch, Rn, St, g, dev):
    """替换 41 号 algo8.meas_t（A8 零件网络、全族网络训练时每步调用）：同 meas(mode='train')，在 GPU 上按样本抽误差。"""
    if WANG:
        assert Rn.shape == RN.shape
        B = St.shape[0]; Pw = P0 * torch.exp((torch.rand(B, device=dev) * 2 - 1) * np.log(2))
        return dev_meas(St.float(), Pw, 'train')
    key = (id(Rn), dev)
    if key not in _TC:
        _TC[key] = (torch.tensor(Rn, dtype=torch.float32, device=dev), torch.tensor(BM, device=dev),
                    torch.tensor(DARK if UDDIN else np.zeros(N_RD), dtype=torch.float32, device=dev))
    Rt, bt, dk = _TC[key]; B = St.shape[0]
    if RERR:                                                # 每个样本各抽一份 R 标定误差（同 rerr_np）
        Lg = torch.tensor(LEG, dtype=torch.float32, device=dev)

        def one(m, c):
            a = c * torch.randn(B, m, 1, device=dev); gl = (torch.randn(B, m, 3, device=dev) / np.sqrt(3) * (c / 2)) @ Lg
            return (1 + a) * (1 + gl) * (1 + c * torch.randn(B, m, Lg.shape[1], device=dev))
        y0 = torch.einsum('bnw,nw,bw->bn', one(1, LEV_R['cc']) * one(N_RD, LEV_R['ci']), Rt, St)
    else:
        y0 = St @ Rt.T
    if M3:                                                  # 每个样本各抽一份 M3 帧增益（同 m3_z，a 对数均匀）
        if 'AK' not in _TC:
            _TC['AK'] = torch.tensor(AK, dtype=torch.float32, device=dev)
        a = torch.exp(np.log(M3_A[0]) + torch.rand(B, 1, device=dev) * np.log(M3_A[1] / M3_A[0]))
        y0 = y0 + torch.einsum('kiw,bw,bk->bi', _TC['AK'], St, a * torch.randn(B, AK.shape[0], device=dev))
    k =St[:, bt].sum(1, keepdim=True) / (SCALE * P0 * torch.exp((torch.rand(B, 1, device=dev) * 2 - 1) * np.log(2)))
    if UDDIN:
        gg = torch.exp(ER['sigma_g'] * torch.randn(B, 1, device=dev))
        e = ER['sigma_b'] * torch.randn(B, 1, device=dev) * dk[None] + ER['sigma_n'] * torch.randn(B, N_RD, device=dev)
    else:
        gg = 1.0
        e = torch.exp(np.log(SIG_LO) + torch.rand(B, 1, device=dev) * np.log(SIG_HI / SIG_LO)) * torch.randn(B, N_RD, device=dev)
    return gg * y0 + k * e


H.meas_t = meas_t
F.READ = 1e-3; F.C_TRUE = 0.02 if RERR else 1e-3; P18.LEV.update(cc=0.0, ci=0.0)   # 不加 --rerr：标定误差项缩到 0.1 %（假设 R 准）；加了：恢复 41 号的 0.02
A.SIG_E[:0] = [x for x in (0.001, 0.002, 0.005) if x not in A.SIG_E]
_cal_cov0 = F.GaussCal.cal_cov


def _cal_cov(self, S2):
    """原协方差 + 器件误差项 (Σ_band x)² · DEVCOV（S2 是光谱二阶矩，Σ_band x 的平方 = S2 在波段内的元素和）。"""
    t = self.t
    if not hasattr(self, '_devc'):
        self._devc = t.tensor(DEVCOV, dtype=t.float64, device=self.dev); self._bm = t.tensor(BM, device=self.dev)
    s = S2[..., self._bm, :][..., self._bm].sum((-2, -1))
    return _cal_cov0(self, S2) + s[..., None, None] * self._devc


F.GaussCal.cal_cov = _cal_cov


# ------------------------------------------------------------------ 谱
def cie_jitter(S0, n, rng):
    """CIE LED 实谱随机平移 ±(4–20) nm、乘 ±20 % 斜坡，波段外置零，峰值归一。"""
    xc = (WL - WL[BM].mean()) / (np.ptp(WL[BM]) / 2); out = []
    for _ in range(n):
        s = S0[rng.integers(len(S0))]; sh = rng.uniform(4, 20) * rng.choice([-1, 1])
        x = np.interp(WL - sh, WL, s, left=0, right=0) * (1 + rng.uniform(-0.2, 0.2) * xc)
        x = np.where(BM, np.clip(x, 0, None), 0); out.append(x / x.max())
    return np.array(out)


def all_fams(seed, n, P, jitter):
    S = Q.fams(BAND, seed, n, P)
    if jitter:
        S[P17.REAL] = cie_jitter(S[P17.REAL], n, np.random.default_rng(seed + 333))
    return S


def to_grid(w, g, lo=None, hi=None):
    """细网格谱按 4 nm 区间平均到 WL（同 44 号），波段外置零，峰值归一。"""
    h = (WL[1] - WL[0]) / 2; f = np.arange(WL[0] - h, WL[-1] + h, 0.02)
    gf = np.interp(f, w, np.clip(g, 0, None), left=0, right=0)
    out = np.array([gf[(f >= l - h) & (f < l + h)].mean() for l in WL]); out[~BM] = 0
    return out / max(out.max(), 1e-30)


def peak_grid(x):
    xb = np.where(BM, np.clip(x, 0, None), 0); i = int(np.argmax(xb))
    if 0 < i < len(x) - 1 and BM[i - 1] and BM[i + 1]:
        y0, y1, y2 = xb[i - 1:i + 2]; den = y0 - 2 * y1 + y2
        return float(WL[i] + (0.5 * (y0 - y2) / den if den != 0 else 0) * (WL[1] - WL[0]))
    return float(WL[i])


def fwhm_grid(x):
    """线性插值找半高点。"""
    w, y = WL[BM], np.clip(x[BM], 0, None)
    if y.max() <= 0:
        return float('nan')
    y = y / y.max(); i = int(np.argmax(y)); l, r = i, i
    while l > 0 and y[l] >= 0.5: l -= 1
    while r < len(y) - 1 and y[r] >= 0.5: r += 1
    xl = w[l] + (0.5 - y[l]) / (y[l + 1] - y[l]) * (w[l + 1] - w[l]) if y[l] < 0.5 else w[l]
    xr = w[r - 1] + (y[r - 1] - 0.5) / (y[r - 1] - y[r]) * (w[r] - w[r - 1]) if y[r] < 0.5 else w[r]
    return float(xr - xl)


def nrmse1(x, s):
    return float(LD.band_nrmse(x[:, None], s[:, None])[0])


# ------------------------------------------------------------------ Wang：共用测试读数（各脚本读同一份，保证只有算法在变）
def _tsfile(nm):
    return os.path.join(DEVDIR, 'results', nm + ('_quick' if ARG.quick else '') + '.npz')


def test_readings(sets, cache=True):
    """Wang 生成谱族的 'device' 档测试读数 {族: (nd, n, NTE)}；第一次生成后存盘，之后核对谱一致再读。"""
    f = _tsfile('testset_device')
    if cache and os.path.exists(f):
        z = np.load(f); out = {}
        for j, (nm, S) in enumerate(sets.items()):
            assert str(z[f'N{j}']) == nm and np.array_equal(z[f'S{j}'], S), f'测试谱和 {f} 里的不一致：{nm}'
            out[nm] = z[f'Y{j}']
        return out
    g = np.random.default_rng(SEED_SIM + 1); out = {nm: meas(S.T, g, 'device') for nm, S in sets.items()}
    if cache:
        np.savez(f, **{f'N{j}': np.array(nm) for j, nm in enumerate(sets)}, **{f'S{j}': S for j, S in enumerate(sets.values())},
                 **{f'Y{j}': out[nm] for j, nm in enumerate(sets)})
    return out


def _peak_fine(w, g):
    i = int(np.argmax(g))
    if 0 < i < len(g) - 1:
        y0, y1, y2 = g[i - 1:i + 2]; den = y0 - 2 * y1 + y2
        return float(w[i] + (0.5 * (y0 - y2) / den if den != 0 else 0) * (w[1] - w[0]))
    return float(w[i])


def fig3_set():
    """Wang 论文 Fig 3 的 25 条测试光：47 号 forward 完整正向模型（细网格积分、real=True），第 r 次用第 r 只测试器件，各测 1 次。
    返回 dict(Y (NSIM, n, 25) 读数 / SCALE, truth (25, NW) 4 nm 网格峰值归一真值, paper (25, NW) 论文 NSF 用实测读数的重建,
              pk (25,) 真值细网格峰位, group, name, grid_dev (25,) 理想读数下 4 nm 网格模型和细网格积分的相对差)。"""
    f = _tsfile('testset_fig3')
    if os.path.exists(f):
        z = np.load(f); return {k: (list(z[k]) if k in ('group', 'name') else z[k]) for k in z.files}
    K = len(DV.test_group); rng = np.random.default_rng(SEED_SIM + 52); u0 = WANG_UNITS['test'][0]
    Y = np.array([[FW.Device(unit=u0 + r).measure(*DV.test_spectrum(k), warn=False, real=True, rng=rng) for k in range(K)] for r in range(NSIM)])
    truth = np.array([to_grid(DV.test_w[k], DV.test_truth[k]) for k in range(K)])
    I0 = np.array([DV.measure(*DV.test_spectrum(k), warn=False) for k in range(K)]) / SCALE; Yg = truth @ RN.T
    a = (Yg * I0).sum(1) / (Yg * Yg).sum(1); gd = np.linalg.norm(a[:, None] * Yg - I0, axis=1) / np.linalg.norm(I0, axis=1)
    out = dict(Y=Y.transpose(0, 2, 1) / SCALE, truth=truth, paper=np.array([to_grid(DV.test_w[k], DV.test_paper[k]) for k in range(K)]),
               pk=np.array([_peak_fine(DV.test_w[k], DV.test_truth[k]) for k in range(K)]), group=np.array(DV.test_group), name=np.array(DV.test_name), grid_dev=gd)
    np.savez(f, **out); out['group'], out['name'] = list(out['group']), list(out['name'])
    return out


F3_GROUPS = ['LED', '双峰', '窄线', '逐 nm 线']
F3_PKG = ('窄线', '逐 nm 线')                                                    # 单峰组：另算峰位误差


def fig3_eval(X, F3):
    """X (NSIM, NW, 25) 某方法在 Fig 3 读数上的重建 → dict(err (NSIM, 25), dpk (NSIM, 25)：峰位 − 真值峰位，只单峰组有值)。"""
    ns, _, K = X.shape
    err = np.array([[nrmse1(X[r, :, k], F3['truth'][k]) for k in range(K)] for r in range(ns)])
    dpk = np.full((ns, K), np.nan)
    for k in range(K):
        if F3['group'][k] in F3_PKG:
            dpk[:, k] = [peak_grid(X[r, :, k]) - F3['pk'][k] for r in range(ns)]
    return dict(err=err, dpk=dpk)


def fig3_summary(R, F3):
    """R = fig3_eval 的结果 → {组: 平均误差}，另加 '_4组平均'、'_峰位中位'（单峰组 |峰位误差| 中位 nm）、'_峰位RMS'。"""
    g = np.array(F3['group']); s = {G: float(R['err'][:, g == G].mean()) for G in F3_GROUPS}
    s['_4组平均'] = float(np.mean([s[G] for G in F3_GROUPS])); d = R['dpk'][:, np.isin(g, F3_PKG)]
    s['_峰位中位'] = float(np.nanmedian(np.abs(d))); s['_峰位RMS'] = float(np.sqrt(np.nanmean(d ** 2)))
    return s


# ------------------------------------------------------------------ A11（照 44 号 build，读数换成器件模拟器）
def build():
    import torch
    t0 = time.time(); Rn = RN
    sv, _, se = P18.SEEDS[0]; g = np.random.default_rng(se)
    Sva = D.gen5(60, sv, BAND)[0]; Yva = meas(Sva.T, g)
    P = H.Parts(Rn, BAND)
    lib = all_fams(P17.TRSEED, NTR, P, True)
    vsets = all_fams(P18.VSEED, NVF, P, True); Yv = {k: meas(S.T, g) for k, S in vsets.items()}
    nd = LD.NDRAW; E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
    lam = min(LD.NN_LAMS, key=lambda l: E(A.dec_snn(Rn, Yva, l), Sva).mean())
    GM0 = D.Gauss5(Rn, BAND); sig = min(A.SIG_E, key=lambda s: E(GM0.solve(Yva, s)[0], Sva).mean())
    GM = S7.GaussMR(Rn, BAND, F.C_TRUE); PM = D.ParamE(Rn, BAND)
    famg = [k for k in lib if k not in P17.PEAKY and k != '同分布五类']
    GF = Q.GaussX(Rn, BAND, {k: lib[k] for k in famg}, {k: P17.GP_EACH for k in famg})
    ipk = [GF.classes.index(c) for c in D.PARAM_CLS]
    Slib = np.concatenate([lib[k] for k in lib] + [D.gen5(len(lib['同分布五类']) * 3, P17.TRSEED + 7, BAND)[0]])
    log(f'λ（A1，生成验证集挑）= {lam:g}；σ_e = {sig}；训练库 {len(lib)} 族、{len(Slib)} 条（全部谱族，CIE LED 用平移版）')
    f8n, fmn = os.path.join(OUT, 'net8.pt'), os.path.join(OUT, 'mlp.pt')
    if os.path.exists(f8n):
        NT8 = H.train_net(P, Rn, steps=1); NT8['net'].load_state_dict(torch.load(f8n)); c = torch.load(f8n + '.musd'); NT8['mu'], NT8['sd'] = c['mu'], c['sd']
        log('  零件网络：读入已训练的 net8.pt')
    else:
        NT8 = H.train_net(P, Rn, steps=STEPS8); torch.save(NT8['net'].state_dict(), f8n); torch.save(dict(mu=NT8['mu'], sd=NT8['sd']), f8n + '.musd')
    if os.path.exists(fmn):
        NN = P17.train_mlp(Rn, Slib[:10], BAND, steps=1); NN['net'].load_state_dict(torch.load(fmn)); log('  全族网络：读入已训练的 mlp.pt')
    else:
        NN = P17.train_mlp(Rn, Slib, BAND, steps=STEPS, log=log); torch.save(NN['net'].state_dict(), fmn)
    log(f'网络就绪 {time.time() - t0:.0f} s')

    def base(Y):
        Xp = PM.solve(Y, sig)[0]; X8 = H.solve_pick(P, Y, H.infer(NT8, Y))[0]
        Xf, dPf, Wf = GF.solve_u(Y); gp = DS.fuse_parts(Rn, Xf, Wf, dPf, [Xp, X8], Y, GF, GF.classes)
        return dict(A1=np.clip(A.dec_snn(Rn, Y, lam), 0, None), gate=np.clip(DS.gate_mean(gp), 0, None), Wf=Wf,
                    net=np.clip(P17.infer_mlp(NN, Rn, Y), 0, None))

    CS = P18.Consist(Rn, BM, GM); vo = {k: base(Y) for k, Y in Yv.items()}
    Sv = {k: np.tile(S, (nd, 1)) for k, S in vsets.items()}
    Xn_all = np.concatenate([P18.flat(vo[k]['net']) for k in vsets]); St_all = np.concatenate([Sv[k] for k in vsets])
    CS.fit_cov(Xn_all, St_all); Yf_all = np.concatenate([P18.flat(Yv[k]) for k in vsets])
    ev_k = {kp: float(LD.band_nrmse(CS.apply(Xn_all, Yf_all, kp).T, St_all.T).mean()) for kp in P18.KAPPAS}; kap = min(ev_k, key=ev_k.get)
    Gv, Xcv, Xgv = [], [], []
    for k in vsets:
        o = vo[k]; o['cons'] = P18.unflat(CS.apply(P18.flat(o['net']), P18.flat(Yv[k]), kap), nd)
        f8, lw = P18.feats(Rn, Yv[k], o['cons'], o['gate'], o['Wf'], GM, ipk)
        Gv.append(np.c_[f8, lw]); Xcv.append(P18.flat(o['cons'])); Xgv.append(P18.flat(o['gate']))
    BN = P18.fit_blend_net(np.concatenate(Gv), np.concatenate(Xcv), np.concatenate(Xgv), St_all, BM)
    log(f'κ = {kap}；融合小网训练完；建模共 {time.time() - t0:.0f} s')

    def scale_y(Y):
        """亮度未知时：全族网络估亮度（读数最小二乘回填后的峰值），读数除以它。"""
        a = P17.infer_mlp(NN, Rn, Y)[:, BM, :].max(1)
        return Y / np.maximum(a, 1e-30)[:, None, :]

    def decode(Y):
        Y = scale_y(Y); o = base(Y); nd_ = Y.shape[0]
        o['cons'] = P18.unflat(CS.apply(P18.flat(o['net']), P18.flat(Y), kap), nd_)
        f8, lw = P18.feats(Rn, Y, o['cons'], o['gate'], o['Wf'], GM, ipk); w2 = P18.blend_w_net(np.c_[f8, lw], *BN)
        o['A11'] = P18.unflat(w2[:, None] * P18.flat(o['cons']) + (1 - w2[:, None]) * P18.flat(o['gate']), nd_)
        o.pop('Wf'); o['w'] = w2.reshape(nd_, -1)
        return o

    gp = np.random.default_rng(11); pool = np.concatenate([P.gen(1024 * 8, gp)[0] for _ in range(8)])   # = algo8.train_net 的训练池
    used = {'调参验证集': Sva, '融合验证集': np.concatenate(list(vsets.values())), '训练库': Slib, '零件网络训练池': pool,
            '高斯先验用谱': D.gen5(A.NTRAIN, 11, BAND)[0]}
    return decode, P, used


# ------------------------------------------------------------------ 论文算法（只 Uddin：作者 MATLAB 代码的 Python 版）
if UDDIN:
    REC = PC.Recon(DV.R, DV.lam, PC.FWHM['vis'])


def paper_decode(Y):
    """Y (nd, n, M) → (nd, NW, M)，重建谱插到 WL 网格。"""
    X = np.zeros((Y.shape[0], len(WL), Y.shape[2]))
    for r in range(Y.shape[0]):
        for m in range(Y.shape[2]):
            sp = REC(Y[r, :, m]); X[r, BM, m] = np.interp(WL[BM], REC.lam_f, sp, left=0, right=0)
    return X


# ------------------------------------------------------------------ 主程序
def main():
    t0 = time.time()
    s = np.linalg.svd(RN[:, BM], compute_uv=False); s = s / s[0]
    log(f'器件 {ARG.device}：读数 {N_RD} 个' + ('' if UDDIN else '（0 到 −10 V 每 0.1 V 一个偏压）' if WANG else f'（4141 个原始读数投影到 R 的 {N_RD} 个左奇异向量）')
        + f'，波段 {BAND}，网格 {WL[1] - WL[0]:.0f} nm、{BM.sum()} 点；参考功率 {P0 * 1e6:.3g} µW；σ_i/σ_1 前 6 个 {np.round(s[:6], 4).tolist()}')
    decode, P, used = build()
    sets = all_fams(TEST_ST, NTE, P, False)
    sets = {k.replace('留出·', ''): v for k, v in sets.items()}                 # 41 号的旧名；这里全部谱族都训练过，输出里去掉"留出"
    leak = P18.overlap(sets, used); log('测试谱与训练库 / 验证集 / 先验用谱 / 零件训练池逐条比对，完全相同的条数：' + ('无' if not leak else str(leak)))
    assert not leak, leak
    J =dict(device=ARG.device, band=BAND, n_read=N_RD, P0=P0, R_sv=s[:8].tolist(), test_seed=TEST_ST, fam={}, real={}, sim={})

    # a. 生成谱族
    modes = (['ideal', 'real'] + (['realR'] if RERR else [])) if UDDIN else ['ideal', 'device'] if WANG else ['ideal', 'recon', 'reading'] + (['m3'] if M3 else [])
    MN = {'ideal': '理想读数（只作诊断）', 'real': '真实模式', 'realR': '真实模式 + R 标定误差（2 % 共模 + 2 % 每行）',
          'recon': '真实模式 recon 档（55 dB）', 'reading': '真实模式 reading 档（41 dB）', 'm3': 'M3：recon 档白噪声 + 逐 Vgs 帧增益漂移（a = 1）',
          'device': '真实器件模式（47 号全部误差项，测试器件，功率 1 µW）'}
    nd = LD.NDRAW; E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
    TR = test_readings(sets) if WANG else None                                  # Wang：共用测试读数（存盘，A12 / A15 / 其他算法读同一份）
    demo = {}                                                                   # 每族第 1 条谱、第 1 次抽样的重建，画示例图
    for mo in modes:
        g = np.random.default_rng(SEED_SIM + modes.index(mo)); J['fam'][mo] = {}; demo[mo] = {}
        for nm, S in sets.items():
            Y = TR[nm] if mo == 'device' else meas(S.T, g, mo); o = decode(Y)
            if UDDIN:
                o['paper'] = paper_decode(Y)
            J['fam'][mo][nm] = {m: float(E(o[m], S).mean()) for m in METH}
            demo[mo][nm] = dict(truth=S[0], **{m: o[m][0, :, 0] for m in METH})
        log(f'\n== 生成谱族 · {MN[mo]}（{len(sets)} 族 × {NTE} 条 × {nd} 次抽样，误差 = 峰值归一均方根 / 真值极差）')
        log('   族                    ' + ''.join(f'{NAME[m]:>12s}' for m in METH))
        for nm in sets:
            log(f'   {nm:<20s}' + ''.join(f'{J["fam"][mo][nm][m]:12.3f}' for m in METH))
        med = {m: float(np.median([J['fam'][mo][k][m] for k in sets])) for m in METH}; J['fam'][mo]['_中位'] = med
        log('   各族中位              ' + ''.join(f'{med[m]:12.3f}' for m in METH))

    # b. 真实读数 vs 同一批测试光的模拟读数
    ex = dict(WL=WL, BM=BM, lam_test=LAM_TEST, demo=demo)
    if WANG:                                                                    # 没有实测读数：论文 Fig 3 的 25 条测试光过完整正向模型
        F3 = fig3_set(); o = decode(F3['Y']); K = len(F3['group'])
        gd = F3['grid_dev']; J['fig3_grid_dev'] = gd.tolist()
        log(f'\n   4 nm 网格模型 vs 47 号细网格完整正向（理想读数、增益对齐后相对差，25 条）：中位 {np.median(gd):.4f}，最大 {gd.max():.4f}')
        log(f'\n== 论文 Fig 3 的 25 条测试光（47 号 forward 完整正向、真实器件模式，{NSIM} 只测试器件各测 1 次）')
        log('   误差 = 4 nm 网格上峰值归一均方根 / 真值极差，组内平均；峰位误差只算单峰组（窄线、逐 nm 线），真值峰位取细网格抛物线顶点')
        RR = {m: fig3_eval(o[m], F3) for m in METH}; RR['论文 NSF（实测读数）'] = fig3_eval(F3['paper'].T[None], F3)
        log('   方法                      ' + ''.join(f'{G:>10s}' for G in F3_GROUPS) + '     4组平均  |峰位|中位  峰位RMS')
        J['fig3'] = {}
        for m, R in RR.items():
            s = fig3_summary(R, F3); J['fig3'][m] = dict(summary=s, err_mean=R['err'].mean(0).tolist(), dpk_mean=np.nanmean(R['dpk'], 0).tolist())
            log(f'   {NAME.get(m, m):<22s}' + ''.join(f'{s[G]:10.3f}' for G in F3_GROUPS) + f'{s["_4组平均"]:12.3f}{s["_峰位中位"]:10.2f}{s["_峰位RMS"]:9.2f}')
        log('   （论文 NSF 那一行用的是作者实测读数，读数来源和上面不同，只作量级参考）')
        log('   A11 融合权重（1 = 网络支，0 = 门控支）各组平均：' + ' '.join(f'{G} {o["w"][:, np.array(F3["group"]) == G].mean():.2f}' for G in F3_GROUPS))
        ex['fig3'] = dict(truth=F3['truth'], paper=F3['paper'], name=F3['name'], group=F3['group'], **{m: o[m][0] for m in METH})
        log(f'\n用时 {time.time() - t0:.0f} s')
        json.dump(J, open(os.path.join(OUT, 'a11.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))
        pickle.dump(ex, open(os.path.join(OUT, 'examples.pkl'), 'wb'))
        figs(ex, J, None)
        return
    Yr = (Y_REAL / SCALE)[None]; o_real = decode(Yr)
    if UDDIN:
        o_real['paper'] = paper_decode(Yr)
        import openpyxl
        fig = PC.read_fig(openpyxl.load_workbook(FW.XLSX, read_only=True, data_only=True), 'vis')
        refs = []
        for j, lt in enumerate(LAM_TEST):
            rl, rr, fl, ff = fig[float(lt)]; refs.append((fl, ff, rl, rr))
        truth = np.array([to_grid(fl, ff) for fl, ff, _, _ in refs]); ref_pk = np.array([PC.peak(fl, ff) for fl, ff, _, _ in refs])
        ref_fw = np.array([PC.fwhm_of(fl, ff) for fl, ff, _, _ in refs])
        pap = dict(err=[nrmse1(to_grid(rl, rr), truth[j]) for j, (_, _, rl, rr) in enumerate(refs)],
                   dpk=[PC.peak(rl, rr) - ref_pk[j] for j, (_, _, rl, rr) in enumerate(refs)])
        J['real']['作者论文里的重建（Fig3c）'] = pap
        ex.update(truth=truth, ref=refs)
    else:
        ref_pk = LAM_TEST.astype(float); truth = None
    real_m = {}
    for m in METH:
        X = o_real[m][0]
        real_m[m] = dict(dpk=[peak_grid(X[:, j]) - ref_pk[j] for j in range(len(LAM_TEST))], fwhm=[fwhm_grid(X[:, j]) for j in range(len(LAM_TEST))])
        if truth is not None:
            real_m[m]['err'] = [nrmse1(X[:, j], truth[j]) for j in range(len(LAM_TEST))]
    J['real'].update(real_m); ex['real'] = {m: o_real[m][0] for m in METH}; ex['w_real'] = o_real['w'][0]

    sim_modes = (['real'] + (['realR'] if RERR else [])) if UDDIN else ['recon', 'reading'] + (['m3'] if M3 else []); sim_m = {}
    for mo in sim_modes + ['ideal']:
        rng = np.random.default_rng(SEED_SIM + 50 + len(sim_m)); sim_m[mo] = {m: dict(dpk=[], err=[], fwhm=[]) for m in METH}
        for j, lt in enumerate(LAM_TEST):
            nn = 1 if mo == 'ideal' else NSIM
            if mo == 'realR':                                                   # 参考谱（4 nm 网格）× 带标定误差的 R，功率取该束拟合值
                Yb = meas(np.tile(truth[j][:, None], (1, nn)), rng, 'realR', nd=1, P=P_FIT[j]).transpose(2, 1, 0)
            elif UDDIN:
                fl, ff = refs[j][0], refs[j][1]; p = np.clip(ff, 0, None); p = p / np.trapezoid(p, fl) * P_FIT[j]
                Yb = np.array([DV.measure(fl, p, dark=False, warn=False, real=mo != 'ideal', rng=rng) for _ in range(nn)])[:, :, None] / SCALE
            elif mo == 'm3':                                                    # 原始 4141 个读数上乘帧增益（a = 1），加 recon 档白噪声，再投影
                I0 = DV.measure([lt], [P0], lines=True, warn=False).ravel()
                Yb = np.array([U.T @ (np.tile(1 + M3_B @ rng.standard_normal(M3_B.shape[1]), _nV) * I0 + FW.REAL['recon'] * rng.standard_normal(I0.size))
                               for _ in range(nn)])[:, :, None] / SCALE
            else:
                Yb = np.array([U.T @ DV.measure([lt], [P0], lines=True, warn=False, real=mo != 'ideal', preset=mo if mo != 'ideal' else 'recon', rng=rng).ravel()
                               for _ in range(nn)])[:, :, None] / SCALE
            o = decode(Yb)
            if UDDIN:
                o['paper'] = paper_decode(Yb)
            for m in METH:
                X = o[m][:, :, 0]
                sim_m[mo][m]['dpk'].append([peak_grid(x) - ref_pk[j] for x in X]); sim_m[mo][m]['fwhm'].append([fwhm_grid(x) for x in X])
                if truth is not None:
                    sim_m[mo][m]['err'].append([nrmse1(x, truth[j]) for x in X])
    J['sim'] = sim_m

    log(f'\n== 作者的真实读数（{len(LAM_TEST)} 束）对比模拟器同一批测试光（真实模式各 {NSIM} 次；理想 1 次）')
    if UDDIN:
        log('   参考 = 商用光谱仪参考谱（Fig3c 右半）；峰位误差 = 重建峰位 − 参考峰位；误差 = 4 nm 网格上峰值归一均方根 / 极差')
        log('   作者论文里的重建（Fig3c 左半）：误差 ' + ' '.join(f'{v:.3f}' for v in pap['err']) + '；峰位误差 ' + ' '.join(f'{v:+.2f}' for v in pap['dpk'])
            + f'；平均 |峰位误差| {np.mean(np.abs(pap["dpk"])):.2f} nm')
        log(f'   参考谱 FWHM：' + ' '.join(f'{v:.1f}' for v in ref_fw) + ' nm')
    else:
        log('   参考 = 标称波长（作者没给参考谱）；峰位误差 = 重建峰位 − 标称波长；论文算法在实测读数上 RMSD 7.86 nm（46 号 paper_check.json）')
    for m in METH:
        r = real_m[m]; dp = np.array(r['dpk'])
        line = f'   {NAME[m]:<10s} 真实读数：峰位误差 ' + ' '.join(f'{v:+.2f}' for v in dp) + f'（平均 |·| {np.mean(np.abs(dp)):.2f}，RMSD {np.sqrt(np.mean(dp ** 2)):.2f} nm）'
        if truth is not None:
            line += '；误差 ' + ' '.join(f'{v:.3f}' for v in r['err']) + f'（平均 {np.mean(r["err"]):.3f}）'
        line += '；FWHM ' + ' '.join(f'{v:.1f}' for v in r['fwhm'])
        log(line)
        for mo in sim_modes + ['ideal']:
            sm = sim_m[mo][m]; dps = np.array(sm['dpk'])                         # (束, 次)
            line = f'   {"":<10s} 模拟·{mo:<7s}：每束平均 |峰位误差| ' + ' '.join(f'{v:.2f}' for v in np.abs(dps).mean(1)) \
                + f'（总平均 {np.abs(dps).mean():.2f}，RMSD {np.sqrt((dps ** 2).mean()):.2f} nm）'
            if truth is not None:
                line += '；误差 ' + ' '.join(f'{v:.3f}' for v in np.mean(sm['err'], 1)) + f'（平均 {np.mean(sm["err"]):.3f}）'
            log(line)
    log('   真实读数上 A11 的融合权重（1 = 全用网络支，0 = 全用门控支）：' + ' '.join(f'{v:.2f}' for v in o_real['w'][0]))

    if not UDDIN:                                                                     # 真实残差有多少落在投影保留的 23 维里
        fr = []
        for j, lt in enumerate(LAM_TEST):
            Im = DV.I_test[j].ravel(); Mo = DV.measure([lt], [P0], lines=True, warn=False).ravel(); a = Mo @ Im / (Mo @ Mo); r = Im - a * Mo
            fr.append(float(np.sum((U.T @ r) ** 2) / np.sum(r ** 2)))
        J['resid_frac_in_span'] = fr
        log(f'\n   "实测 − 模型"残差落在保留的 {N_RD} 维里的能量比例：' + ' '.join(f'{v:.2f}' for v in fr)
            + f'（白噪声应为 {N_RD}/4141 = {N_RD / 4141:.4f}）')
    log(f'\n用时 {time.time() - t0:.0f} s')
    json.dump(J, open(os.path.join(OUT, 'a11.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))
    pickle.dump(ex, open(os.path.join(OUT, 'examples.pkl'), 'wb'))
    figs(ex, J, ref_pk)


def figs(ex, J, ref_pk):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    w = WL[BM]
    if WANG:                                                                    # Fig 3 的 25 条：第 1 只测试器件的重建
        f3 = ex['fig3']; K = len(f3['name']); nc = 5; nr = -(-K // nc)
        fg, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.1 * nr), squeeze=False)
        for k in range(K):
            ax = axs[k // nc, k % nc]; tr = f3['truth'][k]
            ax.plot(w, tr[BM] / tr.max(), 'k-', lw=1.6, label='真值'); ax.plot(w, f3['paper'][k][BM] / max(f3['paper'][k].max(), 1e-30), color='0.55', ls='--', lw=0.9, label='论文 NSF（实测读数）')
            for m, c, lw in (('A1', 'C2', 0.8), ('A11', 'C3', 1.2)):
                x = f3[m][BM, k]; ax.plot(w, x / max(x.max(), 1e-30), color=c, lw=lw, label=NAME[m])
            ax.set_title(f"{f3['group'][k]}：{f3['name'][k]}\nA11 {nrmse1(f3['A11'][:, k], tr):.3f}", fontsize=7); ax.set_xlim(BAND); ax.tick_params(labelsize=6)
        for k in range(K, nr * nc):
            axs[k // nc, k % nc].axis('off')
        axs[0, 0].legend(fontsize=6); fg.suptitle('wang：论文 Fig 3 测试光，真实器件模式读数（测试器件 0）', fontsize=10)
        fg.tight_layout(); fg.savefig(os.path.join(OUT, 'fig_fig3.png'), dpi=130); plt.close(fg)
    n = 0 if WANG else len(LAM_TEST)
    fg, axs = plt.subplots(1, max(n, 1), figsize=(3.2 * max(n, 1), 3.2), squeeze=False)
    for j in range(n):
        ax = axs[0, j]
        if UDDIN:
            fl, ff, rl, rr = ex['ref'][j]
            ax.plot(fl, ff / ff.max(), 'k-', lw=1.5, label='参考谱'); ax.plot(rl, rr / rr.max(), color='0.5', ls='--', lw=1, label='作者论文的重建')
        else:
            ax.axvline(LAM_TEST[j], color='k', lw=1, label='标称波长')
        for m, c in (('A11', 'C3'), ('A1', 'C2')) + ((('paper', 'C0'),) if UDDIN else ()):
            x = ex['real'][m][BM, j]; ax.plot(w, x / max(x.max(), 1e-30), color=c, lw=1.2, label=f'{NAME[m]}（实测读数）')
        ax.set_xlim(BAND); ax.set_title(f'{LAM_TEST[j]:.0f} nm'); ax.set_xlabel('波长 (nm)')
    if not WANG:
        axs[0, 0].legend(fontsize=7); fg.suptitle(f'{ARG.device}：作者实测读数上的重建（A11 用器件模拟器训练）', fontsize=10)
        fg.tight_layout(); fg.savefig(os.path.join(OUT, 'fig_real.png'), dpi=150)
    plt.close(fg)

    mo = 'real' if UDDIN else 'device' if WANG else 'recon'; fam = [k for k in J['fam'][mo] if not k.startswith('_')]
    fg, ax = plt.subplots(figsize=(7.5, 0.28 * len(fam) + 1.2)); yy = np.arange(len(fam))
    for i, (m, c) in enumerate(zip(METH, ('C2', 'C1', 'C4', 'C3', 'C0'))):
        ax.plot([J['fam'][mo][k][m] for k in fam], yy, 'o', color=c, ms=4, label=NAME[m])
    ax.set_yticks(yy); ax.set_yticklabels(fam, fontsize=7); ax.invert_yaxis(); ax.set_xlabel('误差（峰值归一均方根 / 极差）'); ax.legend(fontsize=7)
    ax.set_title(f'{ARG.device}：生成谱族，{mo} 读数', fontsize=10); fg.tight_layout(); fg.savefig(os.path.join(OUT, 'fig_families.png'), dpi=150); plt.close(fg)

    for mo, dm in ex['demo'].items():                                           # 各族一条重建示例（测试集第 1 条、第 1 次抽样，不挑）
        fam = list(dm); nc = 5; nr = -(-len(fam) // nc)
        fg, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.1 * nr), squeeze=False)
        for i, nm in enumerate(fam):
            ax = axs[i // nc, i % nc]; d = dm[nm]; tr = d['truth']
            ax.plot(w, tr[BM], 'k-', lw=1.6, label='真值')
            for m, c, lw in (('A1', 'C2', 0.8), ('paper', 'C0', 1.0), ('A11', 'C3', 1.2)):
                if m in d:
                    ax.plot(w, d[m][BM], color=c, lw=lw, label=NAME[m])
            er = ' / '.join(f'{NAME[m][:4] if m != "A11" else "A11"} {nrmse1(d[m], tr):.2f}' for m in ('A11', 'paper', 'A1') if m in d)
            ax.set_title(f'{nm}\n{er}', fontsize=7); ax.set_xlim(BAND); ax.tick_params(labelsize=6)
        for i in range(len(fam), nr * nc):
            axs[i // nc, i % nc].axis('off')
        axs[0, 0].legend(fontsize=6)
        fg.suptitle(f'{ARG.device}：各族重建示例（{mo} 读数{"，训练含 R 标定误差" if RERR else ""}{"，训练含 M3 帧增益漂移" if M3 else ""}；标题 = 本条误差）', fontsize=10)
        fg.tight_layout(); fg.savefig(os.path.join(OUT, f'fig_demo_{mo}.png'), dpi=130); plt.close(fg)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
