"""通用 OPD 边界函数（无增益有机光电二极管）。

所有量用 SI 单位：d [m]、V [V]、λ [m]、μ [m²/Vs]、τ [s]、ε 为相对介电常数、
J [A/m²]、P [W/m²]、f [Hz]。V 取反偏为正，V_eff = V0 + V。
每个函数对应《42种策略_通用边界模型.md》中的一个编号公式（注释里标 Eq.x）。
参数都需要按目标体系标定；本文件不含任何体系的默认数值。
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import iv

q = 1.602176634e-19
kB = 1.380649e-23
eps0 = 8.8541878128e-12
h = 6.62607015e-34
c = 2.99792458e8


def kT(T=300.0):
    return kB * T / q  # eV


# ---------------------------------------------------------------- 光吸收 η_A
def alpha_from_k(k, lam):
    """Eq.A0  α = 4πk/λ。"""
    return 4 * np.pi * k / lam


def eta_A_incoherent(alpha, d, Rf=0.04, Rb=0.9):
    """Eq.A1  非相干多次反射：前面反射 Rf，背面反射 Rb。"""
    x = np.exp(-alpha * d)
    return (1 - Rf) * (1 - x) * (1 + Rb * x) / (1 - Rf * Rb * x**2)


def eta_A_cavity(alpha, d, lam, n, Rf, Rb, phi=0.0):
    """Eq.A2  Fabry–Pérot 腔平均近似（忽略层内驻波项），phi = φf + φb。"""
    x = np.exp(-alpha * d)
    r = np.sqrt(Rf * Rb) * x
    delta = 4 * np.pi * n * d / lam + phi
    return (1 - Rf) * (1 - x) * (1 + Rb * x) / (1 + r**2 - 2 * r * np.cos(delta))


def d_resonance(m, lam, n, phi=0.0):
    """Eq.A3  第 m 阶共振厚度。"""
    return (m - phi / (2 * np.pi)) * lam / (2 * n)


def d_resonance_fwhm(alpha, d, lam, n, Rf, Rb):
    """Eq.A4  共振的厚度半高宽（厚度容差窗口）。"""
    r = np.sqrt(Rf * Rb) * np.exp(-alpha * d)
    return lam * (1 - r) / (2 * np.pi * n * np.sqrt(r))


def d_abs_floor(alpha, A_target, Rf=0.04, Rb=0.9, dmax=20e-6):
    """Eq.A5  无腔时达到 A* 的最小厚度；达不到返回 nan。"""
    g = lambda d: eta_A_incoherent(alpha, d, Rf, Rb) - A_target
    if g(dmax) < 0:
        return np.nan
    return brentq(g, 1e-12, dmax)


def path_enhancement_limit(n, mode_ratio=1.0):
    """Eq.A6  光陷阱光程增强上限 F_max = 4n²·(M/M_bulk)。"""
    return 4 * n**2 * mode_ratio


def eta_A_trapping(alpha, d, F):
    """Eq.A7  光程增强 F 下的吸收（弱吸收近似）。"""
    return alpha * d * F / (alpha * d * F + 1)


def orientation_factor(sin2_mean):
    """Eq.A8  正入射下的取向吸收因子 (3/2)<sin²θ>，各向同性为 1，上限 1.5。"""
    return 1.5 * sin2_mean


def fresnel_R(n1, n2):
    """Eq.A9  单界面正入射反射率。"""
    return ((n1 - n2) / (n1 + n2)) ** 2


# ---------------------------------------------------------------- 激子 η_reach
def eta_slab(d, L, s=0.0):
    """Eq.X1  一维平板：x=0 为 D/A 界面（完全解离），x=d 处表面复合 s = S·L/D。
    s=0 为反射（理想激子阻挡），s→∞ 为完全猝灭。"""
    x = d / L
    sh, ch = np.sinh(x), np.cosh(x)
    return (L / d) * (sh + s * (ch - 1)) / (ch + s * sh)


def eta_sphere(R, L):
    """Eq.X2  球形相畴（半径 R），表面完全解离。"""
    x = R / L
    return (3 / x) * (1 / np.tanh(x) - 1 / x)


def sensitivity_L(d, L):
    """Eq.X3  ∂lnη/∂lnL（平板，s=0），<ε 时提高 L_D 已无收益。"""
    x = d / L
    return 1 - 2 * x / np.sinh(2 * x)


def fret_efficiency(r, R0):
    """Eq.X4  Förster 转移效率。"""
    return 1 / (1 + (r / R0) ** 6)


def marcus_rate(V_eV, lam_eV, dG_eV, T=300.0):
    """Eq.X5  Marcus 速率 [1/s]；dG<0 为放热。"""
    hbar = 1.054571817e-34
    kt = kT(T)
    pref = 2 * np.pi / hbar * (V_eV * q) ** 2 / np.sqrt(4 * np.pi * lam_eV * q * kt * q)
    return pref * np.exp(-((dG_eV + lam_eV) ** 2) / (4 * lam_eV * kt))


def eta_competition(k_good, *k_loss):
    """Eq.X6  速率竞争产率。"""
    return k_good / (k_good + sum(k_loss))


# ---------------------------------------------------------------- CT 解离 η_diss
def binding_energy(eps_r, a):
    """Eq.C1  点电荷束缚能 [eV]，a 为电荷间距 [m]。"""
    return q / (4 * np.pi * eps0 * eps_r * a)


def coulomb_radius(eps_r, T=300.0):
    """Eq.C2  束缚能等于 kT 的距离 [m]。"""
    return q / (4 * np.pi * eps0 * eps_r * kT(T))


def braun_eta(F, eps_r, a, k_f, mu_sum, T=300.0):
    """Eq.C3  Onsager–Braun 场依赖解离产率。F [V/m]，mu_sum = μe+μh。"""
    gamma = q * mu_sum / (eps0 * eps_r)  # Langevin
    Eb = binding_energy(eps_r, a)
    kt = kT(T)
    b = q**3 * np.asarray(F, float) / (8 * np.pi * eps0 * eps_r * (kB * T) ** 2)
    b = np.maximum(b, 1e-12)
    bessel = iv(1, 2 * np.sqrt(2 * b)) / np.sqrt(2 * b)
    kd = 3 * gamma / (4 * np.pi * a**3) * np.exp(-Eb / kt) * bessel
    return kd / (kd + k_f)


def ct_absorption(E, f_sigma, E_CT, lam_eV, T=300.0):
    """Eq.C4  CT 吸收（Marcus–Gauss 线形，Vandewal），返回 E·α 的相对值。"""
    kt = kT(T)
    return f_sigma / (E * np.sqrt(4 * np.pi * lam_eV * kt)) * np.exp(
        -((E_CT + lam_eV - E) ** 2) / (4 * lam_eV * kt))


# ---------------------------------------------------------------- 收集 η_coll
def hecht_uniform(d, mutau_e, mutau_h, Veff):
    """Eq.K1  均匀场、均匀产生的 Hecht 收集效率（两种载流子之和）。"""
    out = 0.0
    for mt in (mutau_e, mutau_h):
        u = mt * Veff / d**2
        out = out + u * (1 - u * (1 - np.exp(-1 / u)))
    return out


def xi_star(eta_target):
    """Eq.K2  使 η_coll=η* 的 ξ* = d²/(μτV_eff)（μτ_e=μτ_h）。0.9→0.325。"""
    g = lambda u: 2 * u * (1 - u * (1 - np.exp(-1 / u))) - eta_target
    return 1 / brentq(g, 1e-4, 1e4)


def d_collection(mutau, Veff, eta_target=0.9):
    """Eq.K3  收集线 d_c(V) = sqrt(ξ*·μτ·V_eff)。"""
    return np.sqrt(xi_star(eta_target) * mutau * Veff)


def hecht_profile(d, alpha, mutau_e, mutau_h, Veff, light_on_hole_side=True, n=400):
    """Eq.K4  指数产生分布下的收集效率（含 CCN）。
    默认光从空穴接触侧(x=0)入射，电子向 x=d 漂移。"""
    x = (np.arange(n) + 0.5) / n * d
    g = alpha * np.exp(-alpha * x)
    g = g / g.sum()
    Le = mutau_e * Veff / d
    Lh = mutau_h * Veff / d
    se, sh = (d - x, x) if light_on_hole_side else (x, d - x)
    ce = (Le / d) * (1 - np.exp(-se / Le))
    ch = (Lh / d) * (1 - np.exp(-sh / Lh))
    return float(np.sum(g * (ce + ch)))


def w_space_charge(mu_slow, G, Veff, eps_r):
    """Eq.K5  Goodman–Rose 空间电荷区宽度；d > w 进入空间电荷限制。G [1/m³s]。"""
    return (9 * eps0 * eps_r * mu_slow / (8 * q * G)) ** 0.25 * np.sqrt(Veff)


def d_space_charge(mu_slow, Veff, photon_flux, eta, eps_r):
    """Eq.K5'  由 w(G)=d 且 G=Φη/d 得到的空间电荷临界厚度
    d_SC = (9 ε μ_s V_eff² /(8 q Φ η))^(1/3)，Φ [光子/m²s]。"""
    return (9 * eps0 * eps_r * mu_slow * Veff**2 / (8 * q * photon_flux * eta)) ** (1 / 3)


def J_scl_stolterfoht(eps_r, mu_slow, Veff, d, beta_ratio=1.0, kappa=1.0):
    """Eq.K6  S4 型双分子损失起点 J ≈ κ(β_L/β)^½·ε μ_s V²/d³。"""
    return kappa * np.sqrt(beta_ratio) * eps0 * eps_r * mu_slow * Veff**2 / d**3


def theta_bartesaghi(gamma, G, d, mu_n, mu_p, Vint):
    """Eq.K7  S23 的 θ；FF≥0.8 需 θ<1e-4（1 sun 模型）。"""
    return gamma * G * d**4 / (mu_n * mu_p * Vint**2)


def G_from_J(Jph, d):
    """Eq.K8  自由电荷体产生率 G = J_ph/(q d)。"""
    return Jph / (q * d)


def J_extraction_limit(phi_ex_eV, A_rich=1.2e6, T=300.0):
    """Eq.K9  抽取势垒能承载的最大电流 A*T²exp(−Φ_ex/kT)。"""
    return A_rich * T**2 * np.exp(-phi_ex_eV / kT(T))


def V_eff_with_IR(V0, V, Jph, A, Rs):
    """Eq.K10  串联电阻压降后的有效电压。"""
    return V0 + V - Jph * A * Rs


def depletion_width(eps_r, V, N):
    """Eq.K11  掺杂层耗尽宽度（D12 的场区）。N [1/m³]。"""
    return np.sqrt(2 * eps0 * eps_r * V / (q * N))


def P_trap_fill(N_it, lam, eta, tau_it):
    """Eq.K12  界面陷阱被填满的光强 [W/m²]，N_it [1/m²]。"""
    return N_it * h * c / (lam * eta * tau_it)


def f_trap_power(P, f0, P0, gamma_t):
    """Eq.K13  陷阱限制截止频率随光强的经验幂律。S27：γ_t≈1.25。"""
    return f0 * (P / P0) ** gamma_t


# ---------------------------------------------------------------- 带宽
def f_transit(mu_slow, Veff, d, kappa=3.5):
    """Eq.B1  渡越带宽 f_tr = κ μ V_eff /(2π d²)。"""
    return kappa * mu_slow * Veff / (2 * np.pi * d**2)


def f_RC(R, eps_r, A, d):
    """Eq.B2  RC 带宽 f_RC = d /(2π R ε A)。"""
    return d / (2 * np.pi * R * eps0 * eps_r * A)


def f_3dB(*fs):
    """Eq.B3  1/f² 相加（S28 式 4 推广，可加入陷阱项）。"""
    return 1 / np.sqrt(sum(1 / np.asarray(f, float) ** 2 for f in fs))


def d_transit_max(mu_slow, Veff, f_target, kappa=3.5):
    """Eq.B4  渡越上限厚度。"""
    return np.sqrt(kappa * mu_slow * Veff / (2 * np.pi * f_target))


def d_RC_min(R, eps_r, A, f_target):
    """Eq.B5  RC 下限厚度。"""
    return 2 * np.pi * R * eps0 * eps_r * A * f_target


# ---------------------------------------------------------------- 暗电流与噪声
def J_dark(d, Veff, J_inj0, phi_inj_eV, eps_r, g_bulk, R_sh_area=np.inf, T=300.0):
    """Eq.N1  J_d = 注入（Schottky 降势垒）+ 体产生 q·g·d + 分流。
    J_inj0 为 A*T² 前因子；g_bulk 为体热产生率 [1/m³s]（CT 项 + 陷阱项）。"""
    F = Veff / d
    dphi = np.sqrt(q * F / (4 * np.pi * eps0 * eps_r))  # eV
    J_inj = J_inj0 * np.exp(-(phi_inj_eV - dphi) / kT(T))
    return J_inj + q * g_bulk * d + Veff / R_sh_area


def d_dark(Veff, J_target, d_lo=10e-9, d_hi=20e-6, n=400, **kw):
    """Eq.N2  暗电流厚度：J_d(d) ≤ J* 的最小 d。J_d(d) 先随 d 降（注入）后随 d 升（体产生），
    所以先对数网格扫描找第一个满足点，再在相邻格点间求根；无解返回 nan。"""
    ds = np.geomspace(d_lo, d_hi, n)
    g = np.array([J_dark(x, Veff, **kw) for x in ds]) - J_target
    ok = np.nonzero(g <= 0)[0]
    if ok.size == 0:
        return np.nan
    k = ok[0]
    if k == 0:
        return ds[0]
    f = lambda x: J_dark(x, Veff, **kw) - J_target
    return brentq(f, ds[k - 1], ds[k])


def J0_reciprocity(J0_rad, EQE_EL):
    """Eq.N3  互易关系 J0 = J0,rad / EQE_EL（S1 检测率上限的依据）。"""
    return J0_rad / EQE_EL


def responsivity(EQE, lam):
    return EQE * q * lam / (h * c)


def D_star(EQE, lam, Jd, A, f=None, K_flicker=0.0, beta=2.0, R_sh=np.inf, T=300.0):
    """Eq.N4  D* = R√A / i_n，i_n² = 2qI_d + 4kT/R_sh + K·I_d^β/f（每 Hz）。"""
    Id = Jd * A
    noise = 2 * q * Id + 4 * kB * T / R_sh
    if K_flicker and f:
        noise = noise + K_flicker * Id**beta / f
    return responsivity(EQE, lam) * np.sqrt(A) / np.sqrt(noise)


def fowler_ipe(E_ph, phi_B, C_F=1.0):
    """Eq.N5  内光电发射 Fowler 产率（3.1-11）。"""
    return np.where(E_ph > phi_B, C_F * (E_ph - phi_B) ** 2 / E_ph, 0.0)


# ---------------------------------------------------------------- 反馈与转换
def eqe_feedback(EQE0, loop_gain):
    """Eq.F1  正反馈 EQE = EQE0/(1−g)，g≥1 时不稳定（3.1-9）。"""
    return np.where(loop_gain < 1, EQE0 / (1 - loop_gain), np.inf)


def eqe_conversion(A_conv, PLQY, eta_couple, EQE_em, EQE_direct):
    """Eq.F2  光谱转换层（3.1-12）。"""
    return A_conv * PLQY * eta_couple * EQE_em + (1 - A_conv) * EQE_direct


def eqe_tandem(EQE_top, T_top, EQE_bot, series=True):
    """Eq.F3  叠层（3.1-13）：串联取较小值，并联/三端相加。"""
    b = T_top * EQE_bot
    return np.minimum(EQE_top, b) if series else EQE_top + b


# ---------------------------------------------------------------- 重叠分类
def classify_overlap(mask_i, mask_j, axis_step, tol, k=3.0, shift_masks=None, same_variable=False):
    """Eq.O1  两个可行域的重叠类型。
    mask_*：同一网格上的布尔可行域；axis_step：沿判定轴的网格步长；
    tol：工艺/操作容差；shift_masks：使能参数扫描得到的 (mask_i', mask_j') 列表。"""
    if same_variable:
        return "same variable"
    inter = np.logical_and(mask_i, mask_j)
    if inter.any():
        # 沿第 0 轴的最大连续宽度
        w = 0
        for col in np.atleast_2d(inter).T:
            run = best = 0
            for v in col:
                run = run + 1 if v else 0
                best = max(best, run)
            w = max(w, best)
        return "wide" if w * axis_step >= k * tol else "narrow"
    if shift_masks:
        for mi, mj in shift_masks:
            if np.logical_and(mi, mj).any():
                return "shiftable"
    return "none"
